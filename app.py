"""RepoMind — Streamlit UI.

Paste a public GitHub repo URL, watch it index (in a background process so the
UI never blocks), then ask natural-language questions about how the project
evolved. Answers are grounded in retrieved GitHub evidence and verified by a
reference + NLI hallucination guard before display.
"""
from __future__ import annotations

import html
import re
import subprocess
import sys
import time
from pathlib import Path

import streamlit as st

import config
import query_pipeline
import telemetry
from core import manifest as manifest_mod
from core import registry
from core.context import RepositoryContext
from core.repo_url import InvalidRepoURL, parse_repo_url
from jobs import status as status_mod

PROJECT_ROOT = Path(__file__).resolve().parent
ACTIVE_STATES = {"pending", "fetching", "chunking", "linking", "embedding", "indexing"}

# --------------------------------------------------------------------------- #
# Example questions, per repository
#
# Three chips, and each one proves a different thing:
#   1. a grounded "why"  — real citations, guard passes
#   2. a multi-hop question — needs the issue -> PR -> commit graph, which is
#      the part a general-purpose model with a repo URL cannot do
#   3. one the system should DECLINE — a plausible feature the project never
#      had. Refusing that is the product, so it gets a permanent slot.
#
# These are not invented. The two answerable ones are questions this repo's own
# evaluation recorded as faithful with clean citations (see
# results/eval-<slug>/results.json); the third comes from
# eval/datasets/<slug>_abstention.jsonl, where it is annotated "verified absent
# via retrieval".
#
# They are therefore tied to the CURRENTLY INDEXED SNAPSHOT. Re-index a repo
# over a different date window and an answerable one may stop being answerable
# — re-pick from those same two files rather than guessing a replacement.
EXAMPLE_QUESTIONS: dict[str, tuple[str, str, str]] = {
    "acme_widgets": (
        "What fixed the startup crash?",
        "Why was the caching layer added?",
        "Why was the DynamicThemeSwitcher feature removed?",
    ),
    "fastapi_fastapi": (
        "Why were benchmark tests excluded from the coverage check in PR #14965?",
        "Why does computed fields support break when using mixed route types "
        "in FastAPI?",
        "Why was the `starlette_extras` plugin bundled with FastAPI by default?",
    ),
    "pallets_click": (
        "Why is Click considering dropping support for Colorama?",
        "Why does `python foo.py` return True in Click 8.3.0 instead of False?",
        "Why was the click Echo class renamed to Print?",
    ),
    "psf_black": (
        "Why was conditional stripping added for Linux executables?",
        "Why did Black incorrectly parse multi-line code that uses backslash "
        "(`\\`) line continuations without indentation?",
        "Why was colored output mode added to the CLI?",
    ),
    "psf_requests": (
        "Why was the chardet upper limit increased to 7?",
        "Why was the `OSError` changed to `FileNotFoundError` in the code "
        "related to missing TLS material?",
        "Why was the `http2` protocol support removed from the core library?",
    ),
    "pydantic_pydantic": (
        "Why was the `ascii_only` option added to `StringConstraints` and what "
        "problem was it intended to solve?",
        "Why was `require-runtime-dependencies = true` added to the "
        "`[tool.hatch.build.targets.wheel]` section in PR #13037?",
        "Why was `field_validator_v3` added?",
    ),
}


def _generic_questions(manifest: dict) -> tuple[str, str, str]:
    """Chips for a repository nobody has curated questions for.

    Any pasted repo has to work without a code change, so these are built from
    the manifest instead of a lookup. The third asks about the year *before*
    coverage begins, which makes it unanswerable by construction — no model
    call and no guesswork needed to keep the decline slot honest.
    """
    coverage = manifest.get("coverage", {}) or {}
    since, until = (coverage.get("since") or "")[:10], (coverage.get("until") or "")[:10]
    first = (
        f"What was the most significant change between {since} and {until}, "
        f"and why was it made?" if since and until else
        "What was the most significant recent change, and why was it made?"
    )
    second = ("Which issue prompted the most recent merged pull request, and "
              "what did that pull request change?")
    try:
        before = int(since[:4]) - 1
    except ValueError:
        # No usable coverage date: fall back to a topic the guard must decline
        # on evidence rather than on dates.
        return first, second, "Why was the experimental plugin loader removed?"
    return first, second, f"What changed in this project during {before}?"


def example_questions(slug: str, manifest: dict) -> tuple[str, str, str]:
    """The three chips for `slug`, falling back to manifest-derived ones."""
    return EXAMPLE_QUESTIONS.get(slug) or _generic_questions(manifest)

# Evaluation panel (reads/writes eval/run.py output; never imports eval/ code —
# a separate process is launched via subprocess, keeping the live app's import
# graph eval-free, same boundary rule the rest of Phase 2 follows).
EVAL_METRICS_DEFAULT = ("faithfulness,answer_relevancy,citation_precision,"
                        "citation_recall,recall_at_k,mrr,ndcg")
MATRIX_COLUMNS = ["query_type", "n", "recall_at_k", "mrr", "ndcg",
                  "citation_precision", "citation_recall", "faithfulness",
                  "answer_relevancy", "abstention_accuracy", "latency_ms_mean"]

st.set_page_config(page_title="RepoMind", page_icon="🧠", layout="wide")


# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
_CSS = """
<style>
/* ========================================================================== *
 * RepoMind — glass design system
 *
 * Surfaces are translucent and blurred (backdrop-filter) over an ambient
 * gradient field, so panels read as layers of glass rather than flat boxes.
 * Streamlit's own class names churn between releases, so structural selectors
 * use data-testid where possible; anything that stops matching after an
 * upgrade degrades to Streamlit's default styling rather than breaking.
 * ========================================================================== */

:root {
  --rm-accent:      #7c6ff0;
  --rm-accent-2:    #c4b5fd;
  --rm-text:        #e8eaf0;
  --rm-muted:       #93a0b4;
  --rm-glass:       rgba(255,255,255,.045);
  --rm-glass-hi:    rgba(255,255,255,.075);
  --rm-stroke:      rgba(255,255,255,.09);
  --rm-stroke-hi:   rgba(255,255,255,.16);
  --rm-blur:        saturate(150%) blur(18px);
  --rm-shadow:      0 8px 32px rgba(0,0,0,.38);
  --rm-shadow-lift: 0 14px 42px rgba(0,0,0,.5);
  --rm-ease:        cubic-bezier(.22,1,.36,1);
}

/* --- ambient background: two soft colour fields behind everything -------- */
.stApp {
  background:
    radial-gradient(900px 520px at 12% -8%,  rgba(124,111,240,.16), transparent 60%),
    radial-gradient(780px 460px at 92% 4%,   rgba(56,189,248,.10),  transparent 62%),
    radial-gradient(700px 700px at 50% 115%, rgba(167,139,250,.10), transparent 60%),
    #0a0d14;
  background-attachment: fixed;
}

.block-container {padding-top: 2.1rem; max-width: 1120px;}

/* Respect users who prefer reduced motion — disable transitions wholesale. */
@media (prefers-reduced-motion: reduce) {
  * { animation: none !important; transition: none !important; }
}

/* --- hero ---------------------------------------------------------------- */
.rm-hero {display:flex; align-items:center; gap:.75rem; margin-bottom:.1rem;}
.rm-hero .emoji {
  font-size:2.5rem; line-height:1;
  filter: drop-shadow(0 4px 16px rgba(124,111,240,.55));
}
.rm-hero h1 {
  margin:0; font-size:2.5rem; font-weight:800; letter-spacing:-.03em;
  background:linear-gradient(100deg,#a99bff 0%,#e9e3ff 42%,#7dd3fc 78%,#a99bff 100%);
  background-size:220% auto;
  -webkit-background-clip:text; background-clip:text;
  -webkit-text-fill-color:transparent;
  animation: rm-sheen 9s linear infinite;
}
@keyframes rm-sheen {to {background-position:220% center;}}
.rm-sub {color:var(--rm-muted); margin:.2rem 0 1.15rem; font-size:1.03rem;}

/* --- coverage chip ------------------------------------------------------- */
.rm-cov {
  display:inline-block; padding:.4rem .9rem; border-radius:999px;
  background:var(--rm-glass); border:1px solid var(--rm-stroke);
  -webkit-backdrop-filter:var(--rm-blur); backdrop-filter:var(--rm-blur);
  color:#cdd4de; font-size:.88rem; box-shadow:var(--rm-shadow);
}

/* --- stat cards ---------------------------------------------------------- */
/* Grid, not flex: with flex the final card on a wrapped row stretches to the
   full width (6 cards on a 3-wide layout looked broken). auto-fill keeps the
   empty tracks so every card stays the same size however many there are. */
.rm-stats {
  display:grid; gap:.8rem; margin:1rem 0 .3rem;
  grid-template-columns:repeat(auto-fill, minmax(158px, 1fr));
}
.rm-card {
  position:relative; overflow:hidden;
  padding:.95rem 1.05rem; border-radius:16px;
  background:linear-gradient(160deg, var(--rm-glass-hi), var(--rm-glass));
  border:1px solid var(--rm-stroke);
  -webkit-backdrop-filter:var(--rm-blur); backdrop-filter:var(--rm-blur);
  box-shadow:var(--rm-shadow);
  transition:transform .35s var(--rm-ease), box-shadow .35s var(--rm-ease),
             border-color .35s var(--rm-ease);
}
/* specular highlight along the top edge — the "glass" tell */
.rm-card::before {
  content:""; position:absolute; inset:0 0 auto 0; height:1px;
  background:linear-gradient(90deg,transparent,rgba(255,255,255,.5),transparent);
}
.rm-card:hover {
  transform:translateY(-3px);
  box-shadow:var(--rm-shadow-lift);
  border-color:var(--rm-stroke-hi);
}
.rm-card .v {
  font-size:1.7rem; font-weight:760; line-height:1.05;
  background:linear-gradient(180deg,#ffffff,#b9c2d4);
  -webkit-background-clip:text; background-clip:text;
  -webkit-text-fill-color:transparent;
}
.rm-card .l {
  font-size:.7rem; color:var(--rm-muted); margin-top:.3rem;
  text-transform:uppercase; letter-spacing:.08em; font-weight:600;
}

/* --- guard pills --------------------------------------------------------- */
.rm-pills {display:flex; gap:.55rem; flex-wrap:wrap; margin:.55rem 0 .35rem;}
.rm-pill {
  display:inline-flex; align-items:center; gap:.4rem;
  padding:.38rem .85rem; border-radius:999px;
  font-size:.83rem; font-weight:650; letter-spacing:.01em;
  border:1px solid transparent;
  -webkit-backdrop-filter:blur(10px); backdrop-filter:blur(10px);
  animation: rm-rise .45s var(--rm-ease) both;
}
@keyframes rm-rise {from {opacity:0; transform:translateY(6px);} to {opacity:1;}}
.rm-ok {
  background:linear-gradient(180deg,rgba(34,197,94,.20),rgba(34,197,94,.10));
  color:#6ee7a4; border-color:rgba(34,197,94,.38);
  box-shadow:0 0 18px rgba(34,197,94,.14);
}
.rm-warn {
  background:linear-gradient(180deg,rgba(245,158,11,.20),rgba(245,158,11,.10));
  color:#fcd34d; border-color:rgba(245,158,11,.38);
  box-shadow:0 0 18px rgba(245,158,11,.14);
}
.rm-bad {
  background:linear-gradient(180deg,rgba(239,68,68,.20),rgba(239,68,68,.10));
  color:#fca5a5; border-color:rgba(239,68,68,.40);
  box-shadow:0 0 18px rgba(239,68,68,.16);
}
.rm-neu {
  background:var(--rm-glass); color:#c2cad6; border-color:var(--rm-stroke);
}

/* --- source-type badges -------------------------------------------------- */
.rm-src {
  display:inline-block; padding:.1rem .55rem; border-radius:7px;
  font-size:.68rem; font-weight:750; text-transform:uppercase;
  letter-spacing:.05em; margin-right:.5rem; vertical-align:middle;
  border:1px solid transparent;
}
.rm-src-commit  {background:rgba(16,185,129,.16); color:#6ee7b7; border-color:rgba(16,185,129,.3);}
.rm-src-pr      {background:rgba(139,124,246,.18); color:#c4b5fd; border-color:rgba(139,124,246,.34);}
.rm-src-issue   {background:rgba(248,113,113,.15); color:#fca5a5; border-color:rgba(248,113,113,.3);}
.rm-src-review  {background:rgba(56,189,248,.15);  color:#7dd3fc; border-color:rgba(56,189,248,.3);}
.rm-src-release {background:rgba(251,191,36,.15);  color:#fcd34d; border-color:rgba(251,191,36,.3);}

/* ========================================================================== *
 * Streamlit widget restyling
 * ========================================================================== */

/* bordered containers (the answer panel) */
[data-testid="stVerticalBlockBorderWrapper"]:has(> div > div > [data-testid="stMarkdownContainer"]) {
  border-radius:18px;
}
div[data-testid="stVerticalBlockBorderWrapper"] {
  background:linear-gradient(160deg, var(--rm-glass-hi), var(--rm-glass));
  border:1px solid var(--rm-stroke) !important;
  border-radius:18px !important;
  -webkit-backdrop-filter:var(--rm-blur); backdrop-filter:var(--rm-blur);
  box-shadow:var(--rm-shadow);
}

/* buttons */
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {
  border-radius:11px; font-weight:620; letter-spacing:.01em;
  background:var(--rm-glass); border:1px solid var(--rm-stroke);
  color:var(--rm-text);
  -webkit-backdrop-filter:blur(10px); backdrop-filter:blur(10px);
  transition:transform .2s var(--rm-ease), box-shadow .25s var(--rm-ease),
             border-color .25s var(--rm-ease), background .25s var(--rm-ease);
}
.stButton > button:hover, .stDownloadButton > button:hover,
.stFormSubmitButton > button:hover {
  transform:translateY(-2px); border-color:var(--rm-stroke-hi);
  background:var(--rm-glass-hi);
  box-shadow:0 10px 26px rgba(0,0,0,.4);
}
.stButton > button:active {transform:translateY(0);}
/* primary action gets the accent gradient */
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] {
  background:linear-gradient(135deg,#7c6ff0,#9d7bf5);
  border-color:rgba(196,181,253,.45); color:#fff;
  box-shadow:0 8px 24px rgba(124,111,240,.36);
}
.stButton > button[kind="primary"]:hover,
.stFormSubmitButton > button[kind="primary"]:hover {
  box-shadow:0 12px 32px rgba(124,111,240,.5);
}

/* example-question chips
 *
 * Three real questions side by side. Streamlit buttons are single-line and
 * centred by default, which truncates them and leaves three ragged heights;
 * these wrap, read left-to-right like text, and stretch to a shared height so
 * the row stays a row. */
.st-key-example_chips [data-testid="stHorizontalBlock"] {align-items:stretch;}
/* Carry the row's stretch all the way down to the button. `height:100%` does
 * not work here — the row's height is set by its tallest child, so the
 * percentage is indefinite and resolves to auto. Each wrapper has to grow
 * instead. */
.st-key-example_chips [data-testid="stColumn"] > div,
.st-key-example_chips [data-testid="stColumn"] [data-testid="stVerticalBlock"],
.st-key-example_chips [data-testid="stElementContainer"],
.st-key-example_chips .stButton {
  display:flex; flex-direction:column; flex:1 1 auto;
}
.st-key-example_chips .stButton > button {
  flex:1 1 auto; width:100%;
  white-space:normal; text-align:left;
  align-items:flex-start; justify-content:flex-start;
  min-height:5.4rem; padding:.85rem 1rem;
  font-size:.85rem; font-weight:560; line-height:1.5;
}
.st-key-example_chips .stButton > button p {
  text-align:left; margin:0; line-height:1.5;
}
/* inline `code` inside a chip must not blow the line width open */
.st-key-example_chips .stButton > button code {
  font-size:.8em; padding:.05rem .28rem; white-space:normal;
  overflow-wrap:anywhere;
}

/* text inputs */
.stTextInput input, .stNumberInput input, .stTextArea textarea {
  background:rgba(255,255,255,.04) !important;
  border:1px solid var(--rm-stroke) !important;
  border-radius:11px !important; color:var(--rm-text) !important;
  transition:border-color .25s var(--rm-ease), box-shadow .25s var(--rm-ease);
}
.stTextInput input:focus, .stNumberInput input:focus, .stTextArea textarea:focus {
  border-color:rgba(124,111,240,.6) !important;
  box-shadow:0 0 0 3px rgba(124,111,240,.16) !important;
}

/* expanders */
[data-testid="stExpander"] details {
  background:var(--rm-glass); border:1px solid var(--rm-stroke) !important;
  border-radius:14px !important; overflow:hidden;
  -webkit-backdrop-filter:blur(12px); backdrop-filter:blur(12px);
  transition:border-color .25s var(--rm-ease), box-shadow .25s var(--rm-ease);
}
[data-testid="stExpander"] details:hover {
  border-color:var(--rm-stroke-hi); box-shadow:0 8px 26px rgba(0,0,0,.32);
}
[data-testid="stExpander"] summary {font-weight:620;}

/* sidebar as a frosted rail */
[data-testid="stSidebar"] {
  background:rgba(12,16,25,.72);
  -webkit-backdrop-filter:var(--rm-blur); backdrop-filter:var(--rm-blur);
  border-right:1px solid var(--rm-stroke);
}

/* the top header is opaque by default and hides the ambient gradient */
[data-testid="stHeader"] {
  background:transparent !important;
  -webkit-backdrop-filter:blur(10px); backdrop-filter:blur(10px);
}

/* progress bar */
.stProgress > div > div > div > div {
  background:linear-gradient(90deg,#7c6ff0,#a99bff,#7dd3fc);
  background-size:200% auto; animation:rm-sheen 2.2s linear infinite;
}

/* metrics */
[data-testid="stMetric"] {
  background:var(--rm-glass); border:1px solid var(--rm-stroke);
  border-radius:14px; padding:.7rem .9rem;
  -webkit-backdrop-filter:blur(12px); backdrop-filter:blur(12px);
}

/* tables + alerts */
[data-testid="stTable"], .stDataFrame {border-radius:14px; overflow:hidden;}
[data-testid="stAlert"] {
  border-radius:13px; border:1px solid var(--rm-stroke);
  -webkit-backdrop-filter:blur(10px); backdrop-filter:blur(10px);
}

/* code + citation links */
code {
  background:rgba(124,111,240,.13) !important;
  border:1px solid rgba(124,111,240,.22);
  border-radius:6px; padding:.08rem .35rem !important;
  color:#cfc6ff !important;
}
a {transition:opacity .2s var(--rm-ease);} a:hover {opacity:.78;}

/* thin, unobtrusive scrollbar */
::-webkit-scrollbar {width:10px; height:10px;}
::-webkit-scrollbar-track {background:transparent;}
::-webkit-scrollbar-thumb {
  background:rgba(255,255,255,.12); border-radius:99px;
  border:2px solid transparent; background-clip:content-box;
}
::-webkit-scrollbar-thumb:hover {background:rgba(255,255,255,.2); background-clip:content-box;}
</style>
"""


def _pill(text: str, kind: str) -> str:
    return f"<span class='rm-pill rm-{kind}'>{html.escape(text)}</span>"


def _src_badge(stype: str) -> str:
    known = {"commit", "pr", "issue", "review", "release"}
    cls = stype if stype in known else "neu"
    return f"<span class='rm-src rm-src-{cls}'>{html.escape(stype)}</span>"


# --------------------------------------------------------------------------- #
# Cached heavy resources (per repository slug + manifest version)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def _get_retriever(slug: str, version: str):
    from index.embedder import OllamaEmbedder
    from retrieval.retriever import Retriever

    ctx = _ctx_for_slug(slug)
    return Retriever(ctx, embedder=OllamaEmbedder())


@st.cache_resource(show_spinner=False)
def _get_answerer():
    from generation.answerer import Answerer

    return Answerer()


@st.cache_resource(show_spinner=False)
def _get_nli():
    from guard.nli_verifier import NLIVerifier

    return NLIVerifier()


def _ctx_for_slug(slug: str) -> RepositoryContext:
    owner, name = slug.split("_", 1)
    from core.repo_url import RepoRef

    return RepositoryContext(
        ref=RepoRef(owner, name), base_dir=config.REPOSITORIES_DIR / slug
    )


# --------------------------------------------------------------------------- #
# Ingestion control
# --------------------------------------------------------------------------- #
def start_ingestion(repo_input: str, months: int) -> None:
    """Spawn the background runner process (non-blocking)."""
    proc = subprocess.Popen(
        [sys.executable, "-m", "jobs.runner", repo_input, str(months)],
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    st.session_state["job_pid"] = proc.pid


def render_progress(ctx: RepositoryContext) -> str | None:
    """Render the live progress bar. Returns the current stage."""
    status = status_mod.status_for(ctx)
    if not status:
        st.info("Starting…")
        return "pending"
    stage = status.get("stage", "pending")
    st.progress(float(status.get("progress", 0.0)),
                text=f"**{stage.title()}** — {status.get('message', '')}")
    live = status.get("live_counts") or {}
    if live:
        cols = st.columns(4)
        for col, key in zip(cols, ("commits", "prs", "issues", "releases")):
            col.metric(key.title(), live.get(key, 0))
    for w in status.get("warnings", []) or []:
        st.warning(w)
    if status.get("error"):
        st.error(status["error"])
    return stage


# --------------------------------------------------------------------------- #
# Q&A
# --------------------------------------------------------------------------- #
_CITE_RE = re.compile(r"\[([A-Za-z0-9_#\-]+)\]")


def linkify_citations(text: str, chunks: list[dict]) -> str:
    """Replace [chunk_id] with a markdown link to the real GitHub URL."""
    url_by_id = {c["chunk_id"]: c.get("github_url", "") for c in chunks}
    base_url = {c["chunk_id"].split("#", 1)[0]: c.get("github_url", "") for c in chunks}

    def _sub(m):
        cid = m.group(1)
        url = url_by_id.get(cid) or base_url.get(cid.split("#", 1)[0])
        return f"[`{cid}`]({url})" if url else f"`{cid}`"

    return _CITE_RE.sub(_sub, text)


def _year_out_of_range(question: str, since: str, until: str) -> str | None:
    """Heuristic warning if the question names a year outside coverage."""
    years = [int(y) for y in re.findall(r"\b((?:19|20)\d{2})\b", question)]
    if not years or not (since and until):
        return None
    lo, hi = int(since[:4]), int(until[:4])
    outside = [y for y in years if y < lo or y > hi]
    if outside:
        return (f"Your question mentions {outside}, outside the indexed window "
                f"({lo}–{hi}). The answer will flag this rather than guess.")
    return None


def _set_example(q: str) -> None:
    st.session_state["question_input"] = q
    st.session_state["auto_ask"] = True


def _citation_pill_text(ans: dict) -> str:
    """Label for the citation badge.

    Counts the citations the answer actually makes, not the chunks that were
    retrieved. These are not the same number and the difference is not small:
    an answer citing one PR out of a six-chunk evidence set used to read
    "6 citations verified", overstating by 6x the single number a viewer reads
    to decide whether to trust the answer.
    """
    if not ans["citations_ok"]:
        n = len(ans["invalid_citations"])
        return f"✗ {n} fabricated citation{'' if n == 1 else 's'}"
    n = len(ans.get("valid_citations", []))
    return f"✓ {n} citation{'' if n == 1 else 's'} verified"


def _friendly_error(exc: Exception) -> str:
    """Turn an exception into something a viewer can act on.

    Each case below is one that actually happened during development.
    """
    text = str(exc)
    name = type(exc).__name__

    if name == "AttributeError" and "PipelineResult" in text:
        return ("The app is running stale code — Streamlit re-runs app.py on "
                "save but does not reload imported modules. Stop it with Ctrl-C "
                "and start it again.")
    if "Could not reach Ollama" in text or "Connection refused" in text:
        return ("Ollama is not running. Start it (`open -a Ollama`) and ask "
                "again — nothing needs re-indexing.")
    if "not found" in text.lower() and "model" in text.lower():
        return ("A local model is missing. Run `ollama pull "
                f"{config.EMBEDDING_MODEL}` and `ollama pull "
                f"{config.GENERATION_MODEL}`.")
    if "rate limit" in text.lower() or "429" in text:
        return ("A provider is rate-limited. The pipeline normally falls back "
                "to the local model automatically; try again in a moment.")
    return f"{name}: {text[:200]}"


def _log_query_error(slug: str, question: str, exc: Exception) -> None:
    """Record the failure so it is not lost when the UI shows a tidy message."""
    telemetry.record_query({
        "repo": slug, "question": question, "error": type(exc).__name__,
        "error_detail": str(exc)[:300],
    })


def _run_qa(ctx, question, since, until) -> None:
    """Run the answer pipeline (with optional adaptive retry) and store it."""
    warn = _year_out_of_range(question, since, until)
    version = ctx.manifest_path.stat().st_mtime if ctx.manifest_path.exists() else ""

    trace: dict = {} if config.ENABLE_METRICS_LOGGING else None
    _q0 = time.perf_counter()

    retriever = _get_retriever(ctx.slug, str(version))
    answerer = _get_answerer()
    nli = _get_nli()
    try:
        with st.spinner("Retrieving, answering, and verifying…"):
            pr = query_pipeline.answer_query(
                retriever, answerer, nli, question, since, until, trace=trace)
    except Exception as exc:  # noqa: BLE001 - never show a traceback on screen
        # A stack trace in front of an audience is the worst possible failure
        # mode for a demo. Show what went wrong and what to do about it; keep
        # the detail available but folded away.
        _log_query_error(ctx.slug, question, exc)
        st.error(f"**Could not answer that question.** {_friendly_error(exc)}")
        with st.expander("Technical detail"):
            st.exception(exc)
        return

    if pr.empty:
        st.session_state["last_answer"] = {
            "slug": ctx.slug, "question": question, "empty": True, "warn": warn}
        telemetry.record_query({
            "repo": ctx.slug, "question": question, "empty": True,
            "total_ms": (time.perf_counter() - _q0) * 1000.0,
            **(trace or {}),
        })
        return

    ref_report, nli_report = pr.ref_report, pr.nli_report
    answer = {
        "slug": ctx.slug,
        "question": question,
        "empty": False,
        "warn": warn,
        "text": pr.text,
        "chunks": pr.chunks,
        "invalid_citations": ref_report.invalid_citations,
        "valid_citations": ref_report.valid_citations,
        "citations_ok": ref_report.is_valid,
        "grounded": nli_report.is_grounded,
        "contradicted": [(c.claim, c.contradiction) for c in nli_report.contradicted],
        "n_unverified": len(nli_report.unverified),
        # Phase B: adaptive-retry outcome (all False on the happy path / flag off).
        "retry_attempted": pr.retry_attempted,
        "retry_reason": pr.retry_reason,
        "retry_succeeded": pr.retry_succeeded,
        "refusal": pr.refusal,
        # getattr, not attribute access: Streamlit re-runs app.py on save
        # without reloading imported modules, so a newly-added field can be
        # missing from an in-memory PipelineResult. Degrade, don't crash.
        "declined": getattr(pr, "declined", False),
        "model": pr.model,
        "fell_back": pr.fell_back,
        "fallback_reason": pr.fallback_reason,
    }
    st.session_state["last_answer"] = answer
    # Keep a short per-session history (most recent first, capped at 5).
    hist = st.session_state.setdefault("history", [])
    hist.insert(0, answer)
    del hist[5:]

    n_cited = len(ref_report.valid_citations) + len(ref_report.invalid_citations)
    telemetry.record_query({
        "repo": ctx.slug,
        "question": question,
        "empty": False,
        "generation_ms": pr.generation_ms,
        "guard_ms": pr.guard_ms,
        "total_ms": (time.perf_counter() - _q0) * 1000.0,
        "guard_verdict": "pass" if pr.guard_pass else "fail",
        "guard_reason": query_pipeline.guard_reason(ref_report, nli_report),
        "num_citations": n_cited,
        "num_valid_citations": len(ref_report.valid_citations),
        "num_contradicted": len(nli_report.contradicted),
        "num_unverified": len(nli_report.unverified),
        "retry_attempted": pr.retry_attempted,
        "retry_reason": pr.retry_reason,
        "retry_succeeded": pr.retry_succeeded,
        **(trace or {}),
    })


def _answer_markdown(ans: dict) -> str:
    """Serialize an answer + its citations to Markdown (for download)."""
    lines = [
        f"# RepoMind answer — {ans['slug'].replace('_', '/')}",
        "", f"**Question:** {ans['question']}", "", "## Answer", "",
        ans["text"], "",
        f"- Citations verified: {'yes' if ans['citations_ok'] else 'NO'}",
        f"- Grounded (NLI): {'yes' if ans['grounded'] else 'flagged'}",
        f"- Unverified claims: {ans['n_unverified']}", "",
        "## Evidence", "",
    ]
    for c in ans["chunks"]:
        lines.append(f"- **{c['chunk_id']}** ({c['source_type']}, "
                     f"{c.get('date', '')[:10]}) — {c.get('github_url', '')}")
    return "\n".join(lines)


def _render_answer(ans: dict) -> None:
    if ans.get("warn"):
        st.warning(ans["warn"])
    if ans.get("empty"):
        st.info("No evidence matched this question inside the indexed window. "
                "Try rephrasing, or widen the date range when indexing.")
        return

    chunks = ans["chunks"]

    # Phase B: an honest refusal after a failed retry — never shown as a
    # verified answer. Show the refusal text and what was retrieved, nothing else.
    if ans.get("refusal"):
        st.markdown(
            "<div class='rm-pills'>"
            f"{_pill('🔁 answer regenerated after guard rejection', 'warn')}"
            f"{_pill('⚠ could not verify — declined to answer', 'bad')}</div>",
            unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown("#### 🚫 No verified answer")
            st.markdown(ans["text"])
        with st.expander(f"🔎 Evidence retrieved ({len(chunks)} chunks)"):
            for c in chunks:
                url = c.get("github_url", "")
                st.markdown(
                    f"{_src_badge(c['source_type'])} **`{c['chunk_id']}`** · "
                    f"{c.get('date', '')[:10]}  \n[{url}]({url})",
                    unsafe_allow_html=True)
                st.caption(c["text"][:360] + ("…" if len(c["text"]) > 360 else ""))
        return

    # An answer that is itself "there is no evidence for that" is a NON-answer.
    # Its citations are real and its claim is entailed, so the guard passes it —
    # but showing green "verified" badges on it reads as success and misleads.
    # Present it as what it is: nothing found, honestly reported.
    declined = ans.get("declined") and not ans.get("refusal")

    with st.container(border=True):
        st.markdown("#### 🔍 No answer found" if declined else "#### 💬 Answer")
        if declined:
            st.caption("The indexed history contains nothing that answers this. "
                       "The evidence below was searched and did not match.")
        st.markdown(linkify_citations(ans["text"], chunks))

    # Guard pills.
    if declined:
        # The guard verdict is still true, but it is about the *declination*,
        # not about an answer — so it must not be presented as a green tick.
        cite_pill = _pill(f"{len(chunks)} chunks searched, none matched", "neu")
        ground_pill = _pill("✓ declined honestly — no unsupported claim", "warn")
    else:
        cite_pill = _pill(_citation_pill_text(ans),
                          "ok" if ans["citations_ok"] else "bad")
        ground_pill = _pill(
            "✓ Grounded (NLI)" if ans["grounded"]
            else f"⚠ {len(ans['contradicted'])} claim(s) contradict evidence",
            "ok" if ans["grounded"] else "warn",
        )
    unv_pill = _pill(f"{ans['n_unverified']} unverified claim(s)",
                     "neu" if ans["n_unverified"] == 0 else "warn")
    retry_pill = (_pill("🔁 answer regenerated after guard rejection", "warn")
                  if ans.get("retry_attempted") else "")
    fallback_pill = (_pill("☁️→💻 cloud unavailable, answered locally", "warn")
                     if ans.get("fell_back") else "")
    st.markdown(
        f"<div class='rm-pills'>{fallback_pill}{retry_pill}{cite_pill}"
        f"{ground_pill}{unv_pill}</div>",
        unsafe_allow_html=True,
    )

    if ans["invalid_citations"]:
        st.error(f"Fabricated citations (not in retrieved evidence): "
                 f"{ans['invalid_citations']}")
    for claim, score in ans["contradicted"]:
        st.error(f"Claim contradicts its cited evidence (p={score:.2f}): _{claim}_")

    st.download_button(
        "⬇️ Export answer (Markdown)",
        data=_answer_markdown(ans),
        file_name=f"repomind_{ans['slug']}_answer.md",
        mime="text/markdown",
    )

    with st.expander(f"🔎 Evidence used ({len(chunks)} chunks)"):
        for c in chunks:
            score = c.get("rerank_score")
            score_txt = f" · relevance {score:.2f}" if score is not None else ""
            url = c.get("github_url", "")
            st.markdown(
                f"{_src_badge(c['source_type'])} **`{c['chunk_id']}`** · "
                f"{c.get('date', '')[:10]}{score_txt}  \n[{url}]({url})",
                unsafe_allow_html=True,
            )
            st.caption(c["text"][:360] + ("…" if len(c["text"]) > 360 else ""))


def answer_panel(ctx: RepositoryContext, manifest: dict) -> None:
    coverage = manifest.get("coverage", {})
    since, until = coverage.get("since", ""), coverage.get("until", "")

    st.markdown("#### Ask about how this repository evolved")
    examples = example_questions(ctx.slug, manifest)
    st.caption("Try an example — the last one has no answer in the indexed "
               "evidence, and RepoMind should decline it rather than guess.")
    # Side by side. Real questions are long, so the chips are styled to wrap
    # and share one height rather than truncate — see `.st-key-example_chips`.
    with st.container(key="example_chips"):
        chip_cols = st.columns(len(examples), gap="small",
                               vertical_alignment="top")
        for col, ex in zip(chip_cols, examples):
            col.button(ex, key=f"ex_{ex}", on_click=_set_example, args=(ex,),
                       width="stretch")

    question = st.text_input(
        "Your question", key="question_input", label_visibility="collapsed",
        placeholder="Why did the maintainers make this change?",
    )
    ask = st.button("Ask", type="primary", disabled=not question.strip())
    auto = st.session_state.pop("auto_ask", False)

    if (ask or auto) and question.strip():
        _run_qa(ctx, question.strip(), since, until)

    ans = st.session_state.get("last_answer")
    if ans and ans.get("slug") == ctx.slug:
        _render_answer(ans)

    _render_history(ctx.slug)


def _render_history(slug: str) -> None:
    """Show the last few questions asked this session for the active repo."""
    hist = [h for h in st.session_state.get("history", [])
            if h.get("slug") == slug and not h.get("empty")]
    # Skip the first (already shown as the current answer).
    past = hist[1:]
    if not past:
        return
    with st.expander(f"🕑 Recent questions ({len(past)})"):
        for h in past:
            badge = "✅" if (h["citations_ok"] and h["grounded"]) else "⚠️"
            st.markdown(f"**{badge} {h['question']}**")
            st.caption(h["text"][:220] + ("…" if len(h["text"]) > 220 else ""))
            st.divider()


def relationship_graph_section(ctx: RepositoryContext) -> None:
    """Render the issue↔PR↔commit↔release graph from links.json."""
    if not ctx.links_path.exists():
        return
    import json

    from process import linker

    graph = json.loads(ctx.links_path.read_text(encoding="utf-8"))
    n_links = linker.count_closes_links(graph)
    if n_links == 0:
        return
    with st.expander(f"🕸️ Evolution graph — {n_links} issue↔PR↔commit links"):
        st.caption("How issues were resolved: which PR closed them, which "
                   "commits implemented the fix, and which release shipped it.")
        # Render a bounded sample, not the whole graph. Unbounded, a large repo
        # produced 100-200 nodes laid out left-to-right, which is unreadable on
        # screen. Say so plainly rather than letting it look like the full picture.
        n_shown = min(6, sum(1 for i in graph.get("issues", {}).values()
                             if i.get("closed_by_prs") or i.get("closed_by_commits")))
        st.caption(f"Showing the {n_shown} most-connected issues of {n_links} "
                   f"links — enough to read at a glance.")
        st.graphviz_chart(linker.to_dot(graph), width="stretch")


# --------------------------------------------------------------------------- #
# Stats
# --------------------------------------------------------------------------- #
def stats_panel(manifest: dict) -> None:
    cov = manifest.get("coverage", {})
    stats = manifest.get("stats", {})
    size_mb = stats.get("index_bytes", 0) / (1024 * 1024)
    cards = [
        ("Commits", cov.get("commits", {}).get("count", 0)),
        ("Pull Requests", cov.get("prs", {}).get("count", 0)),
        ("Issues", cov.get("issues", {}).get("count", 0)),
        ("Contributors", stats.get("contributors", 0)),
        ("Chunks", stats.get("chunks", 0)),
        ("Index size", f"{size_mb:.1f} MB"),
    ]
    html_cards = "".join(
        f"<div class='rm-card'><div class='v'>{v}</div>"
        f"<div class='l'>{label}</div></div>"
        for label, v in cards
    )
    st.markdown(f"<div class='rm-stats'>{html_cards}</div>", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def _delete_repo(slug: str, display_name: str) -> None:
    """Delete an indexed repository and clear anything still pointing at it."""
    try:
        freed = registry.delete_repository(slug)
    except Exception as exc:  # noqa: BLE001 - surface the reason, never crash
        st.session_state.pop("confirm_delete", None)
        st.error(f"Could not delete {display_name}: {exc}")
        return

    # Drop cached retriever/answer state so nothing references the deleted index.
    _get_retriever.clear()
    st.session_state.pop("confirm_delete", None)
    if st.session_state.get("active_slug") == slug:
        st.session_state.pop("active_slug", None)
    st.session_state.pop("last_answer", None)
    st.session_state["history"] = [
        h for h in st.session_state.get("history", []) if h.get("slug") != slug
    ]
    st.toast(f"Deleted {display_name} ({freed / 1e6:.1f} MB freed)", icon="🗑️")
    st.rerun()


def _sidebar() -> None:
    with st.sidebar:
        st.markdown("### 📚 Indexed repositories")
        entries = registry.list_repositories()
        if entries:
            labels = {}
            for e in entries:
                dot = {"ready": "🟢", "failed": "🔴"}.get(e.status, "🟡")
                labels[f"{dot}  {e.full_name}"] = e.slug
            picked = st.selectbox("Switch repository", list(labels),
                                  label_visibility="collapsed")
            picked_slug = labels[picked]
            col_open, col_del = st.columns([3, 1])
            if col_open.button("Open selected", width="stretch"):
                st.session_state["active_slug"] = picked_slug
                st.session_state.pop("last_answer", None)
                st.rerun()
            if col_del.button("🗑️", width="stretch",
                              help="Delete this repository's index from disk"):
                st.session_state["confirm_delete"] = picked_slug
                st.rerun()

            # Two-step confirmation: deleting an index is destructive (though
            # always recoverable by re-indexing from GitHub).
            pending = st.session_state.get("confirm_delete")
            if pending:
                entry = next((e for e in entries if e.slug == pending), None)
                name = entry.full_name if entry else pending
                st.warning(f"Delete **{name}**? This removes its downloaded "
                           f"data and index from disk. You can re-index it "
                           f"from GitHub at any time.")
                c1, c2 = st.columns(2)
                if c1.button("Yes, delete", type="primary",
                             width="stretch"):
                    _delete_repo(pending, name)
                if c2.button("Cancel", width="stretch"):
                    st.session_state.pop("confirm_delete", None)
                    st.rerun()
        else:
            st.caption("None yet — index one to get started.")

        st.divider()
        st.markdown("#### ⚙️ Pipeline")
        st.caption(f"Embeddings · `{config.EMBEDDING_MODEL}`")
        # Show the model that will actually answer, not just the local default.
        if config.api_generation_enabled():
            st.caption(f"Generation · ☁️ `{config.GENERATION_API_MODEL}`")
            st.caption(f"↳ falls back to `{config.GENERATION_MODEL}` if offline")
        else:
            st.caption(f"Generation · 💻 `{config.GENERATION_MODEL}`")
        st.caption(f"Reranker · `{config.RERANKER_MODEL.split('/')[-1]}`")
        st.caption(f"Guard · NLI `{config.NLI_MODEL.split('/')[-1]}`")

        st.divider()
        if config.GITHUB_TOKEN:
            st.success("GITHUB_TOKEN set — full ingestion enabled.")
        else:
            st.warning("No GITHUB_TOKEN — pull requests (GraphQL) are skipped. "
                       "Add one to `.env` for full ingestion.")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def cross_repo_panel() -> None:
    """Phase F: ask one question across several indexed repositories at once."""
    st.markdown("### 🔀 Compare across repositories")
    entries = registry.list_repositories()
    reusable = [e for e in entries if e.reusable]
    stale = [e for e in entries if not e.reusable and e.status == "ready"]
    if stale:
        st.warning("Excluded (stale/invalid index — re-index to include): "
                   + ", ".join(e.full_name for e in stale))
    if len(reusable) < 2:
        st.caption("Index at least two repositories to compare them here.")
        return

    labels = {e.full_name: e.slug for e in reusable}
    picked = st.multiselect("Repositories to compare (2–4)", list(labels),
                            max_selections=4)
    q = st.text_input("Question to ask each selected repository",
                      key="xrepo_q",
                      placeholder="e.g. How did this project handle caching?")
    go = st.button("Compare", type="primary",
                   disabled=not (q.strip() and len(picked) >= 2))
    if not go:
        return

    answerer, nli = _get_answerer(), _get_nli()
    repos = []
    for full in picked:
        slug = labels[full]
        ctx = _ctx_for_slug(slug)
        m = manifest_mod.read_manifest(ctx.manifest_path)
        version = ctx.manifest_path.stat().st_mtime
        cov = m.get("coverage", {})
        repos.append((full, _get_retriever(slug, str(version)),
                      cov.get("since", ""), cov.get("until", "")))

    with st.spinner("Answering across repositories…"):
        results = query_pipeline.answer_across_repos(q.strip(), repos, answerer, nli)

    cols = st.columns(len(results))
    for col, (label, pr) in zip(cols, results):
        with col:
            st.markdown(f"#### {label}")
            if pr.empty:
                st.info("No evidence found in this repo's window.")
                continue
            if pr.refusal:
                st.warning("Declined — insufficient verified evidence.")
                st.caption(pr.text)
                continue
            st.markdown(linkify_citations(pr.text, pr.chunks))
            ok = pr.guard_pass
            st.markdown(
                "<div class='rm-pills'>"
                + _pill("✓ grounded" if ok else "⚠ guard flagged",
                        "ok" if ok else "warn")
                + "</div>", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Evaluation (RAGAS-style metrics matrix), shown directly in the app
# --------------------------------------------------------------------------- #
def _eval_results_dir() -> Path:
    return PROJECT_ROOT / "results"


def _list_eval_runs(full_name: str) -> list[dict]:
    """Scan results/*/results.json for completed runs of ``full_name``."""
    import json

    root = _eval_results_dir()
    if not root.exists():
        return []
    runs = []
    for p in sorted(root.glob("*/results.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if data.get("repo") != full_name:
            continue
        runs.append({"path": p, "out_dir": p.parent, "data": data,
                     "created_at": data.get("created_at", "")})
    runs.sort(key=lambda r: r["created_at"], reverse=True)
    return runs


def _dataset_path_for_slug(slug: str) -> Path:
    return PROJECT_ROOT / "eval" / "datasets" / f"{slug}.jsonl"


def _start_eval_run(full_name: str, dataset_path: Path, slug: str) -> None:
    """Launch `python -m eval.run` as a background process (never imported)."""
    out_dir = _eval_results_dir() / f"webrun-{slug}-{int(time.time())}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"
    with log_path.open("w", encoding="utf-8") as log_fh:
        proc = subprocess.Popen(
            [sys.executable, "-m", "eval.run", "--repo", full_name,
             "--dataset", str(dataset_path), "--metrics", EVAL_METRICS_DEFAULT,
             "--subset", "full", "--out", str(out_dir)],
            cwd=str(PROJECT_ROOT), stdout=log_fh, stderr=subprocess.STDOUT,
        )
    st.session_state.setdefault("eval_jobs", {})[slug] = {
        "proc": proc, "out_dir": out_dir, "log": log_path}


def _fmt_metric(v):
    if isinstance(v, float):
        return round(v, 3)
    return v


def render_eval_matrix(results: dict) -> None:
    """Render the headline metric cards + the per-category metrics matrix."""
    by_type = results.get("by_query_type", {})
    if not by_type:
        st.info("No per-category results in this run.")
        return

    # Provenance: scores are only comparable across runs using the same models.
    models = results.get("models") or {}
    if models:
        gen = models.get("generation", "?")
        st.caption(
            f"Produced by generation `{gen}` · judge "
            f"`{models.get('judge_provider','?')}:{models.get('judge_model','?')}`"
        )
        if gen != config.GENERATION_MODEL:
            st.warning(
                f"This run used generation model `{gen}`, but the app currently "
                f"answers with `{config.GENERATION_MODEL}`. Scores from different "
                f"models are not directly comparable."
            )
    else:
        st.caption("⚠️ This run predates model-provenance recording — the models "
                   "used are unknown. Re-run to record them.")

    answerable_types = [qt for qt in by_type if qt != "unanswerable"]

    def _avg(key):
        vals = [by_type[qt].get(key) for qt in answerable_types
                if isinstance(by_type[qt].get(key), (int, float))]
        return sum(vals) / len(vals) if vals else None

    un = by_type.get("unanswerable")
    cards = [
        ("Faithfulness", _avg("faithfulness")),
        ("Answer relevancy", _avg("answer_relevancy")),
        ("Citation precision", _avg("citation_precision")),
        ("Citation recall", _avg("citation_recall")),
        ("Recall@k", _avg("recall_at_k")),
    ]
    if un:
        cards.append(("Abstention accuracy", un.get("abstention_accuracy")))
    html_cards = "".join(
        f"<div class='rm-card'><div class='v'>"
        f"{f'{v:.2f}' if isinstance(v, (int, float)) else 'n/a'}</div>"
        f"<div class='l'>{label}</div></div>"
        for label, v in cards
    )
    st.markdown(f"<div class='rm-stats'>{html_cards}</div>", unsafe_allow_html=True)

    rows = []
    for qt in sorted(by_type):
        s = by_type[qt]
        row = {"query_type": qt}
        for col in MATRIX_COLUMNS[1:]:
            row[col] = _fmt_metric(s.get(col))
        rows.append(row)
    st.caption("Metrics matrix — rows are query categories; a blank cell means "
               "that metric doesn't apply to the category (e.g. faithfulness "
               "for unanswerable items, or abstention accuracy elsewhere).")
    st.dataframe(rows, width="stretch", hide_index=True)

    dist = results.get("distribution", {})
    if dist:
        st.caption("Category distribution: "
                   + ", ".join(f"{k}={v}" for k, v in dist.items() if v))
    judge = results.get("judge", {})
    if not judge.get("available", True):
        st.warning(f"Evaluation judge was unavailable during this run: "
                   f"{judge.get('reason', 'unknown')}")


def evaluation_panel(ctx: RepositoryContext, manifest: dict) -> None:
    """Phase C/D: RAGAS-style evaluation metrics, rendered directly in the app."""
    full_name = f"{manifest['repo']['owner']}/{manifest['repo']['name']}"
    slug = ctx.slug
    dataset_path = _dataset_path_for_slug(slug)

    with st.expander("📊 Evaluation (RAGAS-style metrics)", expanded=False):
        if not dataset_path.exists():
            st.info(
                f"No golden set found at `eval/datasets/{slug}.jsonl`. Add one "
                f"(JSONL, one question per line — see README \"Golden sets\") "
                f"to enable in-app evaluation for this repository."
            )
            return

        job = st.session_state.get("eval_jobs", {}).get(slug)
        if job:
            if (job["out_dir"] / "results.json").exists():
                st.session_state["eval_jobs"].pop(slug, None)
                st.rerun()
            elif job["proc"].poll() is not None:
                st.session_state["eval_jobs"].pop(slug, None)
                tail = (job["log"].read_text(encoding="utf-8")[-1500:]
                       if job["log"].exists() else "")
                st.error("Evaluation run failed — see log tail below.")
                st.code(tail or "(no output captured)")
            else:
                st.info("⏳ Running evaluation… this calls the judge model per "
                       "question and can take a minute or two.")
                time.sleep(2.0)
                st.rerun()
                return

        col1, col2 = st.columns([3, 1])
        col1.caption(f"Golden set: `eval/datasets/{slug}.jsonl`")
        if col2.button("▶️ Run evaluation", width="stretch"):
            _start_eval_run(full_name, dataset_path, slug)
            st.rerun()

        runs = _list_eval_runs(full_name)
        if not runs:
            st.caption("No evaluation runs yet for this repository. Click "
                       "**Run evaluation** above.")
            return

        labels = [f"{r['created_at'][:19].replace('T', ' ')} · "
                  f"{len(r['data'].get('rows', []))} questions" for r in runs]
        idx = st.selectbox("Run", range(len(runs)), format_func=lambda i: labels[i])
        render_eval_matrix(runs[idx]["data"])


def main() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)
    st.markdown(
        "<div class='rm-hero'><span class='emoji'>🧠</span>"
        "<h1>RepoMind</h1></div>"
        "<p class='rm-sub'>Ask why a GitHub repository evolved the way it did — "
        "answers grounded in retrieved commits, PRs, issues, reviews &amp; "
        "releases, and verified before you see them.</p>",
        unsafe_allow_html=True,
    )

    _sidebar()

    if config.ENABLE_CROSS_REPO:
        cross_repo_panel()
        st.divider()

    # New-repo input.
    with st.form("index_form"):
        col1, col2, col3 = st.columns([5, 1.2, 1.5])
        repo_input = col1.text_input(
            "GitHub repository URL",
            placeholder="https://github.com/owner/name",
        )
        months = col2.number_input("Months", 1, 60, config.DEFAULT_LOOKBACK_MONTHS)
        col3.markdown("<div style='height:1.75rem'></div>", unsafe_allow_html=True)
        submitted = col3.form_submit_button("Index / Open", type="primary",
                                            width="stretch")

    if submitted and repo_input.strip():
        try:
            ref = parse_repo_url(repo_input.strip())
        except InvalidRepoURL as exc:
            st.error(f"Invalid GitHub URL: {exc}")
            return
        st.session_state["active_slug"] = ref.slug
        st.session_state.pop("last_answer", None)
        entry = registry.find(ref)
        if entry and entry.reusable:
            st.toast(f"{ref.full_name} already indexed — reusing.", icon="♻️")
        else:
            reason = entry.reuse_reason if entry else "not indexed yet"
            st.toast(f"Indexing {ref.full_name} ({reason})…", icon="⏳")
            start_ingestion(repo_input.strip(), int(months))
        st.rerun()

    slug = st.session_state.get("active_slug")
    if not slug:
        st.info("👆 Paste a public GitHub repository URL above to begin.")
        return

    ctx = _ctx_for_slug(slug)
    if not ctx.manifest_path.exists():
        st.info("Preparing…")
        _poll_rerun(ctx)
        return

    manifest = manifest_mod.read_manifest(ctx.manifest_path)
    status = manifest.get("status")

    cov = manifest.get("coverage", {})
    st.markdown(
        f"<h3 style='margin-bottom:.3rem'>{manifest['repo']['owner']}/"
        f"{manifest['repo']['name']}</h3>"
        f"<span class='rm-cov'>📅 Indexed coverage: "
        f"<b>{cov.get('since', '?')[:10]} → {cov.get('until', '?')[:10]}</b></span>",
        unsafe_allow_html=True,
    )

    if status in ACTIVE_STATES:
        st.write("")
        render_progress(ctx)
        _poll_rerun(ctx)
        return
    if status == "failed":
        render_progress(ctx)
        st.error("Indexing failed — see the message above.")
        return

    ok, reason = manifest_mod.is_reusable(manifest)
    if not ok and status == "ready":
        st.warning(f"This index is stale ({reason}). Re-index to refresh.")
    stats_panel(manifest)
    relationship_graph_section(ctx)
    evaluation_panel(ctx, manifest)
    st.divider()
    answer_panel(ctx, manifest)


def _poll_rerun(ctx: RepositoryContext) -> None:
    """Auto-refresh while a job is active."""
    import time

    time.sleep(1.5)
    st.rerun()


if __name__ == "__main__":
    main()
