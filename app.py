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

EXAMPLE_QUESTIONS = [
    "What fixed the startup crash?",
    "Why was the caching layer added?",
    "What changed in the latest release?",
]

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
.block-container {padding-top: 2.2rem; max-width: 1080px;}
/* hero */
.rm-hero {display:flex; align-items:center; gap:.7rem; margin-bottom:.1rem;}
.rm-hero .emoji {font-size:2.3rem;}
.rm-hero h1 {margin:0; font-size:2.3rem; font-weight:800; letter-spacing:-.02em;
  background:linear-gradient(90deg,#8b7ff5,#c4b5fd); -webkit-background-clip:text;
  -webkit-text-fill-color:transparent; background-clip:text;}
.rm-sub {color:#9aa4b2; margin:.15rem 0 1.1rem; font-size:1.02rem;}
/* coverage chip */
.rm-cov {display:inline-block; background:#161b26; border:1px solid #262d3d;
  border-radius:8px; padding:.32rem .8rem; color:#c7cdd6; font-size:.9rem;}
/* stat cards */
.rm-stats {display:flex; gap:.7rem; flex-wrap:wrap; margin:.9rem 0 .2rem;}
.rm-card {flex:1 1 130px; background:#161b26; border:1px solid #242c3c;
  border-radius:12px; padding:.8rem .95rem;}
.rm-card .v {font-size:1.55rem; font-weight:750; color:#f2f4f7; line-height:1.05;}
.rm-card .l {font-size:.74rem; color:#93a0b4; margin-top:.2rem;
  text-transform:uppercase; letter-spacing:.05em;}
/* guard pills */
.rm-pills {display:flex; gap:.5rem; flex-wrap:wrap; margin:.2rem 0 .3rem;}
.rm-pill {display:inline-flex; align-items:center; gap:.4rem; padding:.34rem .75rem;
  border-radius:999px; font-size:.84rem; font-weight:600; border:1px solid transparent;}
.rm-ok {background:rgba(34,197,94,.12); color:#4ade80; border-color:rgba(34,197,94,.3);}
.rm-warn {background:rgba(245,158,11,.13); color:#fbbf24; border-color:rgba(245,158,11,.32);}
.rm-bad {background:rgba(239,68,68,.13); color:#f87171; border-color:rgba(239,68,68,.32);}
.rm-neu {background:#1b2230; color:#c2cad6; border-color:#2a3242;}
/* source-type badges */
.rm-src {display:inline-block; padding:.06rem .5rem; border-radius:6px; font-size:.7rem;
  font-weight:700; text-transform:uppercase; letter-spacing:.03em; margin-right:.45rem;
  vertical-align:middle;}
.rm-src-commit {background:#12331f; color:#6ee7b7;}
.rm-src-pr {background:#241f4d; color:#c4b5fd;}
.rm-src-issue {background:#3a201d; color:#fca5a5;}
.rm-src-review {background:#132b38; color:#7dd3fc;}
.rm-src-release {background:#332a16; color:#fcd34d;}
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


def _run_qa(ctx, question, since, until) -> None:
    """Run the answer pipeline (with optional adaptive retry) and store it."""
    warn = _year_out_of_range(question, since, until)
    version = ctx.manifest_path.stat().st_mtime if ctx.manifest_path.exists() else ""

    trace: dict = {} if config.ENABLE_METRICS_LOGGING else None
    _q0 = time.perf_counter()

    retriever = _get_retriever(ctx.slug, str(version))
    answerer = _get_answerer()
    nli = _get_nli()
    with st.spinner("Retrieving, answering, and verifying…"):
        pr = query_pipeline.answer_query(
            retriever, answerer, nli, question, since, until, trace=trace)

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
        "citations_ok": ref_report.is_valid,
        "grounded": nli_report.is_grounded,
        "contradicted": [(c.claim, c.contradiction) for c in nli_report.contradicted],
        "n_unverified": len(nli_report.unverified),
        # Phase B: adaptive-retry outcome (all False on the happy path / flag off).
        "retry_attempted": pr.retry_attempted,
        "retry_reason": pr.retry_reason,
        "retry_succeeded": pr.retry_succeeded,
        "refusal": pr.refusal,
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

    with st.container(border=True):
        st.markdown("#### 💬 Answer")
        st.markdown(linkify_citations(ans["text"], chunks))

    # Guard pills.
    cite_pill = _pill(
        f"✓ {len(chunks)} citations verified" if ans["citations_ok"]
        else f"✗ {len(ans['invalid_citations'])} fabricated citation(s)",
        "ok" if ans["citations_ok"] else "bad",
    )
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
    st.caption("Try an example:")
    chip_cols = st.columns(len(EXAMPLE_QUESTIONS))
    for col, ex in zip(chip_cols, EXAMPLE_QUESTIONS):
        col.button(ex, key=f"ex_{ex}", on_click=_set_example, args=(ex,),
                   use_container_width=True)

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
        st.graphviz_chart(linker.to_dot(graph), use_container_width=True)


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
            if st.button("Open selected", use_container_width=True):
                st.session_state["active_slug"] = labels[picked]
                st.session_state.pop("last_answer", None)
                st.rerun()
        else:
            st.caption("None yet — index one to get started.")

        st.divider()
        st.markdown("#### ⚙️ Pipeline")
        st.caption(f"Embeddings · `{config.EMBEDDING_MODEL}`")
        st.caption(f"Generation · `{config.GENERATION_MODEL}`")
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
    st.dataframe(rows, use_container_width=True, hide_index=True)

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
        if col2.button("▶️ Run evaluation", use_container_width=True):
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
                                            use_container_width=True)

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
