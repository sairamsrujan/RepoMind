# HANDOFF — read this before changing anything

You are picking up a **finished, working** final-year project. It is not a
half-built prototype. Everything described below has been verified by running
it, not assumed.

**Your default assumption should be: this works. Do not rebuild it.**

The single biggest risk to this project is a well-meaning assistant deciding to
"improve" the architecture, swap a model, or restructure the code. Several of
the design choices below look odd until you know *why* they are that way — and
the reasons are recorded here precisely so they don't get undone.

---

## 1. What this project is

**RepoMind** answers *"why did this GitHub repository evolve this way?"*

You paste a repo URL. It ingests commits, PRs, issues, reviews and releases,
indexes them, and answers natural-language questions with **inline citations
linking to the real GitHub pages** — and every answer is **verified before
display** by a hallucination guard that will refuse to answer rather than guess.

- Final-year B.Tech major project, team of three.
- Runs locally on a MacBook: `streamlit run app.py`. **No deployment, no Docker,
  no cloud, no auth, no database.** That is deliberate, not an omission.
- Demo is roughly nine months away. **Reliability beats new features.**

`CLAUDE.md` holds the original project specification. Read it for intent, but
note the system has since moved beyond it (multi-provider LLMs, Phase-2 work);
where they disagree, **this file and the code are current**.

---

## 2. Current state (verified, not assumed)

| Thing | State |
|---|---|
| Test suite | **216 passing, 0 failures** (`pytest -q`, Ollama up) |
| Demo readiness | `python scripts/demo_check.py` |
| Provider health | `python scripts/check_providers.py` |
| Golden sets | 330 questions across 5 real repos + 1 fixture |
| Abstention | **0.90 mean** over 150 unanswerable questions (30/repo) |
| Failure gallery | 15 real cases in `results/failure_gallery.md` |
| Published | https://github.com/sairamsrujan/RepoMind |

### Indexed repositories

| Repo | Domain | Commits | PRs | Chunks | Graph links |
|---|---|---|---|---|---|
| `fastapi/fastapi` | web framework | 1609 | 2709 | 5170 | 127 |
| `pydantic/pydantic` | validation | 590 | 1446 | 3558 | **650** |
| `psf/black` | formatter | 292 | 901 | 1594 | 197 |
| `psf/requests` | HTTP client | 119 | 658 | 1260 | 156 |
| `pallets/click` | CLI | 286 | 570 | 1059 | 376 |
| `acme/widgets` | *fixture* | 3 | 2 | 11 | 4 |

`pallets/click` is still the **best demo repo** (high link density, familiar
project). `acme/widgets` is a synthetic fixture for fast deterministic tests and
is excluded from headline averages — 11 chunks makes it far easier than any real
corpus.

`fastapi/fastapi` was re-indexed with a working token: **0 → 2709 PRs**, and its
abstention rose from 0.857 to 0.900.

---

## 3. ⚠️ Landmines — things that WILL break if you change them

These are not style preferences. Each one caused a real failure that took time
to diagnose.

### 3.1 `TORCH_DEVICE = "cpu"` — do NOT change to `mps`

PyTorch's Apple-GPU (MPS) backend **segfaults** inside
`at::native::mps::copy_cast_kernel_mps` when loading the reranker / NLI
cross-encoders. It killed the whole app; there were three crash reports in ten
minutes. MPS is ~3× faster but crashes, especially with two processes running.

**CPU is the deliberate trade: slightly slower, never crashes.** A demo that
dies is infinitely worse than a demo that takes an extra second.

### 3.2 `MMR_TOP_N = 12` — do NOT raise it back to 20

Every candidate surviving MMR costs one cross-encoder pass. At 20 on CPU,
reranking took **24 seconds** — 95% of total query time. At 12 it is ~2s.
Raising this makes every query dramatically slower.

### 3.3 The embedding model must stay local and unchanged

`qwen3-embedding:0.6b` built the vectors in every Chroma index on disk.
Changing it **invalidates every indexed repository** and forces a full rebuild
of all four. It cannot be swapped to a cloud provider — the manifest reuse
fingerprint checks it (`core/manifest.py::is_reusable`).

### 3.4 Never wait long on a provider rate limit

Groq sends `Retry-After: 1243` when its daily token cap is hit. An earlier
version honoured that (clamped to 60s) and slept **twice**, adding ~120s to
every query before falling back to a local model that answers in ~10s.

`generation/answerer.py::_is_quota_exhausted` now detects daily-quota 429s and
fails over **immediately**, while still retrying short per-minute limits.
`GENERATION_API_MAX_WAIT = 8.0` is the hard ceiling. **Do not raise it.**

### 3.5 Run ONE heavy job at a time (16 GB machine)

Models are large: chat ~4.7 GB, embeddings ~2.4 GB, reranker ~2.3 GB, NLI
~0.8 GB. Running the app + pytest + an evaluation together causes heavy
swapping; the OS then kills a process **with no error message at all**.

**A job that stops silently with an empty log is almost always this.** It is
not a code bug. Free Ollama's models between jobs:

```bash
curl -s localhost:11434/api/generate -d '{"model":"qwen2.5:7b-instruct","keep_alive":0}'
```

### 3.6 Do not add `ragas` or `langchain`

The RAGAS-style metrics are hand-implemented in `eval/metrics.py` **on purpose**.
Those libraries churn fast and would break the "runs unattended for a year with
every dependency pinned" requirement. This is documented in `DECISIONS.md` and
was a deliberate, defended deviation from the original spec.

### 3.7 `requirements.txt` is pinned with `==` — leave it

Do not upgrade, unpin, or add dependencies without being asked. There is a
3.2 GB offline `wheelhouse/` built from these exact versions.

### 3.8 Tests must not assert exact golden-set sizes

Golden sets are regenerable at any size. Two separate test failures were caused
by hardcoded counts (`len(entries) == 5`, `unanswerable n == 1`). Assert
*structure* — category present, `n >= 1`, required fields exist — never a fixed
count.

### 3.9 The ablation must pin ONE generation model

An ablation isolates one variable: the configuration. If generation is left on
the normal cloud-with-fallback path, the answering model degrades **in the same
order the configurations run**, because both proceed top to bottom while the
daily quota drains:

```
1.retrieval-only    19/20 answers from Groq-70B
2.+MMR              14/20
3.+MMR+reranker      0/20   (13 × NVIDIA-49B, 7 × local-7B)
5.full+guard+retry   4/20   (10 × local-7B)
```

The table then shows "faithfulness falls as you add pipeline stages" — which is
the answerer getting weaker, not the stages hurting. Every cross-config number
is confounded, and it looks completely plausible.

`eval/ablation.py` now pins generation (`GENERATION_PROVIDER=ollama`) unless
`--allow-cloud-generation` is passed, and records `generation_model` plus
per-config `answered_by` counts in `ablation.json`. **Check those before quoting
any ablation number.** The confounded run is kept as
`results/ablation-multi-CONFOUNDED-superseded/` as the evidence for this rule.

For an ablation, a consistent weaker model beats an inconsistent stronger one.

### 3.10 Model choice is a latency decision, not just a quality one

Two reasoning models were chosen for jobs that emit a few tokens, and each cost
hours before being caught:

| Role | Reasoning model | Replacement | Speed-up |
|---|---|---|---|
| Judge | `deepseek-v4-flash` 88s | `groq:gpt-oss-120b` 0.6s | **147×** |
| Generation fallback | `nemotron-super-49b` ~99s | pinned local ~20s | ~5× |

A reasoning model thinks at length before answering. That is invisible per call
and decisive across a 300-question run. **Time a candidate on the real prompt
before putting it in a chain** — `scripts/check_providers.py` lists them, but it
does not time them.

---

## 4. Architecture — what talks to what

```
GitHub URL
   ↓  ingest/          REST (commits, releases) + GraphQL (PRs, reviews, issues)
   ↓                   checkpointed + resumable
raw JSON
   ↓  process/         chunker.py -> tagged chunks;  linker.py -> issue↔PR↔commit graph
chunks.jsonl
   ↓  index/           Ollama embeddings -> ChromaDB (dense) + BM25 (sparse)
                       manifest.json fingerprint decides reuse vs rebuild
─────────────────────────────── per question ───────────────────────────────
question
   ↓  retrieval/       dense(30) + sparse(30) -> RRF(k=60) -> pool 40
                       -> MMR(λ=0.5) -> 12 -> cross-encoder rerank -> 6
   ↓  generation/      prompt.py (evidence delimited as untrusted) -> answerer.py
                       Groq 70B, or local qwen2.5:7b if that fails
   ↓  guard/           ① reference_validator: every [chunk_id] must exist
                       ② nli_verifier: each claim vs its cited evidence
   ↓  query_pipeline.py   guard fails + retry enabled -> ONE widened retry
                          -> still fails -> honest refusal
   ↓  app.py           answer + clickable citations + guard badges
```

### Module map

| Path | Responsibility |
|---|---|
| `config.py` | **All** tunables and model names. Nothing hardcoded elsewhere |
| `providers.py` | LLM provider registry (Groq/Gemini/NVIDIA/OpenRouter/Ollama) |
| `query_pipeline.py` | Retrieve → generate → guard → retry → refusal orchestration |
| `telemetry.py` | Per-query metrics to `data/metrics/queries.jsonl` (fail-silent) |
| `app.py` | Streamlit UI |
| `core/` | repo URL parsing, manifest, registry, paths |
| `ingest/` | GitHub REST + GraphQL fetchers, checkpointing |
| `process/` | chunker, linker (evolution graph) |
| `index/` | embedder, Chroma + BM25 builders |
| `retrieval/` | retriever (RRF), mmr, reranker, filters |
| `generation/` | prompt builder, answerer (cloud + local fallback) |
| `guard/` | reference_validator, nli_verifier |
| `jobs/` | background ingestion runner + status file |
| `eval/` | golden sets, metrics, runner, ablation — **offline only** |
| `scripts/` | demo_check, smoke_test, freeze_environment, summarise_metrics |

### 🚧 Architectural boundary — do not violate

**Nothing in `retrieval/`, `generation/`, `guard/`, `jobs/`, or `app.py` may
import from `eval/`.** If `eval/` were deleted, the app must still run. The
evaluation panel in `app.py` launches `eval/run.py` as a **subprocess** for
exactly this reason. `eval/` imports *from* the pipeline, never the reverse.

---

## 5. Where LLMs are used (five places, only two swappable)

| # | Component | Model | Location | Live path? |
|---|---|---|---|---|
| 1 | Embeddings | `qwen3-embedding:0.6b` | local **only** | ✅ |
| 2 | Reranker | `BAAI/bge-reranker-v2-m3` | local (CPU) | ✅ |
| 3 | **Generation** | Groq 70B ↔ `qwen2.5:7b-instruct` | ☁️/💻 | ✅ |
| 4 | NLI guard | `cross-encoder/nli-deberta-v3-base` | local (CPU) | ✅ |
| 5 | **RAGAS judge** | Groq 70B ↔ local | ☁️/💻 | ❌ offline only |

#2 and #4 are **classifiers, not chatbots**. Only #3 and #5 can use cloud.

### Provider roles — THREE roles, THREE distinct model families

```
GENERATION_CHAIN   groq:llama-3.3-70b-versatile → nvidia:…nemotron-super-49b-v1.5
JUDGE_CHAIN        nvidia:deepseek-v4-flash → deepseek-v4-pro → groq:gpt-oss-120b
QUESTIONGEN_CHAIN  nvidia:nemotron-3-nano-30b-a3b → openrouter:nemotron-nano-9b:free
```

All three must stay **different model families**. The question-author/judge split
was always documented — but a real bug shipped for a while where the **judge was
the same model as the answerer** (both Groq `llama-3.3-70b-versatile`), i.e. the
model was grading its own output. That is the worse of the two collisions.

`config.roles_are_distinct()` now enforces this and every `results.json` records
the outcome. It compares **canonical** model names, because Groq's
`openai/gpt-oss-120b` and Cerebras's `gpt-oss-120b` are the same model wearing
different vendor packaging.

**Model sizing — do not reach for the biggest model.** Free tiers cap
tokens-per-DAY, and input cost is identical whatever model reads it (~950 prompt
tokens + six evidence chunks). A 550B model burns the daily budget several times
faster while adding nothing to a grounding-and-formatting task. Each role uses
the smallest model that does its job: the answerer needs instruction-following,
the judge needs consistency (flash, not pro), and only question generation
genuinely benefits from reasoning — served by a 30B MoE with ~3B active.

**Never guess a cloud model ID.** Run `python scripts/check_providers.py`, which
queries every provider and validates each role's model. Guessed ids fail as a
404 that looks exactly like an outage.

---

## 6. Free-tier limits (real numbers, learned the hard way)

**Groq:** 30 RPM · 12,000 TPM · 1,000 RPD · **100,000 tokens/day**

Each evaluation question costs ~4,000 tokens (≈950 prompt + ~3,000 judge).
That means **~3 questions per minute** sustainable, and roughly **25 questions
per day** before the daily cap. A 175-question evaluation *cannot* run entirely
on Groq's free tier — 132 fallback events occurred during the last full run.

**Practical rule:**
- **Demo** → cloud is fine (you won't ask 25 questions live)
- **Bulk evaluation** → expect fallback to local; that is normal, not a bug
- **Never run evaluations the same day you demo** — you'll exhaust the quota

### Measured provider status — re-check before relying on any of it

Free tiers are **not durable**. In one afternoon of testing, three of five
providers broke:

| Provider | Status | Detail |
|---|---|---|
| **NVIDIA** (build.nvidia.com) | ✅ best | 102 models; Nemotron + DeepSeek. Primary for judge + question-gen |
| **Groq** | ✅ fast, small cap | Daily token cap exhausts after a few dozen questions |
| **OpenRouter** | ✅ works | `:free` Nemotron models fine. Some 404 with *"No endpoints matching your guardrail"* — needs prompt-training enabled in privacy settings |
| **Gemini** | ❌ unreliable | See below |
| **Cerebras** | ❌ unusable | `402 Payment required` on all 3 models — free tier is not on every account |

**Gemini correction (supersedes the old `AIzaSy` advice above).** Google no
longer issues `AIza…` keys; AI Studio now issues `AQ.…` keys and **they
authenticate correctly**. The blocker is availability, not the key format:
`gemini-2.5-flash` and `-flash-lite` return 404 *"no longer available"*,
`2.0-flash` and `2.5-pro` return 429, `3.5-flash` returns 503. Gemini is
therefore kept **out of the default chains**.

### Why chains exist

Because of the table above. Each role walks an ordered chain and the **local
model is always appended last**, so no third-party failure can take the app
down. This is what makes "still works in nine months" true rather than hoped for.

⚠️ **The chain is for OFFLINE roles only.** The interactive answer path still
falls straight back to local — see §3.4. Trying more cloud providers before
local costs a full timeout each and made every query dramatically slower. A test
(`test_api_failure_falls_back_to_local`) enforces this; if it fails, someone has
re-introduced cloud retries into the live query path.

---

## 7. Measured performance (don't re-derive this)

Per query, warm:

| Stage | Time |
|---|---|
| embed query | 0.5s |
| dense + sparse search | 0.1s |
| MMR | 0.03s |
| **rerank** | ~2s (was 24s before `MMR_TOP_N=12`) |
| generation — Groq | **0.7s** |
| generation — local fallback | ~10s |
| NLI guard | 0.1–5s |
| **total** | **~4s cloud / ~25s local** |

**Cold start costs ~40s** because Ollama reloads the embedding model.
**Always ask one warm-up question before demoing.**

---

## 8. Evaluation results (as of the last full run)

Abstention accuracy — the headline number, proving the guard works.
**30 verified-unanswerable questions per repository:**

| Repo | n | Abstention accuracy | Hallucinated |
|---|---|---|---|
| `pallets/click` | 30 | **0.967** | 1 |
| `psf/requests` | 30 | **0.933** | 2 |
| `fastapi/fastapi` | 30 | **0.900** | 3 |
| `pydantic/pydantic` | 30 | **0.867** | 4 |
| `psf/black` | 30 | **0.833** | 5 |
| **mean (real repos)** | **150** | **0.900** | 12 |

⚠️ **Do not restore the old "1.00" figure.** It came from **7** questions per
repo and was a small-sample artifact. The sample was deliberately enlarged to 30
and the number fell to 0.90. That is the *better* result — it survives the
question "on how many questions?", which 1.00 did not. Anyone re-reporting 1.00
is quoting a superseded run.

**`evolution` questions remain weakest** (recall 0.36–0.39 even with graph
expansion) — multi-hop retrieval across time is genuinely the hard case. This is
a documented, honest limitation with 15 failure cases behind it, **not something
to hide or "fix" by tuning metrics**.

⚠️ **Provenance caveat that must stay in any report:** these answers came
**entirely from local qwen2.5:7b**, because the cloud daily token cap was
already exhausted when the run started. They are a floor, not a best case. Each
run now records an `answered_by` count so this is verifiable rather than
remembered — check it before quoting any number.

The **mixed-category** results in `results/eval-*` are STALE: they predate the
fastapi PR re-index and were produced while the judge was the same model as the
answerer. Re-run before citing:

```bash
bash scripts/run_full_evaluation.sh
```

---

## 9. Feature flags

| Flag | Default | Effect |
|---|---|---|
| `ENABLE_METRICS_LOGGING` | `True` | Per-query metrics to JSONL |
| `ENABLE_ADAPTIVE_RETRY` | `False` | One widened retry on guard rejection |
| `ENABLE_CROSS_REPO` | `False` | Compare one question across repos |

With every flag off, behaviour is identical to the Phase-1 baseline. Turning a
flag on must never be required for the app to work.

---

## 10. Commands that matter

```bash
# Run the app
.venv/bin/streamlit run app.py

# BEFORE EVERY DEMO — 9 checks, exits non-zero if anything would fail live
.venv/bin/python scripts/demo_check.py

# MONTHLY — catches environment drift over the 9-month gap
.venv/bin/python scripts/smoke_test.py

# Tests
.venv/bin/python -m pytest -q

# Evaluation (resumes from checkpoint if interrupted — re-run same command)
.venv/bin/python -m eval.run --repo pallets/click \
  --dataset eval/datasets/pallets_click.jsonl --out results/my-run

# Regenerate a golden set
.venv/bin/python -m eval.generate_golden_set --repo pallets/click --n 50
```

---

## 11. What is NOT done (honest list)

- **3-repo ablation table** — `eval/ablation.py` exists and is tested, but the
  full multi-repo run was never completed (it is expensive: 7 configs × 175 q).
- **`fastapi/fastapi` has 0 PRs** — re-index with the working token to fix.
- **README lacks architecture diagrams** — planned as Mermaid (GitHub renders
  natively, no image files to maintain).
- **Not yet pushed to GitHub.**
- **Cross-repo panel** (`ENABLE_CROSS_REPO`) built and unit-tested but never
  demoed live.

---

## 12. Working agreement for the next session

**Do:**
- Run `pytest -q` before and after any change; it must stay at 0 failures.
- Run one heavy job at a time.
- Add tests for new behaviour.
- Read `DECISIONS.md` before changing a design choice — the rationale and cost
  of each is recorded there.
- Be honest in reports about the fallback/provenance caveat.

**Do not:**
- Rebuild, restructure, or "clean up" working modules unprompted.
- Change pinned dependencies or model tags.
- Switch `TORCH_DEVICE` to `mps`, or raise `MMR_TOP_N`.
- Add Docker, cloud deployment, auth, a database, or an agent framework
  (LangChain / LangGraph / CrewAI). All were considered and deliberately
  rejected — they add failure modes without adding marks.
- Let `eval/` be imported by the live app path.
- Paste API keys into chat. The user adds them to `.env` themselves; `.env` is
  gitignored and must never be committed.

**When something silently dies with an empty log:** check memory first (§3.5),
then check Ollama is running (`curl localhost:11434/api/tags`). It is almost
never a code bug.

---

## 13. Priorities for remaining work

In order of value to the final grade:

1. **README with architecture diagram** — most people judge a repo from the
   README alone.
2. **Push to GitHub** with a clear structure.
3. **A 2-minute demo video** ending on the refusal moment — far more persuasive
   than any amount of prose.
4. Complete the 3-repo ablation table (report centrepiece, expensive to run).
5. Re-index `fastapi/fastapi` with PRs.

**The engineering is done. What remains is presentation.** Resist the urge to
add features; polish what exists.
