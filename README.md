# RepoMind

**Ask why a GitHub repository evolved the way it did.** Paste any public GitHub
repo URL and ask natural-language questions ("Why was the caching layer added?",
"What fixed the startup crash?"). Answers are grounded in retrieved GitHub
evidence — commits, pull requests, issues, reviews, releases — with inline
citations to the real GitHub URLs, and every answer is verified by a
hallucination guard before you see it.

Runs **fully locally** on a MacBook (Apple Silicon) via `streamlit run app.py`.
Zero cost, no cloud, no deployment. The only optional external call is the
evaluation judge, which is swappable to a local model with one config change.

---

## What it does

```
GitHub URL ─▶ ingest (REST+GraphQL) ─▶ chunk + link ─▶ embed (Ollama) ─▶ index (Chroma + BM25)
                                                                                    │
question ─▶ hybrid retrieval (dense+sparse, RRF) ─▶ MMR ─▶ rerank ─▶ grounded, cited answer
                                                                                    │
                                                              guard: reference check + NLI entailment
```

- **Hybrid retrieval**: dense (Chroma / Ollama embeddings) + sparse (BM25),
  merged with Reciprocal Rank Fusion, diversified with MMR, reranked with a
  BGE cross-encoder.
- **Grounded generation**: a local Ollama chat model answers *only* from
  retrieved evidence, with inline `[chunk_id]` citations and an explicit
  "outside the indexed window" fallback.
- **Hallucination guard**: a reference validator (every citation must be real)
  plus an NLI entailment check (every claim must be supported by its cited
  evidence — contradictions are flagged).
- **Reuse-aware**: re-opening an already-indexed repo reuses its index; a
  config change (e.g. a new embedding model) is detected and triggers a rebuild.

---

## Prerequisites

1. **Python 3.11+** (tested on 3.11).
2. **[Ollama](https://ollama.com)** installed and running (`ollama serve`).
   Pull the required models:

   ```bash
   ollama pull qwen3-embedding:0.6b     # embeddings (1024-dim)
   ollama pull qwen2.5:7b-instruct      # answer generation
   # optional fallbacks:
   ollama pull nomic-embed-text         # fallback embedder
   ```

   The reranker (`BAAI/bge-reranker-v2-m3`) and NLI model
   (`cross-encoder/nli-deberta-v3-base`) are downloaded automatically from
   Hugging Face on first use and cached locally.

If Ollama is not running, the app stops with clear instructions rather than
proceeding.

---

## Setup

```bash
# 1. Create and activate a virtual environment (Python 3.11)
python3.11 -m venv .venv
source .venv/bin/activate

# 2. Install pinned dependencies
pip install -r requirements.txt

# 3. Configure secrets
cp .env.example .env
#   then edit .env:
#     GITHUB_TOKEN=...   (recommended — see below)
```

### About `GITHUB_TOKEN`

- **Commits / issues / releases** work unauthenticated but are capped at 60
  requests/hour.
- **Pull requests** use GitHub's **GraphQL API, which requires authentication** —
  without a token, PR ingestion is skipped (the app still indexes everything
  else and shows a clear warning).
- With a token you get 5,000 requests/hour and full PR/review/linked-issue
  ingestion. Create a classic or fine-grained PAT with public-repo read scope.

---

## Run the app

```bash
streamlit run app.py
```

Then in the browser:

1. Paste a repo URL (e.g. `https://github.com/pallets/click`) and a date window
   (default: last 12 months). Click **Index / Open**.
2. Watch the live progress (fetching → chunking → embedding → indexing). This
   runs in a **background process** so the UI never blocks.
3. Once ready, see the stats panel and ask questions. Answers show inline
   citation links to GitHub plus a guard summary (valid citations / grounded /
   unverified claims).
4. Switch between indexed repos from the sidebar; re-opening one **reuses** its
   index instead of rebuilding.

The UI also includes: a **relationship graph** of how issues were resolved
(issue ↔ PR ↔ commit ↔ release), a per-session **question history**, an
**export-answer-to-Markdown** button, and **live counts** during indexing.

You can also ingest from the command line:

```bash
python -m jobs.runner pallets/click 4     # repo, months window
```

---

## Per-query metrics (Phase A)

When `ENABLE_METRICS_LOGGING` (default **`True`**) is on, every answered query
appends one JSON line to `data/metrics/queries.jsonl` — latencies (retrieval /
rerank / generation / guard / total), candidate counts at each retrieval stage
(dense, BM25, RRF, MMR, rerank), the guard verdict and reason, citation counts,
and the feature-flag state. Logging is best-effort: a failure to write never
slows or breaks a query, and with the flag off no file is created. `data/metrics/`
is gitignored.

Summarise the log (count / mean / median / p95 latencies, guard pass rate,
citation validity rate) with the standard-library-only script:

```bash
python scripts/summarise_metrics.py
```

---

## Adaptive verification retry (Phase B)

When `ENABLE_ADAPTIVE_RETRY` (default **`False`**) is on and the hallucination
guard **rejects** an answer (a fabricated citation or a contradicted claim), the
system retries **exactly once** with a widened strategy: the dense and BM25
candidate pools are multiplied by `RETRY_POOL_MULTIPLIER`, MMR is skipped (so the
highest-scoring evidence is kept rather than diversified away), and the
cross-encoder reranker stays on. The answer is regenerated and re-checked:

- retry passes the guard → the new answer is shown, badged **🔁 answer
  regenerated after guard rejection**;
- retry still fails → an **honest refusal** is shown (it states there was
  insufficient verifiable evidence and lists what was retrieved) — never an
  unverified answer presented as verified.

Retries are hard-bounded at one (no recursion, no loops). With the flag off the
answer path is byte-for-byte identical to before. The retry decision, trigger
reason, and outcome (`retry_attempted` / `retry_reason` / `retry_succeeded`) are
written to the Phase A metrics line. This feedback step is the honest basis for
calling the system *agentic* — plain deterministic Python, no orchestration
framework.

---

## Run the evaluation / ablation

The gold set is derived automatically: every closed issue resolved by a linked,
merged PR becomes a question whose ground-truth evidence is that PR and its
commits. Then four configurations are compared:

1. retrieval only, 2. + MMR, 3. + MMR + reranker, 4. full system + guard.

```bash
# The repo must already be indexed.
python -m eval.ablation pallets/click 20    # owner/name, max questions
```

Reports mean **faithfulness**, **answer relevancy**, **citation precision**,
**recall@k**, and **average latency** per configuration.

### Swappable judge

Faithfulness and answer relevancy use a judge selected by **one** config
variable, `JUDGE_PROVIDER`:

| `JUDGE_PROVIDER` | Needs           | Cost        |
|------------------|-----------------|-------------|
| `ollama`         | local model     | **free**    |
| `groq`           | `GROQ_API_KEY`  | free tier   |
| `gemini`         | `GEMINI_API_KEY`| free tier   |

Default falls back to the local Ollama judge if no API key is present, so the
whole system runs with **zero paid API calls**. (These are RAGAS-style
faithfulness / answer-relevancy metrics computed through the swappable judge;
we deliberately avoid the heavy, fast-changing `ragas`+`langchain` dependency
stack to keep the project runnable unattended for 1+ year — see
`eval/metrics.py`.)

---

## Golden sets & the evaluation runner (Phase C)

Alongside the auto-derived gold set, a **versioned, category-aware benchmark**
lives at `eval/datasets/<owner>_<name>.jsonl` (JSONL, not YAML — PyYAML is only
a transitive, unpinned dependency here). One object per line:

```json
{"id": "aw-001", "question": "...", "query_type": "factual",
 "ground_truth": "...", "evidence": ["pr_101", "commit_c0ffee1"], "notes": "..."}
```

`query_type` is one of `factual`, `causal`, `cross_commit`, `evolution`,
`unanswerable`. **Unanswerable** entries have **no** `ground_truth` — they test
that the guard suppresses hallucination (handled as a distinct code path). A
validator fails loudly on duplicate/malformed entries before any run.

Target category distribution when hand-building a set:

| Category | Share |
|----------|-------|
| `factual` (single-hop) | ~25% |
| `causal` (why-changed) | ~15% |
| `cross_commit` (multi-hop) | ~25% |
| `evolution` (over time) | ~20% |
| `unanswerable` | ~15% |

Run the evaluation (results broken down by `query_type`, with **abstention
accuracy** on unanswerable items). Judge responses are cached on disk (a re-run
costs zero API calls) and rate-limited; any judge failure degrades to
`evaluation unavailable: <reason>`:

```bash
python -m eval.run --repo acme/widgets --dataset eval/datasets/acme_widgets.jsonl \
    --metrics faithfulness,answer_relevancy,citation_precision,recall_at_k \
    --subset full --out results/run-1/
```

## Ablation harness (Phase D)

The four classic configs plus a **fifth** (full + guard + adaptive retry) and
two **channel-isolation** configs (dense-only, sparse-only) — so BM25's
contribution is *measured*, not asserted. Reports per config and per
`query_type`, writing `ablation.csv` + `ablation.json` for the report:

```bash
python -m eval.ablation --repos acme/widgets,pallets/click --out results/ablation-1/
```

(Each repo auto-loads `eval/datasets/<slug>.jsonl` unless `--datasets` is given.)

## Failure gallery (Phase E)

An honest record of where the system fails — guard rejections, fabricated
citations, NLI contradictions, retrieval misses, incorrect refusals — spread
across categories (default cap 15), each with a blank `**Why:**` line:

```bash
python scripts/export_failure_gallery.py --results results/run-1/results.json \
    --dataset eval/datasets/acme_widgets.jsonl --out results/failure_gallery.md
```

## Cross-repository comparison (Phase F)

With `ENABLE_CROSS_REPO` (default **`False`**) on, the app can ask one question
against several indexed repositories at once — each retrieved from its **own**
index (never merged), answered and guarded independently, shown side by side
with citations attributed to their source repo. Stale-index repos are excluded
with a notice.

## Durability & provenance (Phase G)

- [`DECISIONS.md`](DECISIONS.md) — the design decision log.
- `python scripts/smoke_test.py` — one-command monthly liveness check (Ollama +
  models present, reranker/NLI resolvable **offline**, `pytest`, one end-to-end
  cited answer). Exits non-zero on any failure.
- `bash scripts/freeze_environment.sh` — builds an offline `wheelhouse/` and
  archives the model cache; see [`ENVIRONMENT.md`](ENVIRONMENT.md) for the
  clean-machine restore procedure.

All Phase 2 feature flags (in `config.py`): `ENABLE_METRICS_LOGGING` (`True`),
`ENABLE_ADAPTIVE_RETRY` (`False`), `ENABLE_CROSS_REPO` (`False`). With every flag
at its default-off value the pipeline is byte-for-byte identical to Phase 1.

---

## Chosen hyperparameters

Set in [`config.py`](config.py):

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `MAX_CHUNK_TOKENS` | 512 | Fits the embedding model's context; keeps a chunk to one coherent commit/PR/issue. |
| `DENSE_TOP_K` / `SPARSE_TOP_K` | 30 / 30 | Wide enough candidate pools for RRF to recover both semantic and keyword hits. |
| `RRF_K` | 60 | Standard Reciprocal Rank Fusion constant; dampens the tail so top ranks dominate. |
| `RRF_POOL_SIZE` | 40 | Merged pool handed to MMR — enough diversity without over-feeding the reranker. |
| `MMR_LAMBDA` | 0.5 | Equal weight to relevance and diversity, removing near-duplicate evidence. |
| `MMR_TOP_N` | 20 | Candidates kept after diversification, before the (slower) cross-encoder. |
| `FINAL_TOP_K` | 6 | Final chunks shown to the LLM — enough evidence, few enough to stay grounded. |
| `NLI_ENTAILMENT_THRESHOLD` | 0.5 | Min entailment probability to accept a claim; below it, flag as unverified. |
| `NLI_CONTRADICTION_THRESHOLD` | 0.6 | Higher bar than entailment, so only a *confident* contradiction flags a claim. |
| `GENERATION_TEMPERATURE` | 0.1 | Low temperature for faithful, deterministic, citation-following answers. |
| `RETRY_POOL_MULTIPLIER` | 2 | On an adaptive retry, double the dense+BM25 pools — wider recall without exploding rerank cost. |

Feature flags (Phase 2): `ENABLE_METRICS_LOGGING` (default `True`),
`ENABLE_ADAPTIVE_RETRY` (default `False`).

Diff strategy: commit chunks embed the **message plus a summary** of changed
files/stats, never the full raw diff (too large and noisy to embed usefully).

---

## Project layout

```
app.py            Streamlit UI
config.py         all model names, paths, hyperparameters (nothing hardcoded elsewhere)
core/             repo-URL parsing, manifest (+ reuse rule), RepositoryContext, registry
ingest/           GitHub REST+GraphQL client, per-source fetchers, resumable checkpoint
process/          chunker (schema-tagged chunks), linker (issue↔PR↔commit↔release graph)
index/            Ollama embedder, Chroma vector store + BM25 builder
retrieval/        hybrid retriever (RRF), MMR, cross-encoder reranker, filters
generation/       prompt (evidence delimiting, citations, injection defense), answerer
guard/            reference validator, NLI hallucination verifier
jobs/             background ingestion runner + status file the UI polls
eval/             synthetic gold questions, metrics, 4-config ablation
tests/            pytest suite (mirrors the structure above)
repositories/     per-repo indexed data (gitignored)
```

## Testing

```bash
pytest -q
```

Model- and Ollama-dependent tests skip automatically if those services aren't
available; the pure-logic tests always run.

### Memory note (important on 16 GB machines)

This system holds several models in RAM at once: Ollama's chat model (~4.7 GB
for `qwen2.5:7b-instruct`), the embedding model (~2.4 GB resident), the BGE
reranker (~2.3 GB), and the NLI guard (~0.8 GB). That is fine on its own, but
**do not run the Streamlit app, `pytest`, and an evaluation at the same time** —
on 16 GB that exceeds physical memory, the machine swaps heavily, and the OS may
kill a process silently (a run that stops with no error message is almost always
this). Symptoms: severe lag and a hot machine.

Rules of thumb:

- Run **one** heavy job at a time (app **or** tests **or** an eval).
- Free Ollama's resident models between jobs:
  `curl -s localhost:11434/api/generate -d '{"model":"qwen2.5:7b-instruct","keep_alive":0}'`
- For long evaluations, prefer the smaller generation model
  (`GENERATION_MODEL=qwen2.5:3b`) — roughly half the RAM. Record which model a
  run used; `results.json` stores this under `models`, and the UI warns when a
  displayed run used a different model than the app currently answers with.
- Evaluations checkpoint after every question, so an interrupted run resumes
  from where it stopped — just re-run the same command.

---

## Design guarantees

- Any public GitHub repo URL indexes without code changes.
- Re-visiting an indexed repo reuses the index; a config change forces a rebuild.
- A question outside the indexed date range gets an explicit "not covered"
  answer, never a guess.
- The hallucination guard catches both fabricated citations and claims that
  contradict their cited evidence.
- The entire system runs with zero paid API calls (the judge is optional and
  swappable to local).
