# AGENTS.md — orientation for AI coding tools

Read this first, then [`HANDOFF.md`](HANDOFF.md) before changing anything.
`HANDOFF.md` lists the settings that look wrong but are deliberate.

## What this is

RepoMind answers *"why did this repository evolve this way?"* over a GitHub
project's commits, PRs, issues and reviews. Every claim carries an inline
`[chunk_id]` citation, and a two-stage guard verifies those citations before the
answer is displayed. When the evidence isn't there, it declines.

Final-year B.Tech project. Runs locally via `streamlit run app.py`. **It is
complete and feature-frozen** — bug fixes and documentation only.

## Status

| | |
|---|---|
| Tests | 266 passing (`pytest -q`, Ollama running) |
| Indexed | 5 real repositories + 1 deterministic fixture |
| Benchmark | 400 questions across the 5 real repos |
| Durability | `python scripts/smoke_test.py` — 11 checks, run monthly |

## Layout

```
app.py            Streamlit UI          query_pipeline.py  retrieve→generate→guard→retry
config.py         all tunables          providers.py       LLM providers + fallback chains
core/             URL · manifest · registry · paths
ingest/           GitHub REST + GraphQL, checkpointed
process/          chunker · linker (evolution graph)
index/            embedder · Chroma + BM25
retrieval/        RRF · MMR · reranker · graph expansion
generation/       prompt · answerer
guard/            reference validator · NLI verifier
eval/             golden sets · metrics · ablation   (OFFLINE ONLY)
```

## Rules that must not be broken

1. **Nothing in `retrieval/`, `generation/`, `guard/`, `jobs/` or `app.py` may
   import from `eval/`.** The in-app evaluation panel launches `eval/run.py` as a
   subprocess to preserve this. Deleting `eval/` must leave the app working.
2. **Never walk a provider chain on the live query path.** A user waiting on a
   question gets one cloud attempt, then local. Chains are for offline
   evaluation. The exception is a fast 429, which rotates models — see
   `GENERATION_ROTATION` in `config.py`.
3. **Pin every dependency with `==` and every Ollama tag explicitly.** No
   `:latest`, ever. The project must run unattended for a year.
4. **Never put a reasoning model in the judge or generation chain.** One was
   measured at 88s per call against 0.6s — 147× — turning a 20-minute evaluation
   into 12 hours. Time any candidate on the real prompt first.
5. **The three evaluation roles must be three different model families.**
   `config.roles_are_distinct()` enforces it; every `results.json` records the
   result. A judge sharing a model with the answerer grades its own work.

## Working here

```bash
.venv/bin/python -m pytest -q          # tests (needs Ollama for a few)
.venv/bin/python scripts/smoke_test.py # durability checks
.venv/bin/python scripts/check_providers.py
open -a Ollama                         # the app, not `ollama serve`
```

Streamlit re-executes `app.py` on save but does **not** reload imported modules —
after editing anything outside `app.py`, restart the process or you are testing
stale code. Several bugs here were caused by exactly that.

Feature flags default off and are read from `.env` via `config._flag()`. If you
add one, add a parametrised test — a hardcoded flag that silently ignores its
documented environment variable has shipped here before.
