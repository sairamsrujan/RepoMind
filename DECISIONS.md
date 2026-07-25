# RepoMind — Decision Log

One short entry per significant design decision, so the rationale survives the
gap before the demo. Each has what we **Chose**, what we chose it **Over**, the
reason, the cost, and where the **Evidence** lives. The `**Tried and rejected:**`
line is left blank to fill in by hand.

---

## Hybrid retrieval over dense-only
- **Chose:** dense (Chroma cosine) + sparse (BM25), merged with RRF
- **Over:** dense-only semantic retrieval
- **Because:** identifier-style queries (PR numbers, commit SHAs, flag names)
  are lexical, not semantic — dense embeddings miss exact tokens BM25 nails.
- **Cost:** a second index (BM25 pickle) built and held in memory per repo.
- **Evidence:** ablation `dense-only` vs `sparse-only` vs hybrid, by `query_type`.
- **Tried and rejected:**

## RRF over weighted score fusion
- **Chose:** Reciprocal Rank Fusion (rank-based)
- **Over:** normalising and linearly weighting dense/BM25 *scores*
- **Because:** dense cosine distances and BM25 scores are on different, unstable
  scales; rank fusion needs no per-corpus tuning and is robust across repos.
- **Cost:** discards score magnitude (uses only rank position).
- **Evidence:** `retrieval/retriever.py::rrf_fuse`.
- **Tried and rejected:**

## RRF_K = 60
- **Chose:** `RRF_K = 60`
- **Over:** smaller K (sharper top-rank preference) or larger K (flatter)
- **Because:** the value from the original RRF paper; dampens the tail so the
  top few ranks dominate without one channel steamrolling the other.
- **Cost:** not tuned per repository.
- **Evidence:** `config.py::RRF_K`.
- **Tried and rejected:**

## MMR_LAMBDA = 0.5
- **Chose:** `MMR_LAMBDA = 0.5` (equal relevance vs diversity)
- **Over:** higher lambda (pure relevance) or lower (more diversity)
- **Because:** commit/PR histories contain many near-duplicate chunks; 0.5
  removes redundant evidence without dropping the most relevant hit.
- **Cost:** on tiny corpora, diversification can push a relevant near-duplicate
  out of the top-k (visible on the 11-chunk demo repo).
- **Evidence:** ablation config 1 vs 2; `tests/test_retrieval.py`.
- **Tried and rejected:**

## FINAL_TOP_K = 6
- **Chose:** 6 chunks handed to the generator
- **Over:** fewer (less context) or more (dilution, higher latency, drift)
- **Because:** enough evidence to answer "why did X evolve" questions while
  staying small enough to keep the answer grounded and the prompt short.
- **Cost:** multi-hop questions needing >6 sources can be under-served.
- **Evidence:** `config.py::FINAL_TOP_K`; ablation recall@k / citation recall.
- **Tried and rejected:**

## Cross-encoder reranker over a bi-encoder
- **Chose:** BAAI/bge-reranker-v2-m3 via sentence-transformers CrossEncoder
- **Over:** bi-encoder re-scoring, or no reranking at all
- **Because:** a cross-encoder jointly attends to (query, chunk) and reorders
  the shortlist far more accurately than cosine on independent embeddings.
- **Cost:** the dominant query latency (first call loads a ~2 GB model; warm
  reranks add seconds); only run on the ~20 MMR survivors to bound it.
- **Evidence:** ablation config 2 vs 3.
- **Tried and rejected:**

## Two-stage hallucination guard
- **Chose:** reference validator (citations must be real) + NLI entailment check
- **Over:** a single LLM "is this faithful?" self-check
- **Because:** the two stages catch different failures deterministically — a
  fabricated `chunk_id` vs a claim that contradicts its cited evidence — with no
  extra LLM call and no self-grading bias.
- **Cost:** an NLI model load; some conservative false-positive "unverified".
- **Evidence:** `tests/test_guard.py` (fake citation + contradiction caught).
- **Tried and rejected:**

## Asymmetric NLI thresholds (0.5 entailment vs 0.6 contradiction)
- **Chose:** accept a claim at entailment ≥ 0.5; flag a contradiction only at
  ≥ 0.6, and only when nothing entails it
- **Over:** a single symmetric threshold
- **Because:** sentence-level NLI over short evidence is noisy; a higher bar to
  *reject* avoids falsely flagging well-supported claims (e.g. a cited issue
  phrased as a question), while a lower bar to *accept* keeps recall.
- **Cost:** a genuinely weak-but-not-contradicted claim lands in "unverified".
- **Evidence:** `guard/nli_verifier.py`; the caching-answer false-positive fix.
- **Tried and rejected:**

## Hand-rolled swappable judge over the `ragas` package
- **Chose:** a config-driven LLM-judge (Groq / Gemini / local Ollama) computing
  RAGAS-style faithfulness + answer relevancy
- **Over:** the `ragas` + `langchain` dependency stack
- **Because:** the hard constraint is running 1+ year unattended with every dep
  pinned; `ragas`/`langchain` churn fast and would break a clean rebuild.
- **Cost:** we re-implement two metrics rather than import a maintained library.
- **Evidence:** `eval/metrics.py`; README "Swappable judge".
- **Tried and rejected:**

## Diff-summary chunking over raw diffs
- **Chose:** commit chunks = message + a *summary* of changed files/stats
- **Over:** embedding the full raw diff
- **Because:** full diffs are huge and noisy, blow the embedding context, and
  bury the semantic signal (the message) in boilerplate.
- **Cost:** fine-grained "which exact line changed" questions aren't answerable.
- **Evidence:** `process/chunker.py`; `config.DIFF_SUMMARY_*`.
- **Tried and rejected:**

## Manifest reuse fingerprint
- **Chose:** reuse an index only if `schema_version`, `embedding_model`,
  `embedding_dim`, and `chunker_version` all match current config
- **Over:** always rebuild, or trust the index blindly
- **Because:** re-visiting a repo must be instant, but a changed embedding model
  or chunker silently corrupts retrieval — the fingerprint forces a rebuild only
  when it actually matters.
- **Cost:** bumping any fingerprint field invalidates every existing index.
- **Evidence:** `core/manifest.py::is_reusable`; `tests/test_manifest_and_registry.py`.
- **Tried and rejected:**
