"""Central configuration for RepoMind.

Every model name, path, default, and tunable hyperparameter lives here as a
named constant. Nothing downstream should hardcode these values inline.

Secrets (GITHUB_TOKEN, optional judge API keys) are read from a `.env` file
via python-dotenv. `.env` must never be committed (see .gitignore).
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT: Path = Path(__file__).resolve().parent
REPOSITORIES_DIR: Path = PROJECT_ROOT / "repositories"

# Load .env from project root (does not override real environment variables).
load_dotenv(PROJECT_ROOT / ".env")

# --------------------------------------------------------------------------- #
# Secrets (from .env / environment)
# --------------------------------------------------------------------------- #
GITHUB_TOKEN: str | None = os.getenv("GITHUB_TOKEN")
GROQ_API_KEY: str | None = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY")

# --------------------------------------------------------------------------- #
# Ollama
# --------------------------------------------------------------------------- #
OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# --------------------------------------------------------------------------- #
# Models — pinned tags. NEVER use ":latest" for reproducibility (see CLAUDE.md).
# --------------------------------------------------------------------------- #
# Embedding model (Ollama). Default per spec; fallback via EMBEDDING_MODEL env.
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "qwen3-embedding:0.6b")
EMBEDDING_MODEL_FALLBACK: str = "nomic-embed-text:v1.5"

# Generation chat model (Ollama).
GENERATION_MODEL: str = os.getenv("GENERATION_MODEL", "qwen2.5:7b-instruct")
GENERATION_MODEL_FALLBACK: str = "llama3.1:8b-instruct"

# Reranker cross-encoder (sentence-transformers / HuggingFace).
RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
RERANKER_MODEL_FALLBACK: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Device for the local sentence-transformers models (reranker + NLI guard).
# Defaults to "cpu" deliberately: PyTorch's MPS (Apple GPU) backend segfaults
# when copying tensors for these cross-encoders, especially with more than one
# process running, which crashes the whole app. These models are small, so CPU
# costs a little latency and buys crash-free demos. Set TORCH_DEVICE=mps to
# opt back into the GPU.
TORCH_DEVICE: str = os.getenv("TORCH_DEVICE", "cpu")

# NLI model for the hallucination guard (sentence-transformers CrossEncoder).
NLI_MODEL: str = os.getenv("NLI_MODEL", "cross-encoder/nli-deberta-v3-base")

# --------------------------------------------------------------------------- #
# Evaluation model roles — three DISTINCT model families, on purpose.
#
# There are three LLM roles in the evaluation loop and they must not collapse
# onto one model:
#
#   1. ANSWERER      generates the answer          (GENERATION_API_MODEL)
#   2. JUDGE         grades faithfulness/relevancy (JUDGE_PROVIDER + JUDGE_MODEL)
#   3. QUESTION-GEN  writes the golden questions   (QUESTIONGEN_* )
#
# If the judge is the same model as the answerer it grades its own output and
# scores itself generously; if it is the same model as the question author it
# rewards its own phrasing. Both are forms of self-preference bias and both
# inflate the reported numbers. Keeping all three on different model families
# is what makes the scores defensible.
#
# JUDGE_PROVIDER / QUESTIONGEN_PROVIDER accept any name in providers.REGISTRY:
#   groq | gemini | nvidia | openrouter | ollama
# --------------------------------------------------------------------------- #
JUDGE_PROVIDER: str = os.getenv("JUDGE_PROVIDER", "nvidia")
QUESTIONGEN_PROVIDER: str = os.getenv("QUESTIONGEN_PROVIDER", "nvidia")

# Per-provider judge models (used when JUDGE_MODEL is not set explicitly).
JUDGE_MODEL_GROQ: str = os.getenv("JUDGE_MODEL_GROQ", "openai/gpt-oss-120b")
JUDGE_MODEL_GEMINI: str = os.getenv("JUDGE_MODEL_GEMINI", "gemini-2.0-flash")
JUDGE_MODEL_NVIDIA: str = os.getenv("JUDGE_MODEL_NVIDIA",
                                    "deepseek-ai/deepseek-v4-pro")
JUDGE_MODEL_OPENROUTER: str = os.getenv(
    "JUDGE_MODEL_OPENROUTER", "meta-llama/llama-3.3-70b-instruct:free")
JUDGE_MODEL_OLLAMA: str = os.getenv("JUDGE_MODEL_OLLAMA", "qwen2.5:7b-instruct")

# Explicit overrides. Set these to pin a role to one exact model regardless of
# provider defaults — needed when two roles share a provider (e.g. judge and
# question-gen both on NVIDIA) and must still use different models.
JUDGE_MODEL: str = os.getenv("JUDGE_MODEL", "")
QUESTIONGEN_MODEL: str = os.getenv("QUESTIONGEN_MODEL",
                                   "nvidia/nemotron-3-nano-30b-a3b")

# --------------------------------------------------------------------------- #
# Provider fallback chains — the reliability mechanism.
#
# Free tiers are not durable. Measured in one afternoon: Gemini returned "no
# longer available" for two models and 429 for the rest, Cerebras returned 402
# (its free tier is not on every account), and Groq's daily token cap was
# exhausted after a few dozen questions. Any single-provider setting will be
# broken long before this project is demonstrated.
#
# Each role therefore has an ORDERED chain: the first entry that answers wins,
# and the local model is always appended last (no key, no network). This is what
# makes "the app still works in nine months" true rather than hoped for.
#
# Format: "provider:model" (first colon splits; model ids may contain ':' and '/').
# Families are kept distinct ACROSS roles so no model ever grades its own work.
# --------------------------------------------------------------------------- #
def _chain(env_name: str, default: list[str]) -> list[str]:
    raw = os.getenv(env_name, "")
    return [s.strip() for s in raw.split(",") if s.strip()] if raw else default


# Model sizing note: on these free tiers the cap is tokens-per-DAY, and the
# input cost is the same whatever model reads it (~950 prompt tokens plus six
# evidence chunks). A 550B model therefore burns the daily budget far faster
# than a 70B one while adding nothing to a task that is about following a
# citation format and staying grounded. Each role below picks the SMALLEST
# model that does its job well, not the largest one available.

# ANSWERER — needs instruction-following (cite every claim as [chunk_id], refuse
# when evidence is thin), not raw scale. This is the demo-facing model, so
# quality is visible; ~70B is the sweet spot. Llama family.
GENERATION_CHAIN: list[str] = _chain("GENERATION_CHAIN", [
    "groq:llama-3.3-70b-versatile",                      # fast, strong follower
    "nvidia:nvidia/llama-3.3-nemotron-super-49b-v1.5",   # 49B, cheaper backup
])

# JUDGE — emits two calibrated floats as JSON. Wants consistency, not
# creativity, and runs offline so latency is irrelevant. DeepSeek family, kept
# deliberately different from the answerer it grades. "flash" first: a grading
# task does not need "pro", and flash costs far less of the daily quota.
JUDGE_CHAIN: list[str] = _chain("JUDGE_CHAIN", [
    "nvidia:deepseek-ai/deepseek-v4-flash",   # cheap, sufficient for scoring
    "nvidia:deepseek-ai/deepseek-v4-pro",     # if flash is overloaded (529)
    "groq:openai/gpt-oss-120b",               # last cloud resort
])

# QUESTION AUTHOR — must infer *why* a change happened from scattered evidence,
# so reasoning genuinely helps here. Runs rarely (once per golden set), and
# nemotron-3-nano is a 30B mixture-of-experts with only ~3B active parameters:
# reasoning behaviour at a fraction of the token cost. Nemotron family.
QUESTIONGEN_CHAIN: list[str] = _chain("QUESTIONGEN_CHAIN", [
    "nvidia:nvidia/nemotron-3-nano-30b-a3b",          # 30B MoE / 3B active
    "openrouter:nvidia/nemotron-nano-9b-v2:free",      # 9B free backup
])

GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"

# --------------------------------------------------------------------------- #
# Pipeline versions — bump to invalidate cached indexes (see manifest reuse rule)
# --------------------------------------------------------------------------- #
SCHEMA_VERSION: int = 1
CHUNKER_VERSION: int = 1

# --------------------------------------------------------------------------- #
# Ingestion
# --------------------------------------------------------------------------- #
DEFAULT_LOOKBACK_MONTHS: int = 12       # default date window if none given
GITHUB_API_URL: str = "https://api.github.com"
GITHUB_GRAPHQL_URL: str = "https://api.github.com/graphql"
GITHUB_REST_PER_PAGE: int = 100         # max page size for REST list endpoints
GITHUB_GRAPHQL_PAGE_SIZE: int = 25      # PRs/issues per GraphQL page
HTTP_TIMEOUT_SECONDS: int = 30
MAX_RETRIES: int = 5                    # for 403/429 backoff
BACKOFF_BASE_SECONDS: float = 2.0       # exponential backoff base
MAX_RATE_WAIT_SECONDS: int = 90         # if a rate-limit reset is farther off
                                        # than this, fail fast instead of
                                        # sleeping through pointless retries

# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #
MAX_CHUNK_TOKENS: int = 512             # target ceiling per chunk
DIFF_SUMMARY_MAX_LINES: int = 40        # truncate commit diffs to N lines
DIFF_SUMMARY_MAX_CHARS: int = 2000      # hard char cap on diff summary

# --------------------------------------------------------------------------- #
# Embedding / indexing
# --------------------------------------------------------------------------- #
EMBED_BATCH_SIZE: int = 16              # chunks embedded per batch call

# --------------------------------------------------------------------------- #
# Retrieval hyperparameters
# --------------------------------------------------------------------------- #
DENSE_TOP_K: int = 30                   # candidates from Chroma (dense)
SPARSE_TOP_K: int = 30                  # candidates from BM25 (sparse)
RRF_K: int = 60                         # Reciprocal Rank Fusion constant
RRF_POOL_SIZE: int = 40                 # merged candidate pool after RRF
MMR_LAMBDA: float = 0.5                 # relevance/diversity trade-off
MMR_TOP_N: int = 12                     # candidates kept after MMR
                                        # (each one costs a cross-encoder
                                        # pass; 12 keeps rerank latency
                                        # demo-acceptable on CPU)
FINAL_TOP_K: int = 6                    # final chunks after reranking

# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
GENERATION_TEMPERATURE: float = 0.1
GENERATION_MAX_TOKENS: int = 1024

# Optional cloud generation. Set GENERATION_PROVIDER=api to answer with a hosted
# OpenAI-compatible model instead of local Ollama. Any failure (missing key,
# auth error, rate limit, timeout, deprecated model) falls back to Ollama when
# GENERATION_API_FALLBACK is on, so the app keeps working offline.
#
# Base URLs for common free-tier providers (all OpenAI-compatible):
#   Groq       https://api.groq.com/openai/v1
#   Gemini     https://generativelanguage.googleapis.com/v1beta/openai/
#   NVIDIA NIM https://integrate.api.nvidia.com/v1
#   OpenRouter https://openrouter.ai/api/v1
GENERATION_PROVIDER: str = os.getenv("GENERATION_PROVIDER", "ollama")  # ollama|api
GENERATION_API_BASE_URL: str = os.getenv(
    "GENERATION_API_BASE_URL", "https://api.groq.com/openai/v1")
GENERATION_API_KEY: str = os.getenv("GENERATION_API_KEY", "")
GENERATION_API_MODEL: str = os.getenv("GENERATION_API_MODEL",
                                      "llama-3.3-70b-versatile")
GENERATION_API_FALLBACK: bool = (
    os.getenv("GENERATION_API_FALLBACK", "true").strip().lower()
    not in ("0", "false", "no"))
GENERATION_API_TIMEOUT: int = int(os.getenv("GENERATION_API_TIMEOUT", "60"))
# Free tiers cap tokens-per-minute. A 429 is transient, so retry with backoff
# before falling back to local — otherwise a long evaluation silently ends up
# half-local and the metrics mix two different models.
GENERATION_API_MAX_RETRIES: int = int(
    os.getenv("GENERATION_API_MAX_RETRIES", "2"))
GENERATION_API_BACKOFF_BASE: float = float(
    os.getenv("GENERATION_API_BACKOFF_BASE", "2"))
# Never stall a query longer than this waiting on a provider. If the provider
# asks for longer (e.g. a daily quota reset), fall back to the local model
# immediately rather than making every query slow before failing over anyway.
GENERATION_API_MAX_WAIT: float = float(
    os.getenv("GENERATION_API_MAX_WAIT", "8"))
# Seconds to pause between generation calls (throttle to stay under TPM caps).
GENERATION_API_THROTTLE: float = float(
    os.getenv("GENERATION_API_THROTTLE", "0"))


def api_generation_enabled() -> bool:
    """True only when cloud generation is both selected and configured."""
    return GENERATION_PROVIDER == "api" and bool(GENERATION_API_KEY)


def effective_generation_model() -> str:
    """The model that will actually answer (for provenance/reporting)."""
    return GENERATION_API_MODEL if api_generation_enabled() else GENERATION_MODEL

# --------------------------------------------------------------------------- #
# Hallucination guard
# --------------------------------------------------------------------------- #
NLI_ENTAILMENT_THRESHOLD: float = 0.5   # min entailment prob to accept a claim
NLI_CONTRADICTION_THRESHOLD: float = 0.6  # min contradiction prob to flag a claim
                                          # (higher than entailment so only a
                                          # confident contradiction is flagged)

# --------------------------------------------------------------------------- #
# Phase 2 feature flags (each defaults to a value that preserves today's
# behaviour; setting a flag to its "off" value disables the feature entirely).
# --------------------------------------------------------------------------- #
ENABLE_METRICS_LOGGING: bool = True     # Phase A: per-query metrics -> JSONL

# Where per-query metrics are appended (one JSON object per line).
METRICS_DIR: Path = PROJECT_ROOT / "data" / "metrics"
METRICS_PATH: Path = METRICS_DIR / "queries.jsonl"

# Phase B: adaptive verification retry. When on, a guard rejection triggers
# exactly ONE widened retry (bigger pools, MMR skipped) before an honest refusal.
ENABLE_ADAPTIVE_RETRY: bool = False
RETRY_POOL_MULTIPLIER: int = 2          # widen dense+BM25 pools by this on retry

# Phase F: cross-repository comparison. When on, one question can be asked
# against several indexed repositories at once (each retrieved independently).
ENABLE_CROSS_REPO: bool = False

# Phase G: graph-aware candidate expansion. When on, a retrieved candidate also
# pulls in the records it is linked to in links.json (issue <-> PR <-> commit
# <-> release), so multi-hop "how did this evolve" questions see the whole chain
# instead of the single best-matching link. Purely additive: neighbours are
# appended to the candidate pool and the reranker still decides the final order.
ENABLE_GRAPH_EXPANSION: bool = (
    os.getenv("ENABLE_GRAPH_EXPANSION", "false").strip().lower()
    not in ("0", "false", "no", ""))
# Bounds — each expanded candidate costs one cross-encoder pass at rerank time,
# and reranking is already the dominant query latency (HANDOFF.md §3.2).
GRAPH_EXPANSION_MAX_SEEDS: int = 6
GRAPH_EXPANSION_MAX_PER_SEED: int = 2
GRAPH_EXPANSION_MAX_TOTAL: int = 6
# How many of the FINAL_TOP_K slots a graph neighbour may occupy.
#
# Measured: without this cap, expansion raised recall (0.510 -> 0.564) but LOWERED
# nDCG (0.531 -> 0.509) — the cross-encoder scored plausible-looking neighbours
# above the actual gold evidence and pushed it down the list. Capping the slots
# keeps the multi-hop recall win without letting neighbours crowd out the
# similarity hits that were already correct.
GRAPH_EXPANSION_MAX_IN_TOPK: int = 2


def flag_state() -> dict:
    """The current Phase-2 feature-flag state, recorded with each metrics line."""
    return {
        "ENABLE_METRICS_LOGGING": ENABLE_METRICS_LOGGING,
        "ENABLE_ADAPTIVE_RETRY": ENABLE_ADAPTIVE_RETRY,
        "ENABLE_CROSS_REPO": ENABLE_CROSS_REPO,
        "ENABLE_GRAPH_EXPANSION": ENABLE_GRAPH_EXPANSION,
        "GENERATION_PROVIDER": GENERATION_PROVIDER,
    }


def judge_model_name() -> str:
    """The model that grades answers, for the currently selected provider.

    An explicit ``JUDGE_MODEL`` wins, so a role can be pinned to one exact model
    even when it shares a provider with another role.
    """
    if JUDGE_MODEL:
        return JUDGE_MODEL
    return {
        "groq": JUDGE_MODEL_GROQ,
        "gemini": JUDGE_MODEL_GEMINI,
        "nvidia": JUDGE_MODEL_NVIDIA,
        "openrouter": JUDGE_MODEL_OPENROUTER,
        "ollama": JUDGE_MODEL_OLLAMA,
    }.get(JUDGE_PROVIDER, JUDGE_MODEL_OLLAMA)


def questiongen_model_name() -> str:
    """The model that writes golden-set questions.

    Empty means "use the provider's default", which is only safe while
    question-gen and the judge sit on different providers.
    """
    return QUESTIONGEN_MODEL


def evaluation_roles() -> dict[str, str]:
    """Who fills each evaluation role right now — recorded with every run.

    Reviewers should be able to confirm at a glance that the answerer, judge and
    question author are three different models (see the note above).
    """
    return {
        "answerer": GENERATION_CHAIN[0] if GENERATION_CHAIN
        else effective_generation_model(),
        "judge": JUDGE_CHAIN[0] if JUDGE_CHAIN
        else f"{JUDGE_PROVIDER}:{judge_model_name()}",
        "question_gen": QUESTIONGEN_CHAIN[0] if QUESTIONGEN_CHAIN
        else f"{QUESTIONGEN_PROVIDER}:{questiongen_model_name()}",
    }


def roles_are_distinct() -> tuple[bool, str]:
    """True when no two evaluation roles share a model (self-preference check).

    Comparison is on the *canonical* model, so the same underlying model offered
    under different vendor packaging — Groq's ``openai/gpt-oss-120b`` versus
    Cerebras's ``gpt-oss-120b`` — is correctly detected as a clash rather than
    passing as two different judges.
    """
    import providers

    roles = evaluation_roles()
    bare = {k: providers.canonical_model(v.split(":", 1)[-1])
            for k, v in roles.items()}
    names = list(bare)
    clashes = [
        f"{a} and {b} are both {bare[a]!r}"
        for i, a in enumerate(names) for b in names[i + 1:]
        if bare[a] == bare[b]
    ]
    return (not clashes), "; ".join(clashes) if clashes else "all roles distinct"


def pipeline_fingerprint() -> dict:
    """The subset of config that determines index reusability.

    An existing index is reusable only if these values match the manifest
    (see the reuse rule in CLAUDE.md / core/manifest.py).
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "embedding_model": EMBEDDING_MODEL,
        "chunker_version": CHUNKER_VERSION,
    }
