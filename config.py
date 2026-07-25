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

# NLI model for the hallucination guard (sentence-transformers CrossEncoder).
NLI_MODEL: str = os.getenv("NLI_MODEL", "cross-encoder/nli-deberta-v3-base")

# --------------------------------------------------------------------------- #
# RAGAS judge — swappable. One flag chooses the provider; never hardcoded.
#   JUDGE_PROVIDER in {"groq", "gemini", "ollama"}
# --------------------------------------------------------------------------- #
JUDGE_PROVIDER: str = os.getenv("JUDGE_PROVIDER", "groq")
JUDGE_MODEL_GROQ: str = os.getenv("JUDGE_MODEL_GROQ", "llama-3.3-70b-versatile")
JUDGE_MODEL_GEMINI: str = os.getenv("JUDGE_MODEL_GEMINI", "gemini-1.5-flash")
JUDGE_MODEL_OLLAMA: str = os.getenv("JUDGE_MODEL_OLLAMA", "qwen2.5:7b-instruct")
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
MMR_TOP_N: int = 20                     # candidates kept after MMR
FINAL_TOP_K: int = 6                    # final chunks after reranking

# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
GENERATION_TEMPERATURE: float = 0.1
GENERATION_MAX_TOKENS: int = 1024

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


def flag_state() -> dict:
    """The current Phase-2 feature-flag state, recorded with each metrics line."""
    return {
        "ENABLE_METRICS_LOGGING": ENABLE_METRICS_LOGGING,
        "ENABLE_ADAPTIVE_RETRY": ENABLE_ADAPTIVE_RETRY,
        "ENABLE_CROSS_REPO": ENABLE_CROSS_REPO,
    }


def judge_model_name() -> str:
    """Return the judge model name for the currently selected provider."""
    return {
        "groq": JUDGE_MODEL_GROQ,
        "gemini": JUDGE_MODEL_GEMINI,
        "ollama": JUDGE_MODEL_OLLAMA,
    }.get(JUDGE_PROVIDER, JUDGE_MODEL_OLLAMA)


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
