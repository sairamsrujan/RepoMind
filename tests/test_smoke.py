"""Phase 0 placeholder / smoke tests."""
from __future__ import annotations

import config


def test_config_imports_and_has_constants():
    assert config.SCHEMA_VERSION == 1
    assert config.EMBEDDING_MODEL  # non-empty
    assert config.FINAL_TOP_K > 0
    assert ":latest" not in config.EMBEDDING_MODEL, "no :latest tags allowed"
    assert ":latest" not in config.GENERATION_MODEL, "no :latest tags allowed"


def test_pipeline_fingerprint_shape():
    fp = config.pipeline_fingerprint()
    assert set(fp) == {"schema_version", "embedding_model", "chunker_version"}


def test_judge_model_name_matches_provider():
    # Whatever provider is configured, we get a non-empty model name.
    assert config.judge_model_name()


# --------------------------------------------------------------------------- #
# Feature flags must be settable from .env
#
# ENABLE_ADAPTIVE_RETRY and ENABLE_CROSS_REPO were hardcoded, so setting them in
# .env did nothing and the app behaved as before with no error. Documented
# configuration that silently does nothing is worse than no configuration.
# --------------------------------------------------------------------------- #
import importlib

import pytest


@pytest.mark.parametrize("flag,default", [
    ("ENABLE_ADAPTIVE_RETRY", False),
    ("ENABLE_CROSS_REPO", False),
    ("ENABLE_GRAPH_EXPANSION", False),
    ("ENABLE_METRICS_LOGGING", True),
])
def test_feature_flag_reads_from_environment(flag, default, monkeypatch):
    import config as cfg

    # Setting the opposite of the default in the environment must take effect.
    monkeypatch.setenv(flag, "false" if default else "true")
    reloaded = importlib.reload(cfg)
    assert getattr(reloaded, flag) is (not default), (
        f"{flag} ignored the environment — it is probably hardcoded")

    monkeypatch.delenv(flag, raising=False)
    reloaded = importlib.reload(cfg)
    assert getattr(reloaded, flag) is default, f"{flag} lost its default"


def test_flag_parser_treats_falsey_strings_as_off(monkeypatch):
    import config as cfg

    for off in ("0", "false", "FALSE", "no", ""):
        monkeypatch.setenv("ENABLE_ADAPTIVE_RETRY", off)
        assert importlib.reload(cfg).ENABLE_ADAPTIVE_RETRY is False, off
    for on in ("1", "true", "TRUE", "yes"):
        monkeypatch.setenv("ENABLE_ADAPTIVE_RETRY", on)
        assert importlib.reload(cfg).ENABLE_ADAPTIVE_RETRY is True, on
    monkeypatch.delenv("ENABLE_ADAPTIVE_RETRY", raising=False)
    importlib.reload(cfg)


# --------------------------------------------------------------------------- #
# The UI must never show a raw traceback
#
# A stale module (Streamlit re-runs app.py on save without reloading imports)
# put an AttributeError stack trace on screen. In a demo that is the worst
# possible failure mode, so the query path catches everything and explains it.
# --------------------------------------------------------------------------- #
def test_friendly_error_explains_the_stale_module_case():
    import app
    exc = AttributeError("'PipelineResult' object has no attribute 'declined'")
    msg = app._friendly_error(exc)
    assert "stale code" in msg.lower()
    assert "ctrl-c" in msg.lower(), "must say how to fix it, not just what broke"


def test_friendly_error_explains_ollama_being_down():
    import app
    msg = app._friendly_error(RuntimeError("Could not reach Ollama at localhost"))
    assert "ollama" in msg.lower()
    assert "open -a Ollama" in msg


def test_friendly_error_always_returns_something_useful():
    import app
    msg = app._friendly_error(ValueError("something nobody predicted"))
    assert "ValueError" in msg and "something nobody predicted" in msg


def test_declined_field_access_survives_a_stale_pipeline_result():
    """A PipelineResult built before `declined` existed must not crash the UI."""
    class OldResult:            # no `declined` attribute, as before the change
        refusal = False
    assert getattr(OldResult(), "declined", False) is False


# --------------------------------------------------------------------------- #
# Example questions must belong to the repository on screen
#
# All three chips were once hardcoded to the acme_widgets fixture, so on every
# real repository they asked about a startup crash and a caching layer that do
# not exist there — three buttons that made a working system look broken.
# --------------------------------------------------------------------------- #
import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent


def test_every_indexed_repository_has_its_own_questions():
    import app

    indexed = {p.name for p in (_ROOT / "repositories").iterdir()
               if (p / "manifest.json").exists()}
    missing = indexed - set(app.EXAMPLE_QUESTIONS)
    assert not missing, f"indexed but using generic chips: {sorted(missing)}"


def test_question_sets_are_three_and_distinct_across_repositories():
    import app

    seen: dict[str, str] = {}
    for slug, questions in app.EXAMPLE_QUESTIONS.items():
        assert len(questions) == 3, slug
        assert len(set(questions)) == 3, f"{slug} repeats a question"
        for q in questions:
            assert q.strip(), slug
            # Streamlit keys are f"ex_{question}"; a duplicate across repos is
            # harmless today but would collide if two were ever rendered at once.
            assert q not in seen, f"{slug} reuses {seen[q]}'s question: {q}"
            seen[q] = slug


def test_third_question_is_a_verified_unanswerable_one():
    """The decline slot is drawn from the abstention set, not invented."""
    import app

    for slug, questions in app.EXAMPLE_QUESTIONS.items():
        dataset = _ROOT / "eval" / "datasets" / f"{slug}_abstention.jsonl"
        if not dataset.exists():
            continue
        known = {json.loads(line)["question"] for line in
                 dataset.read_text().splitlines() if line.strip()}
        assert questions[2] in known, (
            f"{slug}: chip 3 is not in {dataset.name}, so nothing shows it is "
            f"actually unanswerable")


# --------------------------------------------------------------------------- #
# The citation badge must count citations, not retrieved chunks
#
# It read `len(chunks)`, so an answer citing one PR out of a six-chunk evidence
# set displayed "6 citations verified" — a 6x overstatement of the one number a
# viewer uses to decide whether to trust the answer.
# --------------------------------------------------------------------------- #
def test_citation_badge_counts_citations_not_chunks():
    import app
    from guard.reference_validator import validate_references

    answer = ("Benchmark tests were excluded to speed up coverage [pr_14965]. "
              "This improved efficiency [pr_14965].")
    chunks = [{"chunk_id": f"pr_{n}", "text": "evidence"}
              for n in (14965, 15656, 14347, 15504, 15272, 16075)]
    report = validate_references(answer, chunks)

    text = app._citation_pill_text({
        "citations_ok": report.is_valid,
        "valid_citations": report.valid_citations,
        "invalid_citations": report.invalid_citations,
    })
    assert text == "✓ 1 citation verified", (
        f"badge said {text!r}; it must not report the {len(chunks)} retrieved "
        f"chunks as citations")


def test_citation_badge_pluralises():
    import app

    base = {"citations_ok": True, "invalid_citations": []}
    assert app._citation_pill_text({**base, "valid_citations": ["pr_1", "pr_2"]}) \
        == "✓ 2 citations verified"
    assert app._citation_pill_text({
        "citations_ok": False, "valid_citations": [],
        "invalid_citations": ["pr_999"]}) == "✗ 1 fabricated citation"


def test_unknown_repository_falls_back_to_manifest_derived_questions():
    """Any pasted repo must get sensible chips without a code change."""
    import app

    manifest = {"coverage": {"since": "2024-01-01", "until": "2024-12-31"}}
    first, second, third = app.example_questions("nobody_knows", manifest)
    assert "2024-01-01" in first and "2024-12-31" in first
    assert second
    # Unanswerable by construction: before coverage starts.
    assert "2023" in third


def test_fallback_survives_a_manifest_with_no_coverage_dates():
    import app

    for manifest in ({}, {"coverage": {}}, {"coverage": {"since": None}}):
        questions = app.example_questions("nobody_knows", manifest)
        assert len(questions) == 3 and all(q.strip() for q in questions)
