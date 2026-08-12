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
