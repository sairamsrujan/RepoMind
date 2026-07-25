"""Phase G test: the smoke-test's model-presence check logic (pure)."""
from __future__ import annotations

import config
from scripts import smoke_test


def test_models_present_matches_base_and_latest(monkeypatch):
    # Ollama reports ":latest"-style tags; the check must match on the base name.
    monkeypatch.setattr(smoke_test, "_ollama_tags",
                        lambda: [config.EMBEDDING_MODEL,
                                 config.GENERATION_MODEL.split(":")[0] + ":latest"])
    ok, msg = smoke_test.check_models_present()
    assert ok, msg


def test_models_present_reports_missing(monkeypatch):
    monkeypatch.setattr(smoke_test, "_ollama_tags", lambda: ["something-else:1"])
    ok, msg = smoke_test.check_models_present()
    assert not ok and "missing" in msg


def test_check_functions_return_tuples(monkeypatch):
    # Ollama unreachable -> reachable check fails gracefully with a message.
    def boom():
        raise RuntimeError("no ollama")
    monkeypatch.setattr(smoke_test, "_ollama_tags", boom)
    ok, msg = smoke_test.check_ollama_reachable()
    assert ok is False and isinstance(msg, str)
