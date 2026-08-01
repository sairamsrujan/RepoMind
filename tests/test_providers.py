"""Multi-provider LLM registry.

The point of the registry is that roles (question generation vs judging) can be
pointed at *different* providers, and that a missing key or a dead free tier
degrades to the local model instead of breaking a run.
"""
from __future__ import annotations

import pytest
import requests

import config
import providers


class _Resp:
    def __init__(self, status=200, body=None, text=""):
        self.status_code = status
        self._body = body or {"choices": [{"message": {"content": "cloud reply"}}]}
        self.text = text or str(self._body)

    def json(self):
        return self._body


# --------------------------------------------------------------------------- #
# Registry shape
# --------------------------------------------------------------------------- #
def test_expected_providers_registered():
    for name in ("groq", "gemini", "nvidia", "openrouter", "ollama"):
        assert name in providers.REGISTRY
    assert providers.REGISTRY["ollama"].is_local


def test_hosted_providers_are_openai_compatible_urls():
    for name, p in providers.REGISTRY.items():
        if p.is_local:
            continue
        assert p.base_url.startswith("https://"), name
        assert p.key_env, f"{name} must name an API-key env var"


def test_unknown_provider_falls_back_to_local():
    assert providers.get("does-not-exist").name == "ollama"
    assert providers.get("").name == "ollama"


def test_availability_depends_on_key(monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    assert providers.REGISTRY["nvidia"].available is False
    monkeypatch.setenv("NVIDIA_API_KEY", "k")
    assert providers.REGISTRY["nvidia"].available is True
    assert providers.REGISTRY["ollama"].available is True   # local always ok


def test_resolve_reports_what_will_actually_run(monkeypatch):
    """A provider with no key must resolve to local, so labels stay honest."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert providers.resolve("gemini").name == "ollama"
    assert providers.describe("gemini").startswith("ollama:")
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    assert providers.resolve("gemini").name == "gemini"
    assert providers.describe("gemini").startswith("gemini:")


# --------------------------------------------------------------------------- #
# chat() behaviour
# --------------------------------------------------------------------------- #
def test_chat_uses_hosted_provider_when_keyed(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    monkeypatch.setattr(requests, "post", lambda *a, **k: _Resp())
    text, used = providers.chat("groq", "hi")
    assert text == "cloud reply"
    assert used.startswith("groq:")


def test_chat_falls_back_to_local_on_http_error(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    calls = {"ollama": 0}

    def fake_post(url, *a, **k):
        if "groq" in url:
            return _Resp(status=429, body={}, text="rate limited")
        calls["ollama"] += 1
        return _Resp(body={"message": {"content": "local reply"}})

    monkeypatch.setattr(requests, "post", fake_post)
    text, used = providers.chat("groq", "hi")
    assert text == "local reply"
    assert calls["ollama"] == 1
    assert used.startswith("ollama:") and "fallback from groq" in used


def test_chat_falls_back_when_cloud_returns_empty(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k")

    def fake_post(url, *a, **k):
        if "groq" in url:
            return _Resp(body={"choices": [{"message": {"content": "   "}}]})
        return _Resp(body={"message": {"content": "local reply"}})

    monkeypatch.setattr(requests, "post", fake_post)
    text, used = providers.chat("groq", "hi")
    assert text == "local reply" and "fallback" in used


def test_chat_can_refuse_to_fall_back(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    monkeypatch.setattr(requests, "post",
                        lambda *a, **k: _Resp(status=500, body={}, text="boom"))
    with pytest.raises(providers.ProviderError):
        providers.chat("groq", "hi", fallback_local=False)


def test_unkeyed_provider_goes_straight_to_local(monkeypatch):
    """No key => never even attempt the hosted call."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    seen = {"urls": []}

    def fake_post(url, *a, **k):
        seen["urls"].append(url)
        return _Resp(body={"message": {"content": "local reply"}})

    monkeypatch.setattr(requests, "post", fake_post)
    text, used = providers.chat("openrouter", "hi")
    assert text == "local reply" and used.startswith("ollama:")
    assert all("openrouter" not in u for u in seen["urls"])


# --------------------------------------------------------------------------- #
# Role separation (the reason this exists)
# --------------------------------------------------------------------------- #
def test_question_generation_and_judging_are_separately_configurable():
    assert hasattr(config, "QUESTIONGEN_PROVIDER")
    assert hasattr(config, "JUDGE_PROVIDER")


def test_roles_can_point_at_different_providers(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setattr(config, "QUESTIONGEN_PROVIDER", "gemini")
    monkeypatch.setattr(config, "JUDGE_PROVIDER", "groq")
    # Different models author and grade -> no self-preference bias.
    assert providers.resolve(config.QUESTIONGEN_PROVIDER).name == "gemini"
    assert providers.resolve(config.JUDGE_PROVIDER).name == "groq"


def test_judge_uses_configured_provider(monkeypatch):
    from eval import metrics
    monkeypatch.setenv("GROQ_API_KEY", "k")
    monkeypatch.setattr(config, "JUDGE_PROVIDER", "groq")
    monkeypatch.setattr(requests, "post", lambda *a, **k: _Resp(
        body={"choices": [{"message": {"content": '{"faithfulness":1,'
                                                  '"answer_relevancy":0.9}'}}]}))
    out = metrics.judge_faithfulness_relevancy("q?", "a [pr_1]", ["evidence"])
    assert out["faithfulness"] == 1.0
    assert out["judge"].startswith("groq:")
