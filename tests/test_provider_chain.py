"""Tests for the provider fallback chain — the 9-month reliability mechanism.

Free tiers retire models, exhaust quotas and return 402/429/503 without notice.
The chain exists so that none of those can take the app down: each entry is
tried in order and the local model always terminates it.
"""
from __future__ import annotations

import pytest

import config
import providers


# --------------------------------------------------------------------------- #
# Spec parsing
# --------------------------------------------------------------------------- #
def test_parse_spec_splits_provider_and_model():
    assert providers.parse_spec("groq:llama-3.3-70b-versatile") == (
        "groq", "llama-3.3-70b-versatile")


def test_parse_spec_keeps_colons_inside_model_ids():
    """OpenRouter free models carry a ':free' suffix — only the FIRST colon splits."""
    name, model = providers.parse_spec(
        "openrouter:nvidia/nemotron-3-super-120b-a12b:free")
    assert name == "openrouter"
    assert model == "nvidia/nemotron-3-super-120b-a12b:free"


def test_parse_spec_without_model():
    assert providers.parse_spec("ollama") == ("ollama", None)


# --------------------------------------------------------------------------- #
# Canonical model identity
# --------------------------------------------------------------------------- #
def test_canonical_model_strips_vendor_prefix():
    """Groq's openai/gpt-oss-120b and Cerebras's gpt-oss-120b are ONE model."""
    assert (providers.canonical_model("openai/gpt-oss-120b")
            == providers.canonical_model("gpt-oss-120b"))


def test_canonical_model_strips_free_suffix_and_google_prefix():
    assert providers.canonical_model("nvidia/nemotron-nano-9b-v2:free") == \
        "nemotron-nano-9b-v2"
    assert providers.canonical_model("models/gemini-2.5-flash") == "gemini-2.5-flash"


def test_canonical_model_distinguishes_genuinely_different_models():
    assert (providers.canonical_model("groq/llama-3.3-70b-versatile")
            != providers.canonical_model("nvidia/deepseek-v4-pro"))


# --------------------------------------------------------------------------- #
# Chain behaviour
# --------------------------------------------------------------------------- #
def test_chain_falls_through_to_the_next_provider(monkeypatch):
    """A dead first provider must cost quality, never availability."""
    calls: list[str] = []

    def fake_openai(p, prompt, model, temperature, timeout):
        calls.append(p.name)
        if p.name == "groq":
            raise providers.ProviderError("429 daily quota exhausted")
        return "second provider answered"

    monkeypatch.setattr(providers, "_chat_openai_compatible", fake_openai)
    monkeypatch.setattr(providers.Provider, "api_key",
                        property(lambda self: "test-key"))

    text, used = providers.chat_chain(
        ["groq:llama-3.3-70b-versatile", "nvidia:deepseek-ai/deepseek-v4-flash"],
        "hello")
    assert text == "second provider answered"
    assert used.startswith("nvidia:")
    assert calls == ["groq", "nvidia"]


def test_chain_always_ends_at_local(monkeypatch):
    """Every cloud provider failing must still produce an answer."""
    def dead(*a, **k):
        raise providers.ProviderError("provider down")

    monkeypatch.setattr(providers, "_chat_openai_compatible", dead)
    monkeypatch.setattr(providers, "_chat_ollama",
                        lambda prompt, model, temp, timeout: "local answered")
    monkeypatch.setattr(providers.Provider, "api_key",
                        property(lambda self: "test-key"))

    text, used = providers.chat_chain(["groq:x", "nvidia:y"], "hello")
    assert text == "local answered"
    assert used.startswith("ollama")


def test_chain_skips_providers_with_no_key(monkeypatch):
    """An unconfigured provider is skipped, not treated as a failure."""
    seen: list[str] = []

    def fake_openai(p, prompt, model, temperature, timeout):
        seen.append(p.name)
        return "ok"

    monkeypatch.setattr(providers, "_chat_openai_compatible", fake_openai)
    monkeypatch.setattr(providers.Provider, "api_key",
                        property(lambda self: "" if self.name == "groq" else "k"))

    _text, used = providers.chat_chain(["groq:a", "nvidia:b"], "hello")
    assert seen == ["nvidia"], "provider without a key must be skipped"
    assert used.startswith("nvidia:")


def test_chain_treats_empty_completion_as_failure(monkeypatch):
    """An empty response is a failure — otherwise a blank answer looks valid."""
    def fake_openai(p, prompt, model, temperature, timeout):
        return "   " if p.name == "groq" else "real answer"

    monkeypatch.setattr(providers, "_chat_openai_compatible", fake_openai)
    monkeypatch.setattr(providers.Provider, "api_key",
                        property(lambda self: "k"))

    text, used = providers.chat_chain(["groq:a", "nvidia:b"], "hello")
    assert text == "real answer"
    assert used.startswith("nvidia:")


def test_chain_raises_only_when_even_local_fails(monkeypatch):
    def dead(*a, **k):
        raise providers.ProviderError("down")

    monkeypatch.setattr(providers, "_chat_openai_compatible", dead)
    monkeypatch.setattr(providers, "_chat_ollama", dead)
    monkeypatch.setattr(providers.Provider, "api_key",
                        property(lambda self: "k"))

    with pytest.raises(providers.ProviderError, match="every provider"):
        providers.chat_chain(["groq:a"], "hello")


# --------------------------------------------------------------------------- #
# Role configuration
# --------------------------------------------------------------------------- #
def test_configured_roles_are_three_distinct_models():
    """The answerer must never grade its own output (self-preference bias)."""
    distinct, detail = config.roles_are_distinct()
    assert distinct, detail


def test_every_role_chain_is_non_empty_and_well_formed():
    for chain in (config.GENERATION_CHAIN, config.JUDGE_CHAIN,
                  config.QUESTIONGEN_CHAIN):
        assert chain, "a role chain must not be empty"
        for spec in chain:
            name, model = providers.parse_spec(spec)
            assert name in providers.REGISTRY, f"unknown provider {name!r}"
            assert model, f"chain entry {spec!r} must pin an explicit model"


def test_role_chains_do_not_collapse_onto_one_model():
    """Even the fallbacks should not make two roles share a model family."""
    heads = {
        "answerer": providers.canonical_model(
            providers.parse_spec(config.GENERATION_CHAIN[0])[1]),
        "judge": providers.canonical_model(
            providers.parse_spec(config.JUDGE_CHAIN[0])[1]),
        "questiongen": providers.canonical_model(
            providers.parse_spec(config.QUESTIONGEN_CHAIN[0])[1]),
    }
    assert len(set(heads.values())) == 3, f"roles collapsed: {heads}"
