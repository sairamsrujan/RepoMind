"""Cloud generation path + automatic local fallback.

The fallback is the safety net that makes cloud generation acceptable for a
demo, so it is tested against every failure mode a provider can throw at us:
missing key, auth error, rate limit, timeout, deprecated model, junk response.
"""
from __future__ import annotations

import pytest
import requests

import config
from generation.answerer import Answerer, AnswerResult, GenerationError

CHUNKS = [{"chunk_id": "pr_1", "source_type": "pr", "text": "PR #1 fixed it.",
           "github_url": "https://example.com/pr/1"}]


@pytest.fixture
def api_on(monkeypatch):
    monkeypatch.setattr(config, "GENERATION_PROVIDER", "api")
    monkeypatch.setattr(config, "GENERATION_API_KEY", "test-key")
    monkeypatch.setattr(config, "GENERATION_API_MODEL", "big-cloud-model")
    monkeypatch.setattr(config, "GENERATION_API_FALLBACK", True)


def _local(monkeypatch, text="Local answer [pr_1]."):
    monkeypatch.setattr(Answerer, "_chat_ollama", lambda self, m: text)


# --------------------------------------------------------------------------- #
# Default behaviour must be unchanged
# --------------------------------------------------------------------------- #
def test_default_uses_local_and_never_calls_api(monkeypatch):
    monkeypatch.setattr(config, "GENERATION_PROVIDER", "ollama")
    _local(monkeypatch)
    monkeypatch.setattr(Answerer, "_chat_api", lambda self, m: pytest.fail(
        "API must not be called when provider is ollama"))
    r = Answerer().answer("q?", CHUNKS)
    assert r.text == "Local answer [pr_1]."
    assert r.fell_back is False
    assert r.cited_chunk_ids == ["pr_1"]


def test_api_selected_without_key_stays_local(monkeypatch):
    monkeypatch.setattr(config, "GENERATION_PROVIDER", "api")
    monkeypatch.setattr(config, "GENERATION_API_KEY", "")   # not configured
    _local(monkeypatch)
    monkeypatch.setattr(Answerer, "_chat_api", lambda self, m: pytest.fail(
        "API must not be called without a key"))
    assert config.api_generation_enabled() is False
    assert Answerer().answer("q?", CHUNKS).fell_back is False


# --------------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------------- #
def test_api_success_reports_api_model(api_on, monkeypatch):
    monkeypatch.setattr(Answerer, "_chat_api",
                        lambda self, m: "Cloud answer [pr_1].")
    r = Answerer().answer("q?", CHUNKS)
    assert r.text == "Cloud answer [pr_1]."
    assert r.model == "big-cloud-model"      # provenance is honest
    assert r.fell_back is False


# --------------------------------------------------------------------------- #
# Every provider failure mode must fall back, not crash
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("boom", [
    GenerationError("API error 401: invalid api key"),        # bad/expired key
    GenerationError("API error 429: rate limit exceeded"),    # rate limited
    GenerationError("API error 404: model decommissioned"),   # model retired
    GenerationError("API request failed: timeout"),           # network timeout
    GenerationError("API returned an empty completion"),      # junk response
    requests.RequestException("connection reset"),            # raw network error
    ValueError("totally unexpected"),                         # anything else
])
def test_api_failure_falls_back_to_local(api_on, monkeypatch, boom):
    def _raise(self, m):
        raise boom
    monkeypatch.setattr(Answerer, "_chat_api", _raise)
    _local(monkeypatch)

    r = Answerer().answer("q?", CHUNKS)
    assert r.text == "Local answer [pr_1]."     # user still gets an answer
    assert r.fell_back is True
    assert r.fallback_reason                    # reason recorded
    assert r.model == config.GENERATION_MODEL   # provenance shows local model


def test_fallback_can_be_disabled(api_on, monkeypatch):
    monkeypatch.setattr(config, "GENERATION_API_FALLBACK", False)

    def _raise(self, m):
        raise GenerationError("API error 500")
    monkeypatch.setattr(Answerer, "_chat_api", _raise)
    with pytest.raises(GenerationError):
        Answerer().answer("q?", CHUNKS)


# --------------------------------------------------------------------------- #
# Config helpers
# --------------------------------------------------------------------------- #
def test_effective_generation_model_tracks_provider(monkeypatch):
    monkeypatch.setattr(config, "GENERATION_PROVIDER", "ollama")
    assert config.effective_generation_model() == config.GENERATION_MODEL
    monkeypatch.setattr(config, "GENERATION_PROVIDER", "api")
    monkeypatch.setattr(config, "GENERATION_API_KEY", "k")
    monkeypatch.setattr(config, "GENERATION_API_MODEL", "cloud-x")
    assert config.effective_generation_model() == "cloud-x"


def test_api_response_parsing(api_on, monkeypatch):
    """_chat_api must parse a normal OpenAI-shaped response."""
    class FakeResp:
        status_code = 200
        text = ""
        def json(self):
            return {"choices": [{"message": {"content": "parsed [pr_1]"}}]}
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResp())
    assert Answerer()._chat_api([{"role": "user", "content": "x"}]) == "parsed [pr_1]"


def test_api_http_error_raises_generation_error(api_on, monkeypatch):
    """A non-retryable 4xx surfaces immediately (429 is covered separately)."""
    class FakeResp:
        status_code = 400
        text = "bad request"
        headers = {}
        def json(self):
            return {}
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResp())
    with pytest.raises(GenerationError, match="400"):
        Answerer()._chat_api([{"role": "user", "content": "x"}])


# --------------------------------------------------------------------------- #
# Rate-limit handling: 429 is transient, retry before falling back
# --------------------------------------------------------------------------- #
class _Resp:
    def __init__(self, status, body=None, headers=None):
        self.status_code = status
        self._body = body or {}
        self.text = str(body)
        self.headers = headers or {}
    def json(self):
        return self._body


_OK = {"choices": [{"message": {"content": "cloud answer [pr_1]"}}]}


def test_429_is_retried_then_succeeds(api_on, monkeypatch):
    monkeypatch.setattr(config, "GENERATION_API_BACKOFF_BASE", 0)  # no real sleep
    calls = {"n": 0}

    def fake_post(*a, **k):
        calls["n"] += 1
        return _Resp(429, {"error": "rate limit"}) if calls["n"] < 3 else _Resp(200, _OK)

    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr("generation.answerer.time.sleep", lambda s: None)
    assert Answerer()._chat_api([{"role": "user", "content": "x"}]) == "cloud answer [pr_1]"
    assert calls["n"] == 3          # retried twice, then succeeded


def test_429_exhausts_retries_then_falls_back(api_on, monkeypatch):
    monkeypatch.setattr(config, "GENERATION_API_MAX_RETRIES", 2)
    monkeypatch.setattr(config, "GENERATION_API_BACKOFF_BASE", 0)
    monkeypatch.setattr(requests, "post", lambda *a, **k: _Resp(429, {"e": "rl"}))
    monkeypatch.setattr("generation.answerer.time.sleep", lambda s: None)
    _local(monkeypatch)

    r = Answerer().answer("q?", CHUNKS)
    assert r.fell_back is True and "429" in r.fallback_reason


def test_retry_after_header_is_honoured(api_on, monkeypatch):
    slept = []
    monkeypatch.setattr("generation.answerer.time.sleep", lambda s: slept.append(s))
    seq = [_Resp(429, {}, {"Retry-After": "7"}), _Resp(200, _OK)]
    monkeypatch.setattr(requests, "post", lambda *a, **k: seq.pop(0))
    Answerer()._chat_api([{"role": "user", "content": "x"}])
    assert slept == [7.0]           # waited exactly what the provider asked


def test_5xx_also_retried(api_on, monkeypatch):
    monkeypatch.setattr(config, "GENERATION_API_BACKOFF_BASE", 0)
    seq = [_Resp(503, {}), _Resp(200, _OK)]
    monkeypatch.setattr(requests, "post", lambda *a, **k: seq.pop(0))
    monkeypatch.setattr("generation.answerer.time.sleep", lambda s: None)
    assert Answerer()._chat_api([{"role": "user", "content": "x"}]) == "cloud answer [pr_1]"


def test_4xx_not_retried(api_on, monkeypatch):
    """A bad key is permanent — retrying wastes time, fail straight to fallback."""
    calls = {"n": 0}
    def fake_post(*a, **k):
        calls["n"] += 1
        return _Resp(401, {"error": "invalid key"})
    monkeypatch.setattr(requests, "post", fake_post)
    with pytest.raises(GenerationError, match="401"):
        Answerer()._chat_api([{"role": "user", "content": "x"}])
    assert calls["n"] == 1          # no pointless retries


# --------------------------------------------------------------------------- #
# Daily-quota exhaustion must fail over FAST, not stall every query
# --------------------------------------------------------------------------- #
def test_daily_quota_falls_back_immediately(api_on, monkeypatch):
    """A 'tokens per day' 429 clears in minutes — never sit on it."""
    slept = []
    monkeypatch.setattr("generation.answerer.time.sleep", lambda s: slept.append(s))
    body = {"error": {"message": "Rate limit reached ... on tokens per day (TPD): "
                                 "Limit 100000, Used 99911"}}
    resp = _Resp(429, body, {"Retry-After": "1243"})
    monkeypatch.setattr(requests, "post", lambda *a, **k: resp)
    _local(monkeypatch)

    r = Answerer().answer("q?", CHUNKS)
    assert r.fell_back is True
    assert sum(slept) == 0, f"must not wait on a daily quota, slept {slept}"


def test_long_retry_after_is_not_honoured(api_on, monkeypatch):
    """Retry-After longer than GENERATION_API_MAX_WAIT => go local now."""
    slept = []
    monkeypatch.setattr("generation.answerer.time.sleep", lambda s: slept.append(s))
    monkeypatch.setattr(requests, "post",
                        lambda *a, **k: _Resp(429, {"e": "slow down"},
                                              {"Retry-After": "600"}))
    _local(monkeypatch)
    Answerer().answer("q?", CHUNKS)
    assert sum(slept) == 0


def test_short_burst_limit_is_still_retried(api_on, monkeypatch):
    """A per-minute burst limit is transient — that one IS worth retrying."""
    slept = []
    monkeypatch.setattr("generation.answerer.time.sleep", lambda s: slept.append(s))
    seq = [_Resp(429, {"error": "per minute burst"}, {"Retry-After": "2"}),
           _Resp(200, _OK)]
    monkeypatch.setattr(requests, "post", lambda *a, **k: seq.pop(0))
    assert Answerer()._chat_api([{"role": "user", "content": "x"}]) == "cloud answer [pr_1]"
    assert slept == [2.0]


def test_wait_is_capped_by_max_wait(api_on, monkeypatch):
    """Even a retryable wait never exceeds the configured ceiling."""
    monkeypatch.setattr(config, "GENERATION_API_MAX_WAIT", 3.0)
    slept = []
    monkeypatch.setattr("generation.answerer.time.sleep", lambda s: slept.append(s))
    seq = [_Resp(503, {}), _Resp(200, _OK)]
    monkeypatch.setattr(requests, "post", lambda *a, **k: seq.pop(0))
    Answerer()._chat_api([{"role": "user", "content": "x"}])
    assert all(s <= 3.0 for s in slept), slept


# --------------------------------------------------------------------------- #
# use_chain: offline evaluation walks the provider chain; interactive does not
#
# The two paths have opposite priorities. Interactive must stay responsive, so
# it drops straight to local (each extra cloud attempt costs a full timeout —
# HANDOFF.md 3.4). Offline evaluation must preserve provenance, so it tries the
# remaining cloud providers before settling for a much smaller local model.
# --------------------------------------------------------------------------- #
def test_interactive_does_not_walk_the_chain(api_on, monkeypatch):
    """The default path must never make extra cloud calls — that costs latency."""
    import providers

    called = []
    monkeypatch.setattr(providers, "chat_chain",
                        lambda *a, **k: called.append(1) or ("x", "nvidia:m"))
    monkeypatch.setattr(Answerer, "_chat_api",
                        lambda self, m: (_ for _ in ()).throw(GenerationError("429")))
    _local(monkeypatch)

    r = Answerer().answer("q?", CHUNKS)          # use_chain defaults to False
    assert not called, "interactive path must not try more cloud providers"
    assert r.model == config.GENERATION_MODEL
    assert r.fell_back is True


def test_evaluation_path_falls_through_to_the_next_cloud_provider(
        api_on, monkeypatch):
    import providers

    monkeypatch.setattr(config, "GENERATION_CHAIN",
                        ["groq:llama-3.3-70b-versatile", "nvidia:backup-model"])
    monkeypatch.setattr(Answerer, "_chat_api",
                        lambda self, m: (_ for _ in ()).throw(
                            GenerationError("429 daily quota")))
    monkeypatch.setattr(providers, "chat_chain",
                        lambda chain, prompt, **k: ("Cloud backup [pr_1].",
                                                    "nvidia:backup-model"))
    _local(monkeypatch)

    r = Answerer(use_chain=True).answer("q?", CHUNKS)
    assert r.text == "Cloud backup [pr_1]."
    assert r.model == "nvidia:backup-model", "provenance must name the real model"
    assert r.fell_back is True
    assert "429" in r.fallback_reason


def test_evaluation_path_still_ends_at_local_when_all_cloud_fails(
        api_on, monkeypatch):
    import providers

    monkeypatch.setattr(config, "GENERATION_CHAIN", ["groq:a", "nvidia:b"])
    monkeypatch.setattr(Answerer, "_chat_api",
                        lambda self, m: (_ for _ in ()).throw(GenerationError("boom")))
    monkeypatch.setattr(providers, "chat_chain",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down")))
    _local(monkeypatch)

    r = Answerer(use_chain=True).answer("q?", CHUNKS)
    assert r.text == "Local answer [pr_1]."
    assert r.model == config.GENERATION_MODEL


def test_chain_result_that_resolved_locally_is_not_reported_as_cloud(
        api_on, monkeypatch):
    """chat_chain appends ollama itself; that must not be mistaken for cloud."""
    import providers

    monkeypatch.setattr(config, "GENERATION_CHAIN", ["groq:a", "nvidia:b"])
    monkeypatch.setattr(Answerer, "_chat_api",
                        lambda self, m: (_ for _ in ()).throw(GenerationError("boom")))
    monkeypatch.setattr(providers, "chat_chain",
                        lambda *a, **k: ("local text", "ollama:qwen2.5:7b-instruct"))
    _local(monkeypatch)

    r = Answerer(use_chain=True).answer("q?", CHUNKS)
    assert r.model == config.GENERATION_MODEL, \
        "an ollama result from the chain must fall through to the explicit local path"


def test_system_prompt_survives_flattening_for_the_chain():
    """The system message carries the citation rules and injection defence."""
    from generation.answerer import _messages_to_prompt

    flat = _messages_to_prompt([
        {"role": "system", "content": "RULE: cite every claim"},
        {"role": "user", "content": "Question: why?"},
    ])
    assert "RULE: cite every claim" in flat
    assert "Question: why?" in flat
