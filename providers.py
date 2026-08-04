"""Registry of interchangeable LLM providers (all free-tier friendly).

One place that knows how to talk to every provider we support, so the rest of
the codebase picks a provider *by name* and never hardcodes a URL or model.

Every hosted provider here speaks the OpenAI chat-completions protocol, so a
single client covers all of them — adding another is one dict entry, not new
code. ``ollama`` is the local escape hatch and needs no key.

Why separate providers for different roles: if the same model both writes the
evaluation questions and grades the answers, it tends to favour its own
phrasing (self-preference bias). Pointing question generation and judging at
*different* providers removes that shared blind spot, which makes the reported
scores more defensible.

Configure in ``.env``:
    QUESTIONGEN_PROVIDER=nvidia     # writes golden-set questions
    JUDGE_PROVIDER=groq             # grades answers
    NVIDIA_API_KEY=...              # whichever keys you actually have
    GROQ_API_KEY=...

A reasoning-tuned model suits question generation: it has to infer *why* a
change happened from scattered evidence, not just summarise one chunk.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import requests

import config


@dataclass(frozen=True)
class Provider:
    """How to reach one hosted provider (or the local model)."""

    name: str
    base_url: str            # empty for the local provider
    key_env: str             # env var holding the API key ("" if none needed)
    default_model: str
    label: str               # human-readable, for reports

    @property
    def api_key(self) -> str:
        return os.getenv(self.key_env, "") if self.key_env else ""

    @property
    def is_local(self) -> bool:
        return self.name == "ollama"

    @property
    def available(self) -> bool:
        """Configured enough to be usable (local is always available)."""
        return True if self.is_local else bool(self.api_key)


# Free tiers as of writing. Limits change — never assume today's numbers hold;
# the code degrades to the next available provider (and finally to local).
REGISTRY: dict[str, Provider] = {
    "groq": Provider(
        name="groq",
        base_url="https://api.groq.com/openai/v1",
        key_env="GROQ_API_KEY",
        default_model="llama-3.3-70b-versatile",
        label="Groq · Llama 3.3 70B",
    ),
    "gemini": Provider(
        name="gemini",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        key_env="GEMINI_API_KEY",
        default_model="gemini-2.5-flash",
        label="Google · Gemini 2.5 Flash",
    ),
    "nvidia": Provider(
        name="nvidia",
        base_url="https://integrate.api.nvidia.com/v1",
        key_env="NVIDIA_API_KEY",
        default_model="nvidia/nemotron-3-nano-30b-a3b",
        label="NVIDIA NIM · Nemotron 3 Nano 30B (reasoning)",
    ),
    "openrouter": Provider(
        name="openrouter",
        base_url="https://openrouter.ai/api/v1",
        key_env="OPENROUTER_API_KEY",
        # Verified present in OpenRouter's :free tier. Deliberately a Gemma
        # model: the other roles use Llama / GPT-OSS / Nemotron, so this stays a
        # distinct family and can stand in for any of them without collapsing
        # two evaluation roles onto the same underlying model.
        default_model="google/gemma-4-31b-it:free",
        label="OpenRouter · Gemma 4 31B (free)",
    ),
    # Cerebras runs a free inference tier on an OpenAI-compatible endpoint.
    # Its value here is capacity, not novelty: every extra provider is another
    # independent daily quota, and quota — not model quality — is what limits a
    # bulk evaluation run (see HANDOFF.md §6).
    "cerebras": Provider(
        name="cerebras",
        base_url="https://api.cerebras.ai/v1",
        key_env="CEREBRAS_API_KEY",
        # Cerebras offers only three models; llama-3.3-70b was retired. NOTE:
        # its gpt-oss-120b is the SAME model Groq serves as openai/gpt-oss-120b,
        # so do not put both on evaluation roles — canonical_model() treats them
        # as identical and roles_are_distinct() will (correctly) fail.
        default_model="zai-glm-4.7",
        label="Cerebras · GLM 4.7 (free tier)",
    ),
    "ollama": Provider(
        name="ollama",
        base_url="",
        key_env="",
        default_model=config.JUDGE_MODEL_OLLAMA,
        label="Local · Ollama",
    ),
}


class ProviderError(RuntimeError):
    """A provider call failed (caller decides whether to fall back)."""


def get(name: str) -> Provider:
    """Look up a provider by name, falling back to local for unknown names."""
    return REGISTRY.get((name or "").strip().lower(), REGISTRY["ollama"])


def resolve(name: str) -> Provider:
    """The provider that will actually be used: requested one, or local.

    Keeps reports honest — if a key is missing we say we used Ollama rather
    than claiming we used the provider that was merely *requested*.
    """
    p = get(name)
    return p if p.available else REGISTRY["ollama"]


def _chat_openai_compatible(p: Provider, prompt: str, model: str,
                            temperature: float, timeout: int) -> str:
    resp = requests.post(
        f"{p.base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {p.api_key}",
                 "Content-Type": "application/json"},
        json={"model": model,
              "messages": [{"role": "user", "content": prompt}],
              "temperature": temperature,
              "max_tokens": config.GENERATION_MAX_TOKENS},
        timeout=timeout,
    )
    if resp.status_code >= 400:
        raise ProviderError(f"{p.name} error {resp.status_code}: {resp.text[:200]}")
    try:
        return resp.json()["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, ValueError) as exc:
        raise ProviderError(f"{p.name}: unexpected response shape ({exc})") from exc


def _chat_ollama(prompt: str, model: str, temperature: float,
                 timeout: int) -> str:
    resp = requests.post(
        f"{config.OLLAMA_HOST}/api/chat",
        json={"model": model, "stream": False,
              "messages": [{"role": "user", "content": prompt}],
              "options": {"temperature": temperature}},
        timeout=timeout,
    )
    if resp.status_code >= 400:
        raise ProviderError(f"ollama error {resp.status_code}: {resp.text[:200]}")
    return resp.json().get("message", {}).get("content", "")


def chat(provider_name: str, prompt: str, *, model: str | None = None,
         temperature: float = 0.0, timeout: int | None = None,
         fallback_local: bool = True) -> tuple[str, str]:
    """Send ``prompt`` to ``provider_name``; return ``(text, label_used)``.

    Falls back to the local model on any failure when ``fallback_local`` is on,
    so an exhausted free tier degrades instead of breaking the run. The second
    return value names the provider/model that actually answered.
    """
    p = resolve(provider_name)
    used_model = model or p.default_model
    timeout = timeout or config.GENERATION_API_TIMEOUT

    if not p.is_local:
        try:
            text = _chat_openai_compatible(p, prompt, used_model, temperature,
                                           timeout)
            if text.strip():
                return text, f"{p.name}:{used_model}"
            raise ProviderError(f"{p.name} returned an empty completion")
        except Exception as exc:  # noqa: BLE001 - degrade, don't crash a run
            if not fallback_local:
                raise
            local = REGISTRY["ollama"]
            text = _chat_ollama(prompt, local.default_model, temperature, timeout)
            return text, (f"ollama:{local.default_model} "
                          f"(fallback from {p.name}: {str(exc)[:80]})")

    text = _chat_ollama(prompt, used_model, temperature, timeout)
    return text, f"ollama:{used_model}"


def parse_spec(spec: str) -> tuple[str, str | None]:
    """Split a ``"provider:model"`` spec. Model is optional.

    Model ids themselves contain ``/`` and ``:`` (``nvidia/nemotron-3-nano:free``),
    so only the FIRST colon separates the provider.
    """
    spec = (spec or "").strip()
    if ":" not in spec:
        return spec, None
    provider, model = spec.split(":", 1)
    return provider.strip(), (model.strip() or None)


def chat_chain(chain, prompt: str, *, temperature: float = 0.0,
               timeout: int | None = None,
               log=None) -> tuple[str, str]:
    """Try each ``provider:model`` in ``chain`` until one answers.

    Free tiers are not durable. In a single afternoon of testing: Gemini
    retired two models mid-flight and rate-limited the rest, Cerebras returned
    402, and Groq's daily token cap was exhausted. Any configuration that
    depends on one provider being up will break long before this project is
    demonstrated.

    So the chain is the reliability story: each entry is tried in order and the
    first that answers wins. The local model is always appended last and needs
    no key or network, which makes "the app still works" independent of every
    third party. Returns ``(text, label_of_what_actually_answered)``.
    """
    specs = list(chain) + ["ollama"]
    errors: list[str] = []
    for spec in specs:
        name, model = parse_spec(spec)
        p = get(name)
        if not p.available:
            errors.append(f"{name}: no key")
            continue
        try:
            used_model = model or p.default_model
            if p.is_local:
                text = _chat_ollama(prompt, used_model, temperature,
                                    timeout or config.GENERATION_API_TIMEOUT)
            else:
                text = _chat_openai_compatible(
                    p, prompt, used_model, temperature,
                    timeout or config.GENERATION_API_TIMEOUT)
            if text and text.strip():
                return text, f"{p.name}:{used_model}"
            errors.append(f"{name}: empty completion")
        except Exception as exc:  # noqa: BLE001 - try the next link
            errors.append(f"{name}: {str(exc)[:80]}")
            if log:
                log(f"provider {name} unavailable, trying next: {str(exc)[:100]}")
    raise ProviderError("every provider in the chain failed: "
                        + "; ".join(errors))


def describe(provider_name: str, model: str | None = None) -> str:
    """Short label for reports, e.g. 'groq:llama-3.3-70b-versatile'.

    ``model`` names the model the caller intends to use. It is only honoured if
    the requested provider is actually available — when :func:`resolve` falls
    back to local, the label reports the local model, so a report never claims a
    cloud model that never ran.
    """
    p = resolve(provider_name)
    requested = get(provider_name)
    if model and p.name == requested.name:
        return f"{p.name}:{model}"
    return f"{p.name}:{p.default_model}"


def available_providers() -> list[str]:
    """Names of every provider that is actually configured right now."""
    return [n for n, p in REGISTRY.items() if p.available]


def list_models(provider_name: str, timeout: int = 20) -> list[str]:
    """Ask a provider which model ids it will actually accept.

    Free tiers retire models without notice, and a guessed id fails as a 404
    that looks exactly like an outage. Query this instead of assuming — see
    HANDOFF.md: ``gemini-2.5-flash`` and ``nvidia/nemotron-nano-3-30b-a3b``
    were both wrong in ways that cost real debugging time.

    Returns an empty list for the local provider (Ollama is queried separately)
    and raises :class:`ProviderError` if the endpoint refuses the key.
    """
    p = get(provider_name)
    if p.is_local:
        resp = requests.get(f"{config.OLLAMA_HOST}/api/tags", timeout=timeout)
        resp.raise_for_status()
        return sorted(m["name"] for m in resp.json().get("models", []))
    if not p.api_key:
        raise ProviderError(f"{p.name}: no key in ${p.key_env}")
    resp = requests.get(f"{p.base_url.rstrip('/')}/models",
                        headers={"Authorization": f"Bearer {p.api_key}"},
                        timeout=timeout)
    if resp.status_code >= 400:
        raise ProviderError(
            f"{p.name} /models returned {resp.status_code}: {resp.text[:160]}")
    return sorted(m.get("id", "") for m in resp.json().get("data", []))


def canonical_model(model: str) -> str:
    """Reduce a model id to the underlying model, ignoring vendor packaging.

    The same model is listed differently by different providers — Groq offers
    ``openai/gpt-oss-120b`` while Cerebras offers plain ``gpt-oss-120b``, and
    OpenRouter appends ``:free``. Comparing raw strings would call those three
    "different models" and let the same model end up judging its own output.
    """
    m = (model or "").strip().lower()
    m = m.split(":free")[0]          # OpenRouter free-tier suffix
    m = m.rsplit("/", 1)[-1]         # vendor prefix (openai/, meta/, nvidia/)
    m = m.removeprefix("models/")    # Google lists ids as models/<id>
    return m


def validate_model(provider_name: str, model: str) -> tuple[bool, str]:
    """Whether ``model`` is currently offered by ``provider_name``."""
    try:
        ids = list_models(provider_name)
    except Exception as exc:  # noqa: BLE001 - report, never crash a run
        return False, f"could not verify: {str(exc)[:120]}"
    if not ids:
        return True, "provider does not expose a model list"
    if model in ids:
        return True, "ok"
    # Google lists ids as "models/gemini-2.5-flash" but accepts the bare id on
    # its OpenAI-compatible endpoint, so compare canonically before failing.
    canon = {canonical_model(i) for i in ids}
    if canonical_model(model) in canon:
        return True, "ok (matched ignoring vendor prefix)"
    return False, f"{model!r} not offered; {len(ids)} available"
