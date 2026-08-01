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
        default_model="gemini-2.0-flash",
        label="Google · Gemini 2.0 Flash",
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
        default_model="meta-llama/llama-3.3-70b-instruct:free",
        label="OpenRouter · Llama 3.3 70B (free)",
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


def describe(provider_name: str) -> str:
    """Short label for reports, e.g. 'groq:llama-3.3-70b-versatile'."""
    p = resolve(provider_name)
    return f"{p.name}:{p.default_model}"


def available_providers() -> list[str]:
    """Names of every provider that is actually configured right now."""
    return [n for n, p in REGISTRY.items() if p.available]
