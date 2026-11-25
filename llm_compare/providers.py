"""Call model APIs (Anthropic and OpenAI-compatible) with plain HTTP requests."""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass

import requests

DEFAULT_TIMEOUT = 120
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
OPENAI_DEFAULT_BASE = "https://api.openai.com/v1"


class ProviderError(Exception):
    """Raised when a model call fails or a model spec is invalid."""


@dataclass(frozen=True)
class ModelSpec:
    provider: str
    model: str

    @property
    def name(self) -> str:
        return f"{self.provider}:{self.model}"


@dataclass
class ModelResponse:
    model: str
    text: str
    seconds: float
    input_tokens: int | None = None
    output_tokens: int | None = None


def parse_model_spec(value: str) -> ModelSpec:
    """Parse 'provider:model', for example 'anthropic:claude-sonnet-5-5'."""
    provider, sep, model = value.partition(":")
    provider, model = provider.strip().lower(), model.strip()
    if not sep or not provider or not model:
        raise ProviderError(f"Model must look like provider:model-name, got '{value}'")
    if provider not in PROVIDERS:
        known = ", ".join(sorted(PROVIDERS))
        raise ProviderError(f"Unknown provider '{provider}'. Supported: {known}")
    return ModelSpec(provider, model)


def _require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ProviderError(f"{name} is not set. Add it to your .env file.")
    return value


def _post(session: requests.Session, url: str, headers: dict, body: dict, timeout: int) -> dict:
    try:
        response = session.post(url, headers=headers, json=body, timeout=timeout)
    except requests.RequestException as exc:
        raise ProviderError(f"Could not reach {url}: {exc}") from exc
    if response.status_code >= 400:
        try:
            error = response.json().get("error", {})
            message = error.get("message") if isinstance(error, dict) else str(error)
        except ValueError:
            message = None
        raise ProviderError(f"API error {response.status_code}: {message or response.text}")
    return response.json()


def call_anthropic(
    spec: ModelSpec, prompt: str, max_tokens: int, session: requests.Session, timeout: int
) -> ModelResponse:
    headers = {
        "x-api-key": _require_env("ANTHROPIC_API_KEY"),
        "anthropic-version": ANTHROPIC_VERSION,
        "content-type": "application/json",
    }
    body = {
        "model": spec.model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    start = time.perf_counter()
    data = _post(session, ANTHROPIC_URL, headers, body, timeout)
    seconds = time.perf_counter() - start

    text = "".join(
        block.get("text", "") for block in data.get("content", []) if block.get("type") == "text"
    )
    usage = data.get("usage", {})
    return ModelResponse(
        model=spec.name,
        text=text,
        seconds=round(seconds, 2),
        input_tokens=usage.get("input_tokens"),
        output_tokens=usage.get("output_tokens"),
    )


def call_openai(
    spec: ModelSpec, prompt: str, max_tokens: int, session: requests.Session, timeout: int
) -> ModelResponse:
    """Works with OpenAI and any API that follows the OpenAI chat completions format."""
    base = os.getenv("OPENAI_BASE_URL", OPENAI_DEFAULT_BASE).rstrip("/")
    headers = {
        "Authorization": f"Bearer {_require_env('OPENAI_API_KEY')}",
        "content-type": "application/json",
    }
    body = {
        "model": spec.model,
        "max_completion_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    start = time.perf_counter()
    data = _post(session, f"{base}/chat/completions", headers, body, timeout)
    seconds = time.perf_counter() - start

    try:
        text = data["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError) as exc:
        raise ProviderError("Unexpected response format from the OpenAI API") from exc
    usage = data.get("usage", {})
    return ModelResponse(
        model=spec.name,
        text=text,
        seconds=round(seconds, 2),
        input_tokens=usage.get("prompt_tokens"),
        output_tokens=usage.get("completion_tokens"),
    )


ProviderFn = Callable[[ModelSpec, str, int, requests.Session, int], ModelResponse]

PROVIDERS: dict[str, ProviderFn] = {
    "anthropic": call_anthropic,
    "openai": call_openai,
}


def ask(
    spec: ModelSpec,
    prompt: str,
    max_tokens: int = 1024,
    session: requests.Session | None = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> ModelResponse:
    """Send a prompt to one model and return its answer."""
    return PROVIDERS[spec.provider](
        spec, prompt, max_tokens, session or requests.Session(), timeout
    )
