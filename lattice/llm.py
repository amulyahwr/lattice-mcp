from __future__ import annotations

import os
from typing import Any

import litellm


def _model_string() -> str:
    provider = os.environ.get("LLM_PROVIDER", "anthropic")
    model = os.environ.get("LLM_MODEL", "claude-sonnet-4-6")
    if provider == "anthropic":
        return f"anthropic/{model}"
    if provider == "openai":
        return f"openai/{model}"
    if provider == "ollama":
        return f"ollama/{model}"
    return model


def complete(messages: list[dict], text_format: type | None = None) -> str:
    provider = os.environ.get("LLM_PROVIDER", "anthropic")
    api_key = os.environ.get("LLM_API_KEY")
    if provider != "ollama" and not api_key:
        raise EnvironmentError(f"LLM_API_KEY is required for provider '{provider}'")

    kwargs: dict[str, Any] = {
        "model": _model_string(),
        "input": messages,
        "api_key": api_key or None,
    }
    if text_format is not None:
        kwargs["text_format"] = text_format

    response = litellm.responses(**kwargs)
    return response.output_text or ""
