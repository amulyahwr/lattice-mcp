from __future__ import annotations

import os

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


def complete(messages: list[dict]) -> str:
    provider = os.environ.get("LLM_PROVIDER", "anthropic")
    api_key = os.environ.get("LLM_API_KEY")
    if provider != "ollama" and not api_key:
        raise EnvironmentError(f"LLM_API_KEY is required for provider '{provider}'")
    response = litellm.completion(
        model=_model_string(),
        messages=messages,
        api_key=api_key or None,
    )
    return response.choices[0].message.content or ""
