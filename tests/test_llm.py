from unittest.mock import MagicMock, patch

import pytest

import lattice.llm as llm_module


def _make_response(text: str):
    msg = MagicMock()
    msg.content = text
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    for var in ("LLM_PROVIDER", "LLM_MODEL", "LLM_API_KEY"):
        monkeypatch.delenv(var, raising=False)


class TestModelString:
    def test_default_anthropic(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("LLM_MODEL", "claude-sonnet-4-6")
        assert llm_module._model_string() == "anthropic/claude-sonnet-4-6"

    def test_openai_prefix(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("LLM_MODEL", "gpt-4o")
        assert llm_module._model_string() == "openai/gpt-4o"

    def test_ollama_prefix(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        monkeypatch.setenv("LLM_MODEL", "llama3")
        assert llm_module._model_string() == "ollama/llama3"

    def test_unknown_provider_passthrough(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "custom")
        monkeypatch.setenv("LLM_MODEL", "my-model")
        assert llm_module._model_string() == "my-model"


class TestComplete:
    def test_returns_content_string(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("LLM_MODEL", "claude-sonnet-4-6")
        monkeypatch.setenv("LLM_API_KEY", "sk-test")
        with patch("lattice.llm.litellm.completion", return_value=_make_response("hello")) as mock:
            result = llm_module.complete([{"role": "user", "content": "hi"}])
        assert result == "hello"
        mock.assert_called_once()

    def test_passes_api_key(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("LLM_MODEL", "claude-sonnet-4-6")
        monkeypatch.setenv("LLM_API_KEY", "sk-mykey")
        with patch("lattice.llm.litellm.completion", return_value=_make_response("ok")) as mock:
            llm_module.complete([{"role": "user", "content": "test"}])
        _, kwargs = mock.call_args
        assert kwargs.get("api_key") == "sk-mykey"

    def test_no_api_key_passes_none(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        monkeypatch.setenv("LLM_MODEL", "llama3")
        with patch("lattice.llm.litellm.completion", return_value=_make_response("ok")) as mock:
            llm_module.complete([{"role": "user", "content": "test"}])
        _, kwargs = mock.call_args
        assert kwargs.get("api_key") is None

    def test_messages_forwarded(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("LLM_MODEL", "claude-sonnet-4-6")
        monkeypatch.setenv("LLM_API_KEY", "sk-test")
        messages = [{"role": "user", "content": "what is 2+2?"}]
        with patch("lattice.llm.litellm.completion", return_value=_make_response("4")) as mock:
            llm_module.complete(messages)
        _, kwargs = mock.call_args
        assert kwargs.get("messages") == messages

    def test_missing_api_key_raises_for_anthropic(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("LLM_MODEL", "claude-sonnet-4-6")
        with pytest.raises(EnvironmentError, match="LLM_API_KEY"):
            llm_module.complete([{"role": "user", "content": "hi"}])

    def test_missing_api_key_raises_for_openai(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("LLM_MODEL", "gpt-4o")
        with pytest.raises(EnvironmentError, match="LLM_API_KEY"):
            llm_module.complete([{"role": "user", "content": "hi"}])

    def test_missing_api_key_ok_for_ollama(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        monkeypatch.setenv("LLM_MODEL", "llama3")
        with patch("lattice.llm.litellm.completion", return_value=_make_response("ok")):
            result = llm_module.complete([{"role": "user", "content": "hi"}])
        assert result == "ok"
