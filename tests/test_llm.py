"""LLM factory: every provider can be constructed (no network calls are made)."""

import pytest

from app.agent import llm
from app.config import LLMProvider, Settings


def _settings(**kwargs) -> Settings:
    return Settings(_env_file=None, **kwargs)


@pytest.mark.parametrize("provider, kwargs, cls_name", [
    (LLMProvider.GEMINI, {"gemini_api_key": "test"}, "ChatGoogleGenerativeAI"),
    (LLMProvider.GROQ, {"groq_api_key": "test"}, "ChatGroq"),
    (LLMProvider.OLLAMA, {}, "ChatOllama"),
    (LLMProvider.OPENROUTER, {"openrouter_api_key": "test"}, "ChatOpenAI"),
])
def test_each_provider_builds(monkeypatch, provider, kwargs, cls_name):
    monkeypatch.setattr(llm, "get_settings", lambda: _settings(llm_provider=provider, **kwargs))
    model = llm.get_llm(temperature=0.0)
    assert type(model).__name__ == cls_name
    assert hasattr(model, "bind_tools")


@pytest.mark.parametrize("provider", [LLMProvider.GEMINI, LLMProvider.GROQ, LLMProvider.OPENROUTER])
def test_missing_key_is_a_clear_error(monkeypatch, provider):
    monkeypatch.setattr(llm, "get_settings", lambda: _settings(llm_provider=provider))
    with pytest.raises(ValueError, match="API_KEY not set"):
        llm.get_llm()
