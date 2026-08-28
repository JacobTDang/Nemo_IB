"""Constructing a model must not require a credential.

Fourteen tests instantiated Financial_Modeling_Agent only to reach
deterministic methods. Validating the key in __init__ held those tests hostage
to a credential their code paths never use. The key is still required -- it is
checked on first real use, and explicitly at boot via validate_credentials().

The OpenRouter half of this file went with `agent/openrouter_template.py` and
`agent/Financial_Modeling_Agent.py` when the LangGraph/OpenRouter layer was
retired (issue #63). `GroqModel` is the surviving template, and it is the one
the news daemons build through `Materiality_Classifier` -- which is where a
credential checked at construction would actually bite.

load_dotenv is patched out in every test because it would otherwise repopulate
the environment from the developer's .env and defeat the point.
"""
import pytest


def _no_dotenv(monkeypatch, module):
    monkeypatch.setattr(module, "load_dotenv", lambda *a, **k: None)


def test_groq_constructs_without_a_key(monkeypatch):
    import agent.groq_template as gt
    _no_dotenv(monkeypatch, gt)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    model = gt.GroqModel()
    assert model.model_name


def test_groq_client_access_raises_without_a_key(monkeypatch):
    import agent.groq_template as gt
    _no_dotenv(monkeypatch, gt)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    model = gt.GroqModel()
    with pytest.raises(ValueError, match="GROQ_API_KEY not found"):
        _ = model.client


def test_groq_empty_key_is_treated_as_missing(monkeypatch):
    """GROQ_API_KEY is present in .env with an empty value. An empty string is
    a missing credential, not a configured one."""
    import agent.groq_template as gt
    _no_dotenv(monkeypatch, gt)
    monkeypatch.setenv("GROQ_API_KEY", "")
    with pytest.raises(ValueError, match="GROQ_API_KEY not found"):
        _ = gt.GroqModel().client


def test_groq_validate_credentials_raises_without_a_key(monkeypatch):
    import agent.groq_template as gt
    _no_dotenv(monkeypatch, gt)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GROQ_API_KEY not found"):
        gt.GroqModel().validate_credentials()


def test_groq_client_is_built_once(monkeypatch):
    import agent.groq_template as gt
    _no_dotenv(monkeypatch, gt)
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-real")
    model = gt.GroqModel()
    assert model.client is model.client
