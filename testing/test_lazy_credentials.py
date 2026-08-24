"""Constructing a model must not require a credential.

Fourteen tests instantiate Financial_Modeling_Agent only to reach deterministic
methods. Validating the key in __init__ held those tests hostage to a
credential their code paths never use. The key is still required -- it is
checked on first real use, and explicitly at boot via validate_credentials().

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


def test_openrouter_constructs_without_a_key(monkeypatch):
    import agent.openrouter_template as ot
    _no_dotenv(monkeypatch, ot)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    model = ot.OpenRouterModel(model_name="vendor/model:free")
    assert model.model_name == "vendor/model:free"


def test_openrouter_client_access_raises_without_a_key(monkeypatch):
    import agent.openrouter_template as ot
    _no_dotenv(monkeypatch, ot)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_NEMOTRON", raising=False)
    model = ot.OpenRouterModel(model_name="vendor/model:free")
    with pytest.raises(ValueError, match="No API key found"):
        _ = model.client


def test_openrouter_fallback_client_raises_without_a_key(monkeypatch):
    """The fallback path must not become a silent way to skip the check."""
    import agent.openrouter_template as ot
    _no_dotenv(monkeypatch, ot)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_GLM", raising=False)
    model = ot.OpenRouterModel(model_name="vendor/model:free")
    with pytest.raises(ValueError, match="No API key found"):
        _ = model.fallback_client


def test_openrouter_validate_credentials_raises_without_a_key(monkeypatch):
    import agent.openrouter_template as ot
    _no_dotenv(monkeypatch, ot)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ValueError, match="No API key found"):
        ot.OpenRouterModel(model_name="vendor/model:free").validate_credentials()


def test_openrouter_client_is_built_once(monkeypatch):
    import agent.openrouter_template as ot
    _no_dotenv(monkeypatch, ot)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real")
    model = ot.OpenRouterModel(model_name="vendor/model:free")
    assert model.client is model.client
    assert model.fallback_client is model.fallback_client


def test_financial_modeling_agent_constructs_without_a_key(monkeypatch):
    """The concrete reason this task exists: 14 pure-logic tests build this."""
    import agent.openrouter_template as ot
    _no_dotenv(monkeypatch, ot)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    from agent.Financial_Modeling_Agent import Financial_Modeling_Agent
    assert Financial_Modeling_Agent() is not None
