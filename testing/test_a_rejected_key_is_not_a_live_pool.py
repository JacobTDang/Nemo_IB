"""A credential the provider rejects cannot verify anything.

`_verify_model_alive` sends a one-token completion and treats every non-404 as
alive, reasoning that a rate limit or a timeout does not prove a model is
dead. That is right for those.

It is wrong for authentication. When the key is rejected, the probe answers a
question about the credential, not about the model — so every model in the
list reads alive and the pool is assembled entirely out of 401s. Observed
live: the configured OPENROUTER_API_KEY is rejected with "Missing
Authentication header", and startup still logged `Pool initialized with 5
models`. Not one had been reached.

A pool that reports five verified models when zero were verified is the
failure this whole codebase is built to avoid: an outage of ours, presented as
a fact about the world. The credential problem must surface once, loudly,
rather than being distributed across five false confirmations.

Rate limits and timeouts keep the existing benefit of the doubt.
"""
import pytest

import agent.openrouter_template as ort


class _Boom:
    def __init__(self, exc):
        self._exc = exc

    class _Chat:
        def __init__(self, exc):
            self.completions = _Boom._Completions(exc)

    class _Completions:
        def __init__(self, exc):
            self._exc = exc

        def create(self, **kwargs):
            raise self._exc

    @property
    def chat(self):
        return _Boom._Chat(self._exc)


def _probe_raising(monkeypatch, exc):
    monkeypatch.setattr(ort, "OpenAI", lambda **kw: _Boom(exc))
    return ort._verify_model_alive("vendor/model:free", api_key="test-key")


def test_a_rejected_credential_is_not_a_live_model(monkeypatch):
    import openai

    err = openai.AuthenticationError.__new__(openai.AuthenticationError)
    Exception.__init__(err, "Missing Authentication header")

    with pytest.raises(ort.CredentialRejected):
        _probe_raising(monkeypatch, err)


def test_a_rate_limit_still_keeps_the_model(monkeypatch):
    """A 429 says the model is known and busy, not absent."""
    import openai

    err = openai.RateLimitError.__new__(openai.RateLimitError)
    Exception.__init__(err, "rate limited")
    assert _probe_raising(monkeypatch, err) is True


def test_a_timeout_still_keeps_the_model(monkeypatch):
    assert _probe_raising(monkeypatch, TimeoutError("slow")) is True


def test_a_404_still_means_dead(monkeypatch):
    import openai

    err = openai.NotFoundError.__new__(openai.NotFoundError)
    Exception.__init__(err, "no such model")
    assert _probe_raising(monkeypatch, err) is False


def test_the_pool_refuses_to_claim_models_it_could_not_reach(monkeypatch):
    """The count is the claim. It must not be built out of failures."""
    import openai

    err = openai.AuthenticationError.__new__(openai.AuthenticationError)
    Exception.__init__(err, "Missing Authentication header")
    monkeypatch.setattr(ort, "OpenAI", lambda **kw: _Boom(err))

    builder = getattr(ort, "_build_pool", None) or getattr(ort, "init_model_pool", None)
    if builder is None:
        pytest.skip("no pool builder found to exercise")
    with pytest.raises(ort.CredentialRejected):
        builder(["vendor/a:free", "vendor/b:free"], api_key="test-key")
