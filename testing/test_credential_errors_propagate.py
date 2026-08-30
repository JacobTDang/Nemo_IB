"""A missing credential must not be reported as a parse failure.

`Materiality_Classifier.classify()` caught every exception and returned None
with "parse failed". Before credentials became lazy, a missing key raised at
construction so this never fired. Now it is reachable, and the failure mode is
bad in a specific way: `rss_aggregator` calls classify in a loop over articles,
so an unset GROQ_API_KEY makes every single call fail identically while the
daemon prints a diagnosis naming the wrong cause and keeps going.

A parse failure is per-article and skippable. A missing credential is neither.
"""
import pytest

from agent.groq_template import CredentialsMissing


def test_credentials_missing_is_a_value_error():
    """Subclassed so existing `except ValueError` handlers keep working."""
    assert issubclass(CredentialsMissing, ValueError)


def test_resolve_raises_the_specific_type(monkeypatch):
    import agent.groq_template as gt
    monkeypatch.setattr(gt, "load_dotenv", lambda *a, **k: None)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(CredentialsMissing):
        gt.GroqModel()._resolve_credential()


def test_classify_propagates_a_missing_credential(monkeypatch):
    """The regression: this must raise, not return None."""
    from agent.Materiality_Classifier import Materiality_Classifier
    import agent.groq_template as gt

    monkeypatch.setattr(gt, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-real")
    classifier = Materiality_Classifier()

    def raise_missing(*a, **k):
        raise CredentialsMissing("GROQ_API_KEY not found in environment.")

    monkeypatch.setattr(classifier, "generate_response", raise_missing)
    with pytest.raises(CredentialsMissing):
        classifier.classify("Some headline")


def test_classify_still_skips_a_genuine_parse_failure(monkeypatch):
    """Per-article failures stay skippable -- one malformed response must not
    stop a daemon working through a feed."""
    from agent.Materiality_Classifier import Materiality_Classifier
    import agent.groq_template as gt

    monkeypatch.setattr(gt, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-real")
    classifier = Materiality_Classifier()

    def raise_parse(*a, **k):
        raise ValueError("Expecting ',' delimiter: line 3 column 9")

    monkeypatch.setattr(classifier, "generate_response", raise_parse)
    assert classifier.classify("Some headline") is None


def test_parse_failure_message_names_the_error_type(monkeypatch, capsys):
    """'parse failed: <message>' hid what actually went wrong."""
    from agent.Materiality_Classifier import Materiality_Classifier
    import agent.groq_template as gt

    monkeypatch.setattr(gt, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-real")
    classifier = Materiality_Classifier()
    monkeypatch.setattr(classifier, "generate_response",
                        lambda *a, **k: (_ for _ in ()).throw(TypeError("bad shape")))
    classifier.classify("Headline")
    assert "TypeError" in capsys.readouterr().err
