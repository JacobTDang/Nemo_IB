"""SEC contact identity must be configured, never guessed.

SEC fair access asks for a real contact address in the User-Agent header. Six
call sites used to default it to `analyst@example.com`, and a seventh
hardcoded `ops@example.com` outright. Each misrepresented the caller to the
SEC on every request and did so silently -- a placeholder address is
syntactically valid, so nothing downstream ever noticed.

The correct behaviour has two halves and both are tested here:

1. Resolution happens on use, not at import. A module must still import
   without credentials, or a missing variable becomes an import traceback in
   whatever imported it rather than a message about the variable.
2. Resolution refuses to invent a value. `sec_series._require_identity` raises
   ValueError naming SEC_EMAIL, and every call site now routes through it.

The three daemons get an extra test: each swallows per-tick exceptions and
retries forever, so a check that only fires inside a tick would print a
traceback every interval instead of failing. Each therefore resolves the
identity eagerly at boot, the same way they already validate the Groq
credential, and the startup tests below run the real entrypoint to prove it.
"""
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# Every site that puts a contact address in front of the SEC.
CALL_SITES = {
    "tools.web_search_server.sec_utils":
        REPO_ROOT / "tools" / "web_search_server" / "sec_utils.py",
    "tools.web_search_server.hf_letters":
        REPO_ROOT / "tools" / "web_search_server" / "hf_letters.py",
    "tools.web_search_server.8K_and_DEF14A_utils":
        REPO_ROOT / "tools" / "web_search_server" / "8K_and_DEF14A_utils.py",
    "daemons.rss_aggregator":
        REPO_ROOT / "daemons" / "rss_aggregator.py",
    "daemons.edgar_firehose":
        REPO_ROOT / "daemons" / "edgar_firehose.py",
    "daemons.gdelt_poller":
        REPO_ROOT / "daemons" / "gdelt_poller.py",
    # Not in the original six: this one hardcoded a different placeholder, so
    # it never showed up in a grep for analyst@example.com.
    "data.risk_factor_diff":
        REPO_ROOT / "data" / "risk_factor_diff.py",
}

PLACEHOLDERS = ("analyst@example.com", "ops@example.com")


def _env_without_sec_email(tmp_path):
    """A child-process environment with SEC_EMAIL removed.

    `.env` on this machine sets SEC_EMAIL, but nothing in the import chain
    calls load_dotenv() at import time, so clearing the process variable is
    enough. The DB paths are redirected so a daemon's init_schema() cannot
    touch the developer's session database.
    """
    env = dict(os.environ)
    env.pop("SEC_EMAIL", None)
    env["PYTHONPATH"] = str(REPO_ROOT)
    env["NEMO_DB_PATH"] = str(tmp_path / "session.db")
    env["NEMO_CACHE_DB_PATH"] = str(tmp_path / "tool_cache.db")
    return env


def test_no_placeholder_contact_address_remains():
    """A default contact address is a misrepresentation, not a fallback."""
    offenders = [
        f"{path.relative_to(REPO_ROOT)} ({placeholder})"
        for path in CALL_SITES.values()
        for placeholder in PLACEHOLDERS
        if placeholder in path.read_text()
    ]
    assert not offenders, (
        "a placeholder still stands in for the SEC contact identity in: "
        f"{', '.join(offenders)}")


def test_every_call_site_imports_without_sec_email(tmp_path):
    """Import must not need the credential -- only use must."""
    script = textwrap.dedent(f"""
        import importlib, sys
        for name in {sorted(CALL_SITES)!r}:
            importlib.import_module(name)
        print("IMPORTS_OK")
    """)
    proc = subprocess.run([sys.executable, "-c", script],
                          cwd=str(REPO_ROOT), env=_env_without_sec_email(tmp_path),
                          capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0, (
        f"importing a call site without SEC_EMAIL failed:\n{proc.stderr}")
    assert "IMPORTS_OK" in proc.stdout


def test_require_identity_names_the_missing_variable(monkeypatch):
    """The shared resolver is the single mechanism every call site reuses."""
    from tools.web_search_server.sec_series import _require_identity
    monkeypatch.delenv("SEC_EMAIL", raising=False)
    with pytest.raises(ValueError, match="SEC_EMAIL"):
        _require_identity()


# ---------------------------------------------------------------------------
# tools/web_search_server call sites
# ---------------------------------------------------------------------------

def test_sec_utils_refuses_without_sec_email(monkeypatch):
    """`get_latest_filing` caches failures, so a swallowed ValueError would
    poison the cache with a permanent "no filing" for the ticker."""
    from tools.web_search_server import sec_utils
    monkeypatch.delenv("SEC_EMAIL", raising=False)
    with pytest.raises(ValueError, match="SEC_EMAIL"):
        sec_utils.get_latest_filing("NEMOTESTNOEMAIL", "10-K")


@pytest.mark.parametrize("func_name,args", [
    ("get_schedule_13d_filings", ("AAPL",)),
    ("diff_10k", ("AAPL",)),
    ("get_company_filings_history", ("AAPL",)),
    ("get_earnings_releases", ("AAPL",)),
])
def test_sec_utils_entrypoints_refuse_without_sec_email(monkeypatch, func_name, args):
    """These wrap the lookup in `except Exception` and return an error dict.
    The identity check has to sit outside that, or a config error arrives
    disguised as a failed company lookup."""
    from tools.web_search_server import sec_utils
    monkeypatch.delenv("SEC_EMAIL", raising=False)
    with pytest.raises(ValueError, match="SEC_EMAIL"):
        getattr(sec_utils, func_name)(*args)


def test_hf_letters_refuses_without_sec_email(monkeypatch):
    from tools.web_search_server import hf_letters
    monkeypatch.delenv("SEC_EMAIL", raising=False)
    with pytest.raises(ValueError, match="SEC_EMAIL"):
        hf_letters.get_fund_holdings("berkshire")


def test_sec_filing_parser_refuses_without_sec_email(monkeypatch):
    import importlib
    module = importlib.import_module("tools.web_search_server.8K_and_DEF14A_utils")
    monkeypatch.delenv("SEC_EMAIL", raising=False)
    with pytest.raises(ValueError, match="SEC_EMAIL"):
        module.SECFilingParser()._set_identity()


def test_sec_filing_parser_still_honours_an_explicit_identity(monkeypatch):
    """The constructor's name/email override predates this fix and still works;
    only the invented default is gone."""
    import importlib
    module = importlib.import_module("tools.web_search_server.8K_and_DEF14A_utils")
    monkeypatch.delenv("SEC_EMAIL", raising=False)
    captured = []
    monkeypatch.setattr(module, "set_identity", captured.append)
    module.SECFilingParser(name="Someone", email="someone@real.test")._set_identity()
    assert captured == ["Someone someone@real.test"]


def test_risk_factor_diff_refuses_without_sec_email(monkeypatch):
    """This one hardcoded its address outright, so no amount of configuration
    could correct it."""
    from data import risk_factor_diff
    monkeypatch.delenv("SEC_EMAIL", raising=False)
    with pytest.raises(ValueError, match="SEC_EMAIL"):
        risk_factor_diff.fetch_and_diff_10k_risks("AAPL")


# ---------------------------------------------------------------------------
# daemons
# ---------------------------------------------------------------------------

def test_edgar_firehose_headers_refuse_without_sec_email(monkeypatch):
    from daemons import edgar_firehose
    monkeypatch.delenv("SEC_EMAIL", raising=False)
    with pytest.raises(ValueError, match="SEC_EMAIL"):
        edgar_firehose._sec_headers()


def test_rss_sec_user_agent_refuses_without_sec_email(monkeypatch):
    from daemons import rss_aggregator
    monkeypatch.delenv("SEC_EMAIL", raising=False)
    with pytest.raises(ValueError, match="SEC_EMAIL"):
        rss_aggregator._sec_user_agent()


def test_gdelt_user_agent_refuses_without_sec_email(monkeypatch):
    from daemons import gdelt_poller
    monkeypatch.delenv("SEC_EMAIL", raising=False)
    with pytest.raises(ValueError, match="SEC_EMAIL"):
        gdelt_poller._user_agent()


@pytest.mark.parametrize("module_name", [
    "daemons.rss_aggregator",
    "daemons.edgar_firehose",
    "daemons.gdelt_poller",
])
def test_daemon_startup_fails_loudly_without_sec_email(tmp_path, module_name):
    """Each daemon catches every per-tick exception and retries forever. The
    identity therefore has to resolve at boot: otherwise a missing SEC_EMAIL
    is a traceback printed once per interval, indefinitely, while the daemon
    reports itself as running."""
    script = textwrap.dedent(f"""
        import importlib, sys
        module = importlib.import_module({module_name!r})
        print("IMPORT_OK", flush=True)
        sys.argv = ["daemon", "--once"]
        module.main()
    """)
    proc = subprocess.run([sys.executable, "-c", script],
                          cwd=str(REPO_ROOT), env=_env_without_sec_email(tmp_path),
                          capture_output=True, text=True, timeout=180)
    assert "IMPORT_OK" in proc.stdout, (
        f"{module_name} failed at import rather than at startup:\n{proc.stderr}")
    assert proc.returncode != 0, (
        f"{module_name} started with no SEC_EMAIL configured:\n{proc.stdout}")
    assert "SEC_EMAIL" in proc.stderr, (
        f"{module_name} failed without naming the missing variable:\n{proc.stderr}")


# ---------------------------------------------------------------------------
# happy path -- the configured address is the one that gets used
# ---------------------------------------------------------------------------

def _capture_identity(monkeypatch):
    """Record identities instead of mutating edgartools' process-wide state."""
    from tools.web_search_server import sec_series
    captured = []
    monkeypatch.setattr(sec_series, "set_identity", captured.append)
    monkeypatch.setattr(sec_series, "_identity_set", False)
    return captured


def test_configured_email_reaches_the_user_agent(monkeypatch):
    from daemons import edgar_firehose, gdelt_poller, rss_aggregator
    _capture_identity(monkeypatch)
    monkeypatch.setenv("SEC_EMAIL", "real.contact@nemo.test")
    monkeypatch.setenv("NAME", "Nemo Tester")

    assert edgar_firehose._sec_headers() == {
        "User-Agent": "Nemo Tester real.contact@nemo.test"}
    assert rss_aggregator._sec_user_agent() == (
        "Nemo Tester real.contact@nemo.test (Nemo Sentry RSS)")
    assert "real.contact@nemo.test" in gdelt_poller._user_agent()


def test_blank_email_counts_as_missing(monkeypatch):
    """SEC_EMAIL= with no value is an unconfigured variable, not a contact."""
    from tools.web_search_server.sec_series import _require_identity
    monkeypatch.setenv("SEC_EMAIL", "   ")
    with pytest.raises(ValueError, match="SEC_EMAIL"):
        _require_identity()
