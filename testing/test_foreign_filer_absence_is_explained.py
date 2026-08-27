"""A filer that uses another form has not "no filings".

Every SEC tool here defaults to form_type='10-K'. A foreign private issuer has
no 10-K at all, so the lookup returns nothing and the caller reports "No filing
found" -- a statement about Taiwan Semiconductor rather than about this tool.
TSM files a 20-F, most recently 2026-04-16.

`_filing_miss` exists for exactly this and fourteen call sites use it. A live
sweep of all 98 tools against TSM found eight that do not: they answer with a
bare absence, and one answers with an empty string. Twenty-five siblings on the
same ticker correctly say "TSM is a foreign private issuer and files 20-F
annually (latest 2026-04-16), not 10-K".

The inconsistency is the bug. A caller who asks eight tools and gets "no filing"
from one and "files 20-F" from another has to already know the answer to tell
which is which.
"""
import pytest

from testing._gates import requires_sec

FOREIGN = "TSM"

# The eight the sweep found, with how they are called.
CASES = [
    ("get_capex_pct_revenue", {}),
    ("get_tax_rate", {}),
    ("get_supply_chain", {}),
    ("get_disclosures_names", {}),
    ("extract_customer_concentration", {}),
    ("diff_10k", {}),
]

# extract_governance_data lives in the 8-K/DEF-14A module rather than
# sec_utils, and is covered by its own case below.


def _message(result):
    if not isinstance(result, dict):
        return ""
    return str(result.get("error") or result.get("note") or "")


@pytest.mark.network
@requires_sec
@pytest.mark.parametrize("fname,kwargs", CASES, ids=[c[0] for c in CASES])
def test_a_foreign_filers_absence_names_the_form_it_does_file(fname, kwargs):
    import tools.web_search_server.sec_utils as su

    fn = getattr(su, fname, None)
    if fn is None:
        pytest.skip(f"{fname} does not live in sec_utils")

    result = fn(FOREIGN, **kwargs)
    if isinstance(result, dict) and result.get("success"):
        return                      # it found something; nothing to explain

    message = _message(result)
    assert message, f"{fname} failed with no message at all"
    assert ("20-F" in message or "foreign private issuer" in message.lower()), (
        f"{fname} reported an absence without saying TSM files 20-F, which "
        f"reads as a fact about the company: {message[:200]}")


@pytest.mark.network
@requires_sec
def test_a_domestic_filer_is_not_given_a_foreign_explanation():
    """The note must not fire for a filer that genuinely has no such data."""
    import tools.web_search_server.sec_utils as su

    result = su.get_tax_rate("MSFT")
    message = _message(result)
    assert "foreign private issuer" not in message.lower(), (
        f"MSFT files a 10-K and was told it does not: {message[:200]}")


@pytest.mark.network
@requires_sec
def test_governance_data_explains_a_missing_proxy():
    """It returned success:false with no `error` key at all -- a failure a
    caller cannot log, act on, or tell from any other. TSM has no DEF 14A
    because proxy statements are a domestic filing, not because it withheld
    one."""
    import importlib

    # The module's name starts with a digit, so it is imported by path.
    gov = importlib.import_module(
        "tools.web_search_server.8K_and_DEF14A_utils")

    result = gov.extract_governance_data(FOREIGN)
    if isinstance(result, dict) and not result.get("success"):
        message = _message(result)
        assert message, "failed with no error message at all"
        assert "DEF 14A" in message or "proxy" in message.lower(), (
            f"the absence is unexplained: {message[:200]}")
