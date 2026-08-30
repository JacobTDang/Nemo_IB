"""The environment a test module sees must not depend on what ran before it.

`sec_utils` refuses to call SEC EDGAR without SEC_EMAIL, and reads it straight
from os.environ. Nothing in that module loads `.env` -- in the container the
variable arrives from the container environment, so it never needed to. Under
pytest it arrived by accident: `research/__init__.py` calls load_dotenv() at
import time, so whether the SEC tests could reach the network came down to
whether some earlier module happened to import `research`.

Run alphabetically the accident held and the file passed. Run on its own it
did not: `pytest testing/test_sec_xbrl_functions.py` failed 20 of 295 with
"SEC_EMAIL is not set", including two error-handling tests that were asserting
on a refusal they never got far enough to produce.

This runs pytest in a subprocess on a single file, which is the only way to
observe the property -- once `.env` is loaded in a process it stays loaded, so
an in-process assertion passes whether or not the fix is there.
"""
import os
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
DOTENV = ROOT / ".env"


def _dotenv_keys():
    if not DOTENV.exists():
        return {}
    found = {}
    for line in DOTENV.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        found[key.strip()] = value.strip().strip("'\"")
    return found


@pytest.mark.skipif(not DOTENV.exists(), reason="no .env in this checkout")
def test_sec_email_reaches_a_module_that_imports_nothing_else(tmp_path):
    """A module importing no application code still sees SEC_EMAIL."""
    if "SEC_EMAIL" not in _dotenv_keys():
        pytest.skip(".env in this checkout does not define SEC_EMAIL")

    probe = ROOT / "testing" / "test_zz_env_probe_generated.py"
    probe.write_text(
        "import os\n\n\n"
        "def test_probe():\n"
        "    assert os.getenv('SEC_EMAIL'), 'SEC_EMAIL absent'\n"
    )
    try:
        # -p no:randomly so the probe is the only thing that runs; a clean
        # environment so the parent's own os.environ cannot supply the answer.
        environment = {k: v for k, v in os.environ.items() if k != "SEC_EMAIL"}
        result = subprocess.run(
            [sys.executable, "-m", "pytest", str(probe), "-q", "-p", "no:randomly"],
            cwd=ROOT, env=environment, capture_output=True, text=True, timeout=300)
    finally:
        probe.unlink(missing_ok=True)

    assert result.returncode == 0, (
        "a test module that imports no application code did not see SEC_EMAIL, "
        "so which tests can reach SEC EDGAR depends on collection order:\n"
        + result.stdout[-2000:])
