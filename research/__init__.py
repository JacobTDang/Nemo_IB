"""Research infrastructure: the point-in-time record and the signals built on it.

Deliberately outside `tools/`. Those five servers ship in the homelab image and
answer questions about the world; this package accumulates a private record and
forms opinions from it. Keeping them apart is the same reason `alpaca` is
excluded from the image -- a data-source host should not be able to trade, and
should not need to.

Every module here loads the repo's `.env` on import. Only `daily_job` did, and
only because it is the one with a `__main__`; `sue` reaches EDGAR too, so a
scan run from anywhere but that entry point rejected all sixteen names with
"SEC_EMAIL is not set". Cron has no shell profile to inherit from, and neither
does a REPL, a notebook, or the next module that reaches an upstream.
"""
from pathlib import Path as _Path

from dotenv import load_dotenv as _load_dotenv

_DOTENV_PATH = _Path(__file__).resolve().parent.parent / ".env"
_load_dotenv(dotenv_path=_DOTENV_PATH)
