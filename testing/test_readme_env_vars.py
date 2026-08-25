"""Every environment variable README tells you to set must be one the code reads.

A wrong name is worse than an absent one. `ALPACA_PAPER_API_KEY` and
`ALPACA_PAPER_SECRET_KEY` were documented for the paper broker; nothing in the
repo has ever read either. Following the setup instructions produced a filled-in
.env and a broker that still could not authenticate, with the credential
apparently right there in the file.

The check is deliberately narrow: backticked SCREAMING_SNAKE tokens in README.md
that look like environment variables, each of which must be read somewhere in
the Python sources or declared in .env.example.
"""
from __future__ import annotations

import os
import re

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# A backticked token that is upper-case and contains at least one underscore.
# Narrow enough to skip prose and CLI flags, wide enough to catch every
# credential the setup section names.
_README_ENV_RE = re.compile(r"`([A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+)`")

_GETENV_RE = re.compile(r"""(?:getenv|environ\.get)\(\s*["']([A-Z0-9_]+)["']""")
_ENVIRON_ITEM_RE = re.compile(r"""environ\[\s*["']([A-Z0-9_]+)["']""")
_DOTENV_RE = re.compile(r"^([A-Z0-9_]+)=", re.MULTILINE)
# Compose reads variables too, and a Python-only scan cannot see them.
# NEMO_BIND_ADDR is read by docker-compose.yml and by nothing else; without
# this the check calls a live, load-bearing variable undocumented-and-unread.
_COMPOSE_ENV_RE = re.compile(r"\$\{([A-Z0-9_]+)[:?}-]")


def _readme_env_names() -> set:
    with open(os.path.join(_REPO, "README.md"), encoding="utf-8") as handle:
        return set(_README_ENV_RE.findall(handle.read()))


def _names_the_code_reads() -> set:
    names = set()
    for directory, subdirs, files in os.walk(_REPO):
        subdirs[:] = [d for d in subdirs
                      if d not in {".venv", ".git", "__pycache__", "node_modules"}]
        for filename in files:
            if not filename.endswith(".py"):
                continue
            path = os.path.join(directory, filename)
            with open(path, encoding="utf-8", errors="ignore") as handle:
                source = handle.read()
            names |= set(_GETENV_RE.findall(source))
            names |= set(_ENVIRON_ITEM_RE.findall(source))
    return names


def _names_the_deploy_config_reads() -> set:
    names = set()
    deploy = os.path.join(_REPO, "deploy")
    for directory, _subdirs, files in os.walk(deploy):
        for filename in files:
            if not filename.endswith((".yml", ".yaml")):
                continue
            with open(os.path.join(directory, filename), encoding="utf-8",
                      errors="ignore") as handle:
                names |= set(_COMPOSE_ENV_RE.findall(handle.read()))
    return names


def _names_in_env_example() -> set:
    with open(os.path.join(_REPO, ".env.example"), encoding="utf-8") as handle:
        return set(_DOTENV_RE.findall(handle.read()))


def test_readme_documents_env_vars_the_code_actually_reads():
    documented = _readme_env_names()
    assert documented, "README env-var scan found nothing -- the regex has drifted"

    known = (_names_the_code_reads() | _names_in_env_example()
             | _names_the_deploy_config_reads())
    unread = sorted(name for name in documented if name not in known)

    assert not unread, (
        "README.md tells the reader to configure variables nothing reads: "
        f"{', '.join(unread)}")


def test_paper_broker_credentials_are_declared():
    """The names AsyncBroker prefers must be the names .env.example gives.

    This used to check README.md. The README now documents only the five MCP
    servers the image ships, and the paper broker is deliberately not one of
    them -- it places orders, and a data-source host should not be able to
    trade. The guarantee still matters, though: the bug that prompted this file
    was a documented credential name nothing read. So the check follows the
    broker to the file that does describe it.
    """
    declared = _names_in_env_example()
    for name in ("ALPACA_PAPER_KEY", "ALPACA_PAPER_SECRET"):
        assert name in declared, (
            f".env.example no longer declares {name}; a filled-in .env would "
            f"leave the broker unable to authenticate with the credential "
            f"apparently right there in the file")
