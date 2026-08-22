"""Agent package.

Deliberately empty. This file used to re-export News_Processing_Agent, which
meant importing anything here -- `agent.cache` from the web-search server, for
instance -- also loaded the LLM templates and, transitively, LangGraph and
LangChain. Nothing consumed the re-export; consumers import submodules
directly (`from agent import sentry_budget`), which needs no help from here.

Keep it empty. A convenience import at this level is paid for by every module
that touches the package.
"""
