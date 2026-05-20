"""Alpaca paper-trading MCP server.

Exposes Risk_Officer-gated tools for Claude Code to read account state and
place paper orders. Production gating logic lives in agent/Risk_Officer.py
and agent/Execution_Agent.py; this package is a thin MCP wrapper.

Entry point:  python -m tools.alpaca.server server
"""
