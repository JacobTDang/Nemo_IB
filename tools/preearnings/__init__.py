"""Deterministic logic for the expectations-centric pre-earnings engine.

These modules hold pure, unit-testable functions (windows, classification,
scoring, aggregation). The sub-agent orchestration that feeds them lives in the
skills, so the math here can be verified without spawning agents or hitting the
network. Nothing in this package hardcodes companies, tickers, peers, or KPIs —
every company-specific input is passed in, derived at runtime by the caller.
"""
