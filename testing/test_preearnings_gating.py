"""Phase F unit tests — deep-research cost gate (no network)."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.preearnings.gating import should_deep_research


def test_gate_passes_within_window_liquid_with_peers():
    out = should_deep_research(days_to_earnings=5, liquid=True, has_peers=True)
    assert out["deep"] is True


def test_gate_blocks_far_dated():
    out = should_deep_research(days_to_earnings=30, liquid=True, has_peers=True)
    assert out["deep"] is False
    assert "window" in out["reason"]


def test_gate_blocks_after_earnings():
    out = should_deep_research(days_to_earnings=-1, liquid=True, has_peers=True)
    assert out["deep"] is False
    assert "passed" in out["reason"]


def test_gate_blocks_illiquid():
    out = should_deep_research(days_to_earnings=3, liquid=False, has_peers=True)
    assert out["deep"] is False
    assert "liquid" in out["reason"]


def test_gate_blocks_no_deep_inputs():
    out = should_deep_research(days_to_earnings=3, liquid=True, has_peers=False, has_options=False)
    assert out["deep"] is False


def test_gate_passes_with_options_only():
    out = should_deep_research(days_to_earnings=3, liquid=True, has_peers=False, has_options=True)
    assert out["deep"] is True


def test_gate_no_date():
    out = should_deep_research(days_to_earnings=None, liquid=True, has_peers=True)
    assert out["deep"] is False
