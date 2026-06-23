"""Unit tests for the tennis page's pure helpers (no Streamlit render needed)."""

from __future__ import annotations

from ui.settings import Settings
from ui.tennis import _evaluate_edge


def _market(yes_ask: str = "0.56", no_ask: str = "0.45") -> dict:
    return {
        "ticker": "KXATPMATCH-TEST-A",
        "series_ticker": "KXATPMATCH",
        "yes_sub_title": "Player A",
        "yes_ask_dollars": yes_ask,
        "no_ask_dollars": no_ask,
    }


def _settings() -> Settings:
    # fallback_fee 0 keeps the breakevens at the raw ask for easy reasoning.
    return Settings(
        bankroll=1000.0,
        kelly_multiplier=0.5,
        vol_adjust=False,
        vol_sensitivity=0.0,
        fallback_fee=0.0,
    )


def test_evaluate_edge_picks_yes_when_model_beats_yes_ask():
    edge = _evaluate_edge(
        _market(), "Player A", "Player B", 0.75, [0.75], 1, None, _settings()
    )
    assert edge is not None
    assert edge.side == "yes"
    assert edge.side_label == "Player A"
    assert edge.contracts > 0


def test_evaluate_edge_picks_no_when_model_low():
    # model P1 = 0.20 -> NO (Player B) at 45c (breakeven 0.45) has a big edge.
    edge = _evaluate_edge(
        _market(), "Player A", "Player B", 0.20, [0.20], 1, None, _settings()
    )
    assert edge is not None
    assert edge.side == "no"
    assert edge.side_label == "Player B"


def test_evaluate_edge_returns_none_without_edge():
    # Between the two breakevens (yes 0.56, no -> 0.55), neither side has an edge.
    edge = _evaluate_edge(
        _market(), "Player A", "Player B", 0.555, [0.555], 1, None, _settings()
    )
    assert edge is None


def test_evaluate_edge_shrinks_used_fraction_with_uncertainty():
    point = _evaluate_edge(
        _market(), "Player A", "Player B", 0.75, [0.75], 1, None, _settings()
    )
    spread = _evaluate_edge(
        _market(),
        "Player A",
        "Player B",
        0.75,
        [0.55, 0.65, 0.75, 0.85, 0.95],
        1,
        None,
        _settings(),
    )
    assert point is not None and spread is not None
    assert point.shrink == 0.0
    assert spread.shrink > 0.0
    assert spread.used_fraction < point.used_fraction


def test_evaluate_edge_respects_yes_player_orientation():
    # YES tracks Player 2, so a low P1 prob makes YES (Player B) the value side.
    edge = _evaluate_edge(
        _market(), "Player A", "Player B", 0.20, [0.20], 2, None, _settings()
    )
    assert edge is not None
    assert edge.side == "yes"
    assert edge.side_label == "Player B"
