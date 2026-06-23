"""End-to-end smoke tests for the multipage app using Streamlit's AppTest.

Each page script is run in isolation with the ``ui.data`` dependency-injection
seam monkeypatched to a fake client and canned fetchers, so the tests exercise
the real Streamlit render path (widgets, layout, the sizer cascade) with no
network. The bar is "no uncaught exception", plus a couple of content assertions
to prove the pages actually rendered their main flow rather than an early stop.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from kalshi.markets import build_game_groups

ROOT = Path(__file__).resolve().parents[1]
FIND = str(ROOT / "app_pages" / "find.py")
WATCH = str(ROOT / "app_pages" / "watch.py")
PORTFOLIO = str(ROOT / "app_pages" / "portfolio.py")
TENNIS = str(ROOT / "app_pages" / "tennis.py")
APP = str(ROOT / "app.py")


class _Creds:
    api_key_id = "test-key-id-1234"


class FakeClient:
    """Minimal stand-in for KalshiClient used by render_sidebar/portfolio."""

    credentials = _Creds()

    def get_balance(self):
        return {"balance": 123_45, "portfolio_value": 500_00}

    def get_positions(self):
        return {
            "market_positions": [
                {
                    "ticker": "KXWCGAME-26JUN20NEDSWE-NED",
                    "position_fp": "10000000000",
                    "market_exposure_dollars": "56.00",
                    "realized_pnl_dollars": "0.00",
                    "fees_paid_dollars": "0.10",
                },
                {
                    "ticker": "KXWCGAME-26JUN20ESPFRA-ESP",
                    "position_fp": "-5000000000",
                    "market_exposure_dollars": "20.00",
                    "realized_pnl_dollars": "0.00",
                    "fees_paid_dollars": "0.05",
                },
            ]
        }

    def get_orders(self, status=None):
        return {
            "orders": [
                {
                    "order_id": "ord-1",
                    "ticker": "KXWCGAME-26JUN20NEDSWE-NED",
                    "action": "buy",
                    "side": "yes",
                    "yes_price": 56,
                    "remaining_count": 3,
                }
            ]
        }


@pytest.fixture
def patch_data(monkeypatch, market):
    """Monkeypatch the ui.data seam: fake client + canned fetchers (no network)."""
    from ui import data

    event_ticker = market["event_ticker"]
    markets_by_ticker = {event_ticker: [market]}

    monkeypatch.setattr(data, "build_client", lambda: (FakeClient(), True, None))
    monkeypatch.setattr(data, "fetch_open_events", lambda _c, **k: [])
    monkeypatch.setattr(data, "fetch_sports_taxonomy", lambda _c: ([], {}))
    monkeypatch.setattr(
        data,
        "fetch_live_window_index",
        lambda _c: ({}, __import__("datetime").datetime.now(__import__("datetime").timezone.utc)),
    )
    monkeypatch.setattr(data, "fetch_resolution_index", lambda _c, s, **k: ({}, False))
    monkeypatch.setattr(
        data, "fetch_markets_for_event_tickers", lambda _c, t: markets_by_ticker
    )
    monkeypatch.setattr(data, "fetch_live_markets", lambda _c, t: markets_by_ticker)
    monkeypatch.setattr(data, "fetch_live_data", lambda _c, mid: None)
    monkeypatch.setattr(data, "fetch_market", lambda _c, t: market)
    monkeypatch.setattr(data, "fetch_fee_model", lambda _c, s: None)
    monkeypatch.setattr(data, "fetch_mid_prices", lambda _c, *a: [])
    monkeypatch.setattr(data, "fetch_ask_price_series", lambda _c, *a, **k: [])
    # Canned correlated mid-price series so the portfolio correlation matrix
    # renders (the two smoke positions get a shared timestamp grid).
    monkeypatch.setattr(
        data,
        "fetch_mid_price_series",
        lambda _c, *a, **k: [(1, 0.40), (2, 0.50), (3, 0.60), (4, 0.70)],
    )
    return markets_by_ticker


def test_find_page_browse_and_size(patch_data, winner_event):
    at = AppTest.from_file(FIND)
    # Seed the loaded events so the browse flow selects a game + market and runs
    # the full sizer cascade without a manual "Load" click.
    at.session_state["events"] = [winner_event]
    at.run()
    assert not at.exception
    # The sizer rendered for the auto-selected market.
    assert any("Place a limit order" in str(m.value) for m in at.markdown)


def test_find_page_without_events(patch_data):
    at = AppTest.from_file(FIND)
    at.run()
    assert not at.exception
    assert any("Load / refresh games" in str(b.label) for b in at.button)


def test_watch_page_with_handoff(patch_data, winner_event):
    group = build_game_groups([winner_event])[0]
    at = AppTest.from_file(WATCH)
    at.session_state["watch_group"] = group
    at.run()
    assert not at.exception
    # The live panel and the body sizer both rendered.
    assert any("Live" in str(m.value) for m in at.markdown)
    assert any("Size a bet on this game" in str(h.value) for h in at.subheader)


def test_portfolio_page(patch_data):
    at = AppTest.from_file(PORTFOLIO)
    at.run()
    assert not at.exception
    assert any("My portfolio" in str(h.value) for h in at.subheader)


def test_tennis_page_sizes_from_model(patch_data, market):
    at = AppTest.from_file(TENNIS)
    # Seed a model result + the auto-filled ticker so the sizing section runs the
    # full MC-priced half-Kelly path against the fake market (yes_sub "Netherlands").
    at.session_state["tn_result"] = {
        "p1_win_prob": 0.70,
        "ci": 0.5,
        "counts": {"2-0": 700, "0-2": 300},
        "n": 1000,
        "p1_name": "Netherlands",
        "p2_name": "Sweden",
        # A win-probability range from the ability sweep -> exercises the
        # uncertainty-aware shrink and the distribution histogram.
        "p1_win_dist": [0.55, 0.62, 0.68, 0.70, 0.72, 0.78, 0.85],
        "ability_unc": 5.0,
    }
    at.session_state["tn_ticker"] = market["ticker"]
    at.run()
    assert not at.exception
    assert any("Half-Kelly sizing" in str(m.value) for m in at.markdown)
    assert any("Uncertainty shrink" in str(m.label) for m in at.metric)


def _button(at, key):
    for b in at.button:
        if b.key == key:
            return b
    raise AssertionError(f"button {key!r} not found")


@pytest.fixture
def live_tennis(monkeypatch, patch_data):
    """Patch the data seam to expose one live tennis match, mispriced for an edge.

    The market implies ~even (50c) but the live score has Player 1 a set and 5-0
    up, so the model is far above the price -> a clear YES edge to scan/find.
    """
    import datetime as dt

    from ui import data

    now = dt.datetime.now(dt.timezone.utc)
    start = now - dt.timedelta(hours=1)
    event_ticker = "KXATPMATCH-26JUN23ALCSIN"
    tennis_event = {
        "event_ticker": event_ticker,
        "series_ticker": "KXATPMATCH",
        "title": "Alcaraz vs Sinner",
        "sub_title": "Who wins?",
        "product_metadata": {"competition": "ATP", "competition_scope": "Game"},
    }
    mk_a = {
        "ticker": "KXATPMATCH-26JUN23ALCSIN-ALC",
        "event_ticker": event_ticker,
        "series_ticker": "KXATPMATCH",
        "yes_sub_title": "Alcaraz",
        "yes_ask_dollars": "0.50",
        "no_ask_dollars": "0.50",
        "custom_strike": {"tennis_competitor": "cid-alc"},
    }
    mk_b = {
        "ticker": "KXATPMATCH-26JUN23ALCSIN-SIN",
        "event_ticker": event_ticker,
        "series_ticker": "KXATPMATCH",
        "yes_sub_title": "Sinner",
        "yes_ask_dollars": "0.50",
        "no_ask_dollars": "0.50",
        "custom_strike": {"tennis_competitor": "cid-sin"},
    }
    details = {
        "competitor1_id": "cid-alc",
        "competitor2_id": "cid-sin",
        "competitor1_overall_score": 1,
        "competitor2_overall_score": 0,
        "competitor1_round_scores": [
            {"outcome": "winner", "score": 6},
            {"outcome": "ongoing", "score": 5},
        ],
        "competitor2_round_scores": [
            {"outcome": "loser", "score": 3},
            {"outcome": "ongoing", "score": 0},
        ],
        "competitor1_current_round_score": 40,
        "competitor2_current_round_score": 0,
        "server": "cid-alc",
    }
    timing = {event_ticker: {"start": start, "status": "live", "milestone_id": "mil-1"}}

    monkeypatch.setattr(data, "fetch_open_events", lambda _c, **k: [tennis_event])
    monkeypatch.setattr(data, "fetch_sports_taxonomy", lambda _c: ([], {"ATP": "Tennis"}))
    monkeypatch.setattr(data, "fetch_live_window_index", lambda _c: (timing, now))
    monkeypatch.setattr(
        data, "fetch_markets_for_event_tickers", lambda _c, t: {event_ticker: [mk_a, mk_b]}
    )
    monkeypatch.setattr(data, "fetch_live_data", lambda _c, mid: details)
    monkeypatch.setattr(data, "fetch_market", lambda _c, t: mk_a)
    monkeypatch.setattr(data, "fetch_fee_model", lambda _c, s: None)
    return mk_a


def test_tennis_scan_finds_edge_and_opens(live_tennis):
    at = AppTest.from_file(TENNIS)
    at.run()
    assert not at.exception
    # Run the scan.
    _button(at, "tn_scan").click()
    at.run()
    assert not at.exception
    # An opportunity table rendered with the YES (Player 1) edge.
    frames = "".join(str(df.value) for df in at.dataframe)
    assert "Alcaraz" in frames
    assert "YES (Alcaraz)" in frames
    # Open the (default-selected) match -> detailed half-Kelly sizing renders.
    _button(at, "tn_open").click()
    at.run()
    assert not at.exception
    assert any("Half-Kelly sizing" in str(m.value) for m in at.markdown)


def test_tennis_scan_no_live_matches(patch_data):
    # patch_data exposes no events, so a scan simply finds nothing (no crash).
    at = AppTest.from_file(TENNIS)
    at.run()
    _button(at, "tn_scan").click()
    at.run()
    assert not at.exception
    assert any("Scanned 0 live match" in str(c.value) for c in at.caption)


def test_router_app_runs(patch_data):
    at = AppTest.from_file(APP)
    at.run()
    assert not at.exception
