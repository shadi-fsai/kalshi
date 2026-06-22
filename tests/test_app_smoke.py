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


def test_router_app_runs(patch_data):
    at = AppTest.from_file(APP)
    at.run()
    assert not at.exception
