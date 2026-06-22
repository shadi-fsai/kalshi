"""Static import checks for the ui/ package.

These guard against typos and broken imports in the Streamlit-bound modules
without needing a running app: importing the modules executes their top-level
code (imports + cache decorators + function defs), and we assert the public
callables the pages rely on are present.
"""

from __future__ import annotations

import importlib


def test_ui_modules_import():
    for name in ("ui.data", "ui.settings", "ui.sizer", "ui.portfolio", "ui.games"):
        assert importlib.import_module(name) is not None


def test_data_exposes_fetchers_and_client_seam():
    from ui import data

    for fn in (
        "build_client",
        "get_client",
        "fetch_open_events",
        "fetch_sports_taxonomy",
        "fetch_live_window_index",
        "fetch_resolution_index",
        "fetch_markets_for_event_tickers",
        "fetch_live_markets",
        "fetch_live_data",
        "fetch_fee_model",
        "fetch_mid_prices",
    ):
        assert callable(getattr(data, fn)), fn


def test_settings_dataclass_defaults():
    from ui.settings import Settings, render_sidebar

    assert callable(render_sidebar)
    s = Settings.defaults()
    assert s.bankroll == 1000.0
    assert 0.0 <= s.kelly_multiplier <= 1.0
    assert s.vol_adjust is True
    assert s.vol_sensitivity == 1.0
    assert s.fallback_fee == 0.01


def test_render_callables_present():
    from ui import games, portfolio, sizer

    assert callable(sizer.render_sizer)
    assert callable(sizer.render_order_ticket)
    assert callable(games.render_find_games)
    assert callable(games.render_manual_ticker)
    assert callable(portfolio.render_portfolio)
