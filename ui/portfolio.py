"""Portfolio view: balance, open positions, and resting orders (with cancel)."""

from __future__ import annotations

import datetime
from typing import Any

import pandas as pd
import streamlit as st

from kalshi.client import KalshiAPIError, KalshiClient
from kalshi.markets import fp_to_float, series_ticker_for_market
from kalshi.risk import correlation_matrix, high_correlation_pairs
from ui import data
from ui.stops import render_stops

# Correlation window selector -> lookback in MINUTES. Short windows use 1-minute
# candles for resolution; longer ones fall back to hourly to keep the per-
# position candlestick payload bounded (see _period_for_window).
_CORR_WINDOWS: dict[str, int] = {
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "2h": 120,
    "6h": 360,
    "24h": 1440,
    "7d": 10080,
}


def _period_for_window(window_minutes: float) -> int:
    """Candle interval (minutes) for a lookback: 1-min up to 3h, else hourly.

    Matches the sizer's price-chart rule so short windows stay high-resolution
    while long windows keep the candlestick payload bounded.
    """
    return 1 if window_minutes <= 180 else 60


def render_portfolio(_client: KalshiClient) -> None:
    """Render balance, current positions, and resting orders (with cancel)."""
    st.subheader("My portfolio")

    top = st.columns([1, 1, 2])
    with top[0]:
        st.button("Refresh", help="Reload balance, positions, and orders.")

    try:
        balance = _client.get_balance()
        positions = _client.get_positions().get("market_positions", [])
        orders = _client.get_orders(status="resting").get("orders", [])
    except KalshiAPIError as exc:
        st.error(
            f"Could not load portfolio ({exc.status_code}): {exc.message}. "
            "Your API key may lack portfolio/trading scope."
        )
        return

    bal_dollars = balance.get("balance", 0) / 100.0
    top[1].metric("Cash balance", f"${bal_dollars:,.2f}")
    if balance.get("portfolio_value") is not None:
        top[2].metric(
            "Portfolio value", f"${balance.get('portfolio_value', 0) / 100.0:,.2f}"
        )

    st.markdown("#### Positions")
    held = [p for p in positions if fp_to_float(p.get("position_fp")) != 0]
    if not held:
        st.caption("No open positions.")
    else:
        rows = []
        for p in held:
            qty = fp_to_float(p.get("position_fp"))
            rows.append(
                {
                    "Ticker": p.get("ticker", ""),
                    "Side": "YES" if qty > 0 else "NO",
                    "Contracts": abs(qty),
                    "Exposure ($)": fp_to_float(p.get("market_exposure_dollars")),
                    "Realized P&L ($)": fp_to_float(p.get("realized_pnl_dollars")),
                    "Fees paid ($)": fp_to_float(p.get("fees_paid_dollars")),
                }
            )
        st.dataframe(rows, use_container_width=True, hide_index=True)
        _render_correlation(_client, held)

    render_stops(_client, positions)

    st.markdown("#### Resting orders")
    if not orders:
        st.caption("No resting (open) orders.")
    else:
        for o in orders:
            order_id = o.get("order_id") or o.get("id") or ""
            cols = st.columns([3, 2, 2, 2, 1])
            cols[0].write(f"`{o.get('ticker', '')}`")
            action = o.get("action") or ""
            side = o.get("side") or ""
            cols[1].write(f"{action.upper()} {side.upper()}".strip())
            # Price field name varies; show whichever is present.
            price_c = (
                o.get("yes_price")
                if o.get("yes_price") is not None
                else o.get("no_price")
            )
            cols[2].write(f"{price_c}c" if price_c is not None else "—")
            remaining = o.get("remaining_count")
            cols[3].write(f"x{remaining}" if remaining is not None else "")
            if cols[4].button("Cancel", key=f"cancel_{order_id}"):
                try:
                    _client.cancel_order(order_id)
                    st.success(f"Canceled order {order_id[:8]}…")
                    st.rerun()
                except KalshiAPIError as exc:
                    st.error(f"Cancel failed ({exc.status_code}): {exc.message}")


def _corr_cell_style(value: object) -> str:
    """Diverging background for one correlation cell (red +, blue -, blank NaN)."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    v = max(-1.0, min(1.0, float(value)))
    alpha = abs(v)
    if v >= 0:
        return f"background-color: rgba(214, 39, 40, {alpha:.2f})"
    return f"background-color: rgba(31, 119, 180, {alpha:.2f})"


def _render_correlation(
    _client: KalshiClient, held: list[dict[str, Any]]
) -> None:
    """Render the empirical position-correlation matrix + high-correlation callout."""
    if len(held) < 2:
        return

    st.markdown("#### Position correlation")
    st.caption(
        "Empirical correlation of each position's mid-price returns over the "
        "selected window, oriented to the side you hold (a NO holding uses "
        "1 - YES). High positive values mean those bets tend to win and lose "
        "together."
    )
    window = st.radio(
        "Window",
        list(_CORR_WINDOWS),
        index=len(_CORR_WINDOWS) - 1,  # default 7d for the most overlap
        horizontal=True,
        key="corr_window",
    )
    now = datetime.datetime.now(datetime.timezone.utc)
    end_ts = int(now.timestamp())
    start_ts = end_ts - _CORR_WINDOWS[window] * 60
    if end_ts <= start_ts:
        st.caption("Invalid correlation window.")
        return
    period_interval = _period_for_window(_CORR_WINDOWS[window])

    series_by_key: dict[str, list[tuple[int, float]]] = {}
    skipped: list[str] = []
    for p in held:
        ticker = p.get("ticker", "")
        if not ticker:
            continue
        qty = fp_to_float(p.get("position_fp"))
        side = "yes" if qty > 0 else "no"
        series_ticker = series_ticker_for_market(p)
        if not series_ticker:
            skipped.append(ticker)
            continue
        try:
            series = data.fetch_mid_price_series(
                _client,
                series_ticker,
                ticker,
                side,
                start_ts,
                end_ts,
                period_interval,
            )
        except KalshiAPIError:
            series = []
        if series:
            series_by_key[ticker] = series
        else:
            skipped.append(ticker)

    if len(series_by_key) < 2:
        st.caption(
            "Not enough positions with price history in this window to compute "
            "correlations. Try a longer window."
        )
        return

    result = correlation_matrix(series_by_key)
    df = pd.DataFrame(
        [[(v if v is not None else float("nan")) for v in row] for row in result.matrix],
        index=result.labels,
        columns=result.labels,
    )
    styled = df.style.format(precision=2, na_rep="—").map(_corr_cell_style)
    st.dataframe(styled, use_container_width=True)
    st.caption(f"Based on {result.overlap} overlapping return sample(s) per pair.")
    if skipped:
        st.caption(
            f"Excluded {len(skipped)} position(s) without usable history: "
            + ", ".join(f"`{t}`" for t in skipped)
        )

    threshold = st.slider(
        "Flag pairs at or above |correlation|",
        min_value=0.5,
        max_value=1.0,
        value=0.7,
        step=0.05,
        key="corr_threshold",
    )
    pairs = high_correlation_pairs(result, threshold)
    if pairs:
        st.warning("Highly correlated positions:")
        for a, b, corr in pairs:
            st.write(f"- `{a}` and `{b}`: **{corr:+.2f}**")
    else:
        st.caption("No position pairs at or above the threshold.")
