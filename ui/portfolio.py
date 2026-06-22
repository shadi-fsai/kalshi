"""Portfolio view: balance, open positions, and resting orders (with cancel)."""

from __future__ import annotations

from typing import Any

import streamlit as st

from kalshi.client import KalshiAPIError, KalshiClient
from kalshi.markets import fp_to_float


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
