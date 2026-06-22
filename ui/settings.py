"""Shared sidebar settings rendered once by the router for every page.

``render_sidebar`` draws the connection status, bankroll, Kelly multiplier,
volatility/time, and fallback-fee controls and returns a :class:`Settings`
snapshot. Pages read the snapshot from ``st.session_state["settings"]`` (falling
back to :meth:`Settings.defaults` so a page can run standalone, e.g. under tests).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import streamlit as st

from kalshi.client import DEFAULT_BASE_URL, KalshiAPIError, KalshiClient
from kalshi.markets import fp_to_float


@dataclass
class Settings:
    """User-controlled sizing inputs shared across pages."""

    bankroll: float
    kelly_multiplier: float
    vol_adjust: bool
    vol_sensitivity: float
    fallback_fee: float

    @classmethod
    def defaults(cls) -> "Settings":
        """Defaults matching the sidebar widget defaults (for standalone/test use)."""
        return cls(
            bankroll=1000.0,
            kelly_multiplier=0.5,
            vol_adjust=True,
            vol_sensitivity=1.0,
            fallback_fee=0.01,
        )


def render_sidebar(
    client: KalshiClient | None, auth_ok: bool, auth_error: str | None = None
) -> Settings:
    """Render the shared sidebar and return the current :class:`Settings`."""
    st.sidebar.title("Kalshi Kelly Sizer")
    st.sidebar.caption(f"Endpoint: `{os.getenv('KALSHI_API_BASE', DEFAULT_BASE_URL)}`")

    if auth_ok and client is not None:
        masked = client.credentials.api_key_id[:8] + "…"
        st.sidebar.success(f"Credentials loaded (key {masked})")
    elif auth_error:
        st.sidebar.error(auth_error)

    st.sidebar.divider()
    st.sidebar.subheader("Bankroll")
    bankroll_source = st.sidebar.radio(
        "Source", ["Manual entry", "Account balance"], index=1
    )

    bankroll = 0.0
    if bankroll_source == "Manual entry":
        bankroll = st.sidebar.number_input(
            "Bankroll ($)", min_value=0.0, value=1000.0, step=50.0
        )
    else:
        if not auth_ok or client is None:
            st.sidebar.warning("Connect credentials to pull your balance.")
        else:
            # Auto-fetch once on first load so the balance shows without a click;
            # the button stays available for manual refreshes afterward.
            auto_fetch = not st.session_state.get("_balance_autofetched")
            if st.sidebar.button("Fetch balance") or auto_fetch:
                st.session_state["_balance_autofetched"] = True
                try:
                    bal = client.get_balance()
                    st.session_state["bankroll_cents"] = bal.get("balance", 0)
                    positions = client.get_positions().get("market_positions", [])
                    st.session_state["positions_value"] = sum(
                        fp_to_float(p.get("market_exposure_dollars")) for p in positions
                    )
                except KalshiAPIError as exc:
                    st.sidebar.error(f"Balance error ({exc.status_code}): {exc.message}")
        if "bankroll_cents" in st.session_state:
            cash = st.session_state["bankroll_cents"] / 100.0
            pos_value = st.session_state.get("positions_value", 0.0)
            # Size off total account equity (cash + the market value of open
            # positions), not just idle cash, so the Kelly fraction reflects the
            # full bankroll at risk rather than understating it.
            bankroll = cash + pos_value
            st.sidebar.metric("Cash balance", f"${cash:,.2f}")
            st.sidebar.metric("Positions value", f"${pos_value:,.2f}")
            st.sidebar.metric(
                "Total bankroll (sizing)",
                f"${bankroll:,.2f}",
                help="Cash balance plus the value of your open positions. Kelly "
                "sizing uses this total.",
            )

    st.sidebar.divider()
    kelly_multiplier = st.sidebar.slider(
        "Fractional Kelly multiplier",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="0.5 = half-Kelly. Lower is more conservative.",
    )

    st.sidebar.divider()
    st.sidebar.subheader("Volatility / time (Sharpe)")
    vol_adjust = st.sidebar.checkbox(
        "Volatility/time-adjust sizing",
        value=True,
        help="Shrink the stake on top of Kelly when the market price is volatile "
        "and lots of time remains to expiry (estimate is still uncertain). Uses "
        "realized volatility from Kalshi candlesticks since the game started.",
    )
    vol_sensitivity = st.sidebar.slider(
        "Volatility sensitivity",
        min_value=0.0,
        max_value=3.0,
        value=1.0,
        step=0.25,
        disabled=not vol_adjust,
        help="How aggressively remaining volatility shrinks the stake. 0 disables "
        "the shrink; higher sizes volatile, far-from-expiry bets smaller.",
    )

    st.sidebar.divider()
    st.sidebar.subheader("Transaction fees")
    st.sidebar.caption(
        "Fees are pulled live from each market's Kalshi series fee model and "
        "modeled round-trip into Kelly edge, breakeven, and sizing. The value below "
        "is only a fallback used when the API fee can't be computed (e.g. flat-fee "
        "markets)."
    )
    fallback_fee_cents = st.sidebar.number_input(
        "Fallback fee (c / contract / side)",
        min_value=0.0,
        max_value=50.0,
        value=1.0,
        step=0.5,
    )
    fallback_fee = fallback_fee_cents / 100.0

    return Settings(
        bankroll=bankroll,
        kelly_multiplier=kelly_multiplier,
        vol_adjust=vol_adjust,
        vol_sensitivity=vol_sensitivity,
        fallback_fee=fallback_fee,
    )
