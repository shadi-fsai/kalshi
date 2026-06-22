"""Portfolio page: balance, positions, and resting orders (with cancel)."""

from __future__ import annotations

import streamlit as st

from ui import data
from ui.portfolio import render_portfolio

st.title("Portfolio")

client, auth_ok, auth_error = data.build_client()
if not auth_ok or client is None:
    st.info(
        "Add your Kalshi credentials to `.env` "
        "(`KALSHI_API_KEY_ID` and `KALSHI_PRIVATE_KEY_PATH`) to view your "
        "portfolio. See the README for setup."
    )
    if auth_error:
        st.caption(auth_error)
    st.stop()

render_portfolio(client)
