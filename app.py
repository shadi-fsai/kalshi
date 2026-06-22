"""Streamlit entrypoint: multipage router for the Kalshi Kelly app.

This script wires the three pages together with ``st.navigation`` and renders the
shared sidebar once per rerun so the connection, bankroll, Kelly, volatility, and
fee settings apply across every page. The actual page logic lives in ``pages/``
and reuses the building blocks in the ``ui`` package; pure market/finance logic
stays in the ``kalshi`` package.

Pages:
- Find games & size: browse/search open games (or enter a ticker) and size a bet.
- Watch a game live: monitor one game's score + opportunities, size on the fly.
- Portfolio: balance, positions, and resting orders (with cancel).
"""

from __future__ import annotations

import streamlit as st
from dotenv import load_dotenv

from ui import data
from ui.settings import render_sidebar

load_dotenv()

st.set_page_config(page_title="Kalshi Kelly Sizer", page_icon="📈", layout="wide")

# Build the client once (cached) and render the shared sidebar; the resulting
# Settings snapshot is stashed for the active page to read.
client, auth_ok, auth_error = data.build_client()
st.session_state["settings"] = render_sidebar(client, auth_ok, auth_error)

find_page = st.Page(
    "app_pages/find.py", title="Find games & size", icon="🔎", default=True
)
watch_page = st.Page("app_pages/watch.py", title="Watch a game live", icon="📺")
portfolio_page = st.Page("app_pages/portfolio.py", title="Portfolio", icon="💼")

st.navigation([find_page, watch_page, portfolio_page]).run()
