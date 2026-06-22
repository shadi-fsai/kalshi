"""Find & size page: browse/search open games (or enter a ticker) and size a bet.

The shared sidebar and connection are rendered by the router (``app.py``); this
page reads the resulting :class:`~ui.settings.Settings` from session state and
falls back to defaults so it can also run standalone (e.g. under AppTest).
"""

from __future__ import annotations

import streamlit as st

from ui import data, games, sizer
from ui.settings import Settings

st.title("Find games & size")
st.write(
    "Pick a live game and market, adjust your win-probability estimate, and the "
    "Kelly criterion sizes the bet. Use **Watch live** to monitor a game in "
    "real time, and **Portfolio** to review positions and place real orders."
)

client, auth_ok, auth_error = data.build_client()
if not auth_ok or client is None:
    st.info(
        "Add your Kalshi credentials to `.env` "
        "(`KALSHI_API_KEY_ID` and `KALSHI_PRIVATE_KEY_PATH`) to load games. "
        "See the README for setup."
    )
    if auth_error:
        st.caption(auth_error)
    st.stop()

settings = st.session_state.get("settings") or Settings.defaults()

mode = st.radio(
    "Find a market by",
    ["Browse open games", "Enter ticker manually"],
    horizontal=True,
)

selected_market = None
favored_side: str | None = None
selected_game_start = None
if mode == "Browse open games":
    selected_market, favored_side, selected_game_start = games.render_find_games(client)
else:
    selected_market = games.render_manual_ticker(client)

if selected_market:
    sizer.render_sizer(
        client,
        selected_market,
        settings,
        selected_game_start=selected_game_start,
        favored_side=favored_side,
    )

st.divider()
st.caption(
    "Kelly sizing tool. Markets are risky; verify all prices on Kalshi before "
    "trading. Order placement submits REAL limit orders to the connected "
    "environment after you confirm."
)
