"""Tennis match pricing page: Monte Carlo pricing vs the Kalshi market.

The simulation works without credentials; only the optional Kalshi market
comparison needs an authenticated client.
"""

from __future__ import annotations

import streamlit as st

from ui import data
from ui.tennis import render_tennis

st.title("Tennis match pricing")

client, auth_ok, auth_error = data.build_client()
if not auth_ok and auth_error:
    st.caption(auth_error)

render_tennis(client, auth_ok)
