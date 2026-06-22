"""Watch live page: monitor one game's score + opportunities and size on the fly.

A game can be handed off from the Find page (via session state ``watch_group``)
or picked here from the live/soon games. The live score and per-market
opportunities panel auto-refreshes on a short cadence via ``st.fragment``; the
market picker and the full sizer live in the static body (so widget state and
order tickets are not disrupted by the refresh).
"""

from __future__ import annotations

import datetime
from typing import Any

import streamlit as st

from kalshi.client import KalshiAPIError
from kalshi.markets import (
    build_game_groups,
    classify_timing,
    evaluate_in_money,
    event_competition,
    in_money_badge,
    live_scores,
    market_type_name,
    price_cents_for_side,
)
from ui import data, sizer
from ui.settings import Settings

st.title("Watch a game live")

client, auth_ok, auth_error = data.build_client()
if not auth_ok or client is None:
    st.info(
        "Add your Kalshi credentials to `.env` "
        "(`KALSHI_API_KEY_ID` and `KALSHI_PRIVATE_KEY_PATH`) to watch a game. "
        "See the README for setup."
    )
    if auth_error:
        st.caption(auth_error)
    st.stop()

settings = st.session_state.get("settings") or Settings.defaults()

# --- Pick the game to watch --------------------------------------------------
load_col, info_col = st.columns([1, 3])
with load_col:
    if st.button("Load / refresh live games", type="primary"):
        try:
            with st.spinner("Loading open games from Kalshi…"):
                st.session_state["events"] = data.fetch_open_events(client)
        except KalshiAPIError as exc:
            st.error(f"Failed to load events ({exc.status_code}): {exc.message}")

timing_index: dict[str, dict[str, Any]] = {}
now_utc = datetime.datetime.now(datetime.timezone.utc)
try:
    timing_index, now_utc = data.fetch_live_window_index(client)
except KalshiAPIError as exc:
    st.warning(
        f"Could not load game start times ({exc.status_code}): {exc.message}."
    )


def _grp_timing(group: dict[str, Any]) -> tuple[str, str] | None:
    for ticker in (group["rep_ticker"], *group["event_tickers"]):
        if ticker in timing_index:
            return classify_timing(timing_index[ticker], now_utc)
    return None


def _group_label(group: dict[str, Any]) -> str:
    timing = _grp_timing(group)
    badge = f"{timing[1]}  ·  " if timing else ""
    bits = [group["matchup"], group["sub_title"], group["competition"]]
    return badge + "  ·  ".join(b for b in bits if b)


events = st.session_state.get("events", [])
sports_events = [e for e in events if event_competition(e)]
groups = build_game_groups(sports_events) if sports_events else []
# Prefer live/soon games for the watch list.
live_groups = [
    g for g in groups if (t := _grp_timing(g)) and t[0] in ("live", "soon")
]

handoff = st.session_state.get("watch_group")
options = list(live_groups)
if handoff:
    handoff_key = handoff.get("rep_ticker")
    if not any(g.get("rep_ticker") == handoff_key for g in options):
        options = [handoff, *options]

if not options:
    st.info(
        "No live or upcoming games loaded yet. Click **Load / refresh live "
        "games** above, or hand a game off from the **Find games & size** page "
        "with the *Watch this game live* button."
    )
    st.stop()

with info_col:
    st.caption(f"{len(options)} game(s) available to watch.")

default_index = 0
if handoff:
    default_index = next(
        (i for i, g in enumerate(options) if g.get("rep_ticker") == handoff.get("rep_ticker")),
        0,
    )

group = st.selectbox(
    "Live game", options, index=default_index, format_func=_group_label
)

refresh_secs = st.select_slider(
    "Live refresh cadence (seconds)",
    options=[10, 15, 20, 30, 60],
    value=20,
    help="How often the live score and opportunities panel below re-fetches "
    "from Kalshi. The market picker and sizer are not disrupted by the refresh.",
)

# Resolve milestone/team ids and kickoff for this game from the live window.
meta = next(
    (
        timing_index[et]
        for et in group["event_tickers"]
        if timing_index.get(et, {}).get("milestone_id")
    ),
    None,
)
grp_start = next(
    (
        timing_index[et]["start"]
        for et in (group["rep_ticker"], *group["event_tickers"])
        if timing_index.get(et, {}).get("start")
    ),
    None,
)
siblings = {e.get("event_ticker"): e for e in group["events"]}
event_tickers = tuple(group["event_tickers"])


def _open_markets(markets_by_ticker: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    combined = [m for ms in markets_by_ticker.values() for m in ms]
    return [m for m in combined if m.get("status") in ("active", "open")] or combined


# --- Live panel (auto-refreshing) -------------------------------------------
@st.fragment(run_every=refresh_secs)
def live_panel() -> None:
    """Live score + per-market opportunities. Display only (no input widgets)."""
    st.markdown("#### Live")
    st.caption(
        f"Updated {datetime.datetime.now().strftime('%H:%M:%S')} · "
        f"auto-refresh every {refresh_secs}s"
    )

    details: dict[str, Any] | None = None
    if meta and meta.get("milestone_id"):
        try:
            details = data.fetch_live_data(client, meta["milestone_id"])
        except KalshiAPIError as exc:
            st.warning(f"Live score unavailable ({exc.status_code}): {exc.message}.")

    try:
        markets_by_ticker = data.fetch_live_markets(client, event_tickers)
    except KalshiAPIError as exc:
        st.error(f"Failed to load markets ({exc.status_code}): {exc.message}.")
        return

    combined = [m for ms in markets_by_ticker.values() for m in ms]
    opens = _open_markets(markets_by_ticker)
    home_id = meta.get("home_team_id") if meta else None
    away_id = meta.get("away_team_id") if meta else None

    if details:
        scores = live_scores(details)
        name_by_id = {
            (m.get("custom_strike") or {}).get("soccer_team"): m.get("yes_sub_title")
            for m in combined
            if (m.get("custom_strike") or {}).get("soccer_team")
        }
        home_name = name_by_id.get(home_id) or "Home"
        away_name = name_by_id.get(away_id) or "Away"
        status_text = (
            details.get("status_text") or details.get("match_status") or "live"
        )
        if scores:
            h, a = scores
            st.info(
                f"🔴 LIVE — {home_name} {h:.0f} : {a:.0f} {away_name}  ·  {status_text}"
            )
        else:
            st.caption(f"Live status: {status_text}")
    else:
        st.caption("No live score available (the game may not be in progress).")

    rows = []
    for m in opens:
        ev = siblings.get(m.get("event_ticker"), {})
        itm = evaluate_in_money(m, details, home_id, away_id) if details else None
        yes_ask = price_cents_for_side(m, "yes", "ask")
        no_ask = price_cents_for_side(m, "no", "ask")
        rows.append(
            {
                "ITM": in_money_badge(itm) or "—",
                "Type": market_type_name(ev),
                "Outcome": m.get("yes_sub_title") or m.get("ticker", ""),
                "YES ask (c)": f"{yes_ask:.0f}" if yes_ask else "—",
                "NO ask (c)": f"{no_ask:.0f}" if no_ask else "—",
            }
        )
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.caption("No open markets for this game right now.")


live_panel()

# --- Size a bet (static body; full sizer) -----------------------------------
st.divider()
st.subheader("Size a bet on this game")

try:
    markets_by_ticker = data.fetch_live_markets(client, event_tickers)
except KalshiAPIError as exc:
    st.error(f"Failed to load markets ({exc.status_code}): {exc.message}.")
    markets_by_ticker = {}

open_markets = _open_markets(markets_by_ticker)


def _mkt_label(m: dict[str, Any]) -> str:
    ev = siblings.get(m.get("event_ticker"), {})
    mtype = market_type_name(ev) if ev else ""
    outcome = m.get("yes_sub_title") or m.get("ticker", "")
    yes_ask = price_cents_for_side(m, "yes", "ask")
    price_part = f"  ·  YES {yes_ask:.0f}c" if yes_ask else ""
    return f"{mtype} — {outcome}{price_part}"


if open_markets:
    open_markets.sort(
        key=lambda m: (
            market_type_name(siblings.get(m.get("event_ticker"), {})),
            m.get("yes_sub_title") or "",
        )
    )
    selected_market = st.selectbox(
        "Market", options=open_markets, format_func=_mkt_label, key="watch_market"
    )
    sizer.render_sizer(
        client, selected_market, settings, selected_game_start=grp_start
    )
else:
    st.info("No open markets found for this game.")

st.divider()
st.caption(
    "Live monitoring + Kelly sizing. Verify all prices on Kalshi before trading. "
    "Order placement submits REAL limit orders after you confirm."
)
