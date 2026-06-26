"""Portfolio stop-loss manager UI.

Renders the "Stop losses" section on the Portfolio page: an engine-health banner,
a live (auto-refreshing) status table of every managed stop with a Remove button,
and a form to add a new stop from a current position.

This UI only OWNS the config file (add/remove). The actual watching and order
firing is done by the out-of-process engine (``kalshi-stop-engine``); this page
reads the engine's status file to show where each stop is. If the engine is not
running, the heartbeat goes stale and we warn loudly -- a stop is only active
while the engine process runs.
"""

from __future__ import annotations

import datetime
import time
from typing import Any

import streamlit as st

from kalshi.client import KalshiAPIError, KalshiClient
from kalshi.markets import fp_to_float
from kalshi.stops import (
    HEARTBEAT_STALE_SECS,
    StopConfig,
    StopStore,
    ref_price_cents,
)
from ui import data

ENGINE_START_CMD = "uv run kalshi-stop-engine"


def _store() -> StopStore:
    return StopStore()


def _held_positions(positions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return [{ticker, side, count}] for non-zero positions.

    ``count`` is the contract size and may be fractional (Kalshi fixed-point
    contracts, e.g. 0.20), so we keep it as a float instead of truncating it.
    """
    out: list[dict[str, Any]] = []
    for p in positions:
        qty = fp_to_float(p.get("position_fp"))
        if qty == 0:
            continue
        out.append(
            {
                "ticker": p.get("ticker", ""),
                "side": "yes" if qty > 0 else "no",
                "count": round(abs(qty), 2),
            }
        )
    return out


def render_stops(client: KalshiClient, positions: list[dict[str, Any]]) -> None:
    """Render the full stop-loss manager section."""
    st.markdown("#### Stop losses")
    st.caption(
        "Synthetic stop-losses managed by a separate engine process. Stops only "
        "fire while that engine is running."
    )
    store = _store()

    _live_status(store, client)
    st.markdown("##### Add a stop")
    _add_stop_form(store, positions)


@st.fragment(run_every=5)
def _live_status(store: StopStore, client: KalshiClient) -> None:
    """Auto-refreshing engine banner + status table (reruns every 5s)."""
    status = store.read_status()
    age = store.heartbeat_age_secs()
    if age is None:
        st.error(
            "Stop engine has never reported in -- stops are NOT being managed. "
            f"Start it with:  `{ENGINE_START_CMD}`"
        )
    elif age > HEARTBEAT_STALE_SECS:
        st.error(
            f"Stop engine heartbeat is stale ({age:.0f}s old) -- stops may NOT be "
            f"managed. Restart it with:  `{ENGINE_START_CMD}`"
        )
    else:
        pid = status.get("engine_pid")
        st.success(f"Stop engine live (heartbeat {age:.0f}s ago, pid {pid}).")

    configs = store.list_configs()
    if not configs:
        st.caption("No stops configured. Add one below.")
        return

    stop_status: dict[str, Any] = status.get("stops", {})
    rows: list[dict[str, Any]] = []
    for cfg in configs:
        s = stop_status.get(cfg.id, {})
        ref = s.get("last_ref_cents")
        if ref is None:
            ref = _fallback_ref(client, cfg)
        distance = None if ref is None else round(ref - cfg.stop_cents, 1)
        updated = s.get("last_update_ts")
        rows.append(
            {
                "Ticker": cfg.ticker,
                "Side": cfg.held_side.upper(),
                "Count": _fmt_count(s.get("count", cfg.count)),
                "Stop (c)": cfg.stop_cents,
                "Now (c)": "—" if ref is None else round(ref, 1),
                "Dist (c)": "—" if distance is None else distance,
                "Ref": cfg.trigger_ref,
                "Env": cfg.env,
                "Armed": "yes" if cfg.armed else "no",
                "State": s.get("state", "pending"),
                "Updated": _ago(updated),
                "Note": s.get("message", ""),
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)

    st.caption("Remove a stop:")
    for cfg in configs:
        cols = st.columns([4, 2, 2, 1])
        cols[0].write(f"`{cfg.ticker}`")
        cols[1].write(f"{cfg.held_side.upper()} @ {cfg.stop_cents:.0f}c")
        cols[2].write(cfg.env)
        if cols[3].button("Remove", key=f"rm_stop_{cfg.id}"):
            if store.remove(cfg.id):
                st.toast(f"Removed stop for {cfg.ticker}")
            st.rerun(scope="fragment")


def _fallback_ref(client: KalshiClient, cfg: StopConfig) -> float | None:
    """Held-side ref price (cents) from REST when the engine has no status yet."""
    try:
        market = data.fetch_market(client, cfg.ticker)
    except KalshiAPIError:
        return None

    def cents(key: str) -> float | None:
        val = market.get(key)
        try:
            return None if val is None else float(val)
        except (TypeError, ValueError):
            return None

    return ref_price_cents(
        cents("yes_bid"), cents("yes_ask"), cents("last_price"), cfg.held_side, cfg.trigger_ref
    )


def _add_stop_form(store: StopStore, positions: list[dict[str, Any]]) -> None:
    held = _held_positions(positions)
    if not held:
        st.caption("No open positions to protect.")
        return

    labels = [f"{h['ticker']} — {h['side'].upper()} x{h['count']:g}" for h in held]
    idx = st.selectbox(
        "Position",
        range(len(held)),
        format_func=lambda i: labels[i],
        key="add_stop_pos",
    )
    chosen = held[idx]

    c1, c2, c3 = st.columns(3)
    stop_cents = c1.number_input(
        f"Stop level ({chosen['side'].upper()} price, c)",
        min_value=1,
        max_value=99,
        value=50,
        key="add_stop_level",
        help="Fires when the held side's price falls to/through this level.",
    )
    trigger_ref = c2.selectbox(
        "Trigger ref", ["bid", "mid", "last"], index=0, key="add_stop_ref",
        help="Which held-side price to compare (bid = price you could sell at).",
    )
    slippage = c3.number_input(
        "Max slippage (c)", min_value=0, max_value=50, value=2, key="add_stop_slip",
        help="How far below the stop the close may fill, to ensure a marketable IOC.",
    )

    c4, c5, c6 = st.columns(3)
    use_auto = c4.checkbox("Auto count (use full position)", value=True, key="add_stop_auto")
    count_val = None
    if not use_auto:
        held_count = float(chosen["count"])
        count_val = c5.number_input(
            "Count", min_value=0.01, max_value=max(0.01, held_count),
            value=max(0.01, held_count), step=0.01, format="%.2f",
            key="add_stop_count",
            help="Contracts to close on trigger (fractional allowed).",
        )
    env = c6.selectbox("Environment", ["prod", "demo"], index=0, key="add_stop_env")
    armed = st.checkbox("Armed (will place orders on trigger)", value=True, key="add_stop_armed")

    confirm = True
    if env == "prod" and armed:
        confirm = st.checkbox(
            "I understand this will place REAL orders with REAL money on production.",
            value=False,
            key="add_stop_confirm",
        )

    if st.button("Add stop", type="primary", key="add_stop_submit"):
        if not confirm:
            st.error("Confirm the real-money acknowledgement to add an armed production stop.")
            return
        try:
            cfg = StopConfig(
                ticker=chosen["ticker"],
                held_side=chosen["side"],
                stop_cents=float(stop_cents),
                count=None if use_auto else round(float(count_val), 2),
                slippage_cents=float(slippage),
                trigger_ref=trigger_ref,
                env=env,
                armed=armed,
            )
        except ValueError as exc:
            st.error(f"Invalid stop: {exc}")
            return
        store.add(cfg)
        st.success(f"Added {chosen['side'].upper()} stop for {chosen['ticker']} at {stop_cents:.0f}c.")
        st.rerun()


def _fmt_count(value: Any) -> str:
    """Format a (possibly fractional) contract count; ``auto`` when unresolved."""
    if isinstance(value, (int, float)):
        return f"{value:g}"
    return "auto"


def _ago(ts: Any) -> str:
    if not isinstance(ts, (int, float)):
        return "—"
    secs = max(0, int(time.time() - ts))
    if secs < 60:
        return f"{secs}s ago"
    if secs < 3600:
        return f"{secs // 60}m ago"
    return datetime.datetime.fromtimestamp(ts).strftime("%H:%M")
