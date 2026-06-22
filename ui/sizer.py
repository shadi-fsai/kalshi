"""The bet sizer: market metrics, Kelly + Sharpe sizing, and the order ticket.

``render_sizer`` is the full per-market flow (formerly the back half of
``app.py``): it reads the shared :class:`~ui.settings.Settings`, pulls fees and
realized volatility, runs the fee-adjusted Kelly sizing and Sharpe metrics, and
hands off to ``render_order_ticket`` for the two-step confirm. It is reused by
both the Find and Watch pages.
"""

from __future__ import annotations

import datetime
import math
import uuid
from typing import Any

import streamlit as st

from kalshi.client import KalshiAPIError, KalshiClient
from kalshi.fees import FeeModel
from kalshi.kelly import kelly_for_contract
from kalshi.markets import (
    market_label,
    parse_ts,
    price_cents_for_side,
    resolution_time,
    series_ticker_for_market,
)
from kalshi.orders import to_book_order
from kalshi.risk import (
    realized_volatility,
    sharpe_metrics,
    sigma_remaining,
    volatility_time_multiplier,
)
from ui import data
from ui.settings import Settings


def render_order_ticket(
    _client: KalshiClient,
    market: dict[str, Any],
    *,
    fee_model: FeeModel | None,
    fallback_fee: float,
    action: str = "buy",
    side: str = "yes",
    count: int = 1,
    price_cents: float = 50.0,
) -> None:
    """Render a limit-order ticket from the Kelly sizer output (no re-entry).

    Side, contract count, and price are taken directly from the sizing above and
    shown read-only; the user only prepares and confirms the order.
    """
    ticker = market.get("ticker", "")
    count = int(count)
    price_c = float(price_cents)
    st.markdown("#### Place a limit order")

    if count < 1:
        st.caption(
            "No contracts are recommended at this price and probability, so "
            "there's nothing to order. Adjust the sizing inputs above to size a "
            "position."
        )
        return

    st.caption(
        "Real order on the connected Kalshi environment. Side, contracts, and "
        "price are taken from the sizing above; review and confirm to submit."
    )

    book = to_book_order(action, side, price_c)
    # Actual fee for THIS count/price, from the Kalshi series fee model.
    price_dollars = price_c / 100.0
    model_fee = fee_model.order_fee(count, price_dollars) if fee_model else None
    if model_fee is not None:
        fee_amt = model_fee
        fee_note = (
            f"{fee_model.fee_type} fee model" if fee_model else "fee model"
        )
    else:
        fee_amt = round(count * fallback_fee, 2)
        fee_note = "manual fallback fee"

    # Read-only summary of the order pulled straight from the sizer.
    summary = st.columns(4)
    summary[0].metric("Action", action.upper())
    summary[1].metric("Side", side.upper())
    summary[2].metric("Contracts", f"{count:,}")
    summary[3].metric("Limit price", f"{price_c:.0f}c")

    if action == "buy":
        est_cost = count * price_dollars + fee_amt
        st.caption(
            f"Maps to YES-book **{book.book_side.upper()}** @ "
            f"{book.yes_price_cents:.0f}c. Est. cost **${est_cost:,.2f}** "
            f"= {count} x {price_c:.0f}c + **${fee_amt:,.2f}** fee ({fee_note})."
        )
    else:
        proceeds = count * price_dollars - fee_amt
        st.caption(
            f"Maps to YES-book **{book.book_side.upper()}** @ "
            f"{book.yes_price_cents:.0f}c. Est. proceeds **${proceeds:,.2f}** "
            f"= {count} x {price_c:.0f}c - **${fee_amt:,.2f}** fee ({fee_note})."
        )

    # Two-step confirm: stash the prepared order, then require a second click.
    ticket_key = (ticker, action, side, count, price_c)
    if st.button("Prepare order", type="secondary", key="ot_prepare"):
        st.session_state["pending_order"] = ticket_key

    pending = st.session_state.get("pending_order")
    if pending == ticket_key:
        st.warning(
            f"Confirm: **{action.upper()} {count} {side.upper()}** on "
            f"`{ticker}` at {price_c:.0f}c "
            f"(YES-book {book.book_side} @ {book.yes_price_cents:.0f}c)."
        )
        cc = st.columns(2)
        if cc[0].button("Confirm & submit", type="primary", key="ot_confirm"):
            try:
                resp = _client.create_order(
                    ticker=ticker,
                    book_side=book.book_side,
                    count=int(count),
                    price_dollars=book.yes_price_dollars,
                    client_order_id=str(uuid.uuid4()),
                )
                order = resp.get("order", resp)
                oid = order.get("order_id") or order.get("id") or "(unknown)"
                status = order.get("status", "submitted")
                st.success(f"Order placed: id `{oid}`, status **{status}**.")
                st.session_state.pop("pending_order", None)
            except KalshiAPIError as exc:
                st.error(f"Order rejected ({exc.status_code}): {exc.message}")
        if cc[1].button("Cancel", key="ot_abort"):
            st.session_state.pop("pending_order", None)


def render_sizer(
    client: KalshiClient,
    selected_market: dict[str, Any],
    settings: Settings,
    *,
    selected_game_start: datetime.datetime | None = None,
    favored_side: str | None = None,
) -> None:
    """Render the full sizing + Sharpe + order-ticket flow for one market."""
    if not selected_market:
        return

    st.divider()
    if st.button(
        "Refresh game data",
        help="Reload live prices, status, fees, and price history for the "
        "currently selected game from Kalshi (clears the cached data).",
    ):
        # Drop the cached market/live/fee/candlestick data so the rerun refetches
        # fresh values for the selected game.
        data.fetch_markets_for_event_tickers.clear()
        data.fetch_live_markets.clear()
        data.fetch_live_data.clear()
        data.fetch_fee_model.clear()
        data.fetch_mid_prices.clear()
        st.rerun()
    st.subheader(market_label(selected_market))

    info_cols = st.columns(4)
    info_cols[0].metric("Status", selected_market.get("status", "?"))
    yes_ask = price_cents_for_side(selected_market, "yes", "ask")
    no_ask = price_cents_for_side(selected_market, "no", "ask")
    info_cols[1].metric("YES ask", f"{yes_ask:.0f}c" if yes_ask else "—")
    info_cols[2].metric("NO ask", f"{no_ask:.0f}c" if no_ask else "—")
    # "last_price_dollars" / legacy "last_price" via the same helper.
    last = price_cents_for_side(selected_market, "last", "price")
    info_cols[3].metric("Last", f"{last:.0f}c" if last else "—")

    side_options = ["yes", "no"]
    # When a favorite market is picked, preselect its favored side. A plain
    # radio keeps its prior selection across reruns and ignores `index`, so we
    # drive it through session_state and only override when the loaded market
    # actually changes — manual flips are preserved until the next new market.
    if favored_side in side_options:
        market_ticker = selected_market.get("ticker")
        if st.session_state.get("_side_for_market") != market_ticker:
            st.session_state["side_to_buy"] = favored_side
            st.session_state["_side_for_market"] = market_ticker
    side = st.radio(
        "Side to buy",
        side_options,
        key="side_to_buy",
        horizontal=True,
        format_func=str.upper,
    )
    default_ask = price_cents_for_side(selected_market, side, "ask")

    # Tradeable prices are 1-99c; clamp the default so a fully-priced ask (100c)
    # doesn't exceed the input bounds.
    if default_ask:
        default_price = min(99.0, max(1.0, float(default_ask)))
        if default_ask >= 100:
            st.caption(
                f"The {side.upper()} ask is 100c (fully priced) — there is no profit "
                "to size here. Using 99c; override below to model a different fill."
            )
    else:
        default_price = 50.0

    price_cents = st.number_input(
        f"Price to buy {side.upper()} (cents)",
        min_value=1.0,
        max_value=99.0,
        value=default_price,
        step=1.0,
        help="Defaults to the current ask. Override to model a different fill.",
    )

    implied = price_cents / 100.0
    est_pct = st.slider(
        f"Your estimated probability that {side.upper()} wins (%)",
        min_value=0.0,
        max_value=100.0,
        value=round(implied * 100.0, 1),
        step=0.5,
        help="Defaults to the market's implied probability (the price). "
        "Move it to express your edge.",
    )
    est_prob = est_pct / 100.0

    hold_to_expiration = st.checkbox(
        "I plan to hold until expiration",
        value=True,
        help="On by default: you hold this position until the market settles, so "
        "there is no exit trade and no sell fee — only the buy fee is modeled "
        "into the breakeven, edge, fees, sizing, and Sharpe. Uncheck if you plan "
        "to sell before expiration; the sell fee is then added round-trip into "
        "all of those calculations.",
    )

    # Pull the real fee from the market's Kalshi series fee model. The marginal
    # per-contract fee (quadratic in price) feeds Kelly; we assume the exit
    # trades near the same price for the round-trip estimate.
    fee_model: FeeModel | None = None
    series_t = series_ticker_for_market(selected_market)
    if series_t:
        try:
            fee_model = data.fetch_fee_model(client, series_t)
        except KalshiAPIError as exc:
            st.warning(
                f"Couldn't load the fee model for `{series_t}` "
                f"({exc.status_code}): {exc.message}. Using the fallback fee."
            )
    per_contract = (
        fee_model.per_contract_fee(price_cents / 100.0) if fee_model else None
    )
    if per_contract is not None:
        fee_buy = per_contract
        fee_source = (
            f"Kalshi {fee_model.fee_type} fee model "
            f"(x{fee_model.fee_multiplier:g})"
        )
    else:
        fee_buy = settings.fallback_fee
        fee_source = "manual fallback fee"
    # Holding to expiration means the contract settles with no exit trade, so no
    # sell fee. When selling before expiration, the sell fee (assumed to trade
    # near the same price) is modeled round-trip into breakeven, edge, and sizing.
    fee_sell = 0.0 if hold_to_expiration else fee_buy

    # --- Volatility / time-to-expiration (Sharpe-aware shrink) -----------
    # Edge and breakeven mirror kelly_for_contract so we can size the vol shrink
    # before calling it (the result is composed into the Kelly multiplier).
    cost = price_cents / 100.0
    breakeven = min(1.0, cost + fee_buy + fee_sell)
    net_edge = est_prob - breakeven

    now_ts = datetime.datetime.now(datetime.timezone.utc)
    # Use this market's own expected settle time. The official close_time is
    # deliberately NOT used as a fallback: for tournament markets it's the
    # series-wide close (e.g. the final's date), which would inflate the horizon
    # to weeks and silently corrupt the Sharpe/volatility-time scaling.
    expected_expiry_dt = resolution_time(selected_market)

    # Let the user override the settlement time used for the time-to-expiry
    # horizon. This drives the annualized Sharpe and the volatility-time sizing,
    # so a more accurate end time (e.g. when Kalshi's expected_expiration_time is
    # missing, coarse, or the game is running long) yields a more accurate Sharpe.
    local_tz = datetime.datetime.now().astimezone().tzinfo
    override_end = st.checkbox(
        "Override expected end time (for a more accurate Sharpe)",
        value=False,
        help="By default the horizon uses Kalshi's expected expiration time for "
        "this market. Check to set your own expected settlement time — the "
        "time-to-expiry drives the annualized Sharpe and the volatility-time "
        "sizing factor.",
    )
    expiry_dt = expected_expiry_dt
    if override_end:
        # Default to the expected expiration expressed as minutes from now, so
        # the box loads with Kalshi's horizon and you just tweak the number.
        if expected_expiry_dt is not None:
            default_minutes = round(
                (expected_expiry_dt - now_ts).total_seconds() / 60.0
            )
        else:
            default_minutes = 60
        override_minutes = st.number_input(
            "Minutes until settlement",
            min_value=1,
            value=max(1, int(default_minutes)),
            step=1,
            help="How many minutes from now you expect this market to settle. "
            "Drives the time-to-expiry behind the annualized Sharpe and the "
            "volatility-time sizing factor.",
        )
        expiry_dt = now_ts + datetime.timedelta(minutes=int(override_minutes))
        if expected_expiry_dt is not None:
            st.caption(
                "Expected expiration (Kalshi): "
                f"{expected_expiry_dt.astimezone(local_tz).strftime('%b %d %H:%M %Z')}"
                f" (~{int(default_minutes)} min) — overridden above."
            )
        else:
            st.caption(
                "Kalshi has no expected expiration for this market; using your "
                "override."
            )

    # Keep the remaining distance at minute (really sub-second) resolution so
    # intraday markets are annualized from their true time-to-expiry, not a
    # rounded number of days. t_days is derived from the same precise value.
    t_minutes = (
        (expiry_dt - now_ts).total_seconds() / 60.0 if expiry_dt else None
    )
    t_days = t_minutes / 1440.0 if t_minutes is not None else None

    vol_per_day: float | None = None
    sig_remaining: float | None = None
    vol_mult = 1.0
    vol_reason = ""
    if not settings.vol_adjust:
        vol_reason = "Volatility/time adjustment is off (sidebar)."
    elif net_edge <= 0:
        vol_reason = "No positive edge, so the volatility shrink doesn't apply."
    elif t_days is None or t_days <= 0:
        vol_reason = (
            "No expiration time on this market, so remaining volatility can't "
            "be scaled — sizing uses Kelly only."
        )
    else:
        ticker_t = selected_market.get("ticker")
        # Bound the lookback window: from kickoff (if known) but never more than
        # 48h back, so the candlestick payload stays small and recent.
        floor_dt = now_ts - datetime.timedelta(hours=48)
        start_dt = selected_game_start or parse_ts(
            selected_market.get("open_time")
        )
        if start_dt is None or start_dt < floor_dt:
            start_dt = floor_dt
        start_ts = int(start_dt.timestamp())
        end_ts = int(now_ts.timestamp())
        window_minutes = (end_ts - start_ts) / 60.0
        period = 1 if window_minutes <= 180 else 60
        if end_ts <= start_ts:
            # The price window starts at/after "now" — e.g. the game hasn't
            # started yet, so there's no trading history to measure. Skip the
            # candlestick fetch (Kalshi rejects end_ts <= start_ts with a 400).
            vol_reason = (
                "This market hasn't started trading in the lookback window yet "
                "(pre-game), so there's no price history to estimate volatility "
                "— sizing uses Kelly only."
            )
        elif not series_t or not ticker_t:
            vol_reason = "Missing series/market ticker; can't fetch price history."
        else:
            try:
                prices = data.fetch_mid_prices(
                    client, series_t, ticker_t, start_ts, end_ts, period
                )
            except KalshiAPIError as exc:
                prices = []
                vol_reason = (
                    f"Couldn't load price history ({exc.status_code}): "
                    f"{exc.message}. Sizing uses Kelly only."
                )
            vol_per_day = realized_volatility(prices, float(period))
            if vol_per_day is None:
                if not vol_reason:
                    vol_reason = (
                        "Not enough price history yet (pre-game or illiquid) to "
                        "estimate volatility — sizing uses Kelly only."
                    )
            else:
                sig_remaining = sigma_remaining(vol_per_day, t_days, est_prob)
                vol_mult = volatility_time_multiplier(
                    edge=net_edge,
                    sigma_remaining=sig_remaining,
                    sensitivity=settings.vol_sensitivity,
                )

    effective_multiplier = settings.kelly_multiplier * vol_mult

    if settings.bankroll <= 0:
        st.warning("Set a positive bankroll in the sidebar to size the bet.")

    result = kelly_for_contract(
        side=side,
        price_cents=price_cents,
        estimated_probability=est_prob,
        bankroll=settings.bankroll,
        kelly_multiplier=effective_multiplier,
        fee_buy=fee_buy,
        fee_sell=fee_sell,
    )

    sm = sharpe_metrics(
        edge=net_edge,
        win_prob=est_prob,
        time_to_expiry_minutes=t_minutes if t_minutes and t_minutes > 0 else 0.0,
    )

    st.divider()
    out = st.columns(5)
    out[0].metric("Implied prob", f"{result.implied_probability * 100:.1f}%")
    fee_scope = "buy-only (hold to expiry)" if hold_to_expiration else "round-trip"
    out[1].metric(
        "Breakeven (incl. fees)",
        f"{result.breakeven_probability * 100:.1f}%",
        help=f"Price plus {fee_scope} fees "
        f"({result.fee_per_contract * 100:.0f}c/contract).",
    )
    out[2].metric(
        "Your prob",
        f"{result.estimated_probability * 100:.1f}%",
        delta=f"{result.edge * 100:+.1f} pts edge",
    )
    out[3].metric("Full Kelly", f"{result.full_kelly_fraction * 100:.2f}%")
    out[4].metric(
        f"Used ({result.kelly_multiplier:.2f}x)",
        f"{result.used_fraction * 100:.2f}%",
        help="Effective fraction = your Kelly multiplier x the volatility/time "
        "factor below.",
    )

    # Sharpe-aware view: per-bet and time-annualized Sharpe (selection), plus the
    # realized-volatility shrink applied to sizing.
    def _fmt_sharpe(value: float, decimals: int) -> str:
        # At a 0%/100% estimate the settlement variance is zero, so the Sharpe
        # (edge / sigma) diverges. Show the signed infinity rather than a
        # misleading finite number.
        if math.isinf(value):
            return "∞" if value > 0 else "−∞"
        return f"{value:.{decimals}f}"

    sh = st.columns(5)
    sh[0].metric(
        "Per-bet Sharpe",
        _fmt_sharpe(sm.sharpe_terminal, 2),
        help="edge / sqrt(q(1-q)) for a hold-to-expiry binary. Independent of "
        "stake; use it to compare bet quality. ∞ means you set a 0%/100% "
        "estimate (zero modeled variance), so the ratio is off the chart.",
    )
    sh[1].metric(
        "Annualized Sharpe",
        _fmt_sharpe(sm.sharpe_annualized, 1)
        if sm.time_to_expiry_minutes > 0
        else "—",
        help="Per-bet Sharpe scaled by sqrt(525600 / minutes-to-expiry): the "
        "horizon is measured to the minute, so a market settling in ~45 min is "
        "annualized from its true remaining distance, not a rounded day count.",
    )
    sh[2].metric(
        "Edge / day",
        f"{sm.edge_per_day * 100:+.2f} pts" if sm.time_to_expiry_days > 0 else "—",
    )
    sh[3].metric(
        "Realized vol",
        f"{vol_per_day * 100:.1f} pts/d" if vol_per_day is not None else "—",
        help="Std. dev. of price changes since the game started (per day), from "
        "Kalshi candlesticks.",
    )
    sh[4].metric(
        "Vol/time factor",
        f"{vol_mult:.2f}x",
        help="Shrink applied on top of your Kelly multiplier. 1.00x = no shrink.",
    )
    if vol_reason:
        st.caption(vol_reason)
    elif vol_per_day is not None:
        if sm.time_to_expiry_days >= 1:
            time_txt = f"{sm.time_to_expiry_days:.2f} days"
        elif sm.time_to_expiry_minutes >= 60:
            time_txt = f"{sm.time_to_expiry_minutes / 60.0:.1f} hours"
        else:
            time_txt = f"{sm.time_to_expiry_minutes:.0f} min"
        st.caption(
            f"Remaining vol over {time_txt} to expiry: "
            f"{(sig_remaining or 0.0) * 100:.1f} pts vs edge "
            f"{net_edge * 100:+.1f} pts -> {vol_mult:.2f}x shrink "
            f"(sensitivity {settings.vol_sensitivity:g}). Effective sizing "
            f"{settings.kelly_multiplier:.2f}x Kelly x {vol_mult:.2f}x = "
            f"{effective_multiplier:.2f}x."
        )

    # Flag the case where fees flip an otherwise-positive raw edge to no-bet.
    raw_edge = est_prob - result.implied_probability
    if not result.has_edge:
        if raw_edge > 0:
            st.info(
                f"Positive raw edge ({raw_edge * 100:+.1f} pts vs price), but "
                f"{fee_scope} fees raise breakeven to "
                f"{result.breakeven_probability * 100:.1f}% — so Kelly recommends "
                "**no bet**. Lower the fees or raise your estimate above breakeven."
            )
        else:
            st.info(
                "No positive edge at this price and probability, so Kelly recommends "
                "**no bet** on this side. Increase your estimated probability above "
                f"the fee-adjusted breakeven {result.breakeven_probability * 100:.1f}% "
                "to size a bet."
            )
    else:
        # Actual total buy fee for the recommended count, from the API model.
        if fee_model is not None and fee_model.order_fee(
            result.contracts, price_cents / 100.0
        ) is not None:
            total_buy_fee = fee_model.order_fee(result.contracts, price_cents / 100.0)
        else:
            total_buy_fee = round(result.contracts * settings.fallback_fee, 2)

        rec = st.columns(4)
        rec[0].metric("Recommended stake", f"${result.recommended_stake:,.2f}")
        rec[1].metric("Contracts", f"{result.contracts:,}")
        rec[2].metric("Actual stake", f"${result.actual_stake:,.2f}")
        rec[3].metric("Buy fee (total)", f"${total_buy_fee:,.2f}")
        if result.contracts == 0:
            st.warning(
                "Bankroll is too small to buy a single contract at this price."
            )
        if hold_to_expiration:
            fee_note = (
                "Held to expiration, so no sell fee is modeled — only the buy "
                "fee affects the Kelly breakeven."
            )
        else:
            fee_note = (
                "Selling before expiration, so round-trip (buy + sell) fees are "
                "modeled into the Kelly breakeven and sizing."
            )
        st.caption(
            f"Buy {result.contracts:,} {side.upper()} contracts at "
            f"{price_cents:.0f}c, est. buy fee **${total_buy_fee:,.2f}** "
            f"({fee_source}); max payout ${result.contracts:,.0f} if it resolves "
            f"{side.upper()}. {fee_note}"
        )

    st.divider()
    render_order_ticket(
        client,
        selected_market,
        fee_model=fee_model,
        fallback_fee=settings.fallback_fee,
        action="buy",
        side=side,
        count=result.contracts,
        price_cents=price_cents,
    )
