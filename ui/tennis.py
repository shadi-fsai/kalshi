"""Tennis match pricing page: live-score inputs, Monte Carlo, market comparison.

Pure scoring/simulation logic lives in ``kalshi.tennis``; this module only wires
the Streamlit inputs, runs the simulation, and compares the model's match-win
probability to the live Kalshi market price.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd
import streamlit as st

from kalshi.client import KalshiAPIError, KalshiClient
from kalshi.fees import FeeModel
from kalshi.kelly import (
    better_side,
    certainty_equivalent_probability,
    kelly_for_contract,
    uncertainty_adjusted_kelly_fraction,
)
from kalshi.markets import (
    live_sport_groups,
    price_cents_for_side,
    series_ticker_for_market,
    tennis_live_score,
)
from kalshi.tennis import (
    DEFAULT_ABILITY_SPREAD,
    POINT_LABELS,
    MatchParams,
    MatchState,
    baselines_from_match_odds,
    match_state_from_live,
    monte_carlo,
    params_from_baselines,
    point_count_from_label,
    point_label_from_score,
    win_prob_distribution,
)
from ui import data
from ui.settings import Settings
from ui.sizer import render_order_ticket

TENNIS_SPORT = "Tennis"

# Parameter-uncertainty sweep: how many ability scenarios to draw and how many
# matches to simulate per scenario when building the win-probability distribution
# that drives uncertainty-aware sizing. Kept modest so a run stays ~1-2s.
_UNC_SCENARIOS = 200
_UNC_SIMS_PER = 400

# Cap on how many live matches a single scan prices (each runs the full ability
# sweep, so this bounds the worst-case scan time).
_SCAN_MAX_MATCHES = 40

# Max model-vs-market disagreement (probability units) tolerated on the
# recommended side before we treat the "edge" as an orientation/staleness error
# rather than a real opportunity and refuse to prime an order.
_ORIENTATION_SANITY_GAP = 0.5


def _percentile(values: Sequence[float], pct: float) -> float:
    """Linear-interpolated percentile of ``values`` (``pct`` in 0-100)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * frac


def _pct_input(label: str, default: float, key: str) -> float:
    """Percentage number input (0-100) returned as a probability in [0, 1].

    State-driven (no ``value=``) so the odds-seeding control can prefill it via
    ``st.session_state`` before the widget is instantiated.
    """
    st.session_state.setdefault(key, default)
    val = st.number_input(label, min_value=0.0, max_value=100.0, step=1.0, key=key)
    return float(val) / 100.0


def _market_implied_prob(market: dict) -> float | None:
    """YES bid/ask midpoint implied probability for a market, or None."""
    bid = price_cents_for_side(market, "yes", "bid")
    ask = price_cents_for_side(market, "yes", "ask")
    if bid is not None and ask is not None:
        return (bid + ask) / 200.0
    if ask is not None:
        return ask / 100.0
    if bid is not None:
        return bid / 100.0
    return None


def _guess_yes_player(market: dict, p1_name: str, p2_name: str) -> int:
    """Best-effort guess of which player the YES side refers to (1 or 2)."""
    text = " ".join(
        str(market.get(k, "")) for k in ("yes_sub_title", "title", "subtitle")
    ).lower()
    if p2_name.lower() in text and p1_name.lower() not in text:
        return 2
    return 1


def _orient_yes_player(market: dict, p1_name: str, p2_name: str) -> int:
    """Which player (1/2) the market's YES side backs, derived authoritatively.

    Kalshi sets ``yes_sub_title`` to the YES outcome's name, so an exact match to
    a player name is decisive; we only fall back to the fuzzy name scan when that
    is unavailable. Getting this right matters: the model's win prob must be
    paired with the matching market side, otherwise a flipped orientation pairs a
    ~certain win prob with the *other* outcome's cheap price and fabricates a huge
    (bogus) edge.
    """
    sub = (market.get("yes_sub_title") or "").strip().lower()
    if sub:
        if sub == (p1_name or "").strip().lower():
            return 1
        if sub == (p2_name or "").strip().lower():
            return 2
    return _guess_yes_player(market, p1_name, p2_name)


def _apply_pending_ability_seed() -> None:
    """Write a queued (baseline1, baseline2) seed into the ability widget state."""
    pending = st.session_state.pop("tn_pending_seed", None)
    if not pending:
        return
    base1, base2 = pending
    spread = DEFAULT_ABILITY_SPREAD * 100.0
    for key, base in (("tn_o1", base1), ("tn_d1", base1), ("tn_o2", base2), ("tn_d2", base2)):
        pct = base * 100.0 + (spread if key.startswith("tn_o") else -spread)
        st.session_state[key] = round(min(100.0, max(0.0, pct)), 1)


def _apply_pending_score() -> None:
    """Write a queued live-score pull (from ``tennis_live_score``) into widgets.

    Runs before the score widgets are created (Streamlit forbids writing a
    widget's state after instantiation). Fills sets, current games, current
    points (game or tiebreak), and the server. Values stay editable so a
    reversed orientation can be corrected; a note records what was filled.
    """
    pending = st.session_state.pop("tn_pending_score", None)
    if not pending:
        return
    filled: list[str] = []

    sets_pair = pending.get("sets")
    if sets_pair is not None:
        st.session_state["tn_s1"] = max(0, min(2, int(sets_pair[0])))
        st.session_state["tn_s2"] = max(0, min(2, int(sets_pair[1])))
        filled.append("sets")

    games_pair = pending.get("games")
    if games_pair is not None:
        st.session_state["tn_g1"] = max(0, min(6, int(games_pair[0])))
        st.session_state["tn_g2"] = max(0, min(6, int(games_pair[1])))
        filled.append("games")

    points_pair = pending.get("points")
    if points_pair is not None:
        if pending.get("in_tiebreak"):
            st.session_state["tn_tb1"] = max(0, int(points_pair[0]))
            st.session_state["tn_tb2"] = max(0, int(points_pair[1]))
        else:
            adv = pending.get("advantage")
            st.session_state["tn_pt1"] = point_label_from_score(
                points_pair[0], adv == 1
            )
            st.session_state["tn_pt2"] = point_label_from_score(
                points_pair[1], adv == 2
            )
        filled.append("points")

    server = pending.get("server")
    if server in (1, 2):
        st.session_state["tn_server"] = server
        filled.append("server")

    if not filled:
        st.session_state["tn_score_note"] = (
            "Kalshi live data did not expose the score; enter it manually."
        )
        return
    note = "Auto-filled " + ", ".join(filled) + " from Kalshi live data"
    note += "." if pending.get("oriented") else (
        " (couldn't confirm player order from the market - verify P1/P2)."
    )
    if pending.get("winner") in (1, 2):
        note += " Kalshi reports this match as already finished."
    st.session_state["tn_score_note"] = note


def _render_odds_seeding(
    client: KalshiClient | None, auth_ok: bool, p1_name: str, p2_name: str
) -> None:
    """Seed offense/defense from the selected market's pre-game odds (+/-12)."""
    ticker = str(st.session_state.get("tn_ticker", "")).strip()
    st.markdown("##### Seed abilities from pre-game odds")
    if not ticker:
        st.caption(
            "Pick a live match above (or enter a market ticker in the compare "
            "section) to derive abilities from the market's odds."
        )
        return
    if not auth_ok or client is None:
        st.caption("Connect Kalshi credentials (.env) to read the market odds.")
        return
    try:
        market = data.fetch_market(client, ticker)
    except KalshiAPIError as exc:
        st.caption(f"Could not read odds ({exc.status_code}): {exc.message}")
        return
    prob = _market_implied_prob(market) if market else None
    if prob is None:
        st.caption("Selected market has no usable price yet.")
        return
    yes_player = _orient_yes_player(market, p1_name, p2_name)
    market_p1 = prob if yes_player == 1 else 1.0 - prob
    spread_pts = int(round(DEFAULT_ABILITY_SPREAD * 100))
    st.caption(
        f"Market implies {p1_name} ~{market_p1 * 100:.0f}% (from `{ticker}`). "
        f"Seeding inverts this to per-player baselines, then sets offense "
        f"+{spread_pts} / defense -{spread_pts} points."
    )
    if st.button(
        f"Seed offense/defense from odds (+/-{spread_pts})", key="tn_seed_odds"
    ):
        base1, base2 = baselines_from_match_odds(market_p1)
        st.session_state["tn_pending_seed"] = (base1, base2)
        st.rerun()


def render_tennis(client: KalshiClient | None, auth_ok: bool) -> None:
    """Render the tennis pricing page: inputs, simulation, and market compare."""
    st.subheader("Tennis match pricing")
    st.caption(
        "Best-of-3, ad scoring, 7-point tiebreaks. Each point's server win "
        "probability combines the server's offense with the returner's defense. "
        "Scan live matches for YES/NO edges, then open one to review and size it "
        "(or enter a score and run the simulation manually)."
    )

    # Seed widget state once so the live-match picker can prefill these.
    for key, default in (("tn_p1", "Player 1"), ("tn_p2", "Player 2"), ("tn_ticker", "")):
        st.session_state.setdefault(key, default)

    # Apply any pending odds-based ability seed and live-score pull BEFORE the
    # corresponding widgets exist (Streamlit forbids writing a widget's state
    # after it is instantiated).
    _apply_pending_ability_seed()
    _apply_pending_score()

    settings = st.session_state.get("settings") or Settings.defaults()
    _render_scanner(client, auth_ok, settings)

    # --- Players + abilities -------------------------------------------
    names = st.columns(2)
    p1_name = names[0].text_input("Player 1", key="tn_p1")
    p2_name = names[1].text_input("Player 2", key="tn_p2")

    st.markdown("##### Point-win abilities (%)")
    ab = st.columns(4)
    with ab[0]:
        o1 = _pct_input(f"{p1_name} offense", 62.0, "tn_o1")
    with ab[1]:
        d1 = _pct_input(f"{p1_name} defense", 38.0, "tn_d1")
    with ab[2]:
        o2 = _pct_input(f"{p2_name} offense", 62.0, "tn_o2")
    with ab[3]:
        d2 = _pct_input(f"{p2_name} defense", 38.0, "tn_d2")

    _render_odds_seeding(client, auth_ok, p1_name, p2_name)

    # --- Live score -----------------------------------------------------
    st.markdown("##### Live score")
    score_note = st.session_state.pop("tn_score_note", None)
    if score_note:
        st.caption(score_note)
    for key in ("tn_s1", "tn_s2", "tn_g1", "tn_g2"):
        st.session_state.setdefault(key, 0)
    sets_cols = st.columns(2)
    p1_sets = int(
        sets_cols[0].number_input(
            f"{p1_name} sets", min_value=0, max_value=2, step=1, key="tn_s1"
        )
    )
    p2_sets = int(
        sets_cols[1].number_input(
            f"{p2_name} sets", min_value=0, max_value=2, step=1, key="tn_s2"
        )
    )
    games_cols = st.columns(2)
    p1_games = int(
        games_cols[0].number_input(
            f"{p1_name} games", min_value=0, max_value=6, step=1, key="tn_g1"
        )
    )
    p2_games = int(
        games_cols[1].number_input(
            f"{p2_name} games", min_value=0, max_value=6, step=1, key="tn_g2"
        )
    )

    in_tiebreak = p1_games == 6 and p2_games == 6
    pts_cols = st.columns(2)
    if in_tiebreak:
        st.caption("At 6-6: enter current tiebreak points.")
        for key in ("tn_tb1", "tn_tb2"):
            st.session_state.setdefault(key, 0)
        p1_points = int(
            pts_cols[0].number_input(
                f"{p1_name} tiebreak pts", min_value=0, step=1, key="tn_tb1"
            )
        )
        p2_points = int(
            pts_cols[1].number_input(
                f"{p2_name} tiebreak pts", min_value=0, step=1, key="tn_tb2"
            )
        )
    else:
        for key in ("tn_pt1", "tn_pt2"):
            st.session_state.setdefault(key, POINT_LABELS[0])
        p1_label = pts_cols[0].selectbox(f"{p1_name} points", POINT_LABELS, key="tn_pt1")
        p2_label = pts_cols[1].selectbox(f"{p2_name} points", POINT_LABELS, key="tn_pt2")
        p1_points = point_count_from_label(p1_label)
        p2_points = point_count_from_label(p2_label)

    server = st.radio(
        "Currently serving",
        [1, 2],
        format_func=lambda x: p1_name if x == 1 else p2_name,
        horizontal=True,
        key="tn_server",
    )

    # --- Simulation controls -------------------------------------------
    st.markdown("##### Simulation")
    sim_cols = st.columns([2, 1, 1])
    n_sims = int(
        sim_cols[0].number_input(
            "Simulations",
            min_value=1000,
            max_value=200000,
            value=20000,
            step=1000,
            key="tn_n",
        )
    )
    seed = int(
        sim_cols[1].number_input("Seed", min_value=0, value=0, step=1, key="tn_seed")
    )
    run = sim_cols[2].button("Run simulation", type="primary", key="tn_run")

    # Ability uncertainty is set once in the scanner controls (key ``tn_unc``) and
    # reused here so the scan and the detailed sizing agree.
    ability_unc = float(st.session_state.get("tn_unc", 5.0))

    # Auto-run once after a live match is selected (set by _apply_match_selection).
    autorun = bool(st.session_state.pop("tn_autorun", False))
    if run or autorun:
        try:
            params = MatchParams(o1=o1, d1=d1, o2=o2, d2=d2)
            state = MatchState(
                p1_sets=p1_sets,
                p2_sets=p2_sets,
                p1_games=p1_games,
                p2_games=p2_games,
                p1_points=p1_points,
                p2_points=p2_points,
                server=server,
                in_tiebreak=in_tiebreak,
            )
        except ValueError as exc:
            st.error(f"Invalid inputs: {exc}")
            return
        with st.spinner(f"Simulating {n_sims:,} matches…"):
            result = monte_carlo(state, params, n_sims=n_sims, seed=seed)
        with st.spinner(
            f"Sweeping {_UNC_SCENARIOS} ability scenarios for the outcome range…"
            if ability_unc > 0
            else "Preparing sizing…"
        ):
            dist = win_prob_distribution(
                state,
                params,
                ability_sd=ability_unc / 100.0,
                n_scenarios=_UNC_SCENARIOS,
                n_sims=_UNC_SIMS_PER,
                seed=seed,
            )
        st.session_state["tn_result"] = {
            "p1_win_prob": result.p1_win_prob,
            "ci": result.ci95_half_width,
            "counts": result.set_score_counts,
            "n": result.n_sims,
            "p1_name": p1_name,
            "p2_name": p2_name,
            "p1_win_dist": dist,
            "ability_unc": ability_unc,
        }

    res = st.session_state.get("tn_result")
    if not res:
        st.info("Set the score and abilities, then run the simulation.")
        return

    _render_result(res)
    st.divider()
    _render_sizing(client, auth_ok, res, settings)


def _group_label(group: dict) -> str:
    """Human-readable label for a live match group (timing, matchup, competition)."""
    return " · ".join(
        bit
        for bit in (
            group.get("timing_label"),
            group.get("matchup"),
            group.get("competition"),
        )
        if bit
    )


def _live_tennis_groups(client: KalshiClient) -> list[dict]:
    """Return the live tennis match groups (each annotated with its milestone id).

    Raises ``KalshiAPIError`` if the scan's underlying event/taxonomy/timing
    fetches fail; the caller surfaces it.
    """
    events = data.fetch_open_events(client)
    _ordering, comp_to_sport = data.fetch_sports_taxonomy(client)
    timing_index, now = data.fetch_live_window_index(client)
    groups = live_sport_groups(
        events, comp_to_sport, timing_index, now, sport=TENNIS_SPORT, states=("live",)
    )

    def _milestone_id(group: dict) -> str | None:
        for ticker in (group["rep_ticker"], *group.get("event_tickers", ())):
            info = timing_index.get(ticker)
            if info and info.get("milestone_id"):
                return info["milestone_id"]
        return None

    return [{**g, "milestone_id": _milestone_id(g)} for g in groups]


def _resolve_match_inputs(client: KalshiClient, group: dict) -> dict:
    """Resolve a live match group into the model + pricing inputs.

    Reads the event's markets (to get player names, the P1 match-winner ticker,
    and competitor ids), the market-implied odds (inverted to ability baselines),
    and the parsed live score (oriented to P1/P2 via competitor ids). Shared by
    the per-match selection and the all-matches scan so both stay consistent. All
    calls go through cached ``data.*`` fetchers. Raises ``KalshiAPIError`` if the
    event's markets cannot be loaded; live-data failures degrade to no score.
    """
    rep = group["rep_ticker"]
    markets = data.fetch_markets_for_event_tickers(client, (rep,)).get(rep, [])
    names: list[str] = []
    ticker_for: dict[str, str] = {}
    competitor_for: dict[str, str] = {}
    for m in markets:
        sub = (m.get("yes_sub_title") or "").strip()
        if sub and sub not in ticker_for:
            ticker_for[sub] = m.get("ticker", "")
            cid = (m.get("custom_strike") or {}).get("tennis_competitor")
            if cid:
                competitor_for[sub] = cid
            names.append(sub)
    if len(names) < 2 and " vs " in group["matchup"]:
        names = [part.strip() for part in group["matchup"].split(" vs ", 1)]

    ticker = ticker_for.get(names[0], "") if names else ""
    if not ticker and markets:
        ticker = markets[0].get("ticker", "")
    # The P1 market's YES side is "P1 wins", so its implied prob is P(P1 wins).
    p1_market = next((m for m in markets if m.get("ticker") == ticker), None)
    prob = _market_implied_prob(p1_market) if p1_market else None
    baselines = baselines_from_match_odds(prob) if prob is not None else None

    parsed_score: dict | None = None
    milestone_id = group.get("milestone_id")
    if milestone_id:
        try:
            details = data.fetch_live_data(client, milestone_id)
        except KalshiAPIError:
            details = None
        if details:
            p1_cid = competitor_for.get(names[0]) if names else None
            p2_cid = competitor_for.get(names[1]) if len(names) > 1 else None
            parsed = tennis_live_score(details, p1_cid, p2_cid)
            if parsed:
                parsed_score = parsed

    return {
        "names": names,
        "ticker": ticker,
        "p1_market": p1_market,
        "competitor_for": competitor_for,
        "baselines": baselines,
        "parsed_score": parsed_score,
    }


def _apply_match_selection(client: KalshiClient, group: dict) -> None:
    """Prefill player names + comparison ticker from a chosen live match."""
    try:
        info = _resolve_match_inputs(client, group)
    except KalshiAPIError as exc:
        st.error(f"Could not load match markets ({exc.status_code}): {exc.message}")
        return

    names = info["names"]
    if len(names) >= 2:
        st.session_state["tn_p1"] = names[0]
        st.session_state["tn_p2"] = names[1]
    if info["ticker"]:
        st.session_state["tn_ticker"] = info["ticker"]
    # Let the comparison re-guess which side YES refers to for the new market.
    st.session_state.pop("tn_yes_player", None)
    if info["baselines"] is not None:
        st.session_state["tn_pending_seed"] = info["baselines"]
    if info["parsed_score"] is not None:
        st.session_state["tn_pending_score"] = info["parsed_score"]

    # Auto-run the simulation on this fresh selection.
    st.session_state["tn_autorun"] = True
    st.rerun()


def _scan_live_opportunities(
    client: KalshiClient, settings: Settings, ability_unc: float, min_edge_pts: float
) -> None:
    """Price every live tennis match and store those with a half-Kelly edge.

    For each live match: resolve inputs, seed abilities from the pre-game odds,
    pull the live score, run the MC point estimate plus the ability sweep, then
    evaluate BOTH the YES and NO sides (via :func:`_evaluate_edge`) and keep the
    better side when its edge clears ``min_edge_pts``. Results, the full match
    list (for drill-in), and scan stats are written to session state. Per-match
    failures are counted/surfaced rather than aborting the whole scan.
    """
    try:
        groups = _live_tennis_groups(client)
    except KalshiAPIError as exc:
        st.error(f"Could not scan live matches ({exc.status_code}): {exc.message}")
        return

    truncated = len(groups) > _SCAN_MAX_MATCHES
    groups = groups[:_SCAN_MAX_MATCHES]
    n_sims = int(st.session_state.get("tn_n", 20000))
    seed = int(st.session_state.get("tn_seed", 0))
    ability_sd = ability_unc / 100.0

    opportunities: list[dict] = []
    all_matches: list[dict] = []
    no_price = 0
    errors: list[str] = []

    progress = st.progress(0.0, text="Scanning live tennis…") if groups else None
    for i, group in enumerate(groups):
        label = _group_label(group)
        all_matches.append({"label": label, "group": group})
        try:
            info = _resolve_match_inputs(client, group)
        except KalshiAPIError as exc:
            errors.append(f"{group.get('matchup', label)} ({exc.status_code}): {exc.message}")
            if progress:
                progress.progress((i + 1) / len(groups), text="Scanning live tennis…")
            continue

        market = info["p1_market"]
        baselines = info["baselines"]
        market_prob = _market_implied_prob(market) if market else None
        if not market or baselines is None or market_prob is None:
            no_price += 1
            if progress:
                progress.progress((i + 1) / len(groups), text="Scanning live tennis…")
            continue

        params = params_from_baselines(*baselines)
        parsed = info["parsed_score"]
        state = match_state_from_live(parsed) if parsed else MatchState()
        result = monte_carlo(state, params, n_sims=n_sims, seed=seed)
        dist = win_prob_distribution(
            state,
            params,
            ability_sd=ability_sd,
            n_scenarios=_UNC_SCENARIOS,
            n_sims=_UNC_SIMS_PER,
            seed=seed,
        )

        fee_model: FeeModel | None = None
        series_t = series_ticker_for_market(market)
        if series_t:
            try:
                fee_model = data.fetch_fee_model(client, series_t)
            except KalshiAPIError:
                fee_model = None

        names = info["names"]
        p1_name = names[0] if names else "Player 1"
        p2_name = names[1] if len(names) > 1 else "Player 2"
        yes_player = _orient_yes_player(market, p1_name, p2_name)
        edge = _evaluate_edge(
            market, p1_name, p2_name, result.p1_win_prob, dist, yes_player,
            fee_model, settings,
        )
        # Drop implausibly large model-vs-market gaps (orientation/staleness), the
        # same sanity guard the detail view applies before priming an order.
        market_side_prob = (
            market_prob if edge and edge.side == "yes" else 1.0 - market_prob
        )
        if (
            edge is not None
            and edge.naive_edge * 100.0 >= min_edge_pts
            and abs(edge.side_mean - market_side_prob) <= _ORIENTATION_SANITY_GAP
        ):
            opportunities.append(
                {
                    "label": label,
                    "group": group,
                    "matchup": group.get("matchup", label),
                    "side": edge.side,
                    "player": edge.side_label,
                    "fair_c": edge.fair_cents,
                    "ask_c": edge.ask,
                    "edge_pts": edge.naive_edge * 100.0,
                    "used_pct": edge.used_fraction * 100.0,
                    "stake": edge.actual_stake,
                    "contracts": edge.contracts,
                    "sort": edge.used_fraction,
                }
            )
        if progress:
            progress.progress((i + 1) / len(groups), text="Scanning live tennis…")
    if progress:
        progress.empty()

    opportunities.sort(key=lambda o: o["sort"], reverse=True)
    st.session_state["tn_opportunities"] = opportunities
    st.session_state["tn_all_matches"] = all_matches
    st.session_state["tn_scan_stats"] = {
        "scanned": len(groups),
        "edges": len(opportunities),
        "no_price": no_price,
        "errors": errors,
        "truncated": truncated,
    }


def _render_scanner(
    client: KalshiClient | None, auth_ok: bool, settings: Settings
) -> None:
    """Scan all live tennis for YES/NO edges, rank them, and open one to drill in."""
    st.markdown("##### Live tennis edge scanner")
    # The detail view's simulation reuses this; keep it defined even when offline.
    st.session_state.setdefault("tn_unc", 5.0)
    if not auth_ok or client is None:
        st.caption("Connect Kalshi credentials (.env) to scan live matches.")
        return

    ctrl = st.columns([2, 1, 1])
    ability_unc = ctrl[0].slider(
        "Ability uncertainty (+/- points)",
        min_value=0.0,
        max_value=20.0,
        step=0.5,
        key="tn_unc",
        help="Resamples each offense/defense input by this standard deviation "
        f"across {_UNC_SCENARIOS} scenarios to build a win-probability range that "
        "shrinks the Kelly stake (more uncertainty -> smaller bet). Used by both "
        "the scan and the detailed view.",
    )
    min_edge = ctrl[1].number_input(
        "Min edge (pts)",
        min_value=0.0,
        max_value=50.0,
        value=0.0,
        step=0.5,
        key="tn_min_edge",
        help="Only list matches whose best side beats its breakeven by at least "
        "this many points.",
    )
    scan = ctrl[2].button("Scan live tennis for edges", type="primary", key="tn_scan")
    if scan:
        _scan_live_opportunities(client, settings, ability_unc, float(min_edge))

    stats = st.session_state.get("tn_scan_stats")
    if stats:
        bits = [
            f"Scanned {stats['scanned']} live match(es)",
            f"{stats['edges']} with an edge",
            f"{stats['no_price']} without a usable price",
        ]
        st.caption(", ".join(bits) + (" (match cap reached)." if stats["truncated"] else "."))
        for err in stats["errors"]:
            st.warning(f"Skipped {err}")

    opportunities = st.session_state.get("tn_opportunities")
    if opportunities is None:
        st.caption(
            "Scan open tennis matches and rank those with a positive YES or NO "
            "half-Kelly edge."
        )
        return

    if opportunities:
        table = pd.DataFrame(
            [
                {
                    "Match": o["matchup"],
                    "Side": f"{o['side'].upper()} ({o['player']})",
                    "Fair (c)": round(o["fair_c"]),
                    "Ask (c)": round(o["ask_c"]),
                    "Edge (pts)": round(o["edge_pts"], 1),
                    "Half-Kelly ($)": round(o["stake"], 2),
                }
                for o in opportunities
            ]
        )
        st.dataframe(table, hide_index=True, use_container_width=True)
    else:
        st.info("No live match currently shows a positive YES or NO half-Kelly edge.")

    show_all = st.checkbox(
        "Show all live matches (incl. no edge)", key="tn_show_all"
    )
    options = list(opportunities)
    if show_all:
        seen = {id(o["group"]) for o in opportunities}
        for match in st.session_state.get("tn_all_matches", []):
            if id(match["group"]) not in seen:
                options.append({"label": match["label"], "group": match["group"]})
    if not options:
        return

    def _fmt(i: int) -> str:
        opt = options[i]
        if "side" in opt:
            return (
                f"{opt['matchup']} - {opt['side'].upper()} ({opt['player']}) "
                f"+{opt['edge_pts']:.1f} pts"
            )
        return opt["label"]

    idx = st.selectbox(
        "Open a match", range(len(options)), format_func=_fmt, key="tn_open_select"
    )
    if st.button("Open match", key="tn_open"):
        _apply_match_selection(client, options[idx]["group"])


def _render_result(res: dict) -> None:
    """Show the model win probabilities and the set-score distribution."""
    p1_name, p2_name = res["p1_name"], res["p2_name"]
    p1 = res["p1_win_prob"]
    ci = res["ci"]
    st.markdown("##### Model output")
    metrics = st.columns(2)
    metrics[0].metric(
        f"{p1_name} win prob", f"{p1 * 100:.1f}%", delta=f"+/- {ci * 100:.1f}%"
    )
    metrics[1].metric(f"{p2_name} win prob", f"{(1 - p1) * 100:.1f}%")
    st.caption(f"Based on {res['n']:,} simulated matches (95% CI).")

    dist = res.get("p1_win_dist") or []
    ability_unc = res.get("ability_unc", 0.0)
    if ability_unc > 0 and len(dist) > 1:
        lo, hi = _percentile(dist, 5.0), _percentile(dist, 95.0)
        sd = statistics.pstdev(dist)
        st.caption(
            f"{p1_name} win-prob range across {len(dist)} ability scenarios "
            f"(+/-{ability_unc:.1f} pts): {lo * 100:.0f}-{hi * 100:.0f}% "
            f"(5-95%), SD +/-{sd * 100:.1f} pts. Wider range -> smaller Kelly stake."
        )
        edges = [i / 100.0 for i in range(0, 101, 5)]
        hist = [0] * (len(edges) - 1)
        for p in dist:
            b = min(int(p * 20), len(hist) - 1)
            hist[b] += 1
        hist_df = pd.DataFrame(
            {
                "Win prob %": [f"{int(edges[i] * 100)}-{int(edges[i + 1] * 100)}" for i in range(len(hist))],
                "Scenarios": hist,
            }
        ).set_index("Win prob %")
        st.bar_chart(hist_df)

    counts: dict[str, int] = res["counts"]
    n = max(res["n"], 1)
    order = ["2-0", "2-1", "1-2", "0-2"]
    rows = [
        {"Set score (P1-P2)": k, "Probability %": counts.get(k, 0) / n * 100.0}
        for k in order
        if k in counts
    ]
    # Include any unexpected scores (e.g. already-decided states) too.
    for k, v in counts.items():
        if k not in order:
            rows.append({"Set score (P1-P2)": k, "Probability %": v / n * 100.0})
    if rows:
        df = pd.DataFrame(rows).set_index("Set score (P1-P2)")
        st.bar_chart(df)


def _side_kelly(
    side: str,
    market: dict,
    model_prob: float,
    fee_model: FeeModel | None,
    settings: Settings,
    model_dist: Sequence[float] | None = None,
):
    """Kelly result for one side, or None when the ask isn't tradeable (1-99c).

    When ``model_dist`` (the win-probability range for this side from the ability
    sweep) is given, the stake is shrunk for that uncertainty: the variance-aware
    Kelly fraction is converted to a certainty-equivalent probability that is fed
    through :func:`kelly_for_contract`, so all downstream numbers stay consistent.
    """
    ask = price_cents_for_side(market, side, "ask")
    if ask is None or not 1 <= ask <= 99:
        return None, ask
    fee_buy = (
        fee_model.per_contract_fee(ask / 100.0)
        if fee_model
        else settings.fallback_fee
    )
    est_prob = model_prob
    if model_dist:
        f_adj = uncertainty_adjusted_kelly_fraction(
            model_dist, price_cents=float(ask), fee_buy=fee_buy, fee_sell=0.0
        )
        est_prob = certainty_equivalent_probability(
            f_adj, price_cents=float(ask), fee_buy=fee_buy, fee_sell=0.0
        )
    result = kelly_for_contract(
        side=side,
        price_cents=float(ask),
        estimated_probability=est_prob,
        bankroll=settings.bankroll,
        kelly_multiplier=settings.kelly_multiplier,
        fee_buy=fee_buy,
        fee_sell=0.0,  # priced to hold to settlement
    )
    return result, ask


@dataclass
class EdgeResult:
    """The best uncertainty-adjusted half-Kelly bet for one match, if any."""

    side: str  # "yes" or "no"
    side_label: str  # player name the side backs
    ask: float  # market ask (cents)
    fair_cents: float  # MC mean fair price (cents)
    naive_edge: float  # edge vs breakeven on the mean (probability units)
    side_mean: float  # mean model win prob for the side
    range_lo: float  # 5th percentile of the side's win-prob range
    range_hi: float  # 95th percentile
    range_std: float  # standard deviation of the range
    naive_full: float  # full Kelly on the mean
    adj_full: float  # uncertainty-adjusted full Kelly
    used_fraction: float  # adjusted fraction x multiplier
    kelly_multiplier: float
    contracts: int
    actual_stake: float
    shrink: float  # 1 - adj_full / naive_full


def _evaluate_edge(
    market: dict,
    p1_name: str,
    p2_name: str,
    model_p1: float,
    dist_p1: Sequence[float],
    yes_player: int,
    fee_model: FeeModel | None,
    settings: Settings,
) -> EdgeResult | None:
    """Pick the better side and size it on the win-prob range; None if no edge.

    Pure (no ``st.*``): given a market, the model's P1 win prob and its range
    (``dist_p1``), and which player YES tracks, this runs the uncertainty-adjusted
    half-Kelly on both sides, returns the better one, and also computes the
    point-estimate ("naive") numbers so callers can show the uncertainty shrink.
    """
    model_yes = model_p1 if yes_player == 1 else 1.0 - model_p1
    model_no = 1.0 - model_yes
    yes_dist = [p if yes_player == 1 else 1.0 - p for p in dist_p1] or [model_yes]
    no_dist = [1.0 - p for p in yes_dist]

    yes_res, yes_ask = _side_kelly(
        "yes", market, model_yes, fee_model, settings, model_dist=yes_dist
    )
    no_res, no_ask = _side_kelly(
        "no", market, model_no, fee_model, settings, model_dist=no_dist
    )

    candidates = [r for r in (yes_res, no_res) if r is not None]
    pick = better_side(*candidates) if len(candidates) == 2 else (
        candidates[0] if candidates and candidates[0].has_edge else None
    )
    if pick is None:
        return None

    side = pick.side
    side_label = (p1_name if yes_player == 1 else p2_name) if side == "yes" else (
        p2_name if yes_player == 1 else p1_name
    )
    ask = yes_ask if side == "yes" else no_ask
    side_dist = yes_dist if side == "yes" else no_dist
    side_mean = model_yes if side == "yes" else model_no
    naive_pick, _ = _side_kelly(side, market, side_mean, fee_model, settings)
    naive_full = naive_pick.full_kelly_fraction if naive_pick else pick.full_kelly_fraction
    shrink = 1.0 - (pick.full_kelly_fraction / naive_full) if naive_full > 0 else 0.0

    return EdgeResult(
        side=side,
        side_label=side_label,
        ask=float(ask),
        fair_cents=side_mean * 100.0,
        naive_edge=naive_pick.edge if naive_pick else pick.edge,
        side_mean=side_mean,
        range_lo=_percentile(side_dist, 5.0),
        range_hi=_percentile(side_dist, 95.0),
        range_std=statistics.pstdev(side_dist) if len(side_dist) > 1 else 0.0,
        naive_full=naive_full,
        adj_full=pick.full_kelly_fraction,
        used_fraction=pick.used_fraction,
        kelly_multiplier=pick.kelly_multiplier,
        contracts=pick.contracts,
        actual_stake=pick.actual_stake,
        shrink=max(0.0, shrink),
    )


def _render_sizing(
    client: KalshiClient | None, auth_ok: bool, res: dict, settings: Settings
) -> None:
    """Compare to Kalshi, then half-Kelly size the best-edge side off the MC price."""
    p1_name, p2_name = res["p1_name"], res["p2_name"]
    st.markdown("##### Price & size vs Kalshi")
    ticker = st.text_input(
        "Kalshi market ticker",
        placeholder="e.g. KXATPMATCH-...-PLR",
        key="tn_ticker",
        help="Pick a live match above to auto-fill this, or paste a ticker.",
    ).strip()
    if not ticker:
        st.caption("Enter a match-winner market ticker to price and size against the model.")
        return
    if not auth_ok or client is None:
        st.info("Connect your Kalshi credentials (.env) to fetch live market prices.")
        return

    try:
        market = data.fetch_market(client, ticker)
    except KalshiAPIError as exc:
        st.error(f"Could not load market ({exc.status_code}): {exc.message}")
        return
    if not market:
        st.warning("No market found for that ticker.")
        return

    market_prob = _market_implied_prob(market)
    if market_prob is None:
        st.warning("That market has no usable price yet.")
        return

    # Reset the YES-orientation default whenever the market changes so a stale
    # choice from a previous match can't invert the new market's pricing.
    if st.session_state.get("tn_yes_player_ticker") != ticker:
        st.session_state.pop("tn_yes_player", None)
        st.session_state["tn_yes_player_ticker"] = ticker
    guess = _orient_yes_player(market, p1_name, p2_name)
    yes_player = st.radio(
        "YES on this market means a win for",
        [1, 2],
        format_func=lambda x: p1_name if x == 1 else p2_name,
        index=0 if guess == 1 else 1,
        horizontal=True,
        key="tn_yes_player",
    )
    model_p1 = res["p1_win_prob"]
    # Win-probability range for P1 from the ability sweep (point estimate if off).
    dist_p1 = res.get("p1_win_dist") or [model_p1]
    ability_unc = res.get("ability_unc", 0.0)
    market_p1 = market_prob if yes_player == 1 else 1.0 - market_prob

    cols = st.columns(3)
    cols[0].metric(f"Model {p1_name}", f"{model_p1 * 100:.1f}%")
    cols[1].metric(f"Market {p1_name}", f"{market_p1 * 100:.1f}%")
    cols[2].metric("Edge (model - market)", f"{(model_p1 - market_p1) * 100:+.1f}%")

    # Fee model from this market's Kalshi series (same source as the main sizer).
    fee_model: FeeModel | None = None
    series_t = series_ticker_for_market(market)
    if series_t:
        try:
            fee_model = data.fetch_fee_model(client, series_t)
        except KalshiAPIError as exc:
            st.caption(
                f"Couldn't load the fee model for `{series_t}` ({exc.status_code}); "
                "using the fallback fee."
            )

    if settings.bankroll <= 0:
        st.warning("Set a positive bankroll in the sidebar to size the bet.")

    edge = _evaluate_edge(
        market, p1_name, p2_name, model_p1, dist_p1, yes_player, fee_model, settings
    )

    st.markdown("##### Half-Kelly sizing (MC-priced)")
    if ability_unc > 0:
        st.caption(
            f"Fair value from the Monte Carlo, sized at {settings.kelly_multiplier:.2f}x "
            f"Kelly on the side with positive edge. The bet is shrunk for model "
            f"uncertainty: each ability was resampled +/-{ability_unc:.1f} pts across "
            f"{len(dist_p1)} scenarios, and the Kelly fraction is reduced by the "
            "variance of the resulting win-probability range (mean-variance Kelly). "
            "Fees from the market's series model; bankroll and multiplier from the sidebar."
        )
    else:
        st.caption(
            f"Fair value from the Monte Carlo, sized at {settings.kelly_multiplier:.2f}x "
            "Kelly on the side with positive edge (ability uncertainty off, so sized "
            "off the point estimate). Fees from the market's series model; bankroll "
            "and multiplier from the sidebar."
        )
    if edge is None:
        st.info(
            "No positive edge on either side at the current asks once fees and model "
            "uncertainty are included, so half-Kelly recommends **no bet**. The model "
            "and market are close, or the spread/fees/uncertainty eat the edge."
        )
        return

    top = st.columns(4)
    top[0].metric("Recommended side", f"{edge.side.upper()} ({edge.side_label})")
    top[1].metric("MC fair price", f"{edge.fair_cents:.0f}c")
    top[2].metric("Market ask", f"{edge.ask:.0f}c")
    top[3].metric("Edge vs breakeven", f"{edge.naive_edge * 100:+.1f} pts")

    rng = st.columns(3)
    rng[0].metric("Win prob (mean)", f"{edge.side_mean * 100:.1f}%")
    rng[1].metric(
        "Range 5-95% / SD",
        f"{edge.range_lo * 100:.0f}-{edge.range_hi * 100:.0f}% +/-{edge.range_std * 100:.1f}",
    )
    rng[2].metric("Uncertainty shrink", f"-{edge.shrink * 100:.0f}%")

    rec = st.columns(4)
    rec[0].metric("Full Kelly (mean)", f"{edge.naive_full * 100:.2f}%")
    rec[1].metric(
        f"Used ({edge.kelly_multiplier:.2f}x, adj.)",
        f"{edge.used_fraction * 100:.2f}%",
    )
    rec[2].metric("Contracts", f"{edge.contracts:,}")
    rec[3].metric("Actual stake", f"${edge.actual_stake:,.2f}")
    if edge.contracts == 0:
        st.warning("Bankroll is too small to buy a single contract at this price.")

    # Sanity guard: in a live match-winner, model and market should broadly agree
    # on who is winning. A huge gap on the recommended side almost always means
    # the YES/NO orientation is wrong (so a ~certain win prob is being paired with
    # the cheap *other* outcome) or the market is stale. Refuse to prime an order.
    market_side_prob = market_prob if edge.side == "yes" else 1.0 - market_prob
    if abs(edge.side_mean - market_side_prob) > _ORIENTATION_SANITY_GAP:
        st.error(
            f"Model ({edge.side_mean * 100:.0f}%) and market "
            f"({market_side_prob * 100:.0f}%) disagree by "
            f"{abs(edge.side_mean - market_side_prob) * 100:.0f} points on the "
            f"recommended {edge.side.upper()} side. That gap is too large to be a "
            "real live edge - the YES/NO orientation is probably wrong (check the "
            "\"YES on this market means a win for\" selection) or the market is "
            "stale. Not priming an order."
        )
        return

    st.divider()
    render_order_ticket(
        client,
        market,
        fee_model=fee_model,
        fallback_fee=settings.fallback_fee,
        action="buy",
        side=edge.side,
        count=edge.contracts,
        price_cents=edge.ask,
    )
