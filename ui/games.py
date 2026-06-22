"""Game discovery: the browse/filter/favorites flow and manual ticker lookup.

``render_find_games`` returns the selected market (plus its favored side and
kickoff time) so the page can hand it to the sizer. ``render_manual_ticker``
resolves a market or event ticker typed by the user. A "Watch this game" button
hands the selected game off to the live Watch page.
"""

from __future__ import annotations

import datetime
from typing import Any

import streamlit as st

from kalshi.client import KalshiAPIError, KalshiClient
from kalshi.markets import (
    RESOLVE_GRACE_MINUTES,
    RESOLVE_LOOKAHEAD_HOURS,
    build_game_groups,
    classify_resolution,
    classify_timing,
    evaluate_in_money,
    event_competition,
    in_money_badge,
    live_scores,
    market_label,
    market_type_name,
    matchup_name,
    price_cents_for_side,
    resolution_time,
    scan_series_for_favorites,
)
from ui import data


def render_manual_ticker(
    client: KalshiClient,
) -> dict[str, Any] | None:
    """Resolve a market (or pick from an event's markets) by typed ticker."""
    selected_market: dict[str, Any] | None = None
    ticker = st.text_input(
        "Market ticker (or event ticker to list markets)", ""
    ).strip()
    if ticker:
        try:
            with st.spinner("Looking up ticker…"):
                # Try as a market ticker first.
                try:
                    selected_market = client.get_market(ticker).get("market")
                except KalshiAPIError:
                    selected_market = None
                if not selected_market:
                    markets = client.get_markets(event_ticker=ticker).get("markets", [])
                    if markets:
                        selected_market = st.selectbox(
                            "Market", options=markets, format_func=market_label
                        )
                    else:
                        st.error(
                            f"No market or event found for ticker '{ticker}'."
                        )
        except KalshiAPIError as exc:
            st.error(f"Lookup failed ({exc.status_code}): {exc.message}")
    return selected_market


def render_find_games(
    client: KalshiClient,
) -> tuple[dict[str, Any] | None, str | None, datetime.datetime | None]:
    """Render the browse/filter/favorites flow.

    Returns ``(selected_market, favored_side, selected_game_start)``; the latter
    two are ``None`` unless a favorite market (favored side) or a game with a
    known kickoff time was selected.
    """
    selected_market: dict[str, Any] | None = None
    favored_side: str | None = None
    selected_game_start: datetime.datetime | None = None

    load_col, info_col = st.columns([1, 3])
    with load_col:
        if st.button("Load / refresh games", type="primary"):
            try:
                with st.spinner("Loading open games from Kalshi…"):
                    st.session_state["events"] = data.fetch_open_events(client)
                data.fetch_sports_taxonomy.clear()
            except KalshiAPIError as exc:
                st.error(f"Failed to load events ({exc.status_code}): {exc.message}")
    with info_col:
        if st.session_state.get("events"):
            st.caption(
                f"{len(st.session_state['events'])} open markets loaded. "
                "Filter by sport/competition or search a team below."
            )

    events = st.session_state.get("events", [])
    if not events:
        st.info("Click **Load / refresh games** to pull current open games from Kalshi.")
        return selected_market, favored_side, selected_game_start

    try:
        sport_ordering, comp_to_sport = data.fetch_sports_taxonomy(client)
    except KalshiAPIError:
        sport_ordering, comp_to_sport = [], {}

    # Only events that belong to a competition are games/sports markets.
    sports_events = [e for e in events if event_competition(e)]

    # Map which competitions (and sports) actually have open events.
    sport_to_comps: dict[str, set[str]] = {}
    for e in sports_events:
        comp = event_competition(e)
        sport = comp_to_sport.get(comp, "Other")
        sport_to_comps.setdefault(sport, set()).add(comp)

    if not sports_events:
        st.warning(
            "No sports games found in the open markets. Try the manual ticker "
            "option, or refresh."
        )
        return selected_market, favored_side, selected_game_start

    # Build sport options in Kalshi's display order, then any extras.
    ordered_present = [s for s in sport_ordering if s in sport_to_comps]
    extras = sorted(s for s in sport_to_comps if s not in sport_ordering)
    sport_options = ["All sports", *ordered_present, *extras]

    # Game start times / live status for the "live or starting soon" filter.
    timing_index: dict[str, dict[str, Any]] = {}
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    try:
        timing_index, now_utc = data.fetch_live_window_index(client)
    except KalshiAPIError as exc:
        st.warning(
            f"Could not load game start times ({exc.status_code}): "
            f"{exc.message}. The live filter is unavailable."
        )

    f1, f2 = st.columns(2)
    with f1:
        sport = st.selectbox("Sport", sport_options)
    if sport == "All sports":
        comp_pool: set[str] = set().union(*sport_to_comps.values())
    else:
        comp_pool = sport_to_comps.get(sport, set())
    with f2:
        competition = st.selectbox(
            "Competition / league", ["All competitions", *sorted(comp_pool)]
        )

    c1, c2, c3 = st.columns(3)
    with c1:
        games_only = st.checkbox(
            "Games only",
            value=True,
            help="Show head-to-head matchups only and hide futures, "
            "season-long props, etc.",
        )
    with c2:
        live_only = st.checkbox(
            "Live / starting ≤1h",
            # Default on when start times are available; off (and disabled)
            # when they aren't, so we never hide every game silently.
            value=bool(timing_index),
            disabled=not timing_index,
            help="Show only games in progress or kicking off within the "
            "next hour (based on Kalshi milestone start times).",
        )
    with c3:
        ending_only = st.checkbox(
            "Resolving / ending ≤2h",
            value=False,
            help="Show only games expected to settle within the next 2 hours "
            "(based on each game's expected expiration time). Requires a quick "
            "scan of the in-scope games.",
        )

    include_resolving = st.checkbox(
        "Include markets resolving now",
        value=False,
        help="Markets whose expected expiration time has already passed have "
        "no remaining time-to-expiry, so the Sharpe / volatility-time figures "
        "can't be computed for them. They're hidden by default; check this to "
        "list them anyway.",
    )

    search = st.text_input(
        "Search by team or game", "", placeholder="e.g. Netherlands, NED, Sweden"
    )

    # Group sibling market-type events (winner/spread/totals/halves/...)
    # into one game each.
    game_groups = build_game_groups(sports_events)

    def _group_timing(group: dict[str, Any]) -> tuple[str, str] | None:
        for ticker in (group["rep_ticker"], *group["event_tickers"]):
            if ticker in timing_index:
                return classify_timing(timing_index[ticker], now_utc)
        return None

    def _group_start(group: dict[str, Any]) -> datetime.datetime:
        for ticker in (group["rep_ticker"], *group["event_tickers"]):
            info = timing_index.get(ticker)
            if info and info.get("start"):
                return info["start"]
        return datetime.datetime.max.replace(tzinfo=datetime.timezone.utc)

    def _matches(group: dict[str, Any]) -> bool:
        comp = group["competition"]
        grp_sport = comp_to_sport.get(comp, "Other")
        if sport != "All sports" and grp_sport != sport:
            return False
        if competition != "All competitions" and comp != competition:
            return False
        if games_only and not group["has_game"]:
            return False
        if live_only:
            timing = _group_timing(group)
            if not timing or timing[0] not in ("live", "soon"):
                return False
        if search:
            hay = " ".join(
                [group["matchup"], group["sub_title"], comp, *group["event_tickers"]]
            ).lower()
            if search.lower() not in hay:
                return False
        return True

    # Games passing the instant (non-network) filters.
    prelim = [g for g in game_groups if _matches(g)]

    # Each game's market resolution times, fetched by scanning ALL the
    # in-scope market-type series (winner, spread, totals, first-half,
    # ...). Each market type reports its own expected_expiration_time, so
    # this lets the "ending soon" filter and badge reflect the soonest-
    # closing market in a game (e.g. a first-half market that settles at
    # halftime), not just the full-match settle time.
    resolution_index: dict[str, datetime.datetime] = {}
    res_truncated = False
    all_series = tuple(
        sorted({s for g in prelim for s in g["series"]})
    )
    if all_series:
        try:
            with st.spinner("Checking when each game's markets resolve…"):
                resolution_index, res_truncated = data.fetch_resolution_index(
                    client, all_series
                )
        except KalshiAPIError as exc:
            st.error(
                f"Could not load resolution times ({exc.status_code}): "
                f"{exc.message}"
            )

    def _group_resolution_at(group: dict[str, Any]) -> datetime.datetime | None:
        """Soonest expected resolution across all of the game's markets."""
        times = [
            resolution_index[et]
            for et in group["event_tickers"]
            if et in resolution_index
        ]
        return min(times) if times else None

    def _group_resolution(group: dict[str, Any]) -> tuple[str, str] | None:
        return classify_resolution(_group_resolution_at(group), now_utc)

    def _is_resolving_now(market: dict[str, Any]) -> bool:
        """True when a market's expected expiration has already passed.

        Such markets have no remaining time-to-expiry, so the Sharpe /
        volatility-time math is undefined; we hide them unless the user
        opts in via ``include_resolving``.
        """
        res = classify_resolution(resolution_time(market), now_utc)
        return bool(res and res[0] == "resolving")

    def _ends_soon(group: dict[str, Any]) -> bool:
        resolve_at = _group_resolution_at(group)
        if resolve_at is None:
            return False
        earliest = now_utc - datetime.timedelta(minutes=RESOLVE_GRACE_MINUTES)
        latest = now_utc + datetime.timedelta(hours=RESOLVE_LOOKAHEAD_HOURS)
        return earliest <= resolve_at <= latest

    def _group_label(group: dict[str, Any]) -> str:
        bits = [group["matchup"], group["sub_title"], group["competition"]]
        base = "  ·  ".join(b for b in bits if b)
        # Start/live badge leads the line; the resolution ("ends in Xm")
        # badge trails it so the timing reads start → end left to right.
        timing = _group_timing(group)
        if timing and timing[0] in ("live", "soon", "later"):
            base = f"{timing[1]}  ·  {base}"
        resolution = _group_resolution(group)
        if resolution and resolution[0] in ("ending", "resolving", "later"):
            base = f"{base}  ·  {resolution[1]}"
        return base

    def _sort_key(group: dict[str, Any]):
        timing = _group_timing(group)
        order = {"live": 0, "soon": 1, "later": 2}.get(
            timing[0] if timing else "", 3
        )
        return (order, _group_start(group), group["competition"], group["matchup"])

    filtered = prelim
    if ending_only:
        filtered = [g for g in prelim if _ends_soon(g)]
    filtered = sorted(filtered, key=_sort_key)
    live_count = sum(
        1 for g in filtered if (t := _group_timing(g)) and t[0] in ("live", "soon")
    )
    caption = f"{len(filtered)} game(s) match your filters."
    if timing_index and not live_only and not ending_only:
        caption += f" {live_count} live or starting within the hour."
    st.caption(caption)

    if res_truncated:
        st.warning(
            "Too many market series in scope to check resolution times for "
            "all — only the first 60 were scanned, so some games' "
            "soonest-closing markets may be missing from the ending-soon "
            "filter. Narrow by sport/competition for full coverage."
        )

    if not filtered:
        msg = "No games match. Broaden the filters or clear the search."
        if live_only:
            msg = (
                "No games are live or starting within the next hour right "
                "now. Uncheck the live filter to see all games."
            )
        if ending_only:
            msg = (
                "No games are expected to resolve within the next 2 hours. "
                "Uncheck the resolving filter to see all games."
            )
        st.info(msg)
        return selected_market, favored_side, selected_game_start

    fav_mode = st.checkbox(
        "Find heavy favorites: scan these games for high-priced markets",
        value=False,
        help="Scans every market type in the games matching your filters "
        "and lists contracts trading at/above a price threshold (e.g. 90c "
        "= 90% implied).",
    )

    # Title lookup across all sibling market-type events in scope.
    event_by_ticker = {
        e.get("event_ticker"): e for g in filtered for e in g["events"]
    }

    if fav_mode:
        scan_scope = st.radio(
            "Scan",
            ["All games in scope", "A specific game"],
            horizontal=True,
            help="Scan every game matching your filters, or just one "
            "game you pick.",
        )
        if scan_scope == "A specific game":
            fav_group = st.selectbox(
                "Game",
                options=filtered,
                format_func=_group_label,
                key="fav_game",
            )
            series_in_scope = set(fav_group["series"])
            allowed_tickers = set(fav_group["event_tickers"])
            scope_count = 1
        else:
            series_in_scope = set().union(*(g["series"] for g in filtered))
            allowed_tickers = set().union(
                *(set(g["event_tickers"]) for g in filtered)
            )
            scope_count = len(filtered)

        fc1, fc2, fc3 = st.columns([2, 1, 1])
        with fc1:
            price_range = st.slider(
                "Price range (c = implied %)",
                min_value=1,
                max_value=99,
                value=(90, 95),
                help="Only show contracts whose price falls in this band, "
                "e.g. 90-95 = above 90% but at/below 95% implied.",
            )
        min_price, max_price = price_range
        with fc2:
            side_choice = st.radio(
                "Side", ["Either", "YES", "NO"], horizontal=True
            )
        with fc3:
            st.write("")
            st.write("")
            scan = st.button(
                f"Scan {scope_count} game(s): {min_price}-{max_price}c",
                type="primary",
            )
        st.caption(
            f"{len(series_in_scope)} market series across {scope_count} "
            "game(s) in scope. Narrow by sport/competition/live for faster, "
            "fuller scans."
        )
        if scan:
            try:
                with st.spinner("Scanning all market types for favorites…"):
                    res, truncated = scan_series_for_favorites(
                        client,
                        series_in_scope,
                        allowed_tickers,
                        min_price=min_price,
                        max_price=max_price,
                        side_choice=side_choice,
                    )
                st.session_state["fav_results"] = res
                st.session_state["fav_truncated"] = truncated
                st.session_state["fav_range"] = (min_price, max_price)
            except KalshiAPIError as exc:
                st.error(f"Scan failed ({exc.status_code}): {exc.message}")

        if st.session_state.get("fav_truncated"):
            st.warning(
                "Too many series in scope — only the first 40 were scanned. "
                "Narrow the filters for full coverage."
            )
        results = st.session_state.get("fav_results", [])
        if results and not include_resolving:
            kept = [r for r in results if not _is_resolving_now(r["market"])]
            resolving_hidden = len(results) - len(kept)
            results = kept
            if resolving_hidden:
                st.caption(
                    f"{resolving_hidden} favorite(s) already resolving now are "
                    "hidden (no time-to-expiry for Sharpe). Check "
                    "\"Include markets resolving now\" above to show them."
                )
        if results:
            lo, hi = st.session_state.get("fav_range", (min_price, max_price))
            st.success(f"{len(results)} market(s) priced {lo}-{hi}c.")

            def _fav_label(r: dict[str, Any]) -> str:
                m = r["market"]
                ev = event_by_ticker.get(m.get("event_ticker"), {})
                game = matchup_name(ev) if ev else m.get("event_ticker", "")
                mtype = market_type_name(ev) if ev else ""
                outcome = m.get("yes_sub_title") or m.get("ticker", "")
                return (
                    f"{r['price']:.0f}c {r['side'].upper()}  ·  {game}  ·  "
                    f"{mtype}: {outcome}"
                )

            chosen = st.selectbox(
                "Favorite markets", options=results, format_func=_fav_label
            )
            if chosen:
                selected_market = chosen["market"]
                favored_side = chosen["side"]
                st.caption(
                    f"Favored side: **{chosen['side'].upper()}** at "
                    f"{chosen['price']:.0f}c — preselected as the side to "
                    "buy below."
                )
        elif scan:
            st.info(
                f"No markets priced {min_price}-{max_price}c in scope. "
                "Widen the price range or the filters."
            )
    else:
        group = st.selectbox(
            "Game", options=filtered, format_func=_group_label
        )
        # Kickoff time bounds the realized-volatility window later.
        grp_start = _group_start(group)
        if grp_start < datetime.datetime.max.replace(
            tzinfo=datetime.timezone.utc
        ):
            selected_game_start = grp_start

        # Hand this game off to the live Watch page (keeps the same group so
        # the Watch page resolves milestone/team ids without a re-pick).
        if st.button("📺 Watch this game live", key="watch_handoff"):
            st.session_state["watch_group"] = group
            st.switch_page("app_pages/watch.py")

        markets_by_ticker: dict[str, list[dict[str, Any]]] = {}
        try:
            with st.spinner("Loading all markets for this game…"):
                markets_by_ticker = data.fetch_markets_for_event_tickers(
                    client, group["event_tickers"]
                )
        except KalshiAPIError as exc:
            st.error(
                f"Failed to load markets ({exc.status_code}): {exc.message}"
            )

        siblings = {e.get("event_ticker"): e for e in group["events"]}
        combined = [m for ms in markets_by_ticker.values() for m in ms]
        open_markets = [
            m for m in combined if m.get("status") in ("active", "open")
        ] or combined
        # Drop markets that are already resolving (expected expiration
        # passed) so the Sharpe / volatility-time calc below always has a
        # positive time-to-expiry. Opt back in via the sidebar-adjacent
        # filter checkbox.
        if not include_resolving:
            kept = [m for m in open_markets if not _is_resolving_now(m)]
            resolving_hidden = len(open_markets) - len(kept)
            open_markets = kept
            if resolving_hidden:
                st.caption(
                    f"{resolving_hidden} market(s) already resolving now are "
                    "hidden (no time-to-expiry for Sharpe). Check "
                    "\"Include markets resolving now\" above to show them."
                )

        # Live "in the money" status: pull the current score from the
        # milestone's live data (only meaningful while the game is on)
        # and map it to each market's settlement rule.
        live_details: dict[str, Any] | None = None
        home_team_id = away_team_id = None
        meta = next(
            (
                timing_index[et]
                for et in group["event_tickers"]
                if timing_index.get(et, {}).get("milestone_id")
            ),
            None,
        )
        timing_state = _group_timing(group)
        is_live_now = bool(timing_state and timing_state[0] == "live")
        if meta and is_live_now:
            home_team_id = meta.get("home_team_id")
            away_team_id = meta.get("away_team_id")
            try:
                live_details = data.fetch_live_data(client, meta["milestone_id"])
            except KalshiAPIError as exc:
                st.warning(
                    f"Live score unavailable ({exc.status_code}): "
                    f"{exc.message}. In-the-money flags are off."
                )

        if live_details:
            scores = live_scores(live_details)
            # Map team ids -> readable names via the winner markets.
            name_by_id = {
                (m.get("custom_strike") or {}).get("soccer_team"): m.get(
                    "yes_sub_title"
                )
                for m in combined
                if (m.get("custom_strike") or {}).get("soccer_team")
            }
            home_name = name_by_id.get(home_team_id) or "Home"
            away_name = name_by_id.get(away_team_id) or "Away"
            status_text = (
                live_details.get("status_text")
                or live_details.get("match_status")
                or "live"
            )
            if scores:
                h, a = scores
                st.info(
                    f"🔴 LIVE — {home_name} {h:.0f} : {a:.0f} {away_name}"
                    f"  ·  {status_text}"
                )
            st.caption(
                "🟢 ITM = currently in the money, 🔴 OTM = out of the "
                "money, based on the live score. Market types we can't "
                "auto-evaluate (corners, halves, player props) show no flag."
            )

        def _in_money(m: dict[str, Any]) -> bool | None:
            if not live_details:
                return None
            return evaluate_in_money(
                m, live_details, home_team_id, away_team_id
            )

        def _mkt_label(m: dict[str, Any]) -> str:
            ev = siblings.get(m.get("event_ticker"), {})
            mtype = market_type_name(ev) if ev else ""
            outcome = m.get("yes_sub_title") or m.get("ticker", "")
            yes_ask = price_cents_for_side(m, "yes", "ask")
            price_part = f"  ·  YES {yes_ask:.0f}c" if yes_ask else ""
            badge = in_money_badge(_in_money(m))
            badge_part = f"{badge}  " if badge else ""
            label = f"{badge_part}{mtype} — {outcome}{price_part}"
            # Trail with the game's start and THIS market's own end
            # time (e.g. a first-half market settles at halftime),
            # mirroring the games list. Start is shared across the
            # game; end comes from each market's expected settle.
            timing = _group_timing(group)
            if timing and timing[0] in ("live", "soon", "later"):
                label = f"{label}  ·  {timing[1]}"
            resolution = classify_resolution(
                resolution_time(m), now_utc
            )
            if resolution and resolution[0] in (
                "ending", "resolving", "later"
            ):
                label = f"{label}  ·  {resolution[1]}"
            return label

        open_markets.sort(
            key=lambda m: (
                market_type_name(siblings.get(m.get("event_ticker"), {})),
                m.get("yes_sub_title") or "",
            )
        )
        if open_markets:
            market_types = len(
                {market_type_name(siblings.get(m.get("event_ticker"), {})) for m in open_markets}
            )
            st.caption(
                f"{len(open_markets)} markets across {market_types} market "
                "types (winner, spread, totals, halves, etc.) for this game."
            )
            selected_market = st.selectbox(
                "Market", options=open_markets, format_func=_mkt_label
            )
            if live_details and selected_market:
                sel_status = _in_money(selected_market)
                if sel_status is True:
                    st.success(
                        "🟢 This market's YES side is currently **in the "
                        "money** at the live score."
                    )
                elif sel_status is False:
                    st.error(
                        "🔴 This market's YES side is currently **out of "
                        "the money** at the live score."
                    )
                else:
                    st.caption(
                        "In-the-money status can't be auto-determined for "
                        "this market type from the live score."
                    )
        else:
            st.warning("No open markets found for this game.")

    return selected_market, favored_side, selected_game_start
