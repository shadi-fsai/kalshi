"""Pure market/event logic for the Kalshi Kelly app.

These helpers contain the business logic that powers the Streamlit UI but have
no Streamlit or caching dependencies, so they can be imported and unit-tested
in isolation. ``app.py`` imports from here and keeps only the Streamlit-bound
wrappers (caching, rendering, network fetch + cache).
"""

from __future__ import annotations

import datetime
import re
from typing import Any

from kalshi.client import KalshiClient

# --- Pricing ------------------------------------------------------------


def price_cents_for_side(
    market: dict[str, Any], side: str, kind: str = "ask"
) -> float | None:
    """Extract a price in cents for ``side`` ("yes"/"no") and ``kind`` ("ask"/"bid").

    Prefers the fixed-point dollar fields (e.g. ``yes_ask_dollars`` = "0.5600")
    and falls back to the legacy integer-cent fields (e.g. ``yes_ask``).
    Returns ``None`` if no usable price is present.
    """
    dollars_key = f"{side}_{kind}_dollars"
    if market.get(dollars_key) not in (None, ""):
        try:
            cents = round(float(market[dollars_key]) * 100.0, 4)
            return cents if cents > 0 else None
        except (TypeError, ValueError):
            pass

    legacy_key = f"{side}_{kind}"
    if market.get(legacy_key) not in (None, ""):
        try:
            cents = float(market[legacy_key])
            return cents if cents > 0 else None
        except (TypeError, ValueError):
            pass
    return None


# --- Event / game grouping ----------------------------------------------


def game_key(event: dict[str, Any]) -> str:
    """Game code shared by all market-type events for one game.

    Event tickers look like ``<SERIES>-<GAMECODE>`` (e.g.
    ``KXWCGAME-26JUN20NEDSWE``), with some variants adding a trailing option
    suffix (``...-NED``). Stripping the series prefix and any trailing suffix
    yields the code that is identical across a game's winner, spread, totals,
    first-half, etc. events.
    """
    ticker = event.get("event_ticker", "")
    series = event.get("series_ticker", "")
    if series and ticker.startswith(series + "-"):
        remainder = ticker[len(series) + 1 :]
    elif "-" in ticker:
        remainder = ticker.split("-", 1)[1]
    else:
        remainder = ticker
    return remainder.split("-")[0]


def matchup_name(event: dict[str, Any]) -> str:
    """Clean matchup title (e.g. 'Netherlands vs Sweden'), dropping ': Spread'."""
    title = event.get("title") or ""
    base = title.split(":", 1)[0].strip()
    return base or event.get("sub_title") or event.get("event_ticker", "")


def market_type_name(event: dict[str, Any]) -> str:
    """Human market-type label for a sibling event (e.g. 'Spread', 'Winner')."""
    title = event.get("title") or ""
    if ":" in title:
        return title.split(":", 1)[1].strip()
    return "Winner"


def event_competition(event: dict[str, Any]) -> str:
    """Human-readable competition for an event (e.g. 'World Soccer Cup')."""
    pm = event.get("product_metadata") or {}
    return pm.get("competition") or ""


def event_scope(event: dict[str, Any]) -> str:
    """Competition scope, e.g. 'Game' for a head-to-head matchup."""
    pm = event.get("product_metadata") or {}
    return pm.get("competition_scope") or ""


def build_game_groups(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group sibling market-type events into one game each.

    Returns a list of group dicts with the representative event, all sibling
    events, their tickers/series, and whether the group has a head-to-head
    (Game-scope) event.
    """
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    for ev in events:
        key = (event_competition(ev), game_key(ev))
        g = groups.setdefault(key, {"competition": key[0], "events": []})
        g["events"].append(ev)

    result: list[dict[str, Any]] = []
    for key, g in groups.items():
        evs = g["events"]
        rep = next((e for e in evs if event_scope(e) == "Game"), evs[0])
        result.append(
            {
                "key": key,
                "competition": g["competition"],
                "events": evs,
                "rep": rep,
                "rep_ticker": rep.get("event_ticker", ""),
                "matchup": matchup_name(rep),
                "sub_title": rep.get("sub_title", ""),
                "event_tickers": tuple(
                    sorted(e.get("event_ticker", "") for e in evs)
                ),
                "series": {
                    e.get("series_ticker") for e in evs if e.get("series_ticker")
                },
                "has_game": any(event_scope(e) == "Game" for e in evs),
            }
        )
    return result


def market_label(market: dict[str, Any]) -> str:
    """Compact label for a market dropdown (name, ticker, YES ask)."""
    name = (
        market.get("yes_sub_title")
        or market.get("title")
        or market.get("ticker", "")
    )
    yes_ask = price_cents_for_side(market, "yes", "ask")
    yes_part = f" - YES ask {yes_ask:.0f}c" if yes_ask else ""
    return f"{name} [{market.get('ticker', '')}]{yes_part}"


def series_ticker_for_market(market: dict[str, Any]) -> str | None:
    """Best-effort series ticker for a market (used to look up its fee model)."""
    series = market.get("series_ticker")
    if series:
        return series
    for key in ("event_ticker", "ticker"):
        value = market.get(key) or ""
        if "-" in value:
            return value.split("-", 1)[0]
    return None


def fp_to_float(value: Any) -> float:
    """Parse a Kalshi fixed-point string/number to float (0.0 on failure)."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


# --- Game timing --------------------------------------------------------
# Start times and live status come from milestones. Provider `status` strings
# are inconsistent and often stale, so we treat `start_date` as authoritative
# and only use status to drop clearly-finished games.

LIVE_LOOKBACK_HOURS = 4  # a started game is assumed possibly-live for this long
LIVE_LOOKAHEAD_HOURS = 1  # "starting soon" window
RESOLVE_LOOKAHEAD_HOURS = 2  # "resolving / ending soon" window
RESOLVE_GRACE_MINUTES = 30  # include games just past expected end (still settling)
FINISHED_STATUSES = {
    "completed", "complete", "final", "ft", "closed", "cancelled", "canceled",
    "postponed", "co", "abandoned", "ended",
}


def parse_ts(value: str | None) -> datetime.datetime | None:
    """Parse an RFC3339/ISO timestamp into an aware datetime, or None."""
    if not value:
        return None
    try:
        return datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def resolution_time(market: dict[str, Any]) -> datetime.datetime | None:
    """When a market is expected to actually resolve/settle, or None if unknown.

    Uses ``expected_expiration_time`` — Kalshi's game-day estimate of when THIS
    market settles. The top-level ``close_time`` / ``expiration_time`` are
    deliberately ignored: for multi-stage events (e.g. a World Cup game) they
    carry the series-wide close (the tournament's final date), not when this
    specific market settles, and they're identical across every market in the
    tournament. First-half/second-half/full-time markets each report their own
    ``expected_expiration_time``, so this is the field to search and size on.
    """
    return parse_ts(market.get("expected_expiration_time"))


def classify_timing(
    info: dict[str, Any] | None, now: datetime.datetime
) -> tuple[str, str] | None:
    """Return ``(state, label)`` where state is 'live'|'soon'|'finished'|'later'.

    ``None`` when no start time is known.
    """
    if not info:
        return None
    start = info.get("start")
    if start is None:
        return None
    status = str(info.get("status") or "").strip().lower()
    if status in FINISHED_STATUSES:
        return ("finished", "ended")
    minutes = (start - now).total_seconds() / 60.0
    if minutes <= 0:
        ago = int(round(-minutes))
        return ("live", f"LIVE · started {ago}m ago")
    if minutes <= LIVE_LOOKAHEAD_HOURS * 60:
        return ("soon", f"starts in {int(round(minutes))}m")
    if minutes < 24 * 60:
        return ("later", f"starts in {minutes / 60:.1f}h")
    return ("later", start.astimezone().strftime("%b %d %H:%M"))


def classify_resolution(
    resolve_at: datetime.datetime | None, now: datetime.datetime
) -> tuple[str, str] | None:
    """Return ``(state, label)`` for a game's expected resolution time.

    State is 'ending' (within the lookahead window), 'resolving' (expected end
    just passed), or 'later'. ``None`` when no resolution time is known.
    """
    if resolve_at is None:
        return None
    minutes = (resolve_at - now).total_seconds() / 60.0
    if minutes <= 0:
        return ("resolving", "resolving now")
    if minutes <= RESOLVE_LOOKAHEAD_HOURS * 60:
        if minutes < 60:
            return ("ending", f"ends in {int(round(minutes))}m")
        return ("ending", f"ends in {minutes / 60:.1f}h")
    if minutes < 24 * 60:
        return ("later", f"ends in {minutes / 60:.1f}h")
    return ("later", resolve_at.astimezone().strftime("%b %d %H:%M"))


# --- Live "in the money" status -----------------------------------------
# Kalshi's live-data endpoint (GET /live_data/milestone/{id}) returns the
# current score for in-progress games. We compare it against each market's
# settlement rule (parsed from custom_strike + yes_sub_title) to flag whether a
# YES bet is currently in the money. We only evaluate market types we can map
# unambiguously and otherwise return None so we never guess silently.


def live_scores(details: dict[str, Any]) -> tuple[float, float] | None:
    """Extract (home, away) score from a live-data details object, or None.

    Prefers this match's score (``*_same_game_score``) over an aggregate that
    may include other legs, then falls back to generic ``home_score`` fields.
    """
    for home_key, away_key in (
        ("home_same_game_score", "away_same_game_score"),
        ("home_aggregate_score", "away_aggregate_score"),
        ("home_score", "away_score"),
    ):
        home, away = details.get(home_key), details.get(away_key)
        if isinstance(home, (int, float)) and isinstance(away, (int, float)):
            return float(home), float(away)
    return None


def evaluate_in_money(
    market: dict[str, Any],
    details: dict[str, Any],
    home_team_id: str | None,
    away_team_id: str | None,
) -> bool | None:
    """Is this market's YES side currently in the money given the live score?

    Returns True (in the money), False (out of the money), or None when the
    market type can't be evaluated from the score alone (e.g. corners, player
    props, half-specific markets). Supports soccer-style winner, spread, total,
    team-total, and both-teams-to-score markets.
    """
    scores = live_scores(details)
    if scores is None:
        return None
    home, away = scores
    sub = (market.get("yes_sub_title") or "").strip()
    sub_l = sub.lower()
    team_id = (market.get("custom_strike") or {}).get("soccer_team")

    def _threshold() -> float | None:
        m = re.search(r"([0-9]+(?:\.[0-9]+)?)", sub_l)
        return float(m.group(1)) if m else None

    # Total goals/points: "Over 2.5 goals scored" (no team attached).
    if not team_id and ("over" in sub_l or "under" in sub_l):
        thr = _threshold()
        if thr is None:
            return None
        total = home + away
        return total < thr if "under" in sub_l else total > thr

    # Both teams to score.
    if not team_id and "both teams" in sub_l:
        return home > 0 and away > 0

    if not team_id:
        return None

    # Team-relative markets need to know which side this team is.
    if team_id == home_team_id:
        team_score, opp_score = home, away
    elif team_id == away_team_id:
        team_score, opp_score = away, home
    else:
        # Sentinel team id that is neither home nor away => the "Tie" outcome.
        return home == away

    # Spread: "<Team> wins by more than 2.5 goals".
    if "more than" in sub_l or "wins by" in sub_l:
        thr = _threshold()
        if thr is None:
            return None
        return (team_score - opp_score) > thr

    # Team total: "<Team> over 1.5 goals".
    if "over" in sub_l or "under" in sub_l:
        thr = _threshold()
        if thr is None:
            return None
        return team_score < thr if "under" in sub_l else team_score > thr

    # Winner: yes_sub_title is just the team name.
    return team_score > opp_score


def in_money_badge(status: bool | None) -> str:
    """Short label for an in-the-money status (empty when unknown)."""
    if status is True:
        return "🟢 ITM"
    if status is False:
        return "🔴 OTM"
    return ""


# --- Favorite scanning --------------------------------------------------


def scan_series_for_favorites(
    client: KalshiClient,
    series_tickers: set[str],
    allowed_event_tickers: set[str],
    *,
    min_price: float,
    max_price: float,
    side_choice: str,
    max_series: int = 40,
) -> tuple[list[dict[str, Any]], bool]:
    """Find open markets whose YES/NO ask falls within ``[min_price, max_price]`` cents.

    Scans by series (one paginated call per series returns markets for all its
    events), keeping only markets belonging to ``allowed_event_tickers``.
    Returns ``(results, truncated)`` where each result is
    ``{market, side, price}`` and ``truncated`` flags that the series cap was hit.
    """

    def _in_range(price: float | None) -> bool:
        return price is not None and min_price <= price <= max_price

    results: list[dict[str, Any]] = []
    series = sorted(series_tickers)
    truncated = len(series) > max_series
    for series_ticker in series[:max_series]:
        cursor: str | None = None
        while True:
            page = client.get_markets(
                series_ticker=series_ticker, status="open", limit=1000, cursor=cursor
            )
            for market in page.get("markets", []):
                if (
                    allowed_event_tickers
                    and market.get("event_ticker") not in allowed_event_tickers
                ):
                    continue
                yes_ask = price_cents_for_side(market, "yes", "ask")
                no_ask = price_cents_for_side(market, "no", "ask")
                if side_choice in ("Either", "YES") and _in_range(yes_ask):
                    results.append({"market": market, "side": "yes", "price": yes_ask})
                if side_choice in ("Either", "NO") and _in_range(no_ask):
                    results.append({"market": market, "side": "no", "price": no_ask})
            cursor = page.get("cursor")
            if not cursor:
                break
    results.sort(key=lambda r: r["price"], reverse=True)
    return results, truncated
