"""Streamlit-cached data fetchers and client construction.

Every network call the UI makes goes through one of these cached wrappers so the
app stays responsive and stays within rate limits. Pure parsing/market logic is
imported from the ``kalshi`` package; nothing here mutates global state.

Pages and the sizer must call these functions through the module object (e.g.
``from ui import data; data.fetch_live_data(...)``) so that the sizer's
"Refresh game data" button can invalidate the exact same cached objects and so
tests can monkeypatch them.
"""

from __future__ import annotations

import datetime
from typing import Any

import streamlit as st

from kalshi.auth import KalshiAuthError, KalshiCredentials
from kalshi.client import KalshiAPIError, KalshiClient
from kalshi.fees import FeeModel, fee_model_from_series
from kalshi.markets import (
    LIVE_LOOKAHEAD_HOURS,
    LIVE_LOOKBACK_HOURS,
    parse_ts,
    resolution_time,
)
from kalshi.risk import (
    ask_price_series_from_candlesticks,
    high_water_marks_cents,
    mid_price_series_from_candlesticks,
    mid_prices_from_candlesticks,
)


@st.cache_resource(show_spinner=False)
def get_client() -> KalshiClient:
    """Build an authenticated client from environment credentials.

    Cached so the key is loaded once per session. Raises on bad/missing creds;
    the caller surfaces the error.
    """
    creds = KalshiCredentials.from_env()
    return KalshiClient(creds)


def build_client() -> tuple[KalshiClient | None, bool, str | None]:
    """Return ``(client, auth_ok, error)`` without raising.

    This is the dependency-injection seam: the router and pages obtain the
    client here, and tests monkeypatch this function to inject a fake client.
    """
    try:
        return get_client(), True, None
    except KalshiAuthError as exc:
        return None, False, f"Auth not configured: {exc}"
    except Exception as exc:  # noqa: BLE001 - surface any credential load failure
        return None, False, f"Failed to load credentials: {exc}"


@st.cache_data(show_spinner=False, ttl=60)
def fetch_markets_for_event_tickers(
    _client: KalshiClient, tickers: tuple[str, ...]
) -> dict[str, list[dict[str, Any]]]:
    """Fetch open markets for each event ticker. Returns ticker -> markets."""
    out: dict[str, list[dict[str, Any]]] = {}
    for ticker in tickers:
        page = _client.get_markets(event_ticker=ticker, limit=1000)
        out[ticker] = page.get("markets", [])
    return out


@st.cache_data(show_spinner=False, ttl=15)
def fetch_live_markets(
    _client: KalshiClient, tickers: tuple[str, ...]
) -> dict[str, list[dict[str, Any]]]:
    """Short-TTL variant of :func:`fetch_markets_for_event_tickers`.

    Used by the live "Watch a game" page so prices refresh on a ~15s cadence
    (the games browser uses the 60s version to stay within rate limits).
    """
    out: dict[str, list[dict[str, Any]]] = {}
    for ticker in tickers:
        page = _client.get_markets(event_ticker=ticker, limit=1000)
        out[ticker] = page.get("markets", [])
    return out


@st.cache_data(show_spinner=False, ttl=120)
def fetch_resolution_index(
    _client: KalshiClient, series_tickers: tuple[str, ...], max_series: int = 60
) -> tuple[dict[str, datetime.datetime], bool]:
    """Map event_ticker -> soonest expected resolution time across its markets.

    Scanning one series returns markets for all that competition's events of
    that market type, so passing every in-scope series (winner, spread, totals,
    first-half, etc.) yields each event's ``expected_expiration_time`` — the
    field that reflects when that specific market settles (a first-half market
    resolves at halftime, not at the full-match close). For each event we keep
    the EARLIEST market resolution so callers can find a game by its soonest-
    closing market. Returns ``(index, truncated)``.
    """
    index: dict[str, datetime.datetime] = {}
    series = sorted(series_tickers)
    truncated = len(series) > max_series
    for series_ticker in series[:max_series]:
        cursor: str | None = None
        while True:
            page = _client.get_markets(
                series_ticker=series_ticker, status="open", limit=1000, cursor=cursor
            )
            for market in page.get("markets", []):
                event_ticker = market.get("event_ticker")
                if not event_ticker:
                    continue
                resolve_at = resolution_time(market)
                if resolve_at and (
                    event_ticker not in index or resolve_at < index[event_ticker]
                ):
                    index[event_ticker] = resolve_at
            cursor = page.get("cursor")
            if not cursor:
                break
    return index, truncated


def fetch_open_events(
    client: KalshiClient,
    *,
    series_ticker: str | None = None,
    max_events: int = 10000,
) -> list[dict[str, Any]]:
    """Fetch open events across all pages, up to ``max_events``.

    Kalshi can return thousands of open events with no useful default ordering,
    so we page through everything (or just one series when ``series_ticker`` is
    given) and let the UI filter client-side. Otherwise specific games (e.g. a
    single World Cup match) can sit past any small cap and never appear.
    """
    events: list[dict[str, Any]] = []
    cursor: str | None = None
    while len(events) < max_events:
        page = client.get_events(
            status="open", limit=200, cursor=cursor, series_ticker=series_ticker or None
        )
        events.extend(page.get("events", []))
        cursor = page.get("cursor")
        if not cursor:
            break
    return events[:max_events]


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_sports_taxonomy(_client: KalshiClient) -> tuple[list[str], dict[str, str]]:
    """Return ``(sport_ordering, competition_to_sport)`` for friendly filtering.

    Cached for an hour. ``_client`` is underscore-prefixed so Streamlit does
    not try to hash it.
    """
    data = _client.get_sports_filters()
    ordering = [s for s in data.get("sport_ordering", []) if s != "All sports"]
    comp_to_sport: dict[str, str] = {}
    for sport, details in (data.get("filters_by_sports") or {}).items():
        if sport == "All sports":
            continue
        for comp in (details.get("competitions") or {}):
            comp_to_sport.setdefault(comp, sport)
    return ordering, comp_to_sport


@st.cache_data(show_spinner=False, ttl=120)
def fetch_live_window_index(
    _client: KalshiClient,
) -> tuple[dict[str, dict[str, Any]], datetime.datetime]:
    """Map event_ticker -> {start, status} for games near the live window.

    Pulls milestones starting from ``LIVE_LOOKBACK_HOURS`` ago up to
    ``LIVE_LOOKAHEAD_HOURS`` ahead. Cached for 2 minutes since liveness changes.
    """
    now = datetime.datetime.now(datetime.timezone.utc)
    min_start = now - datetime.timedelta(hours=LIVE_LOOKBACK_HOURS)
    horizon = now + datetime.timedelta(hours=LIVE_LOOKAHEAD_HOURS)
    index: dict[str, dict[str, Any]] = {}
    cursor: str | None = None
    for _ in range(25):  # safety cap on pagination
        page = _client.get_milestones(
            minimum_start_date=min_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            limit=200,
            cursor=cursor,
        )
        milestones = page.get("milestones", [])
        for m in milestones:
            start = parse_ts(m.get("start_date"))
            details = m.get("details") or {}
            status = details.get("status")
            entry = {
                "start": start,
                "status": status,
                "milestone_id": m.get("id"),
                "home_team_id": details.get("home_team_id"),
                "away_team_id": details.get("away_team_id"),
            }
            for ticker in m.get("related_event_tickers") or []:
                index[ticker] = entry
        cursor = page.get("cursor")
        last_start = parse_ts(milestones[-1].get("start_date")) if milestones else None
        if last_start and last_start > horizon:
            break  # ascending order: nothing further is within the window
        if not cursor:
            break
    return index, now


@st.cache_data(show_spinner=False, ttl=20)
def fetch_live_data(_client: KalshiClient, milestone_id: str) -> dict[str, Any] | None:
    """Return the ``details`` object of a milestone's live data, or None.

    Live "in the money" status is computed from this score via
    ``kalshi.markets.evaluate_in_money``.
    """
    resp = _client.get_live_data(milestone_id)
    live = (resp or {}).get("live_data") or {}
    details = live.get("details")
    return details if isinstance(details, dict) else None


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_fee_model(_client: KalshiClient, series_ticker: str) -> FeeModel | None:
    """Fetch and cache a series' fee model (fee_type + multiplier)."""
    series = _client.get_series(series_ticker).get("series", {})
    return fee_model_from_series(series)


@st.cache_data(show_spinner=False, ttl=120)
def fetch_mid_prices(
    _client: KalshiClient,
    series_ticker: str,
    ticker: str,
    start_ts: int,
    end_ts: int,
    period_interval: int,
) -> list[float]:
    """Return a mid-price series (probability units) from candlesticks, oldest first.

    Parsing of the candlestick OHLC fields lives in
    ``kalshi.risk.mid_prices_from_candlesticks`` (pure + unit-tested). Cached for
    2 minutes.
    """
    resp = _client.get_candlesticks(
        series_ticker,
        ticker,
        start_ts=start_ts,
        end_ts=end_ts,
        period_interval=period_interval,
    )
    return mid_prices_from_candlesticks(resp.get("candlesticks", []))


@st.cache_data(show_spinner=False, ttl=120)
def fetch_high_water_marks(
    _client: KalshiClient,
    series_ticker: str,
    ticker: str,
    start_ts: int,
    end_ts: int,
    period_interval: int,
) -> tuple[float | None, float | None]:
    """Return ``(yes_hwm_cents, no_hwm_cents)`` from a market's candlestick history.

    The per-side high-water-mark math lives in
    ``kalshi.risk.high_water_marks_cents`` (pure + unit-tested). Cached for 2
    minutes and keyed per market, so a market that appears as both a YES and a
    NO favorite only fetches candlesticks once.
    """
    resp = _client.get_candlesticks(
        series_ticker,
        ticker,
        start_ts=start_ts,
        end_ts=end_ts,
        period_interval=period_interval,
    )
    return high_water_marks_cents(resp.get("candlesticks", []))


@st.cache_data(show_spinner=False, ttl=60)
def fetch_ask_price_series(
    _client: KalshiClient,
    series_ticker: str,
    ticker: str,
    side: str,
    start_ts: int,
    end_ts: int,
    period_interval: int,
) -> list[tuple[int, float]]:
    """Return a timestamped ask-price series (cents) for ``side`` from candlesticks.

    Used by the pre-order price chart. The per-side ask math lives in
    ``kalshi.risk.ask_price_series_from_candlesticks`` (pure + unit-tested).
    Cached for 60s and keyed by side + window so toggling either refetches.
    """
    resp = _client.get_candlesticks(
        series_ticker,
        ticker,
        start_ts=start_ts,
        end_ts=end_ts,
        period_interval=period_interval,
    )
    return ask_price_series_from_candlesticks(resp.get("candlesticks", []), side)


@st.cache_data(show_spinner=False, ttl=120)
def fetch_mid_price_series(
    _client: KalshiClient,
    series_ticker: str,
    ticker: str,
    side: str,
    start_ts: int,
    end_ts: int,
    period_interval: int,
) -> list[tuple[int, float]]:
    """Return a timestamped mid-price series (probability units) oriented to ``side``.

    Used by the portfolio correlation matrix. The per-side mid math lives in
    ``kalshi.risk.mid_price_series_from_candlesticks`` (pure + unit-tested).
    Cached for 2 minutes and keyed per market + side.
    """
    resp = _client.get_candlesticks(
        series_ticker,
        ticker,
        start_ts=start_ts,
        end_ts=end_ts,
        period_interval=period_interval,
    )
    return mid_price_series_from_candlesticks(resp.get("candlesticks", []), side)


__all__ = [
    "KalshiAPIError",
    "build_client",
    "get_client",
    "fetch_markets_for_event_tickers",
    "fetch_live_markets",
    "fetch_resolution_index",
    "fetch_open_events",
    "fetch_sports_taxonomy",
    "fetch_live_window_index",
    "fetch_live_data",
    "fetch_fee_model",
    "fetch_mid_prices",
    "fetch_high_water_marks",
    "fetch_ask_price_series",
    "fetch_mid_price_series",
]
