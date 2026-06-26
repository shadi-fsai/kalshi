"""Thin REST client for the Kalshi Trade API.

Implements the endpoints the app needs: events, markets, orderbook,
candlesticks, series fee models, milestones/live data, and portfolio balance +
positions. It also places and cancels orders (``create_order`` / ``cancel_order``,
including ``reduce_only`` for protective exits), so it must be pointed at the
right environment and used with care.

Base URLs (see https://docs.kalshi.com/getting_started/api_environments):
  Production: https://external-api.kalshi.com/trade-api/v2
  Demo:       https://external-api.demo.kalshi.co/trade-api/v2
"""

from __future__ import annotations

import os
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import requests

from kalshi.auth import KalshiCredentials

DEFAULT_BASE_URL = "https://external-api.kalshi.com/trade-api/v2"


class KalshiAPIError(Exception):
    """Raised when the Kalshi API returns a non-success response."""

    def __init__(self, status_code: int, message: str, url: str):
        self.status_code = status_code
        self.message = message
        self.url = url
        super().__init__(f"Kalshi API error {status_code} for {url}: {message}")


class KalshiClient:
    """Authenticated client for Kalshi Trade API read + order endpoints."""

    def __init__(
        self,
        credentials: KalshiCredentials,
        base_url: str | None = None,
        timeout: float = 15.0,
    ):
        self.credentials = credentials
        self.base_url = (base_url or os.getenv("KALSHI_API_BASE") or DEFAULT_BASE_URL).rstrip("/")
        self.timeout = timeout
        # The signed path must include the API root prefix (e.g. /trade-api/v2).
        self._path_prefix = urlsplit(self.base_url).path.rstrip("/")
        self._session = requests.Session()

    def _request(
        self,
        method: str,
        endpoint: str,
        params: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Perform a signed request. ``endpoint`` starts with ``/`` (e.g. ``/markets``).

        Pass ``body`` to send a JSON payload (for POST/PUT). The signature
        covers only timestamp + method + path (never the body or query string).
        """
        full_path = self._path_prefix + endpoint
        url = self.base_url + endpoint
        # Sign the path WITHOUT query parameters (handled inside headers()).
        headers = self.credentials.headers(method, full_path)
        headers["Accept"] = "application/json"
        if body is not None:
            headers["Content-Type"] = "application/json"

        try:
            response = self._session.request(
                method,
                url,
                headers=headers,
                params=params,
                json=body,
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise KalshiAPIError(0, f"Network error: {exc}", url) from exc

        if not response.ok:
            # Surface the API error rather than failing silently.
            try:
                payload = response.json()
                message = (
                    payload.get("message")
                    or payload.get("error")
                    or response.text
                )
            except ValueError:
                message = response.text
            raise KalshiAPIError(response.status_code, str(message), response.url)

        # Some endpoints (e.g. DELETE cancel) return 204/empty bodies on success.
        if response.status_code == 204 or not response.content:
            return {}
        try:
            return response.json()
        except ValueError as exc:
            raise KalshiAPIError(
                response.status_code, f"Invalid JSON response: {exc}", response.url
            ) from exc

    # --- Market data -----------------------------------------------------

    def get_events(
        self,
        *,
        status: str | None = "open",
        limit: int = 100,
        cursor: str | None = None,
        series_ticker: str | None = None,
        with_nested_markets: bool = False,
    ) -> dict[str, Any]:
        """List events. Defaults to open events."""
        params: dict[str, Any] = {"limit": limit}
        if status:
            params["status"] = status
        if cursor:
            params["cursor"] = cursor
        if series_ticker:
            params["series_ticker"] = series_ticker
        if with_nested_markets:
            params["with_nested_markets"] = "true"
        return self._request("GET", "/events", params=params)

    def get_event(self, event_ticker: str, *, with_nested_markets: bool = True) -> dict[str, Any]:
        """Get a single event by ticker, including its markets by default."""
        params = {"with_nested_markets": "true" if with_nested_markets else "false"}
        return self._request("GET", f"/events/{event_ticker}", params=params)

    def get_markets(
        self,
        *,
        event_ticker: str | None = None,
        series_ticker: str | None = None,
        status: str | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        """List markets, optionally scoped to an event or series."""
        params: dict[str, Any] = {"limit": limit}
        if event_ticker:
            params["event_ticker"] = event_ticker
        if series_ticker:
            params["series_ticker"] = series_ticker
        if status:
            params["status"] = status
        if cursor:
            params["cursor"] = cursor
        return self._request("GET", "/markets", params=params)

    def get_market(self, ticker: str) -> dict[str, Any]:
        """Get a single market by ticker."""
        return self._request("GET", f"/markets/{ticker}")

    def get_series(self, series_ticker: str) -> dict[str, Any]:
        """Get a single series, including its ``fee_type`` and ``fee_multiplier``."""
        return self._request("GET", f"/series/{series_ticker}")

    def get_market_orderbook(self, ticker: str, *, depth: int = 10) -> dict[str, Any]:
        """Get the orderbook for a market."""
        return self._request("GET", f"/markets/{ticker}/orderbook", params={"depth": depth})

    def get_candlesticks(
        self,
        series_ticker: str,
        ticker: str,
        *,
        start_ts: int,
        end_ts: int,
        period_interval: int = 60,
        include_latest_before_start: bool = True,
    ) -> dict[str, Any]:
        """Get OHLC candlesticks for a market over ``[start_ts, end_ts]`` (Unix).

        ``period_interval`` is the candle length in minutes and must be one of
        1, 60, or 1440. Each candlestick carries ``end_period_ts`` plus
        ``yes_bid`` / ``yes_ask`` / ``price`` OHLC distributions, used here to
        estimate realized price volatility. See
        https://docs.kalshi.com/api-reference/market/get-market-candlesticks
        """
        params: dict[str, Any] = {
            "start_ts": start_ts,
            "end_ts": end_ts,
            "period_interval": period_interval,
            "include_latest_before_start": str(include_latest_before_start).lower(),
        }
        return self._request(
            "GET",
            f"/series/{series_ticker}/markets/{ticker}/candlesticks",
            params=params,
        )

    # --- Search / taxonomy ----------------------------------------------

    def get_sports_filters(self) -> dict[str, Any]:
        """Get the sport -> competition taxonomy used for friendly filtering.

        Returns ``filters_by_sports`` (sport -> competitions/scopes) and an
        ordered ``sport_ordering`` list for display.
        """
        return self._request("GET", "/search/filters_by_sport")

    def get_milestones(
        self,
        *,
        minimum_start_date: str | None = None,
        limit: int = 200,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        """List milestones (game start times and live status).

        Each milestone carries a ``start_date`` and ``related_event_tickers``,
        which lets us tag events as live or starting soon. ``minimum_start_date``
        is an RFC3339 timestamp; results are ordered by start date ascending.
        """
        params: dict[str, Any] = {"limit": limit}
        if minimum_start_date:
            params["minimum_start_date"] = minimum_start_date
        if cursor:
            params["cursor"] = cursor
        return self._request("GET", "/milestones", params=params)

    def get_live_data(self, milestone_id: str) -> dict[str, Any]:
        """Get live data (current score, status, scoring events) for a milestone.

        Returns ``{"live_data": {"type", "details", "milestone_id"}}`` where
        ``details`` is a sport-specific object. Soccer includes
        ``home_same_game_score`` / ``away_same_game_score``, ``status``,
        ``status_text``, and scoring/card events. Tennis (``type`` ==
        ``tennis_tournament_singles``) is competitor-indexed: per-competitor
        ``*_overall_score`` (sets), ``*_round_scores`` (per-set games, with the
        ``"ongoing"`` entry the current set), ``*_current_round_score`` (current
        game points 0/15/30/40), plus ``server`` / ``winner`` / ``advantage`` as
        competitor ids (see ``markets.tennis_live_score``). The structured
        play-by-play ``game_stats`` endpoint does NOT support tennis. See
        https://docs.kalshi.com/api-reference/live-data/get-live-data
        """
        return self._request("GET", f"/live_data/milestone/{milestone_id}")

    # --- Portfolio -------------------------------------------------------

    def get_balance(self) -> dict[str, Any]:
        """Get the account balance (cents)."""
        return self._request("GET", "/portfolio/balance")

    def get_positions(
        self,
        *,
        count_filter: str | None = "position",
        limit: int = 1000,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        """List current positions.

        ``count_filter="position"`` restricts to markets where you hold a
        non-zero position. Returns ``market_positions`` (each with a signed
        ``position_fp``: positive = YES contracts, negative = NO) and
        ``event_positions``.
        """
        params: dict[str, Any] = {"limit": limit}
        if count_filter:
            params["count_filter"] = count_filter
        if cursor:
            params["cursor"] = cursor
        return self._request("GET", "/portfolio/positions", params=params)

    def get_orders(
        self,
        *,
        status: str | None = "resting",
        ticker: str | None = None,
        limit: int = 200,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        """List orders, defaulting to currently resting (open) orders."""
        params: dict[str, Any] = {"limit": limit}
        if status:
            params["status"] = status
        if ticker:
            params["ticker"] = ticker
        if cursor:
            params["cursor"] = cursor
        return self._request("GET", "/portfolio/orders", params=params)

    # --- Trading (write) -------------------------------------------------

    def create_order(
        self,
        *,
        ticker: str,
        book_side: str,
        count: int,
        price_dollars: float,
        client_order_id: str,
        outcome_side: str | None = None,
        post_only: bool = False,
        reduce_only: bool = False,
        time_in_force: str = "good_till_canceled",
        self_trade_prevention_type: str = "taker_at_cross",
    ) -> dict[str, Any]:
        """Place a limit order via the V2 event-market order endpoint.

        ``book_side`` is the single-book side quoted from the YES leg:
        ``"bid"`` = buy YES, ``"ask"`` = sell YES (buying NO is selling YES at
        ``1 - price``). ``price_dollars`` is the YES-book price in dollars
        (e.g. 0.56). This places a REAL order on the configured environment.

        ``outcome_side`` (``"yes"``/``"no"``) is Kalshi's canonical directional
        field: it lets the Kalshi UI display a buy-NO order as "buy NO" rather
        than the economically equivalent "sell YES". ``post_only=True`` makes the
        order maker-only -- it never crosses the book or pays a taker fee, and is
        auto-cancelled if it would match a resting order. Use
        ``time_in_force="immediate_or_cancel"`` for taker (cross-now) behavior.

        ``reduce_only=True`` guarantees the order can only shrink an existing
        position (never open or flip one) -- used for protective exits/stops so a
        flatten can't accidentally build the opposite side.

        See https://docs.kalshi.com/api-reference/orders/create-order-v2
        """
        if book_side not in ("bid", "ask"):
            raise ValueError(f"book_side must be 'bid' or 'ask' (got {book_side!r}).")
        if outcome_side is not None and outcome_side not in ("yes", "no"):
            raise ValueError(
                f"outcome_side must be 'yes', 'no', or None (got {outcome_side!r})."
            )
        body: dict[str, Any] = {
            "ticker": ticker,
            "side": book_side,
            "count": str(int(count)),
            "price": f"{price_dollars:.4f}",
            "time_in_force": time_in_force,
            "self_trade_prevention_type": self_trade_prevention_type,
            "client_order_id": client_order_id,
        }
        if outcome_side is not None:
            body["outcome_side"] = outcome_side
        if post_only:
            body["post_only"] = True
        if reduce_only:
            body["reduce_only"] = True
        return self._request("POST", "/portfolio/events/orders", body=body)

    def cancel_order(self, order_id: str) -> dict[str, Any]:
        """Cancel (reduce to zero) a resting order by its order id."""
        return self._request("DELETE", f"/portfolio/orders/{order_id}")
