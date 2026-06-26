"""Minimal Kalshi Trade API WebSocket client (ticker channel).

Kalshi has no native stop/trigger order, so a stop must be driven client-side by
watching live prices. This module connects to the authenticated Trade API
WebSocket and streams the public ``ticker`` channel for a single market, yielding
parsed best bid/ask/last in dollars.

Endpoints (see https://docs.kalshi.com/getting_started/api_environments):
  Production: wss://external-api-ws.kalshi.com/trade-api/ws/v2
  Demo:       wss://external-api-ws.demo.kalshi.co/trade-api/ws/v2

The handshake reuses the same RSA-PSS signing as REST: sign ``GET`` over the
path ``/trade-api/ws/v2`` and pass the signature headers on connect.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from urllib.parse import urlsplit

import websockets
from websockets.exceptions import ConnectionClosed

from kalshi.auth import KalshiCredentials

log = logging.getLogger("kalshi.ws")

WS_SIGN_PATH = "/trade-api/ws/v2"


def derive_ws_url(base_url: str) -> str:
    """Map a REST base URL (or env override) to the matching WebSocket URL."""
    override = os.getenv("KALSHI_WS_URL")
    if override:
        return override.rstrip("/")
    host = urlsplit(base_url).netloc or base_url
    mapping = {
        "external-api.kalshi.com": "external-api-ws.kalshi.com",
        "external-api.demo.kalshi.co": "external-api-ws.demo.kalshi.co",
    }
    ws_host = mapping.get(host)
    if ws_host is None:
        # Generic: external-api.* -> external-api-ws.*; otherwise reuse host
        # (shared hosts api.elections.kalshi.com / demo-api.kalshi.co also work).
        if host.startswith("external-api."):
            ws_host = host.replace("external-api.", "external-api-ws.", 1)
        else:
            ws_host = host
    return f"wss://{ws_host}{WS_SIGN_PATH}"


def _to_dollars(value: object) -> float | None:
    """Parse a fixed-point dollars string (e.g. ``"0.95"``) to a float."""
    if value in (None, ""):
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class Tick:
    """A normalized ticker update (prices in dollars)."""

    market_ticker: str
    yes_bid: float | None
    yes_ask: float | None
    last: float | None
    ts_ms: int | None


def _parse_ticker_msg(payload: dict) -> Tick | None:
    msg = payload.get("msg") or {}
    ticker = msg.get("market_ticker")
    if not ticker:
        return None
    last = (
        _to_dollars(msg.get("price_dollars"))
        or _to_dollars(msg.get("last_price_dollars"))
        or _to_dollars(msg.get("price"))
    )
    ts = msg.get("ts_ms")
    try:
        ts = int(ts) if ts is not None else None
    except (TypeError, ValueError):
        ts = None
    return Tick(
        market_ticker=ticker,
        yes_bid=_to_dollars(msg.get("yes_bid_dollars")),
        yes_ask=_to_dollars(msg.get("yes_ask_dollars")),
        last=last,
        ts_ms=ts,
    )


async def stream_ticker(
    credentials: KalshiCredentials,
    base_url: str,
    ticker: str,
):
    """Yield :class:`Tick` updates for ``ticker`` over a single WS connection.

    Raises :class:`websockets.exceptions.ConnectionClosed` (or other connection
    errors) when the socket drops, so the caller can reconnect / fall back. This
    never swallows disconnects silently.
    """
    url = derive_ws_url(base_url)
    headers = credentials.headers("GET", WS_SIGN_PATH)
    log.info("WS connecting to %s for %s", url, ticker)
    async with websockets.connect(url, additional_headers=headers) as ws:
        sub = {
            "id": 1,
            "cmd": "subscribe",
            "params": {"channels": ["ticker"], "market_ticker": ticker},
        }
        await ws.send(json.dumps(sub))
        log.info("WS subscribed to ticker for %s", ticker)
        try:
            async for raw in ws:
                try:
                    payload = json.loads(raw)
                except (ValueError, TypeError):
                    log.warning("WS non-JSON message ignored: %r", raw)
                    continue
                mtype = payload.get("type")
                if mtype == "ticker":
                    tick = _parse_ticker_msg(payload)
                    if tick is not None:
                        yield tick
                elif mtype == "subscribed":
                    log.info("WS subscription confirmed: %s", payload.get("msg"))
                elif mtype == "error":
                    # Surface server-side errors loudly instead of failing silent.
                    log.error("WS error message: %s", payload)
                    raise RuntimeError(f"Kalshi WS error: {payload}")
                else:
                    log.debug("WS other message: %s", payload)
        except ConnectionClosed as exc:
            log.warning("WS connection closed (%s); caller should reconnect.", exc)
            raise
