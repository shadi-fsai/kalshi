"""Shared portfolio + environment helpers for the stop-loss tools.

Small, dependency-light functions used by both the stop engine and the ad-hoc
hedge watcher: which base URL an environment maps to, how to read a signed
position out of ``get_positions``, and a REST price snapshot used as the
WebSocket fallback. Prices are returned in CENTS to match the rest of the stack.
"""

from __future__ import annotations

from kalshi.client import KalshiClient

DEMO_BASE_URL = "https://external-api.demo.kalshi.co/trade-api/v2"
PROD_BASE_URL = "https://external-api.kalshi.com/trade-api/v2"


def base_url_for_env(env: str) -> str:
    """Map ``"prod"``/``"demo"`` to the matching Kalshi REST base URL."""
    return PROD_BASE_URL if env == "prod" else DEMO_BASE_URL


def signed_position_contracts(market_position: dict) -> int:
    """Signed whole contracts for a market position (>0 YES, <0 NO, 0 flat).

    Prefers the integer ``position`` field; falls back to the fixed-point
    ``position_fp`` (2-decimal) when only that is present.
    """
    if market_position.get("position") is not None:
        return int(market_position["position"])
    fp = market_position.get("position_fp")
    if fp is None:
        return 0
    return round(float(fp) / 100.0)


def held_contracts(client: KalshiClient, ticker: str, held_side: str) -> int:
    """Positive contract count held on ``held_side`` for ``ticker`` (0 if none).

    Returns 0 when flat or when the held position is on the other side (so a YES
    query against a NO holding returns 0, never a negative number).
    """
    resp = client.get_positions()
    for mp in resp.get("market_positions", []):
        if mp.get("ticker") == ticker or mp.get("market_ticker") == ticker:
            signed = signed_position_contracts(mp)
            if held_side == "yes":
                return signed if signed > 0 else 0
            return -signed if signed < 0 else 0
    return 0


def rest_yes_prices(
    client: KalshiClient, ticker: str
) -> tuple[float | None, float | None, float | None]:
    """REST snapshot of ``(yes_bid, yes_ask, last)`` in CENTS for ``ticker``.

    Used as the WebSocket fallback so a stop is never blind. Any missing field
    comes back as ``None`` rather than raising.
    """
    market = client.get_market(ticker).get("market", {})

    def cents(key: str) -> float | None:
        val = market.get(key)
        if val is None:
            return None
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    return cents("yes_bid"), cents("yes_ask"), cents("last_price")
