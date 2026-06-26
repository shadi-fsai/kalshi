"""Unit tests for the WebSocket client's pure helpers (no real connection)."""

import pytest

from kalshi.ws import Tick, _parse_ticker_msg, _to_dollars, derive_ws_url


def test_derive_ws_url_prod_and_demo():
    assert (
        derive_ws_url("https://external-api.kalshi.com/trade-api/v2")
        == "wss://external-api-ws.kalshi.com/trade-api/ws/v2"
    )
    assert (
        derive_ws_url("https://external-api.demo.kalshi.co/trade-api/v2")
        == "wss://external-api-ws.demo.kalshi.co/trade-api/ws/v2"
    )


def test_derive_ws_url_env_override(monkeypatch):
    monkeypatch.setenv("KALSHI_WS_URL", "wss://example.test/ws/")
    assert derive_ws_url("https://anything/trade-api/v2") == "wss://example.test/ws"


def test_derive_ws_url_unknown_host_reuses_host(monkeypatch):
    monkeypatch.delenv("KALSHI_WS_URL", raising=False)
    assert (
        derive_ws_url("https://api.elections.kalshi.com/trade-api/v2")
        == "wss://api.elections.kalshi.com/trade-api/ws/v2"
    )


@pytest.mark.parametrize(
    "raw,expected",
    [("0.95", 0.95), (None, None), ("", None), ("oops", None), (0.5, 0.5)],
)
def test_to_dollars(raw, expected):
    assert _to_dollars(raw) == expected


def test_parse_ticker_msg_full():
    payload = {
        "type": "ticker",
        "msg": {
            "market_ticker": "KXT",
            "yes_bid_dollars": "0.40",
            "yes_ask_dollars": "0.42",
            "price_dollars": "0.41",
            "ts_ms": 1234,
        },
    }
    tick = _parse_ticker_msg(payload)
    assert tick == Tick(market_ticker="KXT", yes_bid=0.40, yes_ask=0.42, last=0.41, ts_ms=1234)


def test_parse_ticker_msg_last_price_fallback():
    payload = {"msg": {"market_ticker": "KXT", "last_price_dollars": "0.33"}}
    tick = _parse_ticker_msg(payload)
    assert tick.last == 0.33
    assert tick.yes_bid is None


def test_parse_ticker_msg_requires_market_ticker():
    assert _parse_ticker_msg({"msg": {"yes_bid_dollars": "0.4"}}) is None


def test_tick_is_frozen():
    tick = Tick(market_ticker="KXT", yes_bid=None, yes_ask=None, last=None, ts_ms=None)
    with pytest.raises(Exception):
        tick.yes_bid = 1  # type: ignore[misc]
