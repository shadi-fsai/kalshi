"""Unit tests for shared portfolio + environment helpers."""

import pytest

from kalshi.positions import (
    DEMO_BASE_URL,
    PROD_BASE_URL,
    base_url_for_env,
    held_contracts,
    rest_yes_prices,
    signed_position_contracts,
)


class FakeClient:
    """Minimal stand-in exposing the two methods the helpers call."""

    def __init__(self, positions=None, market=None):
        self._positions = positions or []
        self._market = market or {}

    def get_positions(self):
        return {"market_positions": self._positions}

    def get_market(self, ticker):  # noqa: ARG002 - signature parity only
        return {"market": self._market}


def test_base_url_for_env():
    assert base_url_for_env("prod") == PROD_BASE_URL
    assert base_url_for_env("demo") == DEMO_BASE_URL
    # Anything that isn't prod is treated as demo (safe default).
    assert base_url_for_env("anything-else") == DEMO_BASE_URL


def test_signed_position_prefers_integer_position():
    assert signed_position_contracts({"position": 7, "position_fp": "999"}) == 7
    assert signed_position_contracts({"position": -3}) == -3


def test_signed_position_falls_back_to_fp():
    # position_fp is 2-decimal fixed point: 500 -> 5 contracts.
    assert signed_position_contracts({"position_fp": "500"}) == 5
    assert signed_position_contracts({"position_fp": "-1200"}) == -12


def test_signed_position_missing_is_zero():
    assert signed_position_contracts({}) == 0


def test_held_contracts_no_side():
    client = FakeClient(positions=[{"ticker": "KXT", "position": -10}])
    assert held_contracts(client, "KXT", "no") == 10
    # Querying the side you don't hold returns 0, never negative.
    assert held_contracts(client, "KXT", "yes") == 0


def test_held_contracts_yes_side():
    client = FakeClient(positions=[{"ticker": "KXT", "position": 8}])
    assert held_contracts(client, "KXT", "yes") == 8
    assert held_contracts(client, "KXT", "no") == 0


def test_held_contracts_matches_market_ticker_key():
    client = FakeClient(positions=[{"market_ticker": "KXT", "position_fp": "-400"}])
    assert held_contracts(client, "KXT", "no") == 4


def test_held_contracts_not_found():
    client = FakeClient(positions=[{"ticker": "OTHER", "position": -10}])
    assert held_contracts(client, "KXT", "no") == 0


def test_rest_yes_prices_parses_cents():
    client = FakeClient(market={"yes_bid": 41, "yes_ask": 43, "last_price": 42})
    assert rest_yes_prices(client, "KXT") == (41.0, 43.0, 42.0)


def test_rest_yes_prices_handles_missing_and_bad():
    client = FakeClient(market={"yes_bid": None, "yes_ask": "oops"})
    assert rest_yes_prices(client, "KXT") == (None, None, None)
