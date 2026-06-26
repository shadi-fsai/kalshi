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


def test_signed_position_uses_fixed_point_contracts():
    # position_fp is a fixed-point contract count (NOT scaled): "0.20" = 0.2.
    assert signed_position_contracts({"position_fp": "0.20"}) == pytest.approx(0.20)
    assert signed_position_contracts({"position_fp": "-12.00"}) == pytest.approx(-12.0)
    # position_fp wins over the legacy integer field when both are present.
    assert signed_position_contracts({"position": 7, "position_fp": "0.50"}) == pytest.approx(0.50)


def test_signed_position_falls_back_to_legacy_integer():
    assert signed_position_contracts({"position": -3}) == pytest.approx(-3.0)


def test_signed_position_missing_or_bad_is_zero():
    assert signed_position_contracts({}) == 0.0
    assert signed_position_contracts({"position_fp": "oops"}) == 0.0


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
    client = FakeClient(positions=[{"market_ticker": "KXT", "position_fp": "-4.00"}])
    assert held_contracts(client, "KXT", "no") == pytest.approx(4.0)


def test_held_contracts_fractional():
    client = FakeClient(positions=[{"ticker": "KXT", "position_fp": "0.20"}])
    assert held_contracts(client, "KXT", "yes") == pytest.approx(0.20)
    assert held_contracts(client, "KXT", "no") == 0.0


def test_held_contracts_not_found():
    client = FakeClient(positions=[{"ticker": "OTHER", "position": -10}])
    assert held_contracts(client, "KXT", "no") == 0


def test_rest_yes_prices_parses_cents():
    client = FakeClient(market={"yes_bid": 41, "yes_ask": 43, "last_price": 42})
    assert rest_yes_prices(client, "KXT") == (41.0, 43.0, 42.0)


def test_rest_yes_prices_handles_missing_and_bad():
    client = FakeClient(market={"yes_bid": None, "yes_ask": "oops"})
    assert rest_yes_prices(client, "KXT") == (None, None, None)
