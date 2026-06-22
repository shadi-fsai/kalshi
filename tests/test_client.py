"""Tests for kalshi.client using requests-mock to stub HTTP."""

from __future__ import annotations

import json

import pytest

from kalshi.client import DEFAULT_BASE_URL, KalshiAPIError, KalshiClient


@pytest.fixture
def client(rsa_credentials) -> KalshiClient:
    return KalshiClient(rsa_credentials, base_url=DEFAULT_BASE_URL)


def _assert_signed(request):
    """All requests must carry the three signed headers."""
    assert request.headers["KALSHI-ACCESS-KEY"] == "test-key-id"
    assert request.headers["KALSHI-ACCESS-SIGNATURE"]
    assert request.headers["KALSHI-ACCESS-TIMESTAMP"]
    assert request.headers["Accept"] == "application/json"


def test_get_events_params_and_signing(client, requests_mock):
    requests_mock.get(
        f"{DEFAULT_BASE_URL}/events",
        json={"events": [{"event_ticker": "E1"}]},
    )
    out = client.get_events(status="open", limit=50, series_ticker="KXWCGAME")
    assert out == {"events": [{"event_ticker": "E1"}]}
    req = requests_mock.last_request
    _assert_signed(req)
    assert req.qs["status"] == ["open"]
    assert req.qs["limit"] == ["50"]
    assert req.qs["series_ticker"] == ["kxwcgame"]  # requests-mock lowercases qs


def test_get_markets_scoped_to_event(client, requests_mock):
    requests_mock.get(f"{DEFAULT_BASE_URL}/markets", json={"markets": []})
    client.get_markets(event_ticker="E1", status="open", limit=1000)
    req = requests_mock.last_request
    assert req.qs["event_ticker"] == ["e1"]
    assert req.qs["status"] == ["open"]


def test_get_series_path(client, requests_mock):
    requests_mock.get(
        f"{DEFAULT_BASE_URL}/series/KXWCGAME",
        json={"series": {"fee_type": "quadratic"}},
    )
    out = client.get_series("KXWCGAME")
    assert out["series"]["fee_type"] == "quadratic"
    _assert_signed(requests_mock.last_request)


def test_get_positions_default_count_filter(client, requests_mock):
    requests_mock.get(
        f"{DEFAULT_BASE_URL}/portfolio/positions",
        json={"market_positions": []},
    )
    client.get_positions()
    assert requests_mock.last_request.qs["count_filter"] == ["position"]


def test_get_orders_default_resting(client, requests_mock):
    requests_mock.get(
        f"{DEFAULT_BASE_URL}/portfolio/orders", json={"orders": []}
    )
    client.get_orders()
    assert requests_mock.last_request.qs["status"] == ["resting"]


def test_get_live_data_path(client, requests_mock):
    requests_mock.get(
        f"{DEFAULT_BASE_URL}/live_data/milestone/MS123",
        json={"live_data": {"details": {}}},
    )
    out = client.get_live_data("MS123")
    assert "live_data" in out


def test_create_order_body_and_content_type(client, requests_mock):
    requests_mock.post(
        f"{DEFAULT_BASE_URL}/portfolio/events/orders",
        json={"order": {"order_id": "O1"}},
    )
    out = client.create_order(
        ticker="KXWCGAME-26JUN20NEDSWE-NED",
        book_side="bid",
        count=10,
        price_dollars=0.56,
        client_order_id="cid-1",
    )
    assert out["order"]["order_id"] == "O1"
    req = requests_mock.last_request
    _assert_signed(req)
    assert req.headers["Content-Type"] == "application/json"
    body = json.loads(req.body)
    assert body == {
        "ticker": "KXWCGAME-26JUN20NEDSWE-NED",
        "side": "bid",
        "count": "10",
        "price": "0.5600",
        "time_in_force": "good_till_canceled",
        "self_trade_prevention_type": "taker_at_cross",
        "client_order_id": "cid-1",
    }


def test_create_order_invalid_book_side_raises(client):
    with pytest.raises(ValueError, match="book_side must be"):
        client.create_order(
            ticker="T",
            book_side="buy",
            count=1,
            price_dollars=0.5,
            client_order_id="c",
        )


def test_cancel_order_204_returns_empty(client, requests_mock):
    requests_mock.delete(
        f"{DEFAULT_BASE_URL}/portfolio/orders/O1", status_code=204
    )
    out = client.cancel_order("O1")
    assert out == {}
    assert requests_mock.last_request.method == "DELETE"


def test_empty_body_returns_empty_dict(client, requests_mock):
    requests_mock.get(
        f"{DEFAULT_BASE_URL}/portfolio/balance", status_code=200, content=b""
    )
    assert client.get_balance() == {}


def test_error_response_parses_message(client, requests_mock):
    requests_mock.get(
        f"{DEFAULT_BASE_URL}/markets",
        status_code=400,
        json={"message": "bad request param"},
    )
    with pytest.raises(KalshiAPIError) as exc:
        client.get_markets()
    assert exc.value.status_code == 400
    assert "bad request param" in exc.value.message


def test_error_response_non_json_uses_text(client, requests_mock):
    requests_mock.get(
        f"{DEFAULT_BASE_URL}/markets",
        status_code=500,
        text="upstream exploded",
    )
    with pytest.raises(KalshiAPIError) as exc:
        client.get_markets()
    assert exc.value.status_code == 500
    assert "upstream exploded" in exc.value.message


def test_invalid_json_success_raises(client, requests_mock):
    requests_mock.get(
        f"{DEFAULT_BASE_URL}/markets",
        status_code=200,
        content=b"not json",
        headers={"Content-Type": "application/json"},
    )
    with pytest.raises(KalshiAPIError, match="Invalid JSON"):
        client.get_markets()


def test_network_error_wrapped(client, requests_mock):
    import requests

    requests_mock.get(
        f"{DEFAULT_BASE_URL}/markets", exc=requests.ConnectionError("down")
    )
    with pytest.raises(KalshiAPIError) as exc:
        client.get_markets()
    assert exc.value.status_code == 0
    assert "Network error" in exc.value.message


def test_get_event_with_nested_markets(client, requests_mock):
    requests_mock.get(
        f"{DEFAULT_BASE_URL}/events/E1", json={"event": {"event_ticker": "E1"}}
    )
    client.get_event("E1")
    assert requests_mock.last_request.qs["with_nested_markets"] == ["true"]


def test_get_events_with_nested_and_cursor(client, requests_mock):
    requests_mock.get(f"{DEFAULT_BASE_URL}/events", json={"events": []})
    client.get_events(cursor="abc", with_nested_markets=True)
    req = requests_mock.last_request
    assert req.qs["cursor"] == ["abc"]
    assert req.qs["with_nested_markets"] == ["true"]


def test_get_market_and_orderbook(client, requests_mock):
    requests_mock.get(
        f"{DEFAULT_BASE_URL}/markets/T1", json={"market": {"ticker": "T1"}}
    )
    requests_mock.get(
        f"{DEFAULT_BASE_URL}/markets/T1/orderbook", json={"orderbook": {}}
    )
    assert client.get_market("T1")["market"]["ticker"] == "T1"
    client.get_market_orderbook("T1", depth=5)
    assert requests_mock.last_request.qs["depth"] == ["5"]


def test_get_sports_filters(client, requests_mock):
    requests_mock.get(
        f"{DEFAULT_BASE_URL}/search/filters_by_sport",
        json={"sport_ordering": ["Soccer"]},
    )
    assert client.get_sports_filters()["sport_ordering"] == ["Soccer"]


def test_get_milestones_params(client, requests_mock):
    requests_mock.get(f"{DEFAULT_BASE_URL}/milestones", json={"milestones": []})
    client.get_milestones(minimum_start_date="2026-06-20T00:00:00Z", cursor="c1")
    req = requests_mock.last_request
    assert req.qs["minimum_start_date"] == ["2026-06-20t00:00:00z"]
    assert req.qs["cursor"] == ["c1"]


def test_get_markets_series_and_cursor(client, requests_mock):
    requests_mock.get(f"{DEFAULT_BASE_URL}/markets", json={"markets": []})
    client.get_markets(series_ticker="S1", cursor="c2")
    req = requests_mock.last_request
    assert req.qs["series_ticker"] == ["s1"]
    assert req.qs["cursor"] == ["c2"]


def test_get_candlesticks_path_and_params(client, requests_mock):
    requests_mock.get(
        f"{DEFAULT_BASE_URL}/series/KXWCGAME/markets/KXWCGAME-26JUN20NEDSWE-NED/candlesticks",
        json={"ticker": "KXWCGAME-26JUN20NEDSWE-NED", "candlesticks": []},
    )
    out = client.get_candlesticks(
        "KXWCGAME",
        "KXWCGAME-26JUN20NEDSWE-NED",
        start_ts=1000,
        end_ts=2000,
        period_interval=60,
    )
    assert out["ticker"] == "KXWCGAME-26JUN20NEDSWE-NED"
    req = requests_mock.last_request
    _assert_signed(req)
    assert req.qs["start_ts"] == ["1000"]
    assert req.qs["end_ts"] == ["2000"]
    assert req.qs["period_interval"] == ["60"]
    assert req.qs["include_latest_before_start"] == ["true"]


def test_get_positions_and_orders_extra_params(client, requests_mock):
    requests_mock.get(
        f"{DEFAULT_BASE_URL}/portfolio/positions", json={"market_positions": []}
    )
    requests_mock.get(f"{DEFAULT_BASE_URL}/portfolio/orders", json={"orders": []})
    client.get_positions(cursor="pc")
    assert requests_mock.last_request.qs["cursor"] == ["pc"]
    client.get_orders(ticker="T1", cursor="oc")
    req = requests_mock.last_request
    assert req.qs["ticker"] == ["t1"]
    assert req.qs["cursor"] == ["oc"]
