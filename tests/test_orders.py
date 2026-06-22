"""Unit tests for the YES/NO -> V2 book-order mapping."""

import pytest

from kalshi.orders import BookOrder, to_book_order


def test_buy_yes_is_bid_at_same_price():
    order = to_book_order("buy", "yes", 56)
    assert order == BookOrder(book_side="bid", yes_price_dollars=0.56)
    assert order.yes_price_cents == pytest.approx(56)
    assert order.outcome_side == "yes"


def test_sell_yes_is_ask_at_same_price():
    order = to_book_order("sell", "yes", 56)
    assert order.book_side == "ask"
    assert order.yes_price_dollars == pytest.approx(0.56)
    # Sell YES is positioned for the NO outcome.
    assert order.outcome_side == "no"


def test_buy_no_is_ask_at_complement_price():
    # Buying NO at 30c == selling YES at 70c.
    order = to_book_order("buy", "no", 30)
    assert order.book_side == "ask"
    assert order.yes_price_dollars == pytest.approx(0.70)
    assert order.outcome_side == "no"


def test_sell_no_is_bid_at_complement_price():
    # Selling NO at 30c == buying YES at 70c.
    order = to_book_order("sell", "no", 30)
    assert order.book_side == "bid"
    assert order.yes_price_dollars == pytest.approx(0.70)
    assert order.outcome_side == "yes"


@pytest.mark.parametrize("action", ["BUY", "Sell"])
@pytest.mark.parametrize("side", ["YES", "No"])
def test_case_insensitive(action, side):
    order = to_book_order(action, side, 50)
    assert order.book_side in ("bid", "ask")
    assert order.yes_price_dollars == pytest.approx(0.50)


def test_invalid_action_raises():
    with pytest.raises(ValueError):
        to_book_order("hold", "yes", 50)


def test_invalid_side_raises():
    with pytest.raises(ValueError):
        to_book_order("buy", "maybe", 50)


@pytest.mark.parametrize("price", [0, 100, -5, 150])
def test_out_of_range_price_raises(price):
    with pytest.raises(ValueError):
        to_book_order("buy", "yes", price)
