"""Translate human buy/sell + YES/NO intent into Kalshi V2 book orders.

Kalshi's V2 order endpoint (``POST /portfolio/events/orders``) quotes every
order from the YES leg of a single book:

- ``bid`` means buy YES, ``ask`` means sell YES.
- Buying NO is economically equivalent to selling YES at ``1 - price``; selling
  NO is equivalent to buying YES at ``1 - price``.

This module keeps that conversion pure and testable so the UI can speak in
familiar buy/sell + YES/NO terms.

See https://docs.kalshi.com/getting_started/order_direction
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BookOrder:
    """A V2 order expressed on the YES book.

    ``outcome_side`` is Kalshi's canonical directional field (the outcome you
    profit from). It mirrors ``book_side`` (``bid`` == ``yes``, ``ask`` == ``no``)
    and is what makes Kalshi's UI label a buy-NO order as "buy NO" rather than the
    economically equivalent "sell YES".
    """

    book_side: str  # "bid" (buy YES) or "ask" (sell YES)
    yes_price_dollars: float  # YES-book price in dollars (0.01-0.99)

    @property
    def yes_price_cents(self) -> float:
        return round(self.yes_price_dollars * 100.0, 4)

    @property
    def outcome_side(self) -> str:
        """The outcome the order is positioned for: ``"yes"`` (bid) or ``"no"`` (ask)."""
        return "yes" if self.book_side == "bid" else "no"


def to_book_order(action: str, side: str, price_cents: float) -> BookOrder:
    """Map ``(action, side, price_cents)`` to a YES-book :class:`BookOrder`.

    Args:
        action: ``"buy"`` or ``"sell"``.
        side: ``"yes"`` or ``"no"`` -- the contract you are buying/selling.
        price_cents: The price of THAT side in cents (1-99).

    Returns:
        A :class:`BookOrder` with the YES-book side and price. Buying/selling NO
        is converted to the equivalent YES-book order at ``100 - price``.

    Raises:
        ValueError: if ``action``/``side`` are invalid or ``price_cents`` is not
            in the tradeable 1-99 range.
    """
    action = action.lower()
    side = side.lower()
    if action not in ("buy", "sell"):
        raise ValueError(f"action must be 'buy' or 'sell' (got {action!r}).")
    if side not in ("yes", "no"):
        raise ValueError(f"side must be 'yes' or 'no' (got {side!r}).")
    if not 1 <= price_cents <= 99:
        raise ValueError(
            f"price_cents must be between 1 and 99 (got {price_cents})."
        )

    # Express the order on the YES book.
    if side == "yes":
        yes_price_cents = price_cents
        book_side = "bid" if action == "buy" else "ask"
    else:  # NO -> mirror onto the YES book at (100 - price)
        yes_price_cents = 100.0 - price_cents
        # buy NO == sell YES (ask); sell NO == buy YES (bid)
        book_side = "ask" if action == "buy" else "bid"

    return BookOrder(book_side=book_side, yes_price_dollars=yes_price_cents / 100.0)
