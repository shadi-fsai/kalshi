"""Kalshi trading-fee calculation from a series' fee model.

Each series exposes a ``fee_type`` and ``fee_multiplier`` (see
``GET /series/{series_ticker}``). Kalshi's published fee schedule
(https://kalshi.com/docs/kalshi-fee-schedule.pdf) defines:

- ``quadratic`` / ``quadratic_with_maker_fees`` (General Trading Fees Table):
  ``fee = roundup_to_cent(0.07 * fee_multiplier * C * P * (1 - P))`` where ``C``
  is the contract count and ``P`` the price in dollars. The roundup applies to
  the order total. For ``quadratic_with_maker_fees`` this is the taker fee;
  resting (maker) fills are cheaper, so this is a conservative estimate.
- ``flat`` (Specific Trading Fees Table): a per-contract amount that is not
  derivable from the multiplier alone, so we return ``None`` and let the caller
  fall back to a manual fee.

The base coefficient ``0.07`` is the General Trading Fees rate; ``fee_multiplier``
scales it (most series use ``1.0``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

GENERAL_FEE_RATE = 0.07
QUADRATIC_TYPES = ("quadratic", "quadratic_with_maker_fees")


def _roundup_to_cent(dollars: float) -> float:
    """Round a dollar amount UP to the next whole cent."""
    # Nudge to absorb floating-point noise before ceiling (e.g. 0.0175000001).
    return math.ceil(round(dollars * 100.0, 6)) / 100.0


@dataclass(frozen=True)
class FeeModel:
    """A series' fee model as reported by the Kalshi API."""

    fee_type: str
    fee_multiplier: float

    @property
    def is_quadratic(self) -> bool:
        return self.fee_type in QUADRATIC_TYPES

    def per_contract_fee(self, price_dollars: float) -> float | None:
        """Marginal (pre-roundup) fee for one contract at ``price_dollars``.

        Suitable for Kelly sizing. Returns ``None`` for non-quadratic models.
        """
        if not self.is_quadratic:
            return None
        if not 0.0 < price_dollars < 1.0:
            return 0.0
        return GENERAL_FEE_RATE * self.fee_multiplier * price_dollars * (1.0 - price_dollars)

    def order_fee(self, count: float, price_dollars: float) -> float | None:
        """Total fee (rounded up to the cent) for ``count`` contracts.

        Returns ``None`` for non-quadratic models (caller should fall back).
        """
        if not self.is_quadratic:
            return None
        if count <= 0 or not 0.0 < price_dollars < 1.0:
            return 0.0
        raw = (
            GENERAL_FEE_RATE
            * self.fee_multiplier
            * count
            * price_dollars
            * (1.0 - price_dollars)
        )
        return _roundup_to_cent(raw)


def fee_model_from_series(series: dict) -> FeeModel | None:
    """Build a :class:`FeeModel` from a ``/series`` payload's ``series`` object."""
    if not series:
        return None
    fee_type = series.get("fee_type")
    multiplier = series.get("fee_multiplier")
    if fee_type is None or multiplier is None:
        return None
    try:
        return FeeModel(fee_type=str(fee_type), fee_multiplier=float(multiplier))
    except (TypeError, ValueError):
        return None
