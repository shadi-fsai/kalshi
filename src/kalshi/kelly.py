"""Kelly criterion sizing for Kalshi binary contracts.

A Kalshi contract trades in cents (1-99) and pays out $1.00 if it resolves in
your favor. Buying a contract at ``price_cents`` costs ``price_cents / 100``
dollars and returns $1 on a win (profit ``1 - cost``) or $0 on a loss.

For a binary bet with win probability ``q`` and net decimal odds ``b`` (profit
per dollar staked on a win), the full-Kelly fraction of bankroll to wager is:

    f* = (q * b - (1 - q)) / b = q - (1 - q) / b

When the estimated probability does not exceed the breakeven (market) price,
the edge is non-positive and the recommended bet is zero.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class KellyResult:
    """Result of a Kelly sizing calculation for a single contract side."""

    side: str
    cost_per_contract: float
    fee_per_contract: float
    entry_cost_per_contract: float
    implied_probability: float
    breakeven_probability: float
    estimated_probability: float
    edge: float
    full_kelly_fraction: float
    kelly_multiplier: float
    used_fraction: float
    recommended_stake: float
    contracts: int
    actual_stake: float

    @property
    def has_edge(self) -> bool:
        return self.full_kelly_fraction > 0.0


def _validate_inputs(price_cents: float, estimated_probability: float) -> None:
    if not 1 <= price_cents <= 99:
        raise ValueError(
            f"price_cents must be between 1 and 99 (got {price_cents})."
        )
    if not 0.0 <= estimated_probability <= 1.0:
        raise ValueError(
            f"estimated_probability must be between 0 and 1 (got {estimated_probability})."
        )


def kelly_for_contract(
    *,
    side: str,
    price_cents: float,
    estimated_probability: float,
    bankroll: float,
    kelly_multiplier: float = 1.0,
    fee_buy: float = 0.01,
    fee_sell: float = 0.01,
) -> KellyResult:
    """Compute the Kelly-optimal stake and contract count for one contract side.

    Transaction fees are modeled round-trip: ``fee_buy`` is paid to enter and
    ``fee_sell`` to exit, both in dollars per contract. Together they raise the
    breakeven probability to ``cost + fee_buy + fee_sell`` and shrink the net
    odds, so sizing is conservative.

    Args:
        side: ``"yes"`` or ``"no"``. ``estimated_probability`` and
            ``price_cents`` should describe THAT side (i.e. the side you are
            considering buying).
        price_cents: The ask/price of the chosen side in cents (1-99). This is
            the cost to buy one contract of that side.
        estimated_probability: Your estimated probability (0-1) that this side
            wins. Defaults in the UI to the implied probability (price / 100).
        bankroll: Total bankroll in dollars used for sizing.
        kelly_multiplier: Fractional-Kelly scaler in [0, 1] (e.g. 0.5 for
            half-Kelly).
        fee_buy: Fee in dollars per contract paid on entry (default $0.01).
        fee_sell: Fee in dollars per contract paid on exit (default $0.01).

    Returns:
        A :class:`KellyResult`. If there is no positive edge, the recommended
        stake and contract count are zero.
    """
    side = side.lower()
    if side not in ("yes", "no"):
        raise ValueError(f"side must be 'yes' or 'no' (got {side!r}).")
    if bankroll < 0:
        raise ValueError(f"bankroll must be non-negative (got {bankroll}).")
    if not 0.0 <= kelly_multiplier <= 1.0:
        raise ValueError(
            f"kelly_multiplier must be between 0 and 1 (got {kelly_multiplier})."
        )
    if fee_buy < 0 or fee_sell < 0:
        raise ValueError(
            f"fees must be non-negative (got buy={fee_buy}, sell={fee_sell})."
        )
    _validate_inputs(price_cents, estimated_probability)

    cost = price_cents / 100.0
    implied_probability = cost  # market price; pre-fee breakeven
    fee_per_contract = fee_buy + fee_sell  # round-trip
    entry_cost = cost + fee_buy  # cash outlay to acquire one contract
    # With fees, you win (1 - cost - fees) and lose (cost + fees); since those
    # sum to 1, the fee-adjusted breakeven probability is simply cost + fees.
    breakeven_probability = min(1.0, cost + fee_per_contract)
    q = estimated_probability
    edge = q - breakeven_probability

    net_win = 1.0 - cost - fee_per_contract
    net_loss = cost + fee_per_contract
    if net_win <= 0.0 or net_loss <= 0.0:
        # Fees exceed the spread to $1 (or to $0); no profitable bet is possible.
        full_fraction = 0.0
    else:
        net_odds = net_win / net_loss
        full_fraction = max(0.0, q - (1.0 - q) / net_odds)

    used_fraction = full_fraction * kelly_multiplier
    recommended_stake = used_fraction * bankroll
    # Add a tiny epsilon before flooring so floating-point error (e.g.
    # 199.9999.../0.5) does not drop an otherwise-affordable contract.
    contracts = (
        int(math.floor(recommended_stake / entry_cost + 1e-9))
        if entry_cost > 0
        else 0
    )
    actual_stake = contracts * entry_cost

    return KellyResult(
        side=side,
        cost_per_contract=cost,
        fee_per_contract=fee_per_contract,
        entry_cost_per_contract=entry_cost,
        implied_probability=implied_probability,
        breakeven_probability=breakeven_probability,
        estimated_probability=q,
        edge=edge,
        full_kelly_fraction=full_fraction,
        kelly_multiplier=kelly_multiplier,
        used_fraction=used_fraction,
        recommended_stake=recommended_stake,
        contracts=contracts,
        actual_stake=actual_stake,
    )
