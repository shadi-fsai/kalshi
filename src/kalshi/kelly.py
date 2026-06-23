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
from collections.abc import Sequence
from dataclasses import dataclass


def _net_odds(price_cents: float, fee_buy: float, fee_sell: float) -> float | None:
    """Fee-adjusted net decimal odds (profit per dollar risked), or None if no bet.

    Mirrors :func:`kelly_for_contract`: ``net_win = 1 - cost - fees`` and
    ``net_loss = cost + fees``. Returns ``None`` when fees swallow the spread.
    """
    cost = price_cents / 100.0
    fee = fee_buy + fee_sell
    net_win = 1.0 - cost - fee
    net_loss = cost + fee
    if net_win <= 0.0 or net_loss <= 0.0:
        return None
    return net_win / net_loss


def _full_kelly(q: float, net_odds: float) -> float:
    """Full-Kelly fraction for win prob ``q`` at ``net_odds`` (clamped at 0)."""
    return max(0.0, q - (1.0 - q) / net_odds)


def uncertainty_adjusted_kelly_fraction(
    win_probs: Sequence[float],
    *,
    price_cents: float,
    fee_buy: float = 0.01,
    fee_sell: float = 0.0,
    risk_aversion: float = 1.0,
) -> float:
    """Full-Kelly fraction shrunk for the spread of a win-probability distribution.

    ``win_probs`` is a sample of the win probability for the side being bought
    (e.g. one value per Monte Carlo scenario that perturbs the model inputs).

    Sizing is the standard full-Kelly on the mean probability times a shrink
    factor driven by the distribution's variance:

        f = f_full(mean_p) * mean_p(1 - mean_p) / (mean_p(1 - mean_p) + k * var_p)

    Rationale: for a one-shot binary bet, expected LOG-growth is linear in ``p``,
    so it depends only on the mean and parameter uncertainty would cancel. To let
    the range matter we use the mean-variance view of Kelly (``f ~ edge /
    variance``): by the law of total variance the predictive variance of the
    bet's return is inflated by ``var_p`` on top of the Bernoulli ``p(1-p)``,
    which shrinks the stake. ``risk_aversion`` (k) scales how strongly the spread
    shrinks the bet (0 = ignore uncertainty, 1 = full law-of-total-variance).
    The factor is 1.0 when the distribution is a point (``var_p == 0``) and falls
    toward 0 as the spread grows. Returns 0.0 when there is no edge on the mean
    or fees swallow the spread.
    """
    if not win_probs:
        return 0.0
    net_odds = _net_odds(price_cents, fee_buy, fee_sell)
    if net_odds is None:
        return 0.0
    mean_p = sum(win_probs) / len(win_probs)
    f_full = _full_kelly(mean_p, net_odds)
    if f_full <= 0.0:
        return 0.0
    bernoulli_var = mean_p * (1.0 - mean_p)
    if bernoulli_var <= 0.0:
        return f_full
    var_p = sum((p - mean_p) ** 2 for p in win_probs) / len(win_probs)
    shrink = bernoulli_var / (bernoulli_var + max(0.0, risk_aversion) * var_p)
    return f_full * shrink


def certainty_equivalent_probability(
    fraction: float,
    *,
    price_cents: float,
    fee_buy: float = 0.01,
    fee_sell: float = 0.0,
) -> float:
    """Win probability whose full-Kelly fraction equals ``fraction``.

    Inverts :func:`_full_kelly` so an uncertainty-adjusted fraction can be fed
    back through :func:`kelly_for_contract` as a single "certainty-equivalent"
    probability. At ``fraction == 0`` this is the fee-adjusted breakeven; it
    rises toward 1 as the fraction grows. Returns the breakeven when fees swallow
    the spread (no bet possible).
    """
    cost = price_cents / 100.0
    fee = fee_buy + fee_sell
    breakeven = min(1.0, cost + fee)
    net_odds = _net_odds(price_cents, fee_buy, fee_sell)
    if net_odds is None:
        return breakeven
    inv = 1.0 / net_odds
    p = (max(0.0, fraction) + inv) / (1.0 + inv)
    return min(1.0, max(breakeven, p))


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


def better_side(yes: KellyResult, no: KellyResult) -> KellyResult | None:
    """The side worth betting: the one with the larger positive Kelly fraction.

    Returns ``None`` when neither side has a positive edge. Ties (equal positive
    fractions) resolve to ``yes`` for determinism.
    """
    candidates = [r for r in (yes, no) if r.has_edge]
    if not candidates:
        return None
    return max(candidates, key=lambda r: r.used_fraction)


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
