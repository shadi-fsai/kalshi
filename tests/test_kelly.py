"""Unit tests for the Kelly sizing math."""

import math

import pytest

from kalshi.kelly import (
    better_side,
    certainty_equivalent_probability,
    kelly_for_contract,
    uncertainty_adjusted_kelly_fraction,
)


def test_uncertainty_kelly_equals_point_kelly_for_single_prob():
    # A degenerate one-value "distribution" (var=0) reproduces full Kelly.
    f = uncertainty_adjusted_kelly_fraction(
        [0.70], price_cents=50, fee_buy=0.0, fee_sell=0.0
    )
    point = kelly_for_contract(
        side="yes", price_cents=50, estimated_probability=0.70,
        bankroll=1000.0, fee_buy=0.0, fee_sell=0.0,
    )
    assert f == pytest.approx(point.full_kelly_fraction)


def test_uncertainty_kelly_shrinks_with_spread_at_equal_mean():
    # Same mean (0.70) but wider spread -> smaller fraction.
    tight = uncertainty_adjusted_kelly_fraction([0.70], price_cents=50, fee_buy=0.0)
    wide = uncertainty_adjusted_kelly_fraction(
        [0.60, 0.70, 0.80], price_cents=50, fee_buy=0.0
    )
    assert wide < tight
    wider = uncertainty_adjusted_kelly_fraction(
        [0.50, 0.70, 0.90], price_cents=50, fee_buy=0.0
    )
    assert wider < wide


def test_uncertainty_kelly_risk_aversion_zero_ignores_spread():
    f = uncertainty_adjusted_kelly_fraction(
        [0.50, 0.70, 0.90], price_cents=50, fee_buy=0.0, risk_aversion=0.0
    )
    point = uncertainty_adjusted_kelly_fraction([0.70], price_cents=50, fee_buy=0.0)
    assert f == pytest.approx(point)


def test_uncertainty_kelly_zero_when_no_edge_on_mean():
    assert (
        uncertainty_adjusted_kelly_fraction([0.30, 0.40], price_cents=50, fee_buy=0.0)
        == 0.0
    )


def test_uncertainty_kelly_zero_when_fees_swallow_spread():
    assert (
        uncertainty_adjusted_kelly_fraction([0.99], price_cents=99, fee_buy=0.02) == 0.0
    )


def test_uncertainty_kelly_empty_is_zero():
    assert uncertainty_adjusted_kelly_fraction([], price_cents=50) == 0.0


def test_uncertainty_kelly_certain_win_returns_full():
    # mean_p == 1 -> Bernoulli variance 0 -> no shrink, full fraction.
    assert uncertainty_adjusted_kelly_fraction([1.0], price_cents=50, fee_buy=0.0) == 1.0


def test_certainty_equivalent_breakeven_when_fees_swallow_spread():
    p_eff = certainty_equivalent_probability(0.5, price_cents=99, fee_buy=0.02)
    assert p_eff == pytest.approx(1.0)


def test_certainty_equivalent_round_trips_fraction():
    f = uncertainty_adjusted_kelly_fraction([0.72], price_cents=50, fee_buy=0.0)
    p_eff = certainty_equivalent_probability(f, price_cents=50, fee_buy=0.0)
    assert p_eff == pytest.approx(0.72, abs=1e-9)


def test_certainty_equivalent_zero_fraction_is_breakeven():
    p_eff = certainty_equivalent_probability(0.0, price_cents=60, fee_buy=0.01)
    assert p_eff == pytest.approx(0.61)


def _kelly(side, price_cents, est):
    return kelly_for_contract(
        side=side,
        price_cents=price_cents,
        estimated_probability=est,
        bankroll=1000.0,
        kelly_multiplier=0.5,
        fee_buy=0.0,
        fee_sell=0.0,
    )


def test_better_side_picks_yes_when_only_yes_has_edge():
    yes = _kelly("yes", 50, 0.70)  # edge
    no = _kelly("no", 50, 0.40)  # no edge
    assert better_side(yes, no) is yes


def test_better_side_picks_no_when_only_no_has_edge():
    yes = _kelly("yes", 50, 0.40)
    no = _kelly("no", 50, 0.70)
    assert better_side(yes, no) is no


def test_better_side_picks_larger_fraction_when_both_have_edge():
    yes = _kelly("yes", 50, 0.60)
    no = _kelly("no", 50, 0.80)  # bigger edge -> bigger fraction
    assert better_side(yes, no) is no


def test_better_side_none_when_neither_has_edge():
    yes = _kelly("yes", 50, 0.45)
    no = _kelly("no", 50, 0.45)
    assert better_side(yes, no) is None


def test_better_side_tie_resolves_to_yes():
    yes = _kelly("yes", 50, 0.70)
    no = _kelly("no", 50, 0.70)  # identical fraction
    assert better_side(yes, no) is yes


def test_no_edge_when_estimate_equals_price():
    # Estimate equals breakeven probability -> zero edge -> no bet (no fees).
    result = kelly_for_contract(
        side="yes",
        price_cents=40,
        estimated_probability=0.40,
        bankroll=1000.0,
        fee_buy=0.0,
        fee_sell=0.0,
    )
    assert result.edge == pytest.approx(0.0)
    assert result.full_kelly_fraction == pytest.approx(0.0)
    assert result.contracts == 0
    assert result.actual_stake == 0.0
    assert result.has_edge is False


def test_negative_edge_yields_no_bet():
    # Estimate below breakeven -> negative edge -> clamped to zero.
    result = kelly_for_contract(
        side="yes",
        price_cents=60,
        estimated_probability=0.40,
        bankroll=1000.0,
    )
    assert result.edge < 0
    assert result.full_kelly_fraction == 0.0
    assert result.contracts == 0


def test_positive_edge_matches_kelly_formula():
    # Buy YES at 50c (cost 0.5, net odds 1.0), estimate 0.60, no fees.
    # f* = q - (1-q)/b = 0.6 - 0.4/1.0 = 0.2
    result = kelly_for_contract(
        side="yes",
        price_cents=50,
        estimated_probability=0.60,
        bankroll=1000.0,
        fee_buy=0.0,
        fee_sell=0.0,
    )
    assert result.full_kelly_fraction == pytest.approx(0.20)
    assert result.recommended_stake == pytest.approx(200.0)
    # contracts = floor(stake / cost) = floor(200 / 0.5) = 400
    assert result.contracts == 400
    assert result.actual_stake == pytest.approx(200.0)


def test_fractional_kelly_multiplier_scales_stake():
    full = kelly_for_contract(
        side="yes",
        price_cents=50,
        estimated_probability=0.60,
        bankroll=1000.0,
        kelly_multiplier=1.0,
    )
    half = kelly_for_contract(
        side="yes",
        price_cents=50,
        estimated_probability=0.60,
        bankroll=1000.0,
        kelly_multiplier=0.5,
    )
    assert half.used_fraction == pytest.approx(full.used_fraction * 0.5)
    assert half.recommended_stake == pytest.approx(full.recommended_stake * 0.5)


def test_certain_win_caps_fraction_below_one():
    # q = 1.0 -> f* = q - 0/b = 1.0 (bet whole bankroll), but never exceeds 1.
    result = kelly_for_contract(
        side="yes",
        price_cents=10,
        estimated_probability=1.0,
        bankroll=1000.0,
        fee_buy=0.0,
        fee_sell=0.0,
    )
    assert result.full_kelly_fraction == pytest.approx(1.0)
    assert result.full_kelly_fraction <= 1.0
    # contracts = floor(1000 / 0.10) = 10000
    assert result.contracts == 10000


def test_no_side_uses_its_own_price():
    # Buying NO at 30c with estimate 0.50 that NO wins.
    # cost 0.3, net odds = 0.7/0.3, f* = 0.5 - 0.5/(0.7/0.3)
    result = kelly_for_contract(
        side="no",
        price_cents=30,
        estimated_probability=0.50,
        bankroll=500.0,
        fee_buy=0.0,
        fee_sell=0.0,
    )
    b = 0.7 / 0.3
    expected = 0.5 - 0.5 / b
    assert result.side == "no"
    assert result.cost_per_contract == pytest.approx(0.30)
    assert result.full_kelly_fraction == pytest.approx(expected)


def test_invalid_price_raises():
    with pytest.raises(ValueError):
        kelly_for_contract(
            side="yes",
            price_cents=0,
            estimated_probability=0.5,
            bankroll=100.0,
        )


def test_invalid_side_raises():
    with pytest.raises(ValueError):
        kelly_for_contract(
            side="maybe",
            price_cents=50,
            estimated_probability=0.5,
            bankroll=100.0,
        )


def test_invalid_multiplier_raises():
    with pytest.raises(ValueError):
        kelly_for_contract(
            side="yes",
            price_cents=50,
            estimated_probability=0.5,
            bankroll=100.0,
            kelly_multiplier=1.5,
        )


def test_zero_bankroll_yields_zero_contracts():
    result = kelly_for_contract(
        side="yes",
        price_cents=50,
        estimated_probability=0.70,
        bankroll=0.0,
    )
    assert result.full_kelly_fraction > 0
    assert result.contracts == 0
    assert math.isclose(result.actual_stake, 0.0)


def test_round_trip_fees_raise_breakeven():
    # 1c buy + 1c sell on a 50c contract -> breakeven 0.52, fee 0.02.
    result = kelly_for_contract(
        side="yes",
        price_cents=50,
        estimated_probability=0.60,
        bankroll=1000.0,
        fee_buy=0.01,
        fee_sell=0.01,
    )
    assert result.fee_per_contract == pytest.approx(0.02)
    assert result.breakeven_probability == pytest.approx(0.52)
    assert result.entry_cost_per_contract == pytest.approx(0.51)
    # Edge is measured against the fee-adjusted breakeven.
    assert result.edge == pytest.approx(0.60 - 0.52)


def test_marginal_edge_becomes_no_bet_with_fees():
    # +EV at zero fees (est 0.51 > price 0.50), but fees push breakeven to 0.52.
    no_fees = kelly_for_contract(
        side="yes",
        price_cents=50,
        estimated_probability=0.51,
        bankroll=1000.0,
        fee_buy=0.0,
        fee_sell=0.0,
    )
    assert no_fees.has_edge is True

    with_fees = kelly_for_contract(
        side="yes",
        price_cents=50,
        estimated_probability=0.51,
        bankroll=1000.0,
        fee_buy=0.01,
        fee_sell=0.01,
    )
    assert with_fees.edge < 0
    assert with_fees.full_kelly_fraction == 0.0
    assert with_fees.contracts == 0
    assert with_fees.has_edge is False


def test_contract_count_uses_entry_cost_with_buy_fee():
    # Sizing capital outlay per contract includes the buy fee.
    result = kelly_for_contract(
        side="yes",
        price_cents=50,
        estimated_probability=0.90,
        bankroll=100.0,
        kelly_multiplier=1.0,
        fee_buy=0.01,
        fee_sell=0.01,
    )
    expected = int((result.used_fraction * 100.0) / result.entry_cost_per_contract + 1e-9)
    assert result.contracts == expected
    assert result.actual_stake == pytest.approx(
        result.contracts * result.entry_cost_per_contract
    )


def test_negative_fees_raise():
    with pytest.raises(ValueError):
        kelly_for_contract(
            side="yes",
            price_cents=50,
            estimated_probability=0.5,
            bankroll=100.0,
            fee_buy=-0.01,
        )
