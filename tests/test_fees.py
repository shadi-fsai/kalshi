"""Unit tests for Kalshi fee calculation."""

import pytest

from kalshi.fees import FeeModel, fee_model_from_series


def test_quadratic_order_fee_rounds_up_to_cent():
    model = FeeModel(fee_type="quadratic", fee_multiplier=1.0)
    # 1 contract @ $0.50: 0.07 * 0.5 * 0.5 = 0.0175 -> rounds up to $0.02.
    assert model.order_fee(1, 0.50) == pytest.approx(0.02)


def test_quadratic_order_fee_scales_with_count():
    model = FeeModel(fee_type="quadratic", fee_multiplier=1.0)
    # 100 contracts @ $0.50: 0.07 * 100 * 0.25 = 1.75 (already whole cents).
    assert model.order_fee(100, 0.50) == pytest.approx(1.75)
    # 100 contracts @ $0.90: 0.07 * 100 * 0.9 * 0.1 = 0.63.
    assert model.order_fee(100, 0.90) == pytest.approx(0.63)


def test_multiplier_scales_fee():
    base = FeeModel(fee_type="quadratic", fee_multiplier=1.0)
    doubled = FeeModel(fee_type="quadratic", fee_multiplier=2.0)
    assert doubled.order_fee(100, 0.50) == pytest.approx(2 * base.order_fee(100, 0.50))


def test_per_contract_fee_is_marginal_pre_roundup():
    model = FeeModel(fee_type="quadratic", fee_multiplier=1.0)
    assert model.per_contract_fee(0.50) == pytest.approx(0.0175)
    assert model.per_contract_fee(0.90) == pytest.approx(0.0063)


def test_maker_type_is_treated_as_quadratic():
    model = FeeModel(fee_type="quadratic_with_maker_fees", fee_multiplier=1.0)
    assert model.is_quadratic is True
    assert model.order_fee(100, 0.50) == pytest.approx(1.75)


def test_flat_and_unknown_return_none():
    assert FeeModel("flat", 1.0).order_fee(100, 0.5) is None
    assert FeeModel("flat", 1.0).per_contract_fee(0.5) is None
    assert FeeModel("mystery", 1.0).order_fee(100, 0.5) is None


def test_zero_or_extreme_inputs():
    model = FeeModel(fee_type="quadratic", fee_multiplier=1.0)
    assert model.order_fee(0, 0.5) == 0.0
    assert model.order_fee(100, 1.0) == 0.0  # no fee at fully-priced
    assert model.per_contract_fee(0.0) == 0.0


def test_fee_model_from_series():
    model = fee_model_from_series({"fee_type": "quadratic", "fee_multiplier": 1})
    assert model == FeeModel("quadratic", 1.0)
    assert fee_model_from_series({}) is None
    assert fee_model_from_series({"fee_type": "quadratic"}) is None
