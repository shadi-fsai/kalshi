"""Tests for kalshi.risk: realized volatility, Sharpe metrics, vol shrink."""

from __future__ import annotations

import math

import pytest

from kalshi.risk import (
    ask_price_series_from_candlesticks,
    correlation_matrix,
    high_correlation_pairs,
    high_water_marks_cents,
    mid_price_series_from_candlesticks,
    mid_prices_from_candlesticks,
    realized_volatility,
    sharpe_metrics,
    sigma_remaining,
    volatility_time_multiplier,
)

# --- mid_prices_from_candlesticks ----------------------------------------


def test_mid_prices_from_dollar_bid_ask():
    # Real Kalshi shape: OHLC distributions keyed by *_dollars strings.
    candles = [
        {
            "yes_bid": {"close_dollars": "0.7600"},
            "yes_ask": {"close_dollars": "0.8000"},
            "price": {"previous_dollars": "0.7800"},
        },
        {
            "yes_bid": {"close_dollars": "0.7400"},
            "yes_ask": {"close_dollars": "0.7800"},
            "price": {"close_dollars": "0.7500"},
        },
    ]
    assert mid_prices_from_candlesticks(candles) == [0.78, 0.76]


def test_mid_prices_prefers_bid_ask_mid_over_traded():
    candles = [
        {
            "yes_bid": {"close_dollars": "0.40"},
            "yes_ask": {"close_dollars": "0.50"},
            "price": {"close_dollars": "0.99"},
        }
    ]
    assert mid_prices_from_candlesticks(candles) == [0.45]


def test_mid_prices_falls_back_to_traded_then_single_side():
    candles = [
        {"price": {"close_dollars": "0.33"}},  # only traded price
        {"yes_bid": {"close_dollars": "0.20"}},  # only bid
        {"yes_ask": {"close_dollars": "0.90"}},  # only ask
    ]
    assert mid_prices_from_candlesticks(candles) == [0.33, 0.20, 0.90]


def test_mid_prices_legacy_integer_cents():
    candles = [
        {"yes_bid": {"close": 40}, "yes_ask": {"close": 60}},
    ]
    assert mid_prices_from_candlesticks(candles) == [0.50]


def test_mid_prices_skips_empty_candles():
    candles = [
        {"price": {"previous_dollars": "0.50"}},  # no close anywhere -> skip
        {"yes_bid": {"close_dollars": "0.30"}, "yes_ask": {"close_dollars": "0.34"}},
        "not-a-dict",
    ]
    assert mid_prices_from_candlesticks(candles) == [0.32]

# --- high_water_marks_cents ----------------------------------------------


def test_high_water_marks_from_dollar_high_low():
    # YES hwm = max yes_ask high; NO hwm = 100 - min yes_bid low.
    candles = [
        {
            "yes_bid": {"high_dollars": "0.60", "low_dollars": "0.40"},
            "yes_ask": {"high_dollars": "0.66", "low_dollars": "0.45"},
        },
        {
            "yes_bid": {"high_dollars": "0.72", "low_dollars": "0.30"},
            "yes_ask": {"high_dollars": "0.80", "low_dollars": "0.55"},
        },
    ]
    yes_hwm, no_hwm = high_water_marks_cents(candles)
    assert yes_hwm == pytest.approx(80.0)  # max(0.66, 0.80) * 100
    assert no_hwm == pytest.approx(70.0)  # (1 - min(0.40, 0.30)) * 100


def test_high_water_marks_legacy_integer_cents():
    candles = [{"yes_bid": {"high": 60, "low": 35}, "yes_ask": {"high": 64, "low": 40}}]
    yes_hwm, no_hwm = high_water_marks_cents(candles)
    assert yes_hwm == pytest.approx(64.0)
    assert no_hwm == pytest.approx(65.0)  # 100 - 35


def test_high_water_marks_falls_back_to_traded_price():
    # No bid/ask OHLC: YES high from price.high, NO from price.low complement.
    candles = [{"price": {"high_dollars": "0.97", "low_dollars": "0.12"}}]
    yes_hwm, no_hwm = high_water_marks_cents(candles)
    assert yes_hwm == pytest.approx(97.0)
    assert no_hwm == pytest.approx(88.0)  # 100 - 12


def test_high_water_marks_none_when_no_data():
    assert high_water_marks_cents([]) == (None, None)
    assert high_water_marks_cents(["not-a-dict", {"price": {}}]) == (None, None)


# --- ask_price_series_from_candlesticks ----------------------------------


def test_ask_price_series_yes_uses_yes_ask_close():
    candles = [
        {"end_period_ts": 100, "yes_ask": {"close_dollars": "0.80"}},
        {"end_period_ts": 160, "yes_ask": {"close_dollars": "0.82"}},
    ]
    assert ask_price_series_from_candlesticks(candles, "yes") == [
        (100, pytest.approx(80.0)),
        (160, pytest.approx(82.0)),
    ]


def test_ask_price_series_no_is_complement_of_yes_bid():
    # NO ask = 1 - YES bid, in cents.
    candles = [{"end_period_ts": 100, "yes_bid": {"close_dollars": "0.30"}}]
    series = ask_price_series_from_candlesticks(candles, "no")
    assert series == [(100, pytest.approx(70.0))]


def test_ask_price_series_falls_back_to_traded_price():
    # YES: no ask -> traded price close. NO: no bid -> 1 - traded price close.
    candles = [{"end_period_ts": 100, "price": {"close_dollars": "0.40"}}]
    assert ask_price_series_from_candlesticks(candles, "yes") == [
        (100, pytest.approx(40.0))
    ]
    assert ask_price_series_from_candlesticks(candles, "no") == [
        (100, pytest.approx(60.0))
    ]


def test_ask_price_series_no_falls_back_to_yes_ask():
    # NO side, only yes_ask present: NO ask = 1 - yes_ask close.
    candles = [{"end_period_ts": 100, "yes_ask": {"close_dollars": "0.85"}}]
    assert ask_price_series_from_candlesticks(candles, "no") == [
        (100, pytest.approx(15.0))
    ]


def test_ask_price_series_legacy_cents_and_skips():
    candles = [
        {"yes_ask": {"close": 55}},  # no end_period_ts -> skip
        {"end_period_ts": "bad", "yes_ask": {"close": 55}},  # bad ts -> skip
        {"end_period_ts": 200, "price": {}},  # no usable price -> skip
        "not-a-dict",
        {"end_period_ts": 300, "yes_ask": {"close": 55}},  # legacy cents -> 55c
    ]
    assert ask_price_series_from_candlesticks(candles, "yes") == [
        (300, pytest.approx(55.0))
    ]


# --- realized_volatility -------------------------------------------------


def test_realized_volatility_constant_series_is_zero():
    assert realized_volatility([0.5, 0.5, 0.5, 0.5], 60.0) == 0.0


def test_realized_volatility_scales_to_per_day():
    # Increments alternate +0.1 / -0.1 -> stdev of increments is 0.1 (sample).
    prices = [0.5, 0.6, 0.5, 0.6, 0.5]
    increments = [0.1, -0.1, 0.1, -0.1]
    import statistics

    expected_step = statistics.stdev(increments)
    vol = realized_volatility(prices, 60.0)
    assert vol is not None
    # 60-min periods -> 24 steps/day.
    assert vol == pytest.approx(expected_step * math.sqrt(24.0))


def test_realized_volatility_too_few_points():
    assert realized_volatility([0.5], 60.0) is None
    assert realized_volatility([0.5, 0.6], 60.0) is None  # only 1 increment


def test_realized_volatility_bad_dt():
    assert realized_volatility([0.5, 0.6, 0.7], 0.0) is None
    assert realized_volatility([0.5, 0.6, 0.7], -5.0) is None


def test_realized_volatility_shorter_period_higher_per_day():
    prices = [0.5, 0.55, 0.5, 0.55, 0.5]
    vol_1m = realized_volatility(prices, 1.0)
    vol_60m = realized_volatility(prices, 60.0)
    assert vol_1m > vol_60m  # same steps, more of them per day at 1-min


# --- sigma_remaining -----------------------------------------------------


def test_sigma_remaining_scales_with_sqrt_time():
    # Low per-day vol so the terminal cap doesn't bind.
    one_day = sigma_remaining(0.05, 1.0, 0.5)
    four_days = sigma_remaining(0.05, 4.0, 0.5)
    assert four_days == pytest.approx(2.0 * one_day)


def test_sigma_remaining_capped_at_terminal_bound():
    # Huge vol is capped at sqrt(q(1-q)).
    capped = sigma_remaining(10.0, 5.0, 0.5)
    assert capped == pytest.approx(0.5)  # sqrt(0.25)


def test_sigma_remaining_zero_when_no_time_or_vol():
    assert sigma_remaining(0.0, 5.0, 0.5) == 0.0
    assert sigma_remaining(0.1, 0.0, 0.5) == 0.0


# --- volatility_time_multiplier ------------------------------------------


def test_vol_multiplier_one_when_no_remaining_vol():
    assert volatility_time_multiplier(edge=0.1, sigma_remaining=0.0) == 1.0


def test_vol_multiplier_zero_when_no_edge():
    assert volatility_time_multiplier(edge=0.0, sigma_remaining=0.2) == 0.0
    assert volatility_time_multiplier(edge=-0.05, sigma_remaining=0.2) == 0.0


def test_vol_multiplier_in_unit_interval_and_monotonic():
    m_low = volatility_time_multiplier(edge=0.1, sigma_remaining=0.1)
    m_high = volatility_time_multiplier(edge=0.1, sigma_remaining=0.4)
    assert 0.0 < m_high < m_low < 1.0  # more remaining vol -> smaller stake


def test_vol_multiplier_formula():
    # edge / (edge + sensitivity * sigma)
    m = volatility_time_multiplier(edge=0.1, sigma_remaining=0.2, sensitivity=1.0)
    assert m == pytest.approx(0.1 / (0.1 + 0.2))


def test_vol_multiplier_sensitivity_zero_disables():
    assert volatility_time_multiplier(
        edge=0.1, sigma_remaining=0.5, sensitivity=0.0
    ) == 1.0


def test_vol_multiplier_higher_sensitivity_smaller():
    m1 = volatility_time_multiplier(edge=0.1, sigma_remaining=0.2, sensitivity=1.0)
    m2 = volatility_time_multiplier(edge=0.1, sigma_remaining=0.2, sensitivity=2.0)
    assert m2 < m1


def test_vol_multiplier_negative_sensitivity_raises():
    with pytest.raises(ValueError, match="sensitivity"):
        volatility_time_multiplier(edge=0.1, sigma_remaining=0.2, sensitivity=-1.0)


# --- sharpe_metrics ------------------------------------------------------


def test_sharpe_terminal_formula():
    sm = sharpe_metrics(edge=0.05, win_prob=0.5, time_to_expiry_days=1.0)
    assert sm.terminal_sigma == pytest.approx(0.5)
    assert sm.sharpe_terminal == pytest.approx(0.05 / 0.5)


def test_sharpe_annualized_and_edge_per_day():
    sm = sharpe_metrics(edge=0.04, win_prob=0.5, time_to_expiry_days=4.0)
    assert sm.sharpe_annualized == pytest.approx(
        sm.sharpe_terminal * math.sqrt(365.0 / 4.0)
    )
    assert sm.edge_per_day == pytest.approx(0.04 / 4.0)


def test_sharpe_shorter_dated_higher_annualized():
    base = sharpe_metrics(edge=0.03, win_prob=0.6, time_to_expiry_days=7.0)
    short = sharpe_metrics(edge=0.03, win_prob=0.6, time_to_expiry_days=0.5)
    assert short.sharpe_annualized > base.sharpe_annualized


def test_sharpe_degenerate_probability():
    sm = sharpe_metrics(edge=0.0, win_prob=1.0, time_to_expiry_days=1.0)
    assert sm.terminal_sigma == 0.0
    assert sm.sharpe_terminal == 0.0
    assert sm.sharpe_annualized == 0.0


def test_sharpe_certainty_with_positive_edge_is_infinite():
    # A 100% estimate => zero settlement variance, so a positive-edge bet has a
    # diverging (off-the-chart) Sharpe rather than a clamped 0.0.
    sm = sharpe_metrics(edge=0.009, win_prob=1.0, time_to_expiry_days=1.0)
    assert sm.terminal_sigma == 0.0
    assert sm.sharpe_terminal == math.inf
    assert sm.sharpe_annualized == math.inf


def test_sharpe_certainty_with_negative_edge_is_negative_infinite():
    sm = sharpe_metrics(edge=-0.02, win_prob=0.0, time_to_expiry_days=1.0)
    assert sm.terminal_sigma == 0.0
    assert sm.sharpe_terminal == -math.inf
    assert sm.sharpe_annualized == -math.inf


def test_sharpe_zero_time_to_expiry():
    sm = sharpe_metrics(edge=0.05, win_prob=0.5, time_to_expiry_days=0.0)
    assert sm.sharpe_annualized == 0.0
    assert sm.edge_per_day == 0.0
    # Per-bet Sharpe is still defined.
    assert sm.sharpe_terminal == pytest.approx(0.1)


def test_sharpe_minutes_matches_days_for_same_horizon():
    # Minute and day inputs describing the same horizon must agree exactly.
    by_min = sharpe_metrics(edge=0.04, win_prob=0.5, time_to_expiry_minutes=4 * 1440)
    by_day = sharpe_metrics(edge=0.04, win_prob=0.5, time_to_expiry_days=4.0)
    assert by_min.sharpe_annualized == pytest.approx(by_day.sharpe_annualized)
    assert by_min.edge_per_day == pytest.approx(by_day.edge_per_day)
    assert by_min.time_to_expiry_days == pytest.approx(4.0)
    assert by_min.time_to_expiry_minutes == pytest.approx(5760.0)


def test_sharpe_minute_resolution_subhour():
    # A 45-minute horizon annualizes from its true distance, not a rounded day.
    sm = sharpe_metrics(edge=0.03, win_prob=0.6, time_to_expiry_minutes=45.0)
    assert sm.time_to_expiry_minutes == pytest.approx(45.0)
    expected = sm.sharpe_terminal * math.sqrt(525600.0 / 45.0)
    assert sm.sharpe_annualized == pytest.approx(expected)
    # Distinct from naively rounding the horizon to one day.
    one_day = sharpe_metrics(edge=0.03, win_prob=0.6, time_to_expiry_minutes=1440.0)
    assert sm.sharpe_annualized > one_day.sharpe_annualized


def test_sharpe_requires_a_horizon():
    with pytest.raises(ValueError, match="time_to_expiry"):
        sharpe_metrics(edge=0.03, win_prob=0.5)


# --- mid_price_series_from_candlesticks ----------------------------------


def test_mid_price_series_yes_uses_midpoint():
    candles = [
        {
            "end_period_ts": 100,
            "yes_bid": {"close_dollars": "0.40"},
            "yes_ask": {"close_dollars": "0.60"},
        },
        {
            "end_period_ts": 160,
            "yes_bid": {"close_dollars": "0.50"},
            "yes_ask": {"close_dollars": "0.70"},
        },
    ]
    assert mid_price_series_from_candlesticks(candles, "yes") == [
        (100, pytest.approx(0.50)),
        (160, pytest.approx(0.60)),
    ]


def test_mid_price_series_no_is_complement():
    candles = [
        {
            "end_period_ts": 100,
            "yes_bid": {"close_dollars": "0.40"},
            "yes_ask": {"close_dollars": "0.60"},
        }
    ]
    # YES mid is 0.50, so the NO holding value is 1 - 0.50 = 0.50 here.
    assert mid_price_series_from_candlesticks(candles, "no") == [(100, pytest.approx(0.50))]
    # Asymmetric quote: YES mid 0.70 -> NO value 0.30.
    candles2 = [
        {
            "end_period_ts": 100,
            "yes_bid": {"close_dollars": "0.60"},
            "yes_ask": {"close_dollars": "0.80"},
        }
    ]
    assert mid_price_series_from_candlesticks(candles2, "no") == [
        (100, pytest.approx(0.30))
    ]


def test_mid_price_series_skips_unusable_and_falls_back_to_price():
    candles = [
        {"yes_bid": {"close": 50}},  # no end_period_ts -> skip
        {"end_period_ts": "bad", "price": {"close_dollars": "0.5"}},  # bad ts -> skip
        {"end_period_ts": 200, "price": {}},  # no usable price -> skip
        "not-a-dict",
        {"end_period_ts": 300, "price": {"close_dollars": "0.40"}},  # traded fallback
    ]
    assert mid_price_series_from_candlesticks(candles, "yes") == [
        (300, pytest.approx(0.40))
    ]


# --- correlation_matrix / high_correlation_pairs -------------------------


def test_correlation_matrix_perfectly_correlated():
    a = [(1, 0.10), (2, 0.20), (3, 0.10), (4, 0.30)]  # returns +.1 -.1 +.2
    b = [(1, 0.50), (2, 0.60), (3, 0.50), (4, 0.70)]  # identical returns
    result = correlation_matrix({"A": a, "B": b})
    assert result.labels == ["A", "B"]
    assert result.matrix[0][0] == 1.0
    assert result.matrix[0][1] == pytest.approx(1.0)
    assert result.matrix[1][0] == pytest.approx(1.0)
    assert result.overlap == 3


def test_correlation_matrix_anti_correlated():
    a = [(1, 0.10), (2, 0.20), (3, 0.10), (4, 0.20)]  # returns +.1 -.1 +.1
    b = [(1, 0.80), (2, 0.70), (3, 0.80), (4, 0.70)]  # returns -.1 +.1 -.1
    result = correlation_matrix({"A": a, "B": b})
    assert result.matrix[0][1] == pytest.approx(-1.0)


def test_correlation_matrix_insufficient_overlap_is_none():
    # Only two shared timestamps -> one return sample, below min_points.
    a = [(1, 0.10), (2, 0.20)]
    b = [(1, 0.50), (2, 0.70)]
    result = correlation_matrix({"A": a, "B": b})
    assert result.matrix[0][1] is None
    assert result.overlap == 1


def test_correlation_matrix_constant_series_is_none():
    a = [(1, 0.10), (2, 0.20), (3, 0.30), (4, 0.40)]
    flat = [(1, 0.50), (2, 0.50), (3, 0.50), (4, 0.50)]  # zero variance returns
    result = correlation_matrix({"A": a, "FLAT": flat})
    assert result.matrix[0][1] is None


def test_correlation_matrix_aligns_on_common_timestamps():
    # B is missing ts=2; only 1,3,4,5 are shared -> returns over those.
    a = [(1, 0.10), (2, 0.99), (3, 0.30), (4, 0.20), (5, 0.40)]
    b = [(1, 0.50), (3, 0.70), (4, 0.60), (5, 0.80)]
    result = correlation_matrix({"A": a, "B": b})
    assert result.overlap == 3
    assert result.matrix[0][1] == pytest.approx(1.0)


def test_high_correlation_pairs_threshold_and_order():
    result = correlation_matrix(
        {
            "A": [(1, 0.1), (2, 0.2), (3, 0.1), (4, 0.3)],  # returns +.1 -.1 +.2
            "B": [(1, 0.5), (2, 0.6), (3, 0.5), (4, 0.7)],  # corr ~ +1 with A
            "C": [(1, 0.9), (2, 0.8), (3, 0.9), (4, 0.7)],  # corr ~ -1 with A and B
        }
    )
    pairs = high_correlation_pairs(result, threshold=0.7)
    # All three pairs are |1.0|; just assert they are all surfaced and sorted.
    assert len(pairs) == 3
    assert all(abs(c) >= 0.7 for _, _, c in pairs)
    assert pairs == sorted(pairs, key=lambda p: abs(p[2]), reverse=True)

    # A high threshold above the achievable magnitude yields nothing.
    assert high_correlation_pairs(result, threshold=1.01) == []
