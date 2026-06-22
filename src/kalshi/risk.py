"""Sharpe-aware risk helpers: realized volatility and time-to-expiration.

These pure functions layer on top of the Kelly sizer (``kalshi.kelly``). For a
single binary contract held to expiry, profit per contract has mean
``edge = q - breakeven`` and standard deviation ``sqrt(q * (1 - q))``, so the
per-bet Sharpe ``edge / sqrt(q*(1-q))`` is independent of stake size. Volatility
and time therefore help in two ways:

- Selection: time-to-expiry lets us annualize the per-bet Sharpe, so a
  short-dated favorite with a small edge can be ranked against a longer-dated
  bet on a comparable (per-unit-time) basis.
- Sizing: realized price volatility since the game started, scaled over the
  remaining time, measures how uncertain the outcome still is. We turn it into a
  shrink factor in ``(0, 1]`` applied on top of fractional Kelly, so a noisy bet
  with lots of time left is sized smaller.

All prices here are in probability units (0-1), i.e. cents / 100.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

# Calendar conventions for annualizing per-bet Sharpe / edge.
MINUTES_PER_DAY = 1440.0
DAYS_PER_YEAR = 365.0
MINUTES_PER_YEAR = DAYS_PER_YEAR * MINUTES_PER_DAY  # 525600


def _ohlc_field_prob(dist: object, field: str) -> float | None:
    """One OHLC field (``open``/``high``/``low``/``close``) in probability units.

    Kalshi candlesticks express prices as fixed-point dollar strings
    (``{field}_dollars`` = "0.5600", already in 0-1 probability units) and also
    carry a legacy integer-cents ``{field}`` field. Prefer the dollar field;
    fall back to legacy cents (divided by 100). Returns ``None`` when neither is
    set or parseable.
    """
    if not isinstance(dist, dict):
        return None
    dollars = dist.get(f"{field}_dollars")
    if dollars not in (None, ""):
        try:
            return float(dollars)
        except (TypeError, ValueError):
            pass
    cents = dist.get(field)
    if cents not in (None, ""):
        try:
            return float(cents) / 100.0
        except (TypeError, ValueError):
            pass
    return None


def _ohlc_close_prob(dist: object) -> float | None:
    """Closing value of a candlestick OHLC distribution, in probability units."""
    return _ohlc_field_prob(dist, "close")


def _ohlc_high_prob(dist: object) -> float | None:
    """Highest value of a candlestick OHLC distribution, in probability units."""
    return _ohlc_field_prob(dist, "high")


def _ohlc_low_prob(dist: object) -> float | None:
    """Lowest value of a candlestick OHLC distribution, in probability units."""
    return _ohlc_field_prob(dist, "low")


def high_water_marks_cents(
    candlesticks: list[dict],
) -> tuple[float | None, float | None]:
    """Per-side high-water-marks (highest price ever reached) in cents.

    Mirrors the favorites scan's ask-per-side convention over a market's
    candlestick history:

    - YES high-water-mark = the highest YES ask reached = ``max`` of each
      candle's ``yes_ask`` high (falling back to the traded ``price`` high, then
      the ``yes_bid`` high when the ask is absent).
    - NO high-water-mark = the highest NO ask reached. NO ask is the complement
      of YES bid, so this is ``100 - min`` of each candle's ``yes_bid`` low
      (falling back to the traded ``price`` low, then the ``yes_ask`` low).

    Returns ``(yes_hwm_cents, no_hwm_cents)``; a side is ``None`` when no candle
    carries usable data for it.
    """
    yes_highs: list[float] = []
    yes_lows: list[float] = []
    for candle in candlesticks:
        if not isinstance(candle, dict):
            continue
        # YES side: prefer the ask high, then traded price, then bid.
        high = _ohlc_high_prob(candle.get("yes_ask"))
        if high is None:
            high = _ohlc_high_prob(candle.get("price"))
        if high is None:
            high = _ohlc_high_prob(candle.get("yes_bid"))
        if high is not None:
            yes_highs.append(high)
        # NO side: derived from the YES bid low (NO ask = 100 - YES bid),
        # falling back to the traded price low, then the YES ask low.
        low = _ohlc_low_prob(candle.get("yes_bid"))
        if low is None:
            low = _ohlc_low_prob(candle.get("price"))
        if low is None:
            low = _ohlc_low_prob(candle.get("yes_ask"))
        if low is not None:
            yes_lows.append(low)
    yes_hwm = max(yes_highs) * 100.0 if yes_highs else None
    no_hwm = (1.0 - min(yes_lows)) * 100.0 if yes_lows else None
    return yes_hwm, no_hwm


def ask_price_series_from_candlesticks(
    candlesticks: list[dict], side: str
) -> list[tuple[int, float]]:
    """Timestamped ask-price series (cents) for one side, oldest first.

    Mirrors the ask-per-side convention used elsewhere (what you'd pay to buy
    that side):

    - ``side == "yes"``: the ``yes_ask`` close (falling back to the traded
      ``price`` close, then the ``yes_bid`` close).
    - ``side == "no"``: the NO ask, which is the complement of the YES bid, so
      ``1 - yes_bid`` close (falling back to ``1 - price`` close, then
      ``1 - yes_ask`` close).

    Each entry is ``(end_period_ts, price_cents)``. Candles missing
    ``end_period_ts`` or any usable price are skipped.
    """
    is_yes = side == "yes"
    series: list[tuple[int, float]] = []
    for candle in candlesticks:
        if not isinstance(candle, dict):
            continue
        ts = candle.get("end_period_ts")
        if ts in (None, ""):
            continue
        try:
            ts_int = int(ts)
        except (TypeError, ValueError):
            continue
        if is_yes:
            prob = _ohlc_close_prob(candle.get("yes_ask"))
            if prob is None:
                prob = _ohlc_close_prob(candle.get("price"))
            if prob is None:
                prob = _ohlc_close_prob(candle.get("yes_bid"))
        else:
            base = _ohlc_close_prob(candle.get("yes_bid"))
            if base is None:
                base = _ohlc_close_prob(candle.get("price"))
            if base is None:
                base = _ohlc_close_prob(candle.get("yes_ask"))
            prob = None if base is None else 1.0 - base
        if prob is None:
            continue
        series.append((ts_int, prob * 100.0))
    return series


def mid_prices_from_candlesticks(candlesticks: list[dict]) -> list[float]:
    """Build a price series (probability units, 0-1) from candlesticks, oldest first.

    Uses the YES bid/ask midpoint when both are present (a continuous quote
    series is the cleanest basis for realized volatility), falling back to a
    single available side, then to the traded ``price`` close. Candles with no
    usable price are skipped.
    """
    prices: list[float] = []
    for candle in candlesticks:
        if not isinstance(candle, dict):
            continue
        bid = _ohlc_close_prob(candle.get("yes_bid"))
        ask = _ohlc_close_prob(candle.get("yes_ask"))
        if bid is not None and ask is not None:
            prices.append((bid + ask) / 2.0)
            continue
        traded = _ohlc_close_prob(candle.get("price"))
        if traded is not None:
            prices.append(traded)
        elif bid is not None:
            prices.append(bid)
        elif ask is not None:
            prices.append(ask)
    return prices


def mid_price_series_from_candlesticks(
    candlesticks: list[dict], side: str
) -> list[tuple[int, float]]:
    """Timestamped mid-price series (probability units) oriented to ``side``.

    Uses the same YES bid/ask-midpoint basis as
    :func:`mid_prices_from_candlesticks` (falling back to the traded ``price``
    close, then a single available side), but keeps each candle's
    ``end_period_ts`` and orients the value to the side held:

    - ``side == "yes"``: the YES mid (the value of a YES holding).
    - ``side == "no"``: ``1 - yes_mid`` (the value of a NO holding).

    Each entry is ``(end_period_ts, value_prob)``. Candles missing
    ``end_period_ts`` or any usable price are skipped. This makes two positions
    "move together" precisely when the bets tend to win/lose together.
    """
    is_yes = side == "yes"
    series: list[tuple[int, float]] = []
    for candle in candlesticks:
        if not isinstance(candle, dict):
            continue
        ts = candle.get("end_period_ts")
        if ts in (None, ""):
            continue
        try:
            ts_int = int(ts)
        except (TypeError, ValueError):
            continue
        bid = _ohlc_close_prob(candle.get("yes_bid"))
        ask = _ohlc_close_prob(candle.get("yes_ask"))
        if bid is not None and ask is not None:
            yes_mid: float | None = (bid + ask) / 2.0
        else:
            traded = _ohlc_close_prob(candle.get("price"))
            yes_mid = traded if traded is not None else (bid if bid is not None else ask)
        if yes_mid is None:
            continue
        value = yes_mid if is_yes else 1.0 - yes_mid
        series.append((ts_int, value))
    return series


@dataclass
class CorrelationMatrix:
    """Pairwise correlations across aligned position return series."""

    labels: list[str]
    matrix: list[list[float | None]]
    overlap: int  # number of return samples shared across all series


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    """Pearson correlation of two equal-length samples, or ``None`` if undefined.

    Undefined (returns ``None``) when there are fewer than two points or either
    series has zero variance (a constant series has no correlation).
    """
    if len(xs) < 2:
        return None
    try:
        return statistics.correlation(xs, ys)
    except statistics.StatisticsError:
        return None


def correlation_matrix(
    series_by_key: dict[str, list[tuple[int, float]]], *, min_points: int = 3
) -> CorrelationMatrix:
    """Pairwise Pearson correlation of per-step returns across positions.

    Each input series is ``(timestamp, value)`` oldest-or-any order. Series are
    aligned on the timestamps common to ALL of them, sorted ascending, then
    differenced into per-step returns; correlations are computed on those return
    vectors. The diagonal is ``1.0``. An off-diagonal entry is ``None`` when the
    shared return sample has fewer than ``min_points`` points or either side has
    zero variance. Label order follows ``series_by_key`` insertion order.
    """
    labels = list(series_by_key)
    n = len(labels)
    # Timestamps shared across every series.
    common: set[int] | None = None
    maps: dict[str, dict[int, float]] = {}
    for key in labels:
        ts_to_val = {ts: val for ts, val in series_by_key[key]}
        maps[key] = ts_to_val
        keys_set = set(ts_to_val)
        common = keys_set if common is None else (common & keys_set)
    shared_ts = sorted(common) if common else []
    # Per-step returns on the shared grid.
    returns: dict[str, list[float]] = {}
    for key in labels:
        vals = [maps[key][ts] for ts in shared_ts]
        returns[key] = [b - a for a, b in zip(vals, vals[1:])]
    overlap = len(shared_ts) - 1 if len(shared_ts) >= 1 else 0

    matrix: list[list[float | None]] = [[None] * n for _ in range(n)]
    for i in range(n):
        matrix[i][i] = 1.0
        for j in range(i + 1, n):
            ri, rj = returns[labels[i]], returns[labels[j]]
            corr = _pearson(ri, rj) if len(ri) >= min_points else None
            matrix[i][j] = corr
            matrix[j][i] = corr
    return CorrelationMatrix(labels=labels, matrix=matrix, overlap=overlap)


def high_correlation_pairs(
    result: CorrelationMatrix, threshold: float
) -> list[tuple[str, str, float]]:
    """Upper-triangle pairs with ``abs(corr) >= threshold``, strongest first.

    Returns ``(label_i, label_j, corr)`` tuples sorted by descending magnitude.
    ``None`` (undefined) correlations are skipped.
    """
    pairs: list[tuple[str, str, float]] = []
    labels, matrix = result.labels, result.matrix
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            corr = matrix[i][j]
            if corr is not None and abs(corr) >= threshold:
                pairs.append((labels[i], labels[j], corr))
    pairs.sort(key=lambda p: abs(p[2]), reverse=True)
    return pairs


def realized_volatility(prices: list[float], dt_minutes: float) -> float | None:
    """Per-day realized volatility of a price series in probability units.

    Computes the standard deviation of successive price increments (not log
    returns: the price is a bounded probability, so additive increments are the
    natural scale) and annualizes it to a per-day figure assuming each step
    spans ``dt_minutes``.

    Args:
        prices: Mid prices in probability units (0-1), oldest first.
        dt_minutes: Minutes between consecutive samples (the candlestick
            period). Must be positive.

    Returns:
        Per-day volatility (probability units), or ``None`` when there are too
        few points (< 3 increments) or ``dt_minutes`` is non-positive.
    """
    if dt_minutes <= 0:
        return None
    increments = [b - a for a, b in zip(prices, prices[1:])]
    if len(increments) < 2:
        return None
    per_step = statistics.stdev(increments)
    steps_per_day = MINUTES_PER_DAY / dt_minutes
    return per_step * math.sqrt(steps_per_day)


def sigma_remaining(
    vol_per_day: float, time_to_expiry_days: float, win_prob: float
) -> float:
    """Expected residual price dispersion over the remaining time, capped.

    Scales the per-day volatility by ``sqrt(T_days)`` (diffusive accumulation)
    and caps it at the terminal binary bound ``sqrt(q*(1-q))`` - a contract's
    price can never be more uncertain than its own settlement variance.
    """
    if vol_per_day <= 0 or time_to_expiry_days <= 0:
        return 0.0
    raw = vol_per_day * math.sqrt(time_to_expiry_days)
    terminal_bound = math.sqrt(max(0.0, win_prob * (1.0 - win_prob)))
    return min(raw, terminal_bound) if terminal_bound > 0 else raw


def volatility_time_multiplier(
    *, edge: float, sigma_remaining: float, sensitivity: float = 1.0
) -> float:
    """Fractional-Kelly shrink factor in ``(0, 1]`` from remaining volatility.

    ``edge / (edge + sensitivity * sigma_remaining)``: with no remaining
    volatility the factor is 1 (full Kelly); as the residual dispersion grows
    relative to the edge, the factor shrinks toward 0. Returns ``0.0`` when the
    edge is non-positive (no bet).

    Args:
        edge: Net edge per contract (``q - breakeven``), in probability units.
        sigma_remaining: Residual price dispersion over the remaining time.
        sensitivity: Scales how aggressively volatility shrinks the stake
            (0 disables the adjustment -> factor 1.0). Must be non-negative.
    """
    if sensitivity < 0:
        raise ValueError(f"sensitivity must be non-negative (got {sensitivity}).")
    if edge <= 0:
        return 0.0
    if sigma_remaining <= 0 or sensitivity == 0:
        return 1.0
    return edge / (edge + sensitivity * sigma_remaining)


@dataclass
class SharpeMetrics:
    """Per-bet and time-annualized Sharpe metrics for a binary contract."""

    edge: float
    win_prob: float
    terminal_sigma: float
    sharpe_terminal: float
    time_to_expiry_minutes: float
    time_to_expiry_days: float
    sharpe_annualized: float
    edge_per_day: float


def sharpe_metrics(
    *,
    edge: float,
    win_prob: float,
    time_to_expiry_minutes: float | None = None,
    time_to_expiry_days: float | None = None,
) -> SharpeMetrics:
    """Compute per-bet and annualized Sharpe for a hold-to-expiry binary bet.

    ``sharpe_terminal = edge / sqrt(q*(1-q))`` is the per-bet Sharpe. Annualizing
    assumes capital recycles roughly once per time-to-expiry, giving
    ``sharpe_terminal * sqrt(525600 / T_minutes)``. The horizon is measured in
    MINUTES so that intraday markets (e.g. a soccer first half that settles in
    ~45 min) are annualized from their true remaining distance rather than a
    rounded number of days.

    Pass exactly one of ``time_to_expiry_minutes`` (preferred, minute-resolution)
    or ``time_to_expiry_days`` (kept for convenience; converted to minutes).

    Degenerate inputs (``q`` at 0 or 1, ``T <= 0``) yield zeroed Sharpe figures
    rather than raising, so the UI can display them uniformly.
    """
    if time_to_expiry_minutes is None and time_to_expiry_days is None:
        raise ValueError(
            "pass either time_to_expiry_minutes or time_to_expiry_days."
        )
    if time_to_expiry_minutes is None:
        time_to_expiry_minutes = (time_to_expiry_days or 0.0) * MINUTES_PER_DAY
    t_minutes = time_to_expiry_minutes
    t_days = t_minutes / MINUTES_PER_DAY

    q = win_prob
    terminal_sigma = math.sqrt(max(0.0, q * (1.0 - q)))
    if terminal_sigma > 0:
        sharpe_terminal = edge / terminal_sigma
    elif edge == 0:
        # No edge and no settlement variance: genuinely 0 (and undefined-ish),
        # treat as zero rather than infinite.
        sharpe_terminal = 0.0
    else:
        # q at the 0/1 boundary => zero settlement variance, so the per-bet
        # Sharpe (edge / sigma) diverges. A positive edge on a "certain"
        # outcome is +inf (a sure win); a negative edge is -inf. Returning the
        # signed infinity (instead of clamping to 0.0) lets the UI show this as
        # an off-the-chart bet rather than a worthless one.
        sharpe_terminal = math.copysign(math.inf, edge)
    if t_minutes > 0:
        sharpe_annualized = sharpe_terminal * math.sqrt(
            MINUTES_PER_YEAR / t_minutes
        )
        edge_per_day = edge / t_days
    else:
        sharpe_annualized = 0.0
        edge_per_day = 0.0
    return SharpeMetrics(
        edge=edge,
        win_prob=q,
        terminal_sigma=terminal_sigma,
        sharpe_terminal=sharpe_terminal,
        time_to_expiry_minutes=t_minutes,
        time_to_expiry_days=t_days,
        sharpe_annualized=sharpe_annualized,
        edge_per_day=edge_per_day,
    )
