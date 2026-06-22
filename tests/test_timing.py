"""Tests for kalshi.markets timing helpers (parse_ts + classifiers).

Uses freezegun so "now" is deterministic; the classifiers take ``now`` as an
argument, which we read from a frozen clock.
"""

from __future__ import annotations

import datetime

from freezegun import freeze_time

from kalshi.markets import (
    classify_resolution,
    classify_timing,
    parse_ts,
    resolution_time,
)

NOW_ISO = "2026-06-20T18:00:00Z"


def _now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def _at(minutes: float) -> datetime.datetime:
    """A datetime ``minutes`` from the frozen now (negative = in the past)."""
    return _now() + datetime.timedelta(minutes=minutes)


# --- parse_ts ------------------------------------------------------------


def test_parse_ts_handles_z_suffix():
    dt = parse_ts("2026-06-20T18:00:00Z")
    assert dt is not None
    assert dt.tzinfo is not None
    assert dt.year == 2026 and dt.hour == 18


def test_parse_ts_none_and_invalid():
    assert parse_ts(None) is None
    assert parse_ts("") is None
    assert parse_ts("not-a-date") is None


# --- resolution_time -----------------------------------------------------


def test_resolution_time_prefers_expected_expiration():
    market = {
        "expected_expiration_time": "2026-06-21T19:00:00Z",
        # Tournament-wide close (e.g. the final's date) must be ignored.
        "close_time": "2026-07-05T16:00:00Z",
        "expiration_time": "2026-07-05T16:00:00Z",
    }
    dt = resolution_time(market)
    assert dt is not None
    assert (dt.month, dt.day, dt.hour) == (6, 21, 19)


def test_resolution_time_ignores_close_time_when_expected_missing():
    # No expected_expiration_time => unknown, never the series-wide close.
    market = {"close_time": "2026-07-05T16:00:00Z"}
    assert resolution_time(market) is None


def test_resolution_time_empty_market():
    assert resolution_time({}) is None


# --- classify_timing -----------------------------------------------------


@freeze_time(NOW_ISO)
def test_classify_timing_none_info():
    assert classify_timing(None, _now()) is None
    assert classify_timing({"status": "x"}, _now()) is None  # no start


@freeze_time(NOW_ISO)
def test_classify_timing_finished_status():
    info = {"start": _at(-30), "status": "FT"}
    assert classify_timing(info, _now()) == ("finished", "ended")


@freeze_time(NOW_ISO)
def test_classify_timing_live():
    info = {"start": _at(-20), "status": "live"}
    state, label = classify_timing(info, _now())
    assert state == "live"
    assert "started 20m ago" in label


@freeze_time(NOW_ISO)
def test_classify_timing_soon():
    info = {"start": _at(30), "status": None}
    state, label = classify_timing(info, _now())
    assert state == "soon"
    assert "starts in 30m" in label


@freeze_time(NOW_ISO)
def test_classify_timing_later_hours():
    info = {"start": _at(180), "status": None}  # 3h ahead
    state, label = classify_timing(info, _now())
    assert state == "later"
    assert "h" in label


@freeze_time(NOW_ISO)
def test_classify_timing_later_far_future():
    info = {"start": _at(48 * 60), "status": None}  # 2 days ahead
    state, _label = classify_timing(info, _now())
    assert state == "later"


# --- classify_resolution -------------------------------------------------


@freeze_time(NOW_ISO)
def test_classify_resolution_none():
    assert classify_resolution(None, _now()) is None


@freeze_time(NOW_ISO)
def test_classify_resolution_resolving_now():
    assert classify_resolution(_at(-5), _now()) == ("resolving", "resolving now")


@freeze_time(NOW_ISO)
def test_classify_resolution_ending_minutes():
    state, label = classify_resolution(_at(40), _now())
    assert state == "ending"
    assert "ends in 40m" in label


@freeze_time(NOW_ISO)
def test_classify_resolution_ending_hours():
    state, label = classify_resolution(_at(90), _now())  # within 2h window
    assert state == "ending"
    assert "h" in label


@freeze_time(NOW_ISO)
def test_classify_resolution_later():
    state, _label = classify_resolution(_at(5 * 60), _now())
    assert state == "later"


@freeze_time(NOW_ISO)
def test_classify_resolution_later_far_future():
    state, _label = classify_resolution(_at(48 * 60), _now())
    assert state == "later"
