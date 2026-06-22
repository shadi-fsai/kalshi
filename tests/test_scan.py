"""Tests for kalshi.markets.scan_series_for_favorites with a fake client."""

from __future__ import annotations

from kalshi.markets import passes_high_water_mark, scan_series_for_favorites


class FakeClient:
    """Minimal stand-in for KalshiClient.get_markets with cursor pagination.

    ``pages_by_series`` maps a series ticker to a list of "pages"; each page is
    a list of market dicts. The fake hands them out one per call, setting a
    cursor until the last page.
    """

    def __init__(self, pages_by_series: dict[str, list[list[dict]]]):
        self.pages_by_series = pages_by_series
        self.calls: list[dict] = []

    def get_markets(self, *, series_ticker=None, status=None, limit=None, cursor=None):
        self.calls.append(
            {"series_ticker": series_ticker, "status": status, "cursor": cursor}
        )
        pages = self.pages_by_series.get(series_ticker, [[]])
        idx = int(cursor) if cursor else 0
        markets = pages[idx]
        next_cursor = str(idx + 1) if idx + 1 < len(pages) else None
        return {"markets": markets, "cursor": next_cursor}


def _mkt(ticker, event_ticker, yes_ask=None, no_ask=None):
    m = {"ticker": ticker, "event_ticker": event_ticker}
    if yes_ask is not None:
        m["yes_ask_dollars"] = f"{yes_ask / 100:.4f}"
    if no_ask is not None:
        m["no_ask_dollars"] = f"{no_ask / 100:.4f}"
    return m


def test_price_range_filter_and_sort():
    client = FakeClient(
        {
            "S1": [
                [
                    _mkt("M1", "E1", yes_ask=92),
                    _mkt("M2", "E1", yes_ask=88),  # below range
                    _mkt("M3", "E1", yes_ask=95),
                ]
            ]
        }
    )
    results, truncated = scan_series_for_favorites(
        client,
        {"S1"},
        allowed_event_tickers={"E1"},
        min_price=90,
        max_price=99,
        side_choice="YES",
    )
    assert truncated is False
    # Sorted by price descending, 88 excluded.
    assert [r["market"]["ticker"] for r in results] == ["M3", "M1"]
    assert all(r["side"] == "yes" for r in results)


def test_allowed_event_tickers_filter():
    client = FakeClient(
        {"S1": [[_mkt("M1", "E1", yes_ask=92), _mkt("M2", "E2", yes_ask=93)]]}
    )
    results, _ = scan_series_for_favorites(
        client,
        {"S1"},
        allowed_event_tickers={"E1"},
        min_price=90,
        max_price=99,
        side_choice="YES",
    )
    assert [r["market"]["ticker"] for r in results] == ["M1"]


def test_side_choice_either_includes_both_legs():
    client = FakeClient({"S1": [[_mkt("M1", "E1", yes_ask=91, no_ask=94)]]})
    results, _ = scan_series_for_favorites(
        client,
        {"S1"},
        allowed_event_tickers={"E1"},
        min_price=90,
        max_price=99,
        side_choice="Either",
    )
    sides = sorted(r["side"] for r in results)
    assert sides == ["no", "yes"]


def test_side_choice_no_only():
    client = FakeClient({"S1": [[_mkt("M1", "E1", yes_ask=91, no_ask=94)]]})
    results, _ = scan_series_for_favorites(
        client,
        {"S1"},
        allowed_event_tickers={"E1"},
        min_price=90,
        max_price=99,
        side_choice="NO",
    )
    assert [r["side"] for r in results] == ["no"]


def test_pagination_follows_cursor():
    client = FakeClient(
        {
            "S1": [
                [_mkt("M1", "E1", yes_ask=92)],
                [_mkt("M2", "E1", yes_ask=93)],
            ]
        }
    )
    results, _ = scan_series_for_favorites(
        client,
        {"S1"},
        allowed_event_tickers={"E1"},
        min_price=90,
        max_price=99,
        side_choice="YES",
    )
    assert {r["market"]["ticker"] for r in results} == {"M1", "M2"}
    # Two pages -> two calls, second carries the cursor.
    assert client.calls[0]["cursor"] is None
    assert client.calls[1]["cursor"] == "1"


def test_empty_allowed_set_keeps_all():
    client = FakeClient({"S1": [[_mkt("M1", "E1", yes_ask=92)]]})
    results, _ = scan_series_for_favorites(
        client,
        {"S1"},
        allowed_event_tickers=set(),
        min_price=90,
        max_price=99,
        side_choice="YES",
    )
    assert len(results) == 1


def test_max_series_truncation_flag():
    client = FakeClient(
        {
            "S1": [[_mkt("M1", "E1", yes_ask=92)]],
            "S2": [[_mkt("M2", "E2", yes_ask=93)]],
            "S3": [[_mkt("M3", "E3", yes_ask=94)]],
        }
    )
    results, truncated = scan_series_for_favorites(
        client,
        {"S1", "S2", "S3"},
        allowed_event_tickers=set(),
        min_price=90,
        max_price=99,
        side_choice="YES",
        max_series=2,
    )
    assert truncated is True
    # Only first 2 series (sorted) are scanned.
    scanned = {c["series_ticker"] for c in client.calls}
    assert scanned == {"S1", "S2"}
    assert len(results) == 2


# --- passes_high_water_mark ----------------------------------------------


def test_passes_high_water_mark_uses_side_and_threshold():
    yes_result = {"side": "yes", "price": 60}
    no_result = {"side": "no", "price": 60}
    pair = (96.0, 40.0)  # (yes_hwm, no_hwm)
    # YES side: 96 >= 95 -> passes; NO side: 40 < 95 -> excluded.
    assert passes_high_water_mark(yes_result, pair, 95) is True
    assert passes_high_water_mark(no_result, pair, 95) is False


def test_passes_high_water_mark_boundary_is_inclusive():
    assert passes_high_water_mark({"side": "yes"}, (95.0, None), 95) is True
    assert passes_high_water_mark({"side": "yes"}, (94.999, None), 95) is False


def test_passes_high_water_mark_none_is_excluded():
    # No usable candle data for the side -> never passes.
    assert passes_high_water_mark({"side": "yes"}, (None, 99.0), 50) is False
    assert passes_high_water_mark({"side": "no"}, (99.0, None), 50) is False
