"""How many ATP matches had a player above 80% win prob who still LOST?

For each finalized ATP match-winner market we build the in-play YES-mid series
(prob the YES player wins, in cents) and read the settled outcome from
settlement_value_dollars (1.0 = YES won, 0.0 = NO won; anything else = voided/
walkover, skipped).

The eventual LOSER's peak in-play probability is:
    if YES won  -> loser is the NO player, peak = 100 - min(YES mid)
    if NO won   -> loser is the YES player, peak = max(YES mid)

A "blown lead" = that loser peak went above 80%. We report any-tick and a
sustained (>=3 in-play minutes >=80%) version, and list the matches.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
from dotenv import load_dotenv

load_dotenv()

from kalshi.auth import KalshiCredentials
from kalshi.client import KalshiClient

SERIES_LIST = [("ATP", "KXATPMATCH"), ("WTA", "KXWTAMATCH")]
LOOKBACK_HOURS = 60
MIN_INPLAY_MINUTES = 30
GAP_BREAK_MIN = 25
MAX_CANDLES = 4900  # API hard cap is 5000 candles per request
THRESHOLD = 95.0


def _ts(s: str) -> int:
    return int(dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp())


def _close_dollars(dist: object) -> float | None:
    if not isinstance(dist, dict):
        return None
    v = dist.get("close_dollars")
    try:
        return float(v) if v not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _vol(c: dict) -> float:
    try:
        return float(c.get("volume_fp") or 0)
    except (TypeError, ValueError):
        return 0.0


def recent_finalized(client: KalshiClient, series: str) -> list[dict]:
    out: list[dict] = []
    cutoff = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=LOOKBACK_HOURS)).timestamp()
    cursor: str | None = None
    seen: set[str] = set()
    while True:
        page = client.get_markets(series_ticker=series, limit=1000, cursor=cursor)
        for m in page.get("markets", []):
            if m.get("status") != "finalized":
                continue
            ev = m.get("event_ticker", "")
            if ev in seen:
                continue
            ct = m.get("close_time")
            if not ct or _ts(ct) < cutoff:
                continue
            seen.add(ev)
            out.append(m)
        cursor = page.get("cursor")
        if not cursor:
            break
    return out


def inplay_window(candles: list[dict]) -> tuple[int, int] | None:
    traded = [int(c["end_period_ts"]) for c in candles if _vol(c) > 0 and c.get("end_period_ts")]
    if len(traded) < MIN_INPLAY_MINUTES:
        return None
    traded.sort()
    end = traded[-1]
    start = end
    prev = end
    for t in reversed(traded[:-1]):
        if prev - t > GAP_BREAK_MIN * 60:
            break
        start = t
        prev = t
    return start, end


def per_minute_mid(candles: list[dict], start_ts: int, end_ts: int) -> np.ndarray:
    rows: list[tuple[int, float]] = []
    for c in candles:
        ts = c.get("end_period_ts")
        if ts is None or not (start_ts <= int(ts) <= end_ts):
            continue
        bid = _close_dollars(c.get("yes_bid"))
        ask = _close_dollars(c.get("yes_ask"))
        if bid is None or ask is None:
            continue
        bid_c, ask_c = bid * 100.0, ask * 100.0
        spread = ask_c - bid_c
        if spread < 0 or spread > 60:
            continue
        rows.append((int(ts), (bid_c + ask_c) / 2.0))
    rows.sort(key=lambda r: r[0])
    return np.array([r[1] for r in rows], dtype=float)


def outcome(market: dict) -> str | None:
    """'yes', 'no', or None (voided/unknown) from settlement_value_dollars."""
    v = market.get("settlement_value_dollars")
    try:
        sv = float(v)
    except (TypeError, ValueError):
        return None
    if sv >= 0.999:
        return "yes"
    if sv <= 0.001:
        return "no"
    return None  # fractional => void / walkover


def analyze(client: KalshiClient, market: dict, series: str) -> dict | None:
    ticker = market["ticker"]
    o, c = _ts(market["open_time"]), _ts(market["close_time"])
    o = max(o, c - MAX_CANDLES * 60)  # keep under the candlestick cap
    wide = client.get_candlesticks(
        series, ticker, start_ts=o, end_ts=c, period_interval=1
    ).get("candlesticks", [])
    win = inplay_window(wide)
    if win is None:
        return None
    start, end = win
    tight = client.get_candlesticks(
        series, ticker, start_ts=start - 120, end_ts=end + 120, period_interval=1
    ).get("candlesticks", [])
    mid = per_minute_mid(tight, start, end)
    if len(mid) < MIN_INPLAY_MINUTES:
        return None
    res = outcome(market)
    if res is None:
        return {"voided": True, "ticker": ticker}

    # Loser's probability path (cents).
    loser_prob = mid if res == "no" else 100.0 - mid
    loser_name = market.get("yes_sub_title") if res == "no" else market.get("no_sub_title")
    peak = float(np.max(loser_prob))
    minutes_over = int(np.sum(loser_prob > THRESHOLD))
    return {
        "voided": False,
        "ticker": ticker,
        "title": market.get("title", ""),
        "loser": loser_name,
        "loser_peak": peak,
        "minutes_over_80": minutes_over,
        "blown_any": peak > THRESHOLD,
        "blown_sustained": minutes_over >= 3,
    }


def run_tour(client: KalshiClient, tour: str, series: str) -> None:
    markets = recent_finalized(client, series)
    analyzed: list[dict] = []
    voided = 0
    for m in markets:
        try:
            r = analyze(client, m, series)
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {m.get('ticker')}: {exc}")
            continue
        if r is None:
            continue
        if r["voided"]:
            voided += 1
            continue
        analyzed.append(r)

    peaks = np.array([r["loser_peak"] for r in analyzed]) if analyzed else np.array([])
    blown = [r for r in analyzed if r["loser_peak"] > THRESHOLD]

    print(f"\n{'='*68}\n{tour}  ({series})")
    print(f"  matches with clean win/loss outcome: {len(analyzed)}  "
          f"(voided/walkover skipped: {voided})")
    n = int(np.sum(peaks > THRESHOLD)) if len(peaks) else 0
    pct = n / len(analyzed) * 100 if analyzed else 0.0
    print(f"  >>> matches where the LOSER was once above {THRESHOLD:.0f}%: "
          f"{n}/{len(analyzed)} = {pct:.1f}%")
    print(f"  blown leads (loser peaked above {THRESHOLD:.0f}%), highest first:")
    for r in sorted(blown, key=lambda x: x["loser_peak"], reverse=True):
        print(f"    {r['loser_peak']:5.1f}%  {r['minutes_over_80']:3d}m>{THRESHOLD:.0f}  "
              f"{r['loser']:24s} LOST  [{r['ticker']}]")


def main() -> None:
    client = KalshiClient(KalshiCredentials.from_env())
    for tour, series in SERIES_LIST:
        run_tour(client, tour, series)


if __name__ == "__main__":
    main()
