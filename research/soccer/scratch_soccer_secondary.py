"""Deep-dive: comeback rates across ALL World Cup soccer market types (secondary
markets too: goals totals, half winner, correct score, spreads, BTTS, corners,
mentions, etc.), not just the match winner.

For every binary market in recent WC games we build the in-play YES-mid series
and compute the eventual WINNER's trough:
    trough = min(YES mid)        if the market settled YES
           = 100 - max(YES mid)  if it settled NO
i.e. the lowest the winning side ever traded. Then per market type we report the
share of markets where that trough dipped below each level (a comeback to 100%).
"""

from __future__ import annotations

import datetime as dt
import math
import sys
from collections import defaultdict

import numpy as np
from dotenv import load_dotenv

load_dotenv()

from kalshi.auth import KalshiCredentials
from kalshi.client import KalshiClient

LOOKBACK_HOURS = 168  # 7 days of World Cup
MIN_INPLAY_MINUTES = 12
GAP_BREAK_MIN = 25
MAX_CANDLES = 4900
MIN_VOL = 200.0  # contracts; skip illiquid markets (bounds candlestick calls)
CAP_PER_TYPE = 100  # analyze the N most-liquid markets per type (bounds runtime)
BLOWN_LEVELS = [80, 90, 95]   # YES led above this, then settled NO ("went to zero")
COMEBACK_LEVELS = [20, 10, 5]  # YES dipped below this, then settled YES (comeback)

# Derivative (secondary) market types only -- exclude the straight match winner.
PRIMARY_TYPES = {"KXWCGAME", "KXUSLGAME", "KXUSLCUPGAME", "KXCHLLDPGAME",
                 "KXLALIGA2GAME"}

SOCCER_TYPES = {"soccer_tournament_multi_leg", "soccer_group", "soccer_game"}

TYPE_NAMES = {
    "KXWCGAME": "Match winner",
    "KXWCSPREAD": "Spread (handicap)",
    "KXWCTOTAL": "Total goals O/U",
    "KXWCTEAMTOTAL": "Team total goals",
    "KXWCBTTS": "Both teams to score",
    "KXWCFTTS": "First team to score",
    "KXWCFIRSTGOAL": "First goal (which team)",
    "KXWCTEAMFIRSTGOAL": "Team scores first goal",
    "KXWC1H": "1st-half winner",
    "KXWC2H": "2nd-half winner",
    "KXWC1HSPREAD": "1st-half spread",
    "KXWC2HSPREAD": "2nd-half spread",
    "KXWC1HTOTAL": "1st-half total",
    "KXWC2HTOTAL": "2nd-half total",
    "KXWC1HBTTS": "1st-half BTTS",
    "KXWC2HBTTS": "2nd-half BTTS",
    "KXWCSCORE": "Correct score",
    "KXWC1HSCORE": "1st-half correct score",
    "KXWCMENTION": "Mention",
    "KXWCGOAL": "Goals (player/total)",
    "KXWCCORNERS": "Total corners",
    "KXWCTCORNERS": "Team corners",
    "KXWCSOA": "Shots on goal",
    "KXWCAST": "Assists",
}


def _ts(s: str) -> int:
    return int(dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp())


def _f(dist, key="close_dollars"):
    if not isinstance(dist, dict):
        return None
    v = dist.get(key)
    try:
        return float(v) if v not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _vol(c):
    try:
        return float(c.get("volume_fp") or 0)
    except (TypeError, ValueError):
        return 0.0


def soccer_games(client):
    now = dt.datetime.now(dt.timezone.utc)
    min_start = (now - dt.timedelta(hours=LOOKBACK_HOURS)).strftime("%Y-%m-%dT%H:%M:%SZ")
    allowed_events, series_set = set(), set()
    cursor = None
    n_games = 0
    for _ in range(120):
        page = client.get_milestones(minimum_start_date=min_start, limit=200, cursor=cursor)
        for m in page.get("milestones", []):
            if m.get("type") not in SOCCER_TYPES:
                continue
            sd = m.get("start_date")
            try:
                start = dt.datetime.fromisoformat((sd or "").replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                continue
            if start > now:
                continue
            n_games += 1
            for ev in m.get("related_event_tickers") or []:
                allowed_events.add(ev)
                series_set.add(ev.split("-", 1)[0])
        cursor = page.get("cursor")
        if not cursor:
            break
    return allowed_events, series_set, n_games


def collect_markets(client, allowed_events, series_set):
    out = []
    for series in sorted(series_set):
        cursor = None
        for _ in range(15):
            try:
                page = client.get_markets(series_ticker=series, limit=1000, cursor=cursor)
            except Exception:  # noqa: BLE001
                break
            for mk in page.get("markets", []):
                if mk.get("event_ticker") not in allowed_events:
                    continue
                if mk.get("status") != "finalized":
                    continue
                out.append(mk)
            cursor = page.get("cursor")
            if not cursor:
                break
    return out


def outcome(mk):
    try:
        sv = float(mk.get("settlement_value_dollars"))
    except (TypeError, ValueError):
        return None
    if sv >= 0.999:
        return "yes"
    if sv <= 0.001:
        return "no"
    return None


def inplay_window(candles):
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


def series(candles, a, b):
    """Return time-ordered (mid, bid, ask) arrays in cents for the window."""
    rows = []
    for c in candles:
        ts = c.get("end_period_ts")
        if ts is None or not (a <= int(ts) <= b):
            continue
        bid = _f(c.get("yes_bid"))
        ask = _f(c.get("yes_ask"))
        if bid is None or ask is None:
            continue
        bid_c, ask_c = bid * 100, ask * 100
        if not (0 <= ask_c - bid_c <= 60):
            continue
        rows.append((int(ts), (bid_c + ask_c) / 2, bid_c, ask_c))
    rows.sort()
    mid = np.array([r[1] for r in rows], dtype=float)
    bid = np.array([r[2] for r in rows], dtype=float)
    ask = np.array([r[3] for r in rows], dtype=float)
    return mid, bid, ask


def fee_cents(price_cents: float) -> float:
    """Kalshi quadratic fee for ONE contract, rounded up to the cent."""
    p = price_cents / 100.0
    if not 0.0 < p < 1.0:
        return 0.0
    return math.ceil(round(0.07 * p * (1.0 - p) * 100.0, 6))


def longshot_profit_gt10(bid, ask, entry_cap=5.0, thr=0.10) -> bool:
    """Could a cheap longshot LONG net > thr after fees?

    Buy YES at the ask when ask <= entry_cap (the trough), then sell at the best
    later bid. Cost = ask + buy fee; proceeds = sell_bid - sell fee. Realistic
    (pay the ask, hit the bid). Returns True if any such trade clears thr.
    """
    n = len(ask)
    if n < 2:
        return False
    suff_bid = np.empty(n)
    suff_bid[-1] = bid[-1]
    for i in range(n - 2, -1, -1):
        suff_bid[i] = max(bid[i], suff_bid[i + 1])
    best = -1e9
    for i in range(n - 1):
        a = ask[i]
        if 0 < a <= entry_cap:
            cost = a + fee_cents(a)
            s = suff_bid[i + 1]
            proceeds = s - fee_cents(s)
            if cost > 0:
                best = max(best, (proceeds - cost) / cost)
    return best > thr


def market_stats(client, mk):
    """Return (settlement, peak_yes, trough_yes, longshot_flag) or None."""
    res = outcome(mk)
    if res is None:
        return None
    try:
        o, c = _ts(mk["open_time"]), _ts(mk["close_time"])
    except (KeyError, AttributeError):
        return None
    o = max(o, c - MAX_CANDLES * 60)
    s_ticker = mk.get("series_ticker") or mk["ticker"].split("-", 1)[0]
    candles = client.get_candlesticks(
        s_ticker, mk["ticker"], start_ts=o, end_ts=c, period_interval=1
    ).get("candlesticks", [])
    win = inplay_window(candles)
    if win is None:
        return None
    mid, bid, ask = series(candles, win[0], win[1])
    if len(mid) < MIN_INPLAY_MINUTES:
        return None
    return res, float(np.max(mid)), float(np.min(mid)), longshot_profit_gt10(bid, ask)


def main():
    client = KalshiClient(KalshiCredentials.from_env())
    allowed, series_set, n_games = soccer_games(client)
    print(f"{n_games} soccer games started in last {LOOKBACK_HOURS}h; "
          f"{len(series_set)} market-type series; {len(allowed)} events")
    markets = collect_markets(client, allowed, series_set)
    liquid = [m for m in markets if float(m.get("volume_fp") or 0) >= MIN_VOL]
    print(f"{len(markets)} finalized markets, {len(liquid)} liquid (>= {MIN_VOL:.0f} vol)\n")
    if "--count" in sys.argv:
        by_type = defaultdict(int)
        for m in liquid:
            by_type[m["ticker"].split("-", 1)[0]] += 1
        for t, n in sorted(by_type.items(), key=lambda x: -x[1]):
            print(f"  {TYPE_NAMES.get(t, t):26s} {n}")
        return

    # Narrow to derivative (secondary) markets only.
    liquid = [m for m in liquid if m["ticker"].split("-", 1)[0] not in PRIMARY_TYPES]

    # Cap to the most-liquid markets per type to bound candlestick calls.
    by_type_markets = defaultdict(list)
    for m in liquid:
        by_type_markets[m["ticker"].split("-", 1)[0]].append(m)
    sample = []
    for t, ms in by_type_markets.items():
        ms.sort(key=lambda m: float(m.get("volume_fp") or 0), reverse=True)
        sample.extend(ms[:CAP_PER_TYPE])
    print(f"analyzing {len(sample)} DERIVATIVE markets (<= {CAP_PER_TYPE} per type)\n")
    liquid = sample

    stats = defaultdict(list)  # type -> list of (settle, peak, trough, longshot)
    for i, mk in enumerate(liquid):
        t = mk["ticker"].split("-", 1)[0]
        try:
            v = market_stats(client, mk)
        except Exception:  # noqa: BLE001
            v = None
        if v is not None:
            stats[t].append(v)
        if (i + 1) % 100 == 0:
            done = sum(len(x) for x in stats.values())
            print(f"  ...{i+1}/{len(liquid)} markets ({done} measured)")

    all_rows = [r for rs in stats.values() for r in rs]
    print(f"\nDerivative markets measured (liquid + in-play data): {len(all_rows)}\n")
    if not all_rows:
        return

    def blown_row(rows):
        n = len(rows)
        return " ".join(
            f"{sum(1 for s, pk, tr, ls in rows if s == 'no' and pk > lv) / n * 100:>4.0f}"
            for lv in BLOWN_LEVELS
        )

    def comeback_row(rows):
        n = len(rows)
        return " ".join(
            f"{sum(1 for s, pk, tr, ls in rows if s == 'yes' and tr < lv) / n * 100:>4.0f}"
            for lv in COMEBACK_LEVELS
        )

    def longshot_pct(rows):
        n = len(rows)
        return sum(1 for r in rows if r[3]) / n * 100

    order = sorted(stats, key=lambda k: -len(stats[k]))

    print("BLOWN LEADS -- % of markets where YES led above the level then settled NO "
          "('went to zero'):\n")
    print(f"{'derivative market type':26s} {'N':>4s}   " +
          "  ".join(f">{lv}" for lv in BLOWN_LEVELS))
    print(f"{'ALL derivatives':26s} {len(all_rows):>4d}   {blown_row(all_rows)}")
    print("-" * 60)
    for t in order:
        if len(stats[t]) < 4:
            continue
        print(f"{TYPE_NAMES.get(t, t):26s} {len(stats[t]):>4d}   {blown_row(stats[t])}")

    print("\nCOMEBACKS -- % of markets where YES dipped below the level then settled YES "
          "('came back to 100%'),\n"
          "plus >10%net = % where buying the <=5c dip nets >10% after Kalshi "
          "buy+sell fees (sell at later bid):\n")
    print(f"{'derivative market type':26s} {'N':>4s}   " +
          "  ".join(f"<{lv}" for lv in COMEBACK_LEVELS) + "   >10%net")
    print(f"{'ALL derivatives':26s} {len(all_rows):>4d}   {comeback_row(all_rows)}"
          f"     {longshot_pct(all_rows):>4.0f}")
    print("-" * 66)
    for t in order:
        if len(stats[t]) < 4:
            continue
        print(f"{TYPE_NAMES.get(t, t):26s} {len(stats[t]):>4d}   "
              f"{comeback_row(stats[t])}     {longshot_pct(stats[t]):>4.0f}")


if __name__ == "__main__":
    main()
