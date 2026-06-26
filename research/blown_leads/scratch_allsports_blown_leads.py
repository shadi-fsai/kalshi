"""Across every head-to-head game type (played in the last 48h),
how often did the eventual winner dip below various win-probability levels,
and how often was a fee-aware <=5c longshot on the winning side profitable?

Handles 2-way sports and 3-way soccer (Team A / Tie / Team B): for each game we
look at every TEAM win market (excluding the Tie outcome), build its in-play
YES-mid series, and record each non-winning team's peak win probability. A team
that led above the threshold but did NOT win the match (lost OR drew) is a blown
lead. Voided/walkover markets (fractional settlement) are skipped.
"""

from __future__ import annotations

import datetime as dt
import math
from collections import defaultdict

import numpy as np
from dotenv import load_dotenv

load_dotenv()

from kalshi.auth import KalshiCredentials
from kalshi.client import KalshiClient

LOOKBACK_HOURS = 48
MIN_INPLAY_MINUTES = 15
GAP_BREAK_MIN = 25
MAX_CANDLES = 4900
MAX_PER_SPORT = 150  # cap measured games per sport (bounds runtime; tennis is huge)
LEVELS = [2, 4, 6, 8, 10, 15, 20]  # trough levels (winner dipped BELOW these)

GAME_TYPES = {
    "tennis_tournament_singles": "Tennis",
    "soccer_tournament_multi_leg": "Soccer",
    "soccer_group": "Soccer",
    "esports_match": "Esports",
    "baseball_game": "Baseball",
    "cricket_match": "Cricket",
    "basketball_game": "Basketball",
    "football_game": "Football",
    "mma_match": "MMA",
    "rugby_match": "Rugby",
    "boxing_match": "Boxing",
    "afl_match": "AFL",
    "lacrosse_match": "Lacrosse",
}

BANNED = ("map ", "set ", "period", "quarter", "1st half", "2nd half", "half",
          "1st", "2nd", "inning", "frame", "leg ", "race ", "game 1", "game 2",
          "game 3", "game 4", "game 5", "first ", "total", "spread", "over ",
          "under", "by more", "margin", "corner", "goal", "score", "assist",
          "mention", "delay", "card", "both teams")
TIE_NAMES = {"tie", "draw"}


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


def pull_game_milestones(client: KalshiClient):
    now = dt.datetime.now(dt.timezone.utc)
    min_start = (now - dt.timedelta(hours=LOOKBACK_HOURS)).strftime("%Y-%m-%dT%H:%M:%SZ")
    out = []
    cursor = None
    for _ in range(80):
        page = client.get_milestones(minimum_start_date=min_start, limit=200, cursor=cursor)
        for m in page.get("milestones", []):
            if m.get("type") not in GAME_TYPES:
                continue
            sd = m.get("start_date")
            try:
                start = dt.datetime.fromisoformat((sd or "").replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                continue
            if start <= now:
                out.append(m)
        cursor = page.get("cursor")
        if not cursor:
            break
    return out


def is_team_winner_market(mk: dict) -> bool:
    title = (mk.get("title") or "").lower()
    if "winner" not in title and not ("win the" in title and "match" in title):
        return False
    if any(b in title for b in BANNED):
        return False
    sub = (mk.get("yes_sub_title") or "").strip().lower()
    if sub in TIE_NAMES:
        return False
    return True


def find_team_markets(client: KalshiClient, milestone: dict):
    """Return (event_ticker, [team winner markets]) for the game, or None."""
    for ev in milestone.get("related_event_tickers") or []:
        try:
            mks = client.get_markets(event_ticker=ev, limit=30).get("markets", [])
        except Exception:  # noqa: BLE001
            continue
        team = [m for m in mks if is_team_winner_market(m)]
        if len(team) >= 2:
            return ev, team
    return None


def outcome(market: dict) -> str | None:
    v = market.get("settlement_value_dollars")
    try:
        sv = float(v)
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


def mid_series(candles, start_ts, end_ts):
    rows = []
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
        rows.append((int(ts), (bid_c + ask_c) / 2.0, bid_c, ask_c))
    rows.sort(key=lambda r: r[0])
    mid = np.array([r[1] for r in rows], dtype=float)
    bid = np.array([r[2] for r in rows], dtype=float)
    ask = np.array([r[3] for r in rows], dtype=float)
    return mid, bid, ask


def series_of(market: dict) -> str:
    return market.get("series_ticker") or (market.get("ticker") or "").split("-", 1)[0]


def fee_cents(price_cents: float) -> float:
    p = price_cents / 100.0
    if not 0.0 < p < 1.0:
        return 0.0
    return math.ceil(round(0.07 * p * (1.0 - p) * 100.0, 6))


def longshot_profit_gt10(bid, ask, entry_cap=5.0, thr=0.10) -> bool:
    """Buy YES at the ask when ask <= entry_cap, sell at best later bid; net > thr
    after Kalshi buy+sell fees?"""
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
            best = max(best, (s - fee_cents(s) - cost) / cost)
    return best > thr


def team_metrics(client: KalshiClient, market: dict):
    """Return (winner_trough_estimate, longshot_flag) for a team market, or None.

    settle YES -> winning team's own trough = min(mid); longshot uses its quotes.
    settle NO  -> 100 - max(mid) (implied winner trough); longshot uses the
                  complementary (winner-side) quotes 100 - yes quotes.
    """
    res = outcome(market)
    if res is None:
        return None
    try:
        o, c = _ts(market["open_time"]), _ts(market["close_time"])
    except (KeyError, AttributeError):
        return None
    o = max(o, c - MAX_CANDLES * 60)
    candles = client.get_candlesticks(
        series_of(market), market["ticker"], start_ts=o, end_ts=c, period_interval=1
    ).get("candlesticks", [])
    win = inplay_window(candles)
    if win is None:
        return None
    mid, bid, ask = mid_series(candles, win[0], win[1])
    if len(mid) < MIN_INPLAY_MINUTES:
        return None
    if res == "yes":
        trough = float(np.min(mid))
        ls = longshot_profit_gt10(bid, ask)
    else:
        trough = float(100.0 - np.max(mid))
        # Winner is the complementary (NO) side: its bid/ask are 100 - yes ask/bid.
        ls = longshot_profit_gt10(100.0 - ask, 100.0 - bid)
    return trough, ls


def main() -> None:
    client = KalshiClient(KalshiCredentials.from_env())
    milestones = pull_game_milestones(client)
    print(f"{len(milestones)} game milestones started in last "
          f"{LOOKBACK_HOURS}h\n")

    # sport -> list of (winner_trough, longshot_flag)
    by_sport: dict[str, list[tuple[float, bool]]] = defaultdict(list)
    seen: set[str] = set()
    voided = 0
    draws = 0
    for i, m in enumerate(milestones):
        sport = GAME_TYPES[m["type"]]
        if len(by_sport[sport]) >= MAX_PER_SPORT:
            continue  # sport already sampled to the cap
        found = find_team_markets(client, m)
        if found is None:
            continue
        ev, team = found
        if ev in seen:
            continue
        seen.add(ev)
        if not any(mk.get("status") == "finalized" for mk in team):
            continue
        # A "flip to 100%" requires an actual team winner (exclude draws).
        if not any(outcome(mk) == "yes" for mk in team):
            if any(outcome(mk) == "no" for mk in team):
                draws += 1
            else:
                voided += 1
            continue
        ests = []
        ls_any = False
        for mk in team:
            try:
                v = team_metrics(client, mk)
            except Exception:  # noqa: BLE001
                v = None
            if v is not None:
                ests.append(v[0])
                ls_any = ls_any or v[1]
        if not ests:
            continue
        by_sport[sport].append((min(ests), ls_any))
        if (i + 1) % 40 == 0:
            done = sum(len(v) for v in by_sport.values())
            print(f"  ...scanned {i+1}/{len(milestones)} ({done} games measured)")

    by_sport = {s: rows for s, rows in by_sport.items() if rows}
    troughs_by_sport = {s: [t for t, _ in rows] for s, rows in by_sport.items()}
    all_tr = np.array([t for ts in troughs_by_sport.values() for t in ts])
    print(f"\nDecisive games measured (a team won + in-play data): {len(all_tr)}")
    print(f"Draws excluded: {draws}   Voided/walkover skipped: {voided}")
    print(f"(capped at {MAX_PER_SPORT} measured games per sport)\n")

    if len(all_tr) == 0:
        print("No measurable games.")
        return

    print("Share of games where the eventual WINNER once dipped BELOW each level\n"
          "(i.e. came back from that deep a trough to win 100%):\n")
    print("level :  " + "   ".join(f"<{lv}" for lv in LEVELS))
    pcts = [np.sum(all_tr < lv) / len(all_tr) * 100 for lv in LEVELS]
    cnts = [int(np.sum(all_tr < lv)) for lv in LEVELS]
    print("ALL % : " + "  ".join(f"{p:5.1f}" for p in pcts))
    print("(n)   : " + "  ".join(f"{c:5d}" for c in cnts) + f"   of {len(all_tr)}")

    all_ls = [ls for rows in by_sport.values() for _, ls in rows]
    print(f"\n>10%net (fee-aware longshot): buy the winning side at the ask when it "
          f"dips to <=5c,\nsell at the best later bid; net >10% after Kalshi buy+sell "
          f"fees. ALL sports: {sum(all_ls)/len(all_ls)*100:.0f}% "
          f"({sum(all_ls)}/{len(all_ls)})\n")

    print("per sport (% of that sport's decisive games):")
    print(f"{'sport':10s} {'N':>4s}  " + " ".join(f"<{lv:>2d}" for lv in LEVELS)
          + "   >10%net")
    for sport in sorted(troughs_by_sport, key=lambda s: -len(troughs_by_sport[s])):
        arr = np.array(troughs_by_sport[sport])
        row = " ".join(f"{np.sum(arr<lv)/len(arr)*100:>3.0f}" for lv in LEVELS)
        ls_rows = [ls for _, ls in by_sport[sport]]
        ls_pct = sum(ls_rows) / len(ls_rows) * 100
        print(f"{sport:10s} {len(arr):>4d}  {row}     {ls_pct:>4.0f}")


if __name__ == "__main__":
    main()
