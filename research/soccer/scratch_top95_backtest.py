"""Backtest: buy every soccer TOP GAME (match-winner market, any league) the
moment the favorite's YES ask reaches an entry level, hold to settlement -- then
SLICE the results to look for any positive-EV subset.

Universe : ALL soccer leagues -- every series whose ticker ends in "GAME"
           (KXWCGAME, KXMLSGAME, KXUSLGAME, KXINTLFRIENDLYGAME, ...) for matches
           started in the last LOOKBACK_DAYS.

Two phases:
  1. fetch+cache  -- pull the ask series for every match-winner market once and
                     dump to CACHE_PATH (re-run with --refresh to refetch).
  2. analyze      -- from the cache, build positions at each entry level and slice
                     by World Cup vs all soccer, high vs low volume (median split),
                     fast-cross momentum, cheap-open-then-rose, and per league.

Fees: flat 1c on entry, 0c on resolution (settlement is fee-free on Kalshi).
"""

from __future__ import annotations

import datetime as dt
import json
import os
import sys
from collections import defaultdict

import numpy as np
from dotenv import load_dotenv

load_dotenv()

from kalshi.auth import KalshiCredentials
from kalshi.client import KalshiClient
from scratch_soccer_secondary import (
    MAX_CANDLES,
    MIN_INPLAY_MINUTES,
    _f,
    _ts,
    collect_markets,
    inplay_window,
    outcome,
)

LOOKBACK_DAYS = 45
ENTRY_FEE = 1.0  # flat 1c entry fee (per user)
ENTRY_HI = 99.0  # must still be buyable below 100
TIE_NAMES = {"tie", "draw"}
LEVELS = [90.0, 92.0, 95.0, 97.0]
FRACS = np.arange(0.0, 1.0001, 0.01)
CACHE_PATH = "/tmp/soccer_fav_cache_v2.json"  # v2: rows carry per-candle volume
CAPTURES = [0.10, 0.25, 0.50]  # fraction of others' volume we assume we can take

# Slice definitions (per plan)
FAST_RISE = 15.0   # cents gained...
FAST_WINDOW = 600  # ...within this many seconds before hitting the level
CHEAP_OPEN = 70.0  # kickoff ask below this = "cheap open then rose"
MIN_SLICE_N = 50   # buckets below this flagged as noisy


# --------------------------------------------------------------------------- #
# Phase 1: fetch + cache
# --------------------------------------------------------------------------- #
def soccer_games_wide(client, days):
    """All soccer match-winner events + their GAME series over `days`."""
    now = dt.datetime.now(dt.timezone.utc)
    min_start = (now - dt.timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    allowed, series_set = set(), set()
    cursor, n = None, 0
    for page_i in range(400):
        page = client.get_milestones(minimum_start_date=min_start, limit=200,
                                      cursor=cursor)
        if page_i % 20 == 0:
            print(f"  ...milestones page {page_i} ({n} soccer matches so far)",
                  flush=True)
        for m in page.get("milestones", []):
            if "soccer" not in (m.get("type") or ""):
                continue
            sd = m.get("start_date")
            try:
                start = dt.datetime.fromisoformat((sd or "").replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                continue
            if start > now:
                continue
            n += 1
            for ev in m.get("related_event_tickers") or []:
                if ev.split("-", 1)[0].endswith("GAME"):
                    allowed.add(ev)
                    series_set.add(ev.split("-", 1)[0])
        cursor = page.get("cursor")
        if not cursor:
            break
    return allowed, series_set, n


def quotes_with_ts(candles, a, b):
    """Rows of (ts, ask_cents, volume_contracts) within the window."""
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
        try:
            vol = float(c.get("volume_fp") or 0)
        except (TypeError, ValueError):
            vol = 0.0
        rows.append((int(ts), ask_c, vol))
    rows.sort()
    return rows


def side_record(client, mk):
    """Return {rows, win, vol} for one match-winner side market, or None."""
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
    rows = quotes_with_ts(candles, win[0], win[1])
    if len(rows) < MIN_INPLAY_MINUTES:
        return None
    try:
        vol = float(mk.get("volume_fp") or 0)
    except (TypeError, ValueError):
        vol = 0.0
    return {"rows": rows, "win": res == "yes", "vol": vol}


def fetch_and_cache(path):
    client = KalshiClient(KalshiCredentials.from_env())
    allowed, series_set, n_games = soccer_games_wide(client, LOOKBACK_DAYS)
    print(f"{n_games} soccer matches over {LOOKBACK_DAYS}d; {len(series_set)} "
          f"leagues; pulling match-winner markets...")
    markets = collect_markets(client, allowed, series_set)
    game_mk = []
    for m in markets:
        if not m["ticker"].split("-", 1)[0].endswith("GAME"):
            continue
        if (m.get("yes_sub_title") or "").strip().lower() in TIE_NAMES:
            continue  # exclude the draw outcome in 3-way markets
        game_mk.append(m)
    by_event = defaultdict(list)
    for m in game_mk:
        by_event[m.get("event_ticker")].append(m)
    print(f"{len(game_mk)} match-winner markets across {len(by_event)} games\n")

    games = []
    for i, (ev, team) in enumerate(by_event.items()):
        prefix = team[0]["ticker"].split("-", 1)[0]
        sides = []
        for mk in team:
            try:
                r = side_record(client, mk)
            except Exception:  # noqa: BLE001
                r = None
            if r is not None:
                sides.append(r)
        if sides:
            games.append({"wc": prefix.startswith("KXWC"), "series": prefix,
                          "event": ev, "sides": sides})
        if (i + 1) % 25 == 0:
            print(f"  ...fetched {i+1}/{len(by_event)} games")

    with open(path, "w") as fh:
        json.dump({"lookback_days": LOOKBACK_DAYS, "games": games}, fh)
    print(f"\ncached {len(games)} games -> {path}")
    return games


# --------------------------------------------------------------------------- #
# Phase 2: analyze
# --------------------------------------------------------------------------- #
def entry_for_level(rows, level):
    """First (ts, entry_ask) where ask in [level, ENTRY_HI], or None."""
    for r in rows:
        ts, ask = r[0], r[1]
        if level <= ask <= ENTRY_HI:
            return int(ts), float(ask)
    return None


def side_position(side, level):
    """Build a position from one side at `level`, or None if it never qualifies."""
    rows = side["rows"]
    e = entry_for_level(rows, level)
    if e is None:
        return None
    entry_ts, entry_ask = e
    opening = float(rows[0][1])

    # fast cross: 10 min before entry, the ask was >= FAST_RISE cents lower.
    prior = [r[1] for r in rows if r[0] <= entry_ts - FAST_WINDOW]
    fast = bool(prior) and (entry_ask - prior[-1] >= FAST_RISE)

    # Empirical buy-side supply: contracts that traded at each ask level from the
    # entry price upward (a depth proxy for "how much could I have bought, and at
    # what price"). volume in the rows is per-candle; we sum by integer cent.
    supply = defaultdict(float)
    for r in rows:
        p = int(round(r[1]))
        if entry_ask <= p <= int(ENTRY_HI) and len(r) > 2:
            supply[p] += r[2]

    return {
        "cost": entry_ask + ENTRY_FEE,
        "settle": 100.0 if side["win"] else 0.0,
        "win": side["win"],
        "entry": entry_ask,
        "ts": entry_ts,
        "opening": opening,
        "fast_cross": fast,
        "cheap_open": opening < CHEAP_OPEN,
        "vol": side["vol"],
        "supply": dict(supply),
    }


def build_positions(games, level):
    """One favorite position per game (first qualifying side). Tags hi_vol via
    the per-level median of game volume."""
    pos = []
    for g in games:
        chosen = None
        for side in g["sides"]:
            p = side_position(side, level)
            if p is not None:
                chosen = p
                break
        if chosen is None:
            continue
        # game volume = sum across the game's sides
        chosen["game_vol"] = sum(s["vol"] for s in g["sides"])
        chosen["wc"] = g["wc"]
        chosen["series"] = g["series"]
        pos.append(chosen)
    if pos:
        med = float(np.median([p["game_vol"] for p in pos]))
        for p in pos:
            p["hi_vol"] = p["game_vol"] >= med
    return pos


def metrics(rows):
    n = len(rows)
    if n == 0:
        return None
    cost = sum(r["cost"] for r in rows)
    proc = sum(r["settle"] for r in rows)
    wins = sum(r["win"] for r in rows)
    return {
        "n": n,
        "win": wins / n * 100,
        "be": cost / n,  # breakeven win% == avg cost in cents
        "roi": (proc - cost) / cost * 100,
        "edge": wins / n * 100 - cost / n,
    }


def best_fraction(rows):
    sized = sorted(rows, key=lambda r: r["ts"])
    sized = [(r["cost"], r["settle"]) for r in sized]
    res = [(f, bankroll_sim(sized, f)[0]) for f in FRACS]
    f, b = max(res, key=lambda x: x[1])
    return f, (b - 1) * 100


def fill_position(p, target_cents, kappa):
    """Walk the empirical supply curve from the entry price upward, buying up to
    `target_cents` worth of contracts (we can take `kappa` of the volume that
    traded at each price). Returns (cost_cents, payoff_cents, contracts)."""
    remaining = target_cents
    contracts = 0.0
    cost = 0.0
    for price in sorted(p["supply"]):
        avail = kappa * p["supply"][price]
        if avail <= 0:
            continue
        per = price + ENTRY_FEE
        take = min(avail, remaining / per)
        if take <= 0:
            break
        contracts += take
        cost += take * per
        remaining -= take * per
        if remaining <= 1e-6:
            break
    payoff = contracts * p["settle"]
    return cost, payoff, contracts


def bucket_scale(positions, target_cents, kappa):
    """Aggregate (capital$, roi%, avg_fill_c, filled_frac) buying `target_cents`
    per bet across a bucket, limited by empirical depth."""
    tot_cost = tot_pay = tot_ctr = 0.0
    want = target_cents * len(positions)
    for p in positions:
        c, pay, ctr = fill_position(p, target_cents, kappa)
        tot_cost += c
        tot_pay += pay
        tot_ctr += ctr
    if tot_cost <= 0:
        return 0.0, 0.0, 0.0, 0.0
    roi = (tot_pay - tot_cost) / tot_cost * 100
    avg_fill = tot_cost / tot_ctr if tot_ctr else 0.0
    filled_frac = tot_cost / want if want else 0.0
    return tot_cost / 100.0, roi, avg_fill, filled_frac


def max_capital_positive(positions, kappa, targets_cents):
    """Largest deployed capital ($) across the bucket that keeps ROI > 0."""
    best_cap = 0.0
    best_roi = None
    for t in targets_cents:
        cap, roi, _, _ = bucket_scale(positions, t, kappa)
        if roi > 0:
            best_cap = cap
            best_roi = roi
    return best_cap, best_roi


def bankroll_sim(trades, frac):
    b, peak, dd = 1.0, 1.0, 0.0
    for cost, settle in trades:
        stake = frac * b
        b = b - stake + stake * settle / cost
        peak = max(peak, b)
        if peak > 0:
            dd = max(dd, (peak - b) / peak)
    return b, dd


def grid(title, buckets, pos_by_level):
    """buckets: list of (name, predicate). Prints ROI%(N) per level."""
    print(f"\n{title}")
    print(f"  {'bucket':18s} " + "  ".join(f"{int(lv):>13d}c" for lv in LEVELS))
    for name, pred in buckets:
        cells = []
        for lv in LEVELS:
            m = metrics([p for p in pos_by_level[lv] if pred(p)])
            if m is None:
                cells.append(f"{'-':>14s}")
            else:
                flag = "*" if m["n"] < MIN_SLICE_N else " "
                cells.append(f"{m['roi']:>+7.1f}%({m['n']:>4d}){flag}")
        print(f"  {name:18s} " + "  ".join(cells))


def edge_grid(title, buckets, pos_by_level):
    """Same buckets, but show win% vs breakeven% (edge in points)."""
    print(f"\n{title}  [cells: win% - be% = edge_pts]")
    print(f"  {'bucket':18s} " + "  ".join(f"{int(lv):>15d}c" for lv in LEVELS))
    for name, pred in buckets:
        cells = []
        for lv in LEVELS:
            m = metrics([p for p in pos_by_level[lv] if pred(p)])
            if m is None:
                cells.append(f"{'-':>16s}")
            else:
                cells.append(f"{m['win']:>4.1f}/{m['be']:>4.1f}={m['edge']:>+4.1f}")
        print(f"  {name:18s} " + "  ".join(cells))


def analyze(games):
    print(f"\nLoaded {len(games)} games "
          f"({sum(g['wc'] for g in games)} World Cup, "
          f"{sum(not g['wc'] for g in games)} other).")

    pos_by_level = {lv: build_positions(games, lv) for lv in LEVELS}

    print("\n" + "=" * 78)
    print(f"BASELINE + UNIVERSE: flat ROI% (N) per entry level   "
          f"[* = N < {MIN_SLICE_N}, noisy]")
    print("=" * 78)
    grid("Universe", [
        ("ALL soccer", lambda p: True),
        ("World Cup", lambda p: p["wc"]),
        ("non-WC", lambda p: not p["wc"]),
    ], pos_by_level)

    grid("Volume (median split, per level)", [
        ("High volume", lambda p: p["hi_vol"]),
        ("Low volume", lambda p: not p["hi_vol"]),
    ], pos_by_level)

    grid("Momentum A: fast cross (>=15c in <=10min)", [
        ("fast cross", lambda p: p["fast_cross"]),
        ("no fast cross", lambda p: not p["fast_cross"]),
    ], pos_by_level)

    grid("Momentum B: cheap open then rose (<70c kickoff)", [
        ("cheap open", lambda p: p["cheap_open"]),
        ("not cheap open", lambda p: not p["cheap_open"]),
    ], pos_by_level)

    grid("Stacked probes", [
        ("WC + high vol", lambda p: p["wc"] and p["hi_vol"]),
        ("fast + high vol", lambda p: p["fast_cross"] and p["hi_vol"]),
        ("cheap + high vol", lambda p: p["cheap_open"] and p["hi_vol"]),
        ("fast + cheap", lambda p: p["fast_cross"] and p["cheap_open"]),
        ("WC + fast", lambda p: p["wc"] and p["fast_cross"]),
    ], pos_by_level)

    print("\n" + "=" * 78)
    print("EDGE CHECK (where, if anywhere, win% beats the price you paid)")
    print("=" * 78)
    edge_grid("Universe", [
        ("ALL soccer", lambda p: True),
        ("World Cup", lambda p: p["wc"]),
        ("non-WC", lambda p: not p["wc"]),
    ], pos_by_level)
    edge_grid("Volume", [
        ("High volume", lambda p: p["hi_vol"]),
        ("Low volume", lambda p: not p["hi_vol"]),
    ], pos_by_level)
    edge_grid("Momentum", [
        ("fast cross", lambda p: p["fast_cross"]),
        ("cheap open", lambda p: p["cheap_open"]),
    ], pos_by_level)

    # Per-league at one representative level (95c).
    lv = 95.0
    print("\n" + "=" * 78)
    print(f"PER-LEAGUE at {int(lv)}c entry (top 12 by N)")
    print("=" * 78)
    by_series = defaultdict(list)
    for p in pos_by_level[lv]:
        by_series[p["series"]].append(p)
    rows = []
    for s, ps in by_series.items():
        m = metrics(ps)
        rows.append((s, m))
    rows.sort(key=lambda x: -x[1]["n"])
    print(f"  {'league':22s} {'N':>4s} {'win%':>6s} {'be%':>6s} {'ROI%':>8s}")
    for s, m in rows[:12]:
        flag = "*" if m["n"] < MIN_SLICE_N else " "
        print(f"  {s:22s} {m['n']:>4d} {m['win']:>6.1f} {m['be']:>6.1f} "
              f"{m['roi']:>+8.1f}{flag}")

    # Positive-EV summary across every slice we computed.
    print("\n" + "=" * 78)
    print(f"POSITIVE-EV SLICES (flat ROI > 0 and N >= {MIN_SLICE_N})")
    print("=" * 78)
    named = {
        "ALL soccer": lambda p: True,
        "World Cup": lambda p: p["wc"],
        "non-WC": lambda p: not p["wc"],
        "High volume": lambda p: p["hi_vol"],
        "Low volume": lambda p: not p["hi_vol"],
        "fast cross": lambda p: p["fast_cross"],
        "cheap open": lambda p: p["cheap_open"],
        "WC+highvol": lambda p: p["wc"] and p["hi_vol"],
        "fast+highvol": lambda p: p["fast_cross"] and p["hi_vol"],
        "cheap+highvol": lambda p: p["cheap_open"] and p["hi_vol"],
        "fast+cheap": lambda p: p["fast_cross"] and p["cheap_open"],
        "WC+fast": lambda p: p["wc"] and p["fast_cross"],
    }
    hits = []
    for lv in LEVELS:
        for nm, pred in named.items():
            m = metrics([p for p in pos_by_level[lv] if pred(p)])
            if m and m["roi"] > 0 and m["n"] >= MIN_SLICE_N:
                f, fret = best_fraction([p for p in pos_by_level[lv] if pred(p)])
                hits.append((m["roi"], nm, int(lv), m, f, fret))
    if not hits:
        print("  None. No slice with adequate N is positive-EV.")
    else:
        hits.sort(reverse=True)
        print(f"  {'slice':16s} {'lvl':>4s} {'N':>5s} {'win%':>6s} {'ROI%':>7s} "
              f"{'optf%':>6s} {'optRet%':>8s}")
        for roi, nm, lv, m, f, fret in hits:
            print(f"  {nm:16s} {lv:>4d} {m['n']:>5d} {m['win']:>6.1f} "
                  f"{roi:>+7.1f} {f*100:>6.0f} {fret:>+8.1f}")
    print("\nCaveat: this is in-sample exploration across many slices; positive "
          "buckets (especially small N) may be multiple-comparison noise.")

    scale_report(pos_by_level)


def scale_report(pos_by_level):
    """How much capital can the (positive-EV) LOW-VOLUME favorite edge absorb
    before walking the book up kills it? Uses the empirical per-cent traded
    volume as a depth proxy; we assume we can take `kappa` of it at each price."""
    print("\n" + "=" * 78)
    print("MAX EXPLOITABLE SCALE -- LOW-VOLUME bucket, hold to settlement")
    print("  Depth = contracts that actually traded at each ask cent (>= entry).")
    print("  We take `kappa` of that volume per price; bigger bets walk up the")
    print("  book -> higher avg fill -> the thin edge erodes.")
    print("=" * 78)

    targets = [t * 100 for t in
               (1, 2, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000)]

    for lv in LEVELS:
        low = [p for p in pos_by_level[lv] if not p["hi_vol"]]
        base = metrics(low)
        print(f"\n-- entry {int(lv)}c | low-vol N={base['n']} | "
              f"flat edge {base['roi']:+.2f}% (tiny size) --")
        # Detailed sweep at kappa=0.25.
        k = 0.25
        print(f"   kappa={k:.2f}:  {'$/bet':>7s} {'capital$':>10s} "
              f"{'ROI%':>7s} {'avgFill':>8s} {'filled%':>8s}")
        for t in targets:
            cap, roi, avg, frac = bucket_scale(low, t, k)
            print(f"            {t/100:>7.0f} {cap:>10,.0f} {roi:>+7.2f} "
                  f"{avg:>8.2f} {frac*100:>7.0f}%")
        # kappa sensitivity: max capital that stays ROI>0.
        print(f"   max capital with ROI>0 (per {LOOKBACK_DAYS}d window):")
        for k in CAPTURES:
            cap, roi = max_capital_positive(low, k, targets)
            sat_cap, sat_roi, _, _ = bucket_scale(low, targets[-1] * 100, k)
            tag = (f"${cap:,.0f} (ROI {roi:+.2f}%)" if cap > 0
                   else "none (negative at all sizes)")
            print(f"      kappa={k:.2f}: {tag};  full-depth cap "
                  f"${sat_cap:,.0f} @ ROI {sat_roi:+.2f}%")

    print("\nReading it: 'capital$' is total deployed across the whole bucket for a "
          f"{LOOKBACK_DAYS}-day window; divide by ~{LOOKBACK_DAYS} for per-day. "
          "Once ROI crosses 0 the edge is gone -- that capital is the ceiling.")


def main():
    refresh = "--refresh" in sys.argv
    if not refresh and os.path.exists(CACHE_PATH):
        with open(CACHE_PATH) as fh:
            data = json.load(fh)
        print(f"Using cache {CACHE_PATH} (lookback {data.get('lookback_days')}d). "
              f"Pass --refresh to refetch.")
        games = data["games"]
    else:
        games = fetch_and_cache(CACHE_PATH)
    analyze(games)


if __name__ == "__main__":
    main()
