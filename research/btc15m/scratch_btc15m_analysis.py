"""Across every Kalshi Bitcoin 15-minute window ("BTC price up in next 15 mins?",
series KXBTC15M), how often did the eventual WINNER once dip below various
win-probability levels (a comeback), how often did the eventual LOSER once climb
above various levels (a favorite blow-up), and how often was a fee-aware <=5c
longshot on the winning side profitable by >10% after Kalshi fees?

Each KXBTC15M market is a single binary per 15-minute window with a clean
``result`` (yes/no), ~16 one-minute candlesticks, and a quadratic fee model
(multiplier 1), so the per-market machinery mirrors the sports blown-leads
scripts but without team/draw/void handling.

Results are reported pooled and broken out by trading session (UTC hour of
close) and by outcome direction (up-settled vs down-settled).
"""

from __future__ import annotations

import datetime as dt
from collections import defaultdict

import numpy as np
from dotenv import load_dotenv

load_dotenv()

from kalshi.auth import KalshiCredentials
from kalshi.client import KalshiClient
from kalshi.fees import FeeModel

SERIES = "KXBTC15M"
LOOKBACK_HOURS: int | None = None  # None = all available finalized windows
MAX_WINDOWS = 6500  # hard cap to bound runtime (one candlestick call per window)
MIN_INPLAY_MINUTES = 6  # need at least this many quoted minutes in the 15-min window
MAX_SPREAD_CENTS = 60.0  # drop crossed / degenerate quotes wider than this

LEVELS_LOW = [40, 30, 20, 10, 5, 2]  # winner dipped BELOW these (comebacks)
LEVELS_HIGH = [60, 70, 80, 90, 95]  # loser climbed ABOVE these (blow-ups)
SUSTAIN_MIN = 2  # minutes a loser must hold above a level to count as "sustained"

ENTRY_CAP_CENTS = 5.0  # buy the winning side only when its ask dips to <= this
SPIKE_NET_THR = 0.10  # net return threshold (>10%) after buy+sell fees

# KXBTC15M is quadratic with multiplier 1 (confirmed via GET /series/KXBTC15M).
FEE = FeeModel(fee_type="quadratic", fee_multiplier=1.0)


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


def session_of(close_ts: int) -> str:
    """Coarse trading session from the UTC hour of the window's close."""
    hour = dt.datetime.fromtimestamp(close_ts, dt.timezone.utc).hour
    if 0 <= hour < 8:
        return "Asia"      # ~00:00-08:00 UTC
    if 8 <= hour < 14:
        return "EU"        # ~08:00-14:00 UTC
    return "US"            # ~14:00-24:00 UTC


def fee_cents(price_cents: float) -> float:
    """Per-contract Kalshi fee (cents) for one contract at ``price_cents``."""
    fee = FEE.order_fee(1, price_cents / 100.0)
    return 0.0 if fee is None else fee * 100.0


# --- data pull -----------------------------------------------------------


def pull_finalized(client: KalshiClient) -> list[dict]:
    """All finalized KXBTC15M markets (optionally within LOOKBACK_HOURS)."""
    cutoff = None
    if LOOKBACK_HOURS is not None:
        cutoff = (
            dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=LOOKBACK_HOURS)
        ).timestamp()
    out: list[dict] = []
    cursor: str | None = None
    while True:
        page = client.get_markets(series_ticker=SERIES, limit=1000, cursor=cursor)
        for m in page.get("markets", []):
            if m.get("status") != "finalized":
                continue
            ct = m.get("close_time")
            if not ct:
                continue
            if cutoff is not None and _ts(ct) < cutoff:
                continue
            out.append(m)
        cursor = page.get("cursor")
        if not cursor or len(out) >= MAX_WINDOWS:
            break
    out.sort(key=lambda m: m.get("close_time", ""), reverse=True)
    return out[:MAX_WINDOWS]


# --- per-window series + outcome ----------------------------------------


def outcome(market: dict) -> str | None:
    """'yes', 'no', or None (non-decisive) for a window.

    Prefer the explicit ``result`` field; fall back to
    ``settlement_value_dollars`` (1.0 -> yes, 0.0 -> no).
    """
    res = (market.get("result") or "").strip().lower()
    if res in ("yes", "no"):
        return res
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


def mid_series(candles, start_ts, end_ts):
    """Aligned YES (mid, bid, ask) arrays in cents over the in-play window."""
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
        if spread < 0 or spread > MAX_SPREAD_CENTS:
            continue
        rows.append((int(ts), (bid_c + ask_c) / 2.0, bid_c, ask_c))
    rows.sort(key=lambda r: r[0])
    mid = np.array([r[1] for r in rows], dtype=float)
    bid = np.array([r[2] for r in rows], dtype=float)
    ask = np.array([r[3] for r in rows], dtype=float)
    return mid, bid, ask


# --- the >10% spike opportunity (fee-aware) -----------------------------


def spike_profit_gt_thr(bid, ask, entry_cap=ENTRY_CAP_CENTS, thr=SPIKE_NET_THR) -> bool:
    """Buy the (winning) side at the ask when ask <= entry_cap, sell at the best
    LATER bid; is the net return > thr after Kalshi buy+sell fees?"""
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


def window_metrics(client: KalshiClient, market: dict):
    """Return per-window metrics dict, or None if not measurable.

    winner prob path = YES mid (if result yes) else 100 - YES mid.
    loser prob path  = the complement. Comebacks use the winner's trough;
    blow-ups use the loser's peak; the spike test uses the winner side's quotes.
    """
    res = outcome(market)
    if res is None:
        return None
    try:
        o, c = _ts(market["open_time"]), _ts(market["close_time"])
    except (KeyError, AttributeError):
        return None
    candles = client.get_candlesticks(
        SERIES, market["ticker"], start_ts=o - 60, end_ts=c + 60, period_interval=1
    ).get("candlesticks", [])
    mid, bid, ask = mid_series(candles, o, c)
    if len(mid) < MIN_INPLAY_MINUTES:
        return None

    if res == "yes":
        winner_prob = mid
        win_bid, win_ask = bid, ask
    else:
        winner_prob = 100.0 - mid
        # Winner is the NO side: its bid/ask are 100 - yes ask/bid.
        win_bid, win_ask = 100.0 - ask, 100.0 - bid
    loser_prob = 100.0 - winner_prob

    trough = float(np.min(winner_prob))  # comeback depth
    peak = float(np.max(loser_prob))     # blow-up height
    sustained = {
        lv: int(np.sum(loser_prob > lv)) >= SUSTAIN_MIN for lv in LEVELS_HIGH
    }
    spike = spike_profit_gt_thr(win_bid, win_ask)
    return {
        "ticker": market["ticker"],
        "session": session_of(c),
        "direction": "up" if res == "yes" else "down",
        "trough": trough,
        "peak": peak,
        "sustained": sustained,
        "spike": spike,
    }


# --- aggregation + output ------------------------------------------------


def _comeback_row(troughs: np.ndarray) -> str:
    return " ".join(
        f"{np.sum(troughs < lv) / len(troughs) * 100:>4.1f}" for lv in LEVELS_LOW
    )


def _blowup_row(peaks: np.ndarray) -> str:
    return " ".join(
        f"{np.sum(peaks > lv) / len(peaks) * 100:>4.1f}" for lv in LEVELS_HIGH
    )


def report(rows: list[dict], skipped: int, non_decisive: int, total: int) -> None:
    n = len(rows)
    troughs = np.array([r["trough"] for r in rows])
    peaks = np.array([r["peak"] for r in rows])
    spikes = [r["spike"] for r in rows]

    print(f"\nFinalized windows pulled: {total}")
    print(f"Measured (>= {MIN_INPLAY_MINUTES} in-play min, decisive): {n}")
    print(f"Skipped (insufficient in-play data): {skipped}   "
          f"Non-decisive (no clean result): {non_decisive}")
    if LOOKBACK_HOURS is not None:
        print(f"(lookback {LOOKBACK_HOURS}h)")
    print(f"(capped at {MAX_WINDOWS} windows)\n")

    if n == 0:
        print("No measurable windows.")
        return

    print("COMEBACKS - share of windows where the eventual WINNER once dipped "
          "BELOW each level\n(then still settled to 100%):\n")
    print("level :  " + "  ".join(f"<{lv:>2d}" for lv in LEVELS_LOW))
    print("ALL % :  " + _comeback_row(troughs))
    print("(n)   :  " + "  ".join(
        f"{int(np.sum(troughs < lv)):>3d}" for lv in LEVELS_LOW) + f"   of {n}")

    print("\nFAVORITE BLOW-UPS - share of windows where the eventual LOSER once "
          "climbed ABOVE each level\n(then still lost):\n")
    print("level :  " + "  ".join(f">{lv:>2d}" for lv in LEVELS_HIGH))
    print("ALL % :  " + _blowup_row(peaks))
    sus_pct = [
        np.mean([r["sustained"][lv] for r in rows]) * 100 for lv in LEVELS_HIGH
    ]
    print(f"sus%* :  " + " ".join(f"{p:>4.1f}" for p in sus_pct))
    print(f"   *sustained = loser held above the level for >= {SUSTAIN_MIN} in-play min")

    print(f"\n>10%net (fee-aware spike): buy the winning side at the ask when it "
          f"dips to <={ENTRY_CAP_CENTS:.0f}c,\nsell at the best later bid; net "
          f">{SPIKE_NET_THR*100:.0f}% after Kalshi buy+sell fees. "
          f"ALL: {sum(spikes)/n*100:.1f}% ({sum(spikes)}/{n})\n")

    # Per-segment breakdown (session rows + direction rows).
    print("per segment:")
    head_low = " ".join(f"<{lv:>2d}" for lv in LEVELS_LOW)
    head_high = " ".join(f">{lv:>2d}" for lv in LEVELS_HIGH)
    print(f"{'segment':12s} {'N':>5s}  | comeback {head_low}  | blow-up {head_high}  | >10%")

    def _seg(label: str, sel: list[dict]) -> None:
        if not sel:
            return
        tr = np.array([r["trough"] for r in sel])
        pk = np.array([r["peak"] for r in sel])
        sp = [r["spike"] for r in sel]
        print(f"{label:12s} {len(sel):>5d}  |          {_comeback_row(tr)}  "
              f"|         {_blowup_row(pk)}  | {sum(sp)/len(sel)*100:>4.1f}")

    by_session: dict[str, list[dict]] = defaultdict(list)
    by_direction: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_session[r["session"]].append(r)
        by_direction[r["direction"]].append(r)

    for s in ("Asia", "EU", "US"):
        _seg(f"session:{s}", by_session.get(s, []))
    for d in ("up", "down"):
        _seg(f"dir:{d}", by_direction.get(d, []))

    _plot(troughs, peaks)


def _plot(troughs: np.ndarray, peaks: np.ndarray) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"\n(skipped chart: matplotlib unavailable: {exc})")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(troughs, bins=50, range=(0, 100), color="#2ca02c", alpha=0.85)
    axes[0].set_title("Winner's lowest in-play probability (comeback depth)")
    axes[0].set_xlabel("winner trough (cents / %)")
    axes[0].set_ylabel("windows")
    axes[1].hist(peaks, bins=50, range=(0, 100), color="#d62728", alpha=0.85)
    axes[1].set_title("Loser's highest in-play probability (blow-up height)")
    axes[1].set_xlabel("loser peak (cents / %)")
    axes[1].set_ylabel("windows")
    fig.suptitle("KXBTC15M - BTC 15-minute windows: comeback / blow-up distributions")
    fig.tight_layout()
    out = "btc15m_analysis.png"
    fig.savefig(out, dpi=130)
    print(f"\nsaved chart -> {out}")


def main() -> None:
    client = KalshiClient(KalshiCredentials.from_env())
    markets = pull_finalized(client)
    print(f"{len(markets)} finalized {SERIES} windows pulled; measuring...")

    rows: list[dict] = []
    skipped = 0
    non_decisive = 0
    for i, m in enumerate(markets):
        if outcome(m) is None:
            non_decisive += 1
            continue
        try:
            r = window_metrics(client, m)
        except Exception as exc:  # noqa: BLE001 - surface, never swallow silently
            print(f"  skip {m.get('ticker')}: {exc}")
            skipped += 1
            continue
        if r is None:
            skipped += 1
            continue
        rows.append(r)
        if (i + 1) % 250 == 0:
            print(f"  ...scanned {i+1}/{len(markets)} ({len(rows)} measured)")

    report(rows, skipped, non_decisive, len(markets))


if __name__ == "__main__":
    main()
