"""Does the price of 'volatility' (yes_ask + no_ask) rise during in-game swings?

For a Kalshi binary match-winner market:
    no_ask = 100 - yes_bid
    => straddle cost = yes_ask + no_ask = 100 + (yes_ask - yes_bid) = 100 + spread

So buying BOTH sides ("volatility") always costs 100c + the bid/ask spread. The
question becomes: does the spread widen in minutes when the mid-price swings?

We pull recent finalized ATP/WTA match-winner markets, isolate the in-play
window (contiguous run of traded minutes), and per minute compute:
    mid     = (yes_bid + yes_ask) / 2          (cents)
    spread  = yes_ask - yes_bid                (cents) == straddle cost - 100
    swing   = |mid_t - mid_{t-1}|              (cents, this minute's move)

Then we test whether spread is larger in high-swing minutes (correlation +
quartile buckets), per match and pooled.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
from dotenv import load_dotenv

load_dotenv()

from kalshi.auth import KalshiCredentials
from kalshi.client import KalshiClient

TENNIS_SERIES = ["KXATPMATCH", "KXWTAMATCH"]
MAX_MATCHES = 8
MIN_INPLAY_MINUTES = 40
GAP_BREAK_MIN = 25  # minutes of no trades that splits the in-play run


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


def recent_finalized_markets(client: KalshiClient) -> list[dict]:
    """One finalized match-winner market per recent event, newest close first."""
    out: list[dict] = []
    now = dt.datetime.now(dt.timezone.utc)
    cutoff = now - dt.timedelta(hours=36)
    for series in TENNIS_SERIES:
        page = client.get_markets(series_ticker=series, limit=1000)
        seen: set[str] = set()
        for m in page.get("markets", []):
            if m.get("status") != "finalized":
                continue
            ev = m.get("event_ticker", "")
            if ev in seen:
                continue
            ct = m.get("close_time")
            if not ct or _ts(ct) < cutoff.timestamp():
                continue
            seen.add(ev)
            m["_series"] = series
            out.append(m)
    out.sort(key=lambda m: m.get("close_time", ""), reverse=True)
    return out


def inplay_window(candles: list[dict]) -> tuple[int, int] | None:
    """Return (start_ts, end_ts) of the last contiguous traded run (the match)."""
    traded = [int(c["end_period_ts"]) for c in candles if _vol(c) > 0 and c.get("end_period_ts")]
    if len(traded) < MIN_INPLAY_MINUTES:
        return None
    traded.sort()
    # Walk back from the last trade while gaps stay under GAP_BREAK_MIN.
    end = traded[-1]
    start = end
    prev = traded[-1]
    for t in reversed(traded[:-1]):
        if prev - t > GAP_BREAK_MIN * 60:
            break
        start = t
        prev = t
    return start, end


def per_minute_series(candles: list[dict], start_ts: int, end_ts: int):
    """Return aligned arrays (ts, mid, spread) for in-play minutes with a 2-sided quote."""
    rows: list[tuple[int, float, float]] = []
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
        # Guard against crossed/degenerate quotes.
        if spread < 0 or spread > 60:
            continue
        rows.append((int(ts), (bid_c + ask_c) / 2.0, spread))
    rows.sort(key=lambda r: r[0])
    ts = np.array([r[0] for r in rows], dtype=float)
    mid = np.array([r[1] for r in rows], dtype=float)
    spread = np.array([r[2] for r in rows], dtype=float)
    return ts, mid, spread


def analyze_match(client: KalshiClient, market: dict):
    series = market["_series"]
    ticker = market["ticker"]
    o, c = _ts(market["open_time"]), _ts(market["close_time"])
    # First pass: wide window to locate the in-play run.
    wide = client.get_candlesticks(
        series, ticker, start_ts=o, end_ts=c, period_interval=1
    ).get("candlesticks", [])
    win = inplay_window(wide)
    if win is None:
        return None
    start, end = win
    # Second pass: tight window for full minute resolution.
    tight = client.get_candlesticks(
        series, ticker, start_ts=start - 120, end_ts=end + 120, period_interval=1
    ).get("candlesticks", [])
    ts, mid, spread = per_minute_series(tight, start, end)
    if len(mid) < MIN_INPLAY_MINUTES:
        return None
    swing = np.abs(np.diff(mid))  # this-minute move
    sp = spread[1:]              # spread aligned to the same minute as the swing
    if np.std(swing) == 0 or np.std(sp) == 0:
        return None
    corr = float(np.corrcoef(swing, sp)[0, 1])
    # Quartile buckets by swing magnitude.
    q75 = np.quantile(swing, 0.75)
    q25 = np.quantile(swing, 0.25)
    hi = sp[swing >= q75]
    lo = sp[swing <= q25]
    return {
        "ticker": ticker,
        "title": market.get("title", ""),
        "minutes": int(len(mid)),
        "median_spread": float(np.median(spread)),
        "corr_swing_spread": corr,
        "spread_lo_swing": float(np.mean(lo)) if len(lo) else float("nan"),
        "spread_hi_swing": float(np.mean(hi)) if len(hi) else float("nan"),
        "ts": ts,
        "mid": mid,
        "spread": spread,
        "swing": swing,
        "sp_aligned": sp,
    }


def main() -> None:
    client = KalshiClient(KalshiCredentials.from_env())
    candidates = recent_finalized_markets(client)
    print(f"{len(candidates)} recent finalized match-winner markets found.\n")

    results = []
    for m in candidates:
        if len(results) >= MAX_MATCHES:
            break
        try:
            r = analyze_match(client, m)
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {m['ticker']}: {exc}")
            continue
        if r is None:
            continue
        results.append(r)
        print(
            f"{r['ticker']:34s} {r['minutes']:4d}m  "
            f"med_spread={r['median_spread']:4.1f}c  "
            f"corr(swing,spread)={r['corr_swing_spread']:+.2f}  "
            f"spread: calm={r['spread_lo_swing']:.2f}c -> swing={r['spread_hi_swing']:.2f}c"
        )

    if not results:
        print("No analyzable matches.")
        return

    # Pooled analysis across all matches.
    all_swing = np.concatenate([r["swing"] for r in results])
    all_sp = np.concatenate([r["sp_aligned"] for r in results])
    pooled_corr = float(np.corrcoef(all_swing, all_sp)[0, 1])
    q75 = np.quantile(all_swing, 0.75)
    q25 = np.quantile(all_swing, 0.25)
    calm = all_sp[all_swing <= q25]
    swingy = all_sp[all_swing >= q75]

    print("\n" + "=" * 70)
    print(f"POOLED over {len(results)} matches, {len(all_swing)} minute-steps")
    print(f"  corr(|mid move|, spread)        = {pooled_corr:+.3f}")
    print(f"  mean spread, calm minutes (Q1)  = {np.mean(calm):.2f}c  (n={len(calm)})")
    print(f"  mean spread, swing minutes (Q4) = {np.mean(swingy):.2f}c  (n={len(swingy)})")
    print(f"  ratio swing/calm                = {np.mean(swingy)/np.mean(calm):.2f}x")
    print("=" * 70)

    _plot(results, all_swing, all_sp)


def _plot(results, all_swing, all_sp) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Pick the match with the most minutes for the time-series panel.
    rep = max(results, key=lambda r: r["minutes"])
    t0 = rep["ts"][0]
    mins = (rep["ts"] - t0) / 60.0

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    ax = axes[0]
    ax.plot(mins, rep["mid"], color="#1f77b4", lw=1.6, label="YES mid (cents)")
    ax.set_xlabel("minutes into in-play window")
    ax.set_ylabel("YES mid price (cents)", color="#1f77b4")
    ax.tick_params(axis="y", labelcolor="#1f77b4")
    ax.set_title(f"{rep['ticker']}\nprice vs bid/ask spread", fontsize=10)
    ax2 = ax.twinx()
    ax2.plot(mins, rep["spread"], color="#d62728", lw=1.0, alpha=0.7,
             label="spread = cost of both sides - 100c")
    ax2.set_ylabel("spread (cents)", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")

    ax = axes[1]
    ax.scatter(all_swing, all_sp, s=8, alpha=0.25, color="#2ca02c")
    # Binned means.
    bins = np.quantile(all_swing, np.linspace(0, 1, 9))
    bins = np.unique(bins)
    idx = np.digitize(all_swing, bins[1:-1])
    bx, by = [], []
    for b in range(len(bins) - 1):
        sel = idx == b
        if sel.sum() >= 5:
            bx.append(all_swing[sel].mean())
            by.append(all_sp[sel].mean())
    ax.plot(bx, by, "o-", color="black", lw=2, label="binned mean spread")
    ax.set_xlabel("|YES mid move| this minute (cents)  [the 'swing']")
    ax.set_ylabel("bid/ask spread (cents)")
    ax.set_title(f"pooled: bigger swings -> wider spread (corr="
                 f"{np.corrcoef(all_swing, all_sp)[0,1]:+.2f})", fontsize=10)
    ax.legend()

    fig.tight_layout()
    out = "tennis_vol_swing.png"
    fig.savefig(out, dpi=130)
    print(f"\nsaved chart -> {out}")


if __name__ == "__main__":
    main()
