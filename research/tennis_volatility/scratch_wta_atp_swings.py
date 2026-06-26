"""Do WTA matches swing between leader/loser more than ATP?

A "lead change" = the YES match-winner mid price crossing 50% (the favorite
flips from one player to the other). We pull as many recent finalized ATP and
WTA match-winner markets as possible, isolate each match's in-play window, build
the per-minute mid series, and count:

    lead_changes      : # times mid crosses 50c (with a +/-1c deadband to kill
                        tick flicker around the midpoint)
    lead_changes/hr   : rate, to compare matches of different lengths
    path_per_hr       : sum(|mid move|) per hour (overall swinginess, cents)
    close_minutes_frac: fraction of in-play minutes spent inside 40-60c (a coin
                        flip / contested)

Then compare ATP vs WTA distributions (means, medians, Mann-Whitney U).
"""

from __future__ import annotations

import datetime as dt

import numpy as np
from dotenv import load_dotenv

load_dotenv()

from kalshi.auth import KalshiCredentials
from kalshi.client import KalshiClient

TENNIS_SERIES = ["KXATPMATCH", "KXWTAMATCH"]
LOOKBACK_HOURS = 48
MIN_INPLAY_MINUTES = 40
GAP_BREAK_MIN = 25
DEADBAND = 1.0  # cents around 50 treated as "tied" for lead-change counting


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
    out: list[dict] = []
    now = dt.datetime.now(dt.timezone.utc)
    cutoff = (now - dt.timedelta(hours=LOOKBACK_HOURS)).timestamp()
    for series in TENNIS_SERIES:
        cursor: str | None = None
        seen: set[str] = set()
        while True:
            page = client.get_markets(
                series_ticker=series, limit=1000, cursor=cursor
            )
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
                m["_series"] = series
                out.append(m)
            cursor = page.get("cursor")
            if not cursor:
                break
    return out


def inplay_window(candles: list[dict]) -> tuple[int, int] | None:
    traded = [
        int(c["end_period_ts"])
        for c in candles
        if _vol(c) > 0 and c.get("end_period_ts")
    ]
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


def count_lead_changes(mid: np.ndarray) -> int:
    """Lead changes = sign(mid-50) transitions, with a deadband + carry-forward."""
    lead = 0  # -1 underdog side, +1 favorite (YES) side, 0 unknown
    changes = 0
    for m in mid:
        if m >= 50 + DEADBAND:
            cur = 1
        elif m <= 50 - DEADBAND:
            cur = -1
        else:
            cur = 0  # tied zone -> hold previous lead
        if cur == 0:
            continue
        if lead != 0 and cur != lead:
            changes += 1
        lead = cur
    return changes


def analyze(client: KalshiClient, market: dict) -> dict | None:
    series = market["_series"]
    ticker = market["ticker"]
    o, c = _ts(market["open_time"]), _ts(market["close_time"])
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
    hours = max((end - start) / 3600.0, 1e-6)
    path = float(np.sum(np.abs(np.diff(mid))))
    return {
        "tour": "ATP" if series == "KXATPMATCH" else "WTA",
        "ticker": ticker,
        "minutes": int(len(mid)),
        "hours": hours,
        "lead_changes": count_lead_changes(mid),
        "lead_changes_per_hr": count_lead_changes(mid) / hours,
        "path_per_hr": path / hours,
        "close_frac": float(np.mean((mid >= 40) & (mid <= 60))),
    }


def mann_whitney_u(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Mann-Whitney U with normal approximation -> (U, two-sided p). No scipy."""
    n1, n2 = len(a), len(b)
    combined = np.concatenate([a, b])
    order = np.argsort(combined, kind="mergesort")
    ranks = np.empty(len(combined), dtype=float)
    ranks[order] = np.arange(1, len(combined) + 1)
    # Average ranks for ties.
    s = combined[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            avg = (ranks[order][i] + ranks[order][j]) / 2.0
            ranks[order[i : j + 1]] = avg
        i = j + 1
    r1 = np.sum(ranks[:n1])
    u1 = r1 - n1 * (n1 + 1) / 2.0
    u2 = n1 * n2 - u1
    u = min(u1, u2)
    mu = n1 * n2 / 2.0
    sigma = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
    if sigma == 0:
        return u, 1.0
    z = (u - mu) / sigma
    # two-sided p via normal CDF
    p = 2.0 * 0.5 * (1.0 - _erf(abs(z) / np.sqrt(2.0)))
    return u, float(min(1.0, p))


def _erf(x: float) -> float:
    # Abramowitz-Stegun 7.1.26
    t = 1.0 / (1.0 + 0.3275911 * x)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
                - 0.284496736) * t + 0.254829592) * t * np.exp(-x * x)
    return y


def summarize(label: str, vals: np.ndarray) -> str:
    return (
        f"  {label:18s} n={len(vals):3d}  mean={np.mean(vals):6.2f}  "
        f"median={np.median(vals):6.2f}  "
        f"p75={np.quantile(vals,0.75):6.2f}  max={np.max(vals):6.2f}"
    )


def main() -> None:
    client = KalshiClient(KalshiCredentials.from_env())
    candidates = recent_finalized_markets(client)
    print(f"{len(candidates)} recent finalized match-winner markets; analyzing...\n")

    results: list[dict] = []
    for i, m in enumerate(candidates):
        try:
            r = analyze(client, m)
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {m.get('ticker')}: {exc}")
            continue
        if r:
            results.append(r)
        if (i + 1) % 20 == 0:
            print(f"  ...processed {i+1}/{len(candidates)} ({len(results)} usable)")

    atp = [r for r in results if r["tour"] == "ATP"]
    wta = [r for r in results if r["tour"] == "WTA"]
    print(f"\nUsable: ATP={len(atp)}  WTA={len(wta)}\n")

    def col(rows, key):
        return np.array([r[key] for r in rows], dtype=float)

    for metric, name in [
        ("lead_changes", "lead changes"),
        ("lead_changes_per_hr", "lead chg / hour"),
        ("path_per_hr", "price path c/hr"),
        ("close_frac", "frac in 40-60c"),
    ]:
        a, w = col(atp, metric), col(wta, metric)
        _, p = mann_whitney_u(a, w)
        print(f"{name}:")
        print(summarize("ATP", a))
        print(summarize("WTA", w))
        print(f"  Mann-Whitney two-sided p = {p:.4f}\n")

    # Share of matches with at least one lead change.
    a_any = np.mean(col(atp, "lead_changes") >= 1)
    w_any = np.mean(col(wta, "lead_changes") >= 1)
    print(f"matches with >=1 lead change: ATP {a_any:.0%}  WTA {w_any:.0%}")
    a_2 = np.mean(col(atp, "lead_changes") >= 2)
    w_2 = np.mean(col(wta, "lead_changes") >= 2)
    print(f"matches with >=2 lead changes: ATP {a_2:.0%}  WTA {w_2:.0%}")

    _plot(atp, wta, col)


def _plot(atp, wta, col) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    ax = axes[0]
    bins = np.arange(0, max(col(atp, "lead_changes").max(),
                            col(wta, "lead_changes").max()) + 2) - 0.5
    ax.hist(col(atp, "lead_changes"), bins=bins, alpha=0.5, label="ATP",
            color="#1f77b4", density=True)
    ax.hist(col(wta, "lead_changes"), bins=bins, alpha=0.5, label="WTA",
            color="#d62728", density=True)
    ax.set_xlabel("lead changes per match (mid crosses 50%)")
    ax.set_ylabel("share of matches")
    ax.set_title("Lead changes per match")
    ax.legend()

    ax = axes[1]
    ax.boxplot([col(atp, "lead_changes_per_hr"), col(wta, "lead_changes_per_hr")],
               tick_labels=["ATP", "WTA"], showmeans=True)
    ax.set_ylabel("lead changes per hour")
    ax.set_title("Lead-change rate (length-normalized)")

    ax = axes[2]
    ax.boxplot([col(atp, "path_per_hr"), col(wta, "path_per_hr")],
               tick_labels=["ATP", "WTA"], showmeans=True)
    ax.set_ylabel("price path (cents) per hour")
    ax.set_title("Overall swinginess")

    fig.suptitle("WTA vs ATP in-play swinginess (Kalshi match-winner mid)", fontsize=13)
    fig.tight_layout()
    out = "wta_vs_atp_swings.png"
    fig.savefig(out, dpi=130)
    print(f"\nsaved chart -> {out}")


if __name__ == "__main__":
    main()
