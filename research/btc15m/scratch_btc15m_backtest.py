"""Backtest a "buy the 90% favorite, scalp to 98%" strategy on Kalshi's Bitcoin
15-minute windows (KXBTC15M), compounded sequentially over all available history.

Strategy (per 15-minute window):
  - Watch the YES mid price. The first side (YES or NO) to reach ENTRY (90c) is
    the "favorite"; buy it, deploying DEPLOY (80%) of the current bankroll and
    keeping the rest (20%) in a cash reserve.
  - Entry: fill at ENTRY cents + ENTRY_FEE (1c) -> cost basis 91c per contract.
  - Take-profit: if that side later reaches TAKE (98c), sell -> TAKE - EXIT_FEE
    (97c).
  - Otherwise hold to settlement: 100c if that side wins, 0c if it loses
    (no trading fee on settlement, per Kalshi: holding to expiry pays only the
    buy fee).
  - Compound: bankroll *= (1 + DEPLOY * per_contract_return) each traded window.

Because all deployed capital buys the same contract at the same price, the
fractional bankroll return for a window equals DEPLOY * (proceeds/cost - 1).

Two fill models are reported:
  - IDEALIZED: fills exactly at 90c / 98c (literally the stated rule; optimistic).
  - REALISTIC: pay the ask to enter (ask >= 90c), sell into the bid (bid >= 98c).

The first run pulls every finalized window once and caches the per-window price
paths to btc15m_cache.json; later runs (e.g. tweaking thresholds) read the cache.
"""

from __future__ import annotations

import datetime as dt
import json
import os

import numpy as np
from dotenv import load_dotenv

load_dotenv()

from kalshi.auth import KalshiCredentials
from kalshi.client import KalshiClient

import scratch_btc15m_analysis as base

CACHE = "btc15m_cache.json"

ENTRY = 90.0      # buy when the favorite side reaches this (cents)
TAKE = 98.0       # take-profit sell level (cents)
DEPLOY = 0.80     # fraction of bankroll deployed per trade (20% reserve)
ENTRY_FEE = 1.0   # flat entry fee (cents / contract)
EXIT_FEE = 1.0    # flat exit fee on a take-profit SELL (cents / contract)


# --- cache build (one slow pull) ----------------------------------------


def build_cache(client: KalshiClient) -> list[dict]:
    markets = base.pull_finalized(client)
    print(f"{len(markets)} finalized {base.SERIES} windows; caching price paths...")
    data: list[dict] = []
    for i, m in enumerate(markets):
        res = base.outcome(m)
        if res is None:
            continue
        try:
            o, c = base._ts(m["open_time"]), base._ts(m["close_time"])
        except (KeyError, AttributeError):
            continue
        try:
            candles = client.get_candlesticks(
                base.SERIES, m["ticker"], start_ts=o - 60, end_ts=c + 60,
                period_interval=1,
            ).get("candlesticks", [])
        except Exception as exc:  # noqa: BLE001 - surface, do not swallow
            print(f"  skip {m.get('ticker')}: {exc}")
            continue
        rows = []
        for cd in candles:
            ts = cd.get("end_period_ts")
            if ts is None or not (o <= int(ts) <= c):
                continue
            bid = base._close_dollars(cd.get("yes_bid"))
            ask = base._close_dollars(cd.get("yes_ask"))
            if bid is None or ask is None:
                continue
            bid_c, ask_c = bid * 100.0, ask * 100.0
            spread = ask_c - bid_c
            if spread < 0 or spread > base.MAX_SPREAD_CENTS:
                continue
            rows.append((int(ts), bid_c, ask_c))
        rows.sort(key=lambda r: r[0])
        if len(rows) < base.MIN_INPLAY_MINUTES:
            continue
        data.append({
            "ticker": m["ticker"],
            "close_ts": c,
            "result": res,
            "bid": [r[1] for r in rows],
            "ask": [r[2] for r in rows],
        })
        if (i + 1) % 250 == 0:
            print(f"  ...cached {i+1}/{len(markets)} ({len(data)} usable)")
    with open(CACHE, "w") as fh:
        json.dump(data, fh)
    print(f"wrote {len(data)} windows -> {CACHE}\n")
    return data


def load_windows() -> list[dict]:
    if os.path.exists(CACHE):
        with open(CACHE) as fh:
            data = json.load(fh)
        print(f"loaded {len(data)} windows from {CACHE}\n")
        return data
    client = KalshiClient(KalshiCredentials.from_env())
    return build_cache(client)


# --- single-window trade -------------------------------------------------


def trade_window(w: dict, idealized: bool) -> dict | None:
    """Return {'r', 'side', 'win', 'exit'} for the window, or None if no entry.

    ``exit`` is 'tp' (take-profit), 'settle_win', or 'settle_loss'.
    """
    bid = np.array(w["bid"], dtype=float)
    ask = np.array(w["ask"], dtype=float)
    mid = (bid + ask) / 2.0
    res = w["result"]
    n = len(mid)

    # First side (in time) to reach ENTRY on the mid. YES reaches it at mid>=ENTRY,
    # NO reaches it at (100-mid)>=ENTRY i.e. mid<=100-ENTRY.
    entry_idx = None
    side = None
    for i in range(n):
        if mid[i] >= ENTRY:
            entry_idx, side = i, "yes"
            break
        if (100.0 - mid[i]) >= ENTRY:
            entry_idx, side = i, "no"
            break
    if entry_idx is None:
        return None

    side_wins = (side == res)

    # Entry cost.
    if idealized:
        entry_px = ENTRY
    else:
        entry_px = ask[entry_idx] if side == "yes" else (100.0 - bid[entry_idx])
    cost = entry_px + ENTRY_FEE

    # Take-profit search after entry.
    proceeds = None
    exit_kind = None
    for j in range(entry_idx + 1, n):
        if idealized:
            side_mid = mid[j] if side == "yes" else (100.0 - mid[j])
            if side_mid >= TAKE:
                proceeds, exit_kind = TAKE - EXIT_FEE, "tp"
                break
        else:
            sell_px = bid[j] if side == "yes" else (100.0 - ask[j])
            if sell_px >= TAKE:
                proceeds, exit_kind = sell_px - EXIT_FEE, "tp"
                break

    if proceeds is None:
        proceeds = 100.0 if side_wins else 0.0
        exit_kind = "settle_win" if side_wins else "settle_loss"

    return {
        "r": proceeds / cost - 1.0,
        "side": side,
        "win": side_wins,
        "exit": exit_kind,
    }


# --- backtest ------------------------------------------------------------


def run(windows: list[dict], idealized: bool, label: str) -> None:
    ordered = sorted(windows, key=lambda w: w["close_ts"])
    bankroll = 1.0
    peak = 1.0
    max_dd = 0.0
    trades = 0
    tp = held_win = held_loss = 0
    rs: list[float] = []
    equity = [1.0]
    for w in ordered:
        t = trade_window(w, idealized)
        if t is None:
            continue
        trades += 1
        rs.append(t["r"])
        bankroll *= (1.0 + DEPLOY * t["r"])
        equity.append(bankroll)
        peak = max(peak, bankroll)
        max_dd = max(max_dd, (peak - bankroll) / peak)
        if t["exit"] == "tp":
            tp += 1
        elif t["exit"] == "settle_win":
            held_win += 1
        else:
            held_loss += 1

    span_lo = dt.datetime.fromtimestamp(ordered[0]["close_ts"], dt.timezone.utc)
    span_hi = dt.datetime.fromtimestamp(ordered[-1]["close_ts"], dt.timezone.utc)
    days = max((span_hi - span_lo).total_seconds() / 86400.0, 1e-9)
    rs_arr = np.array(rs)
    wins = tp + held_win  # any non-loss exit (favorite did not blow up)

    print(f"==== {label} fills (entry={ENTRY:.0f}c, take={TAKE:.0f}c, "
          f"deploy={DEPLOY*100:.0f}%, fees {ENTRY_FEE:.0f}c/{EXIT_FEE:.0f}c) ====")
    print(f"  window range        : {span_lo:%Y-%m-%d %H:%M} -> "
          f"{span_hi:%Y-%m-%d %H:%M} UTC  ({days:.1f} days)")
    print(f"  windows in sample   : {len(ordered)}")
    print(f"  trades taken        : {trades}  "
          f"({trades/len(ordered)*100:.1f}% of windows hit {ENTRY:.0f}c)")
    print(f"  take-profit @ {TAKE:.0f}c   : {tp}  ({tp/trades*100:.1f}%)")
    print(f"  held to win (100c)  : {held_win}  ({held_win/trades*100:.1f}%)")
    print(f"  held to LOSS (0c)   : {held_loss}  ({held_loss/trades*100:.1f}%)")
    print(f"  favorite win rate   : {wins/trades*100:.2f}%  "
          f"(blow-up / loss rate {held_loss/trades*100:.2f}%)")
    print(f"  mean per-trade ret  : {rs_arr.mean()*100:+.3f}%   "
          f"median {np.median(rs_arr)*100:+.3f}%")
    print(f"  per-trade win/loss  : +{TAKE-EXIT_FEE-(ENTRY+ENTRY_FEE):.0f}c..."
          f"+{100-(ENTRY+ENTRY_FEE):.0f}c on a win, "
          f"-{ENTRY+ENTRY_FEE:.0f}c on a loss (idealized)")
    print(f"  FINAL bankroll      : {bankroll:.4f}x  "
          f"({(bankroll-1)*100:+.1f}% total)")
    if bankroll > 0 and days > 0:
        cagr = bankroll ** (365.0 / days) - 1.0
        print(f"  annualized (CAGR)   : {cagr*100:+.1f}%/yr")
    print(f"  max drawdown        : {max_dd*100:.1f}%")
    print()


def main() -> None:
    windows = load_windows()
    if not windows:
        print("No windows to backtest.")
        return
    run(windows, idealized=True, label="IDEALIZED")
    run(windows, idealized=False, label="REALISTIC")


if __name__ == "__main__":
    main()
