"""Backtest: buy every World Cup CORRECT SCORE market the moment YES trades < 5c,
then compare two exit rules.

Entry  : first in-play candle where the YES ask is < ENTRY_CAP cents. Pay the ask
         plus the Kalshi buy fee.
Exit A : hold to settlement (100c if it settles YES, 0c if NO; no settlement fee).
Exit B : sell the first time the YES bid reaches >= TRIPLE x the entry ask (pay the
         sell fee). If it never triples, fall back to settlement.

We report aggregate ROI = total proceeds / total cost across every position
(equal 1-contract stake per market).
"""

from __future__ import annotations

import numpy as np
from dotenv import load_dotenv

load_dotenv()

from kalshi.auth import KalshiCredentials
from kalshi.client import KalshiClient
from scratch_soccer_secondary import (
    MAX_CANDLES,
    MIN_INPLAY_MINUTES,
    _ts,
    collect_markets,
    fee_cents,
    inplay_window,
    outcome,
    series,
    soccer_games,
)

SCORE_PREFIX = "KXWCSCORE"  # full-match correct-score markets
ENTRY_CAP = 5.0  # buy when the YES ask is below this (cents)
TRIPLE = 3.0


def backtest_market(client, mk):
    res = outcome(mk)
    if res is None:
        return None  # voided / walkover
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
    _mid, bid, ask = series(candles, win[0], win[1])
    n = len(ask)
    if n < MIN_INPLAY_MINUTES:
        return None

    entry_idx = None
    for i in range(n):
        if 0 < ask[i] < ENTRY_CAP:
            entry_idx = i
            break
    if entry_idx is None:
        return None  # never traded below the cap

    entry = float(ask[entry_idx])
    cost = entry + fee_cents(entry)
    settle = 100.0 if res == "yes" else 0.0

    proceeds_hold = settle

    target = TRIPLE * entry
    proceeds_triple = None
    for j in range(entry_idx + 1, n):
        if bid[j] >= target:
            proceeds_triple = float(bid[j]) - fee_cents(float(bid[j]))
            break
    tripled = proceeds_triple is not None
    if not tripled:
        proceeds_triple = settle

    return {
        "cost": cost,
        "entry": entry,
        "settled_yes": res == "yes",
        "tripled": tripled,
        "proceeds_hold": proceeds_hold,
        "proceeds_triple": proceeds_triple,
    }


def main():
    client = KalshiClient(KalshiCredentials.from_env())
    allowed, series_set, n_games = soccer_games(client)
    print(f"{n_games} WC games; pulling correct-score markets...")
    markets = collect_markets(client, allowed, series_set)
    score = [m for m in markets if m["ticker"].split("-", 1)[0] == SCORE_PREFIX]
    print(f"{len(score)} finalized correct-score markets found\n")

    trades = []
    for i, mk in enumerate(score):
        try:
            t = backtest_market(client, mk)
        except Exception:  # noqa: BLE001
            t = None
        if t is not None:
            trades.append(t)
        if (i + 1) % 100 == 0:
            print(f"  ...{i+1}/{len(score)} scanned ({len(trades)} traded < {ENTRY_CAP:.0f}c)")

    if not trades:
        print("No qualifying trades.")
        return

    n = len(trades)
    cost = sum(t["cost"] for t in trades)
    hold = sum(t["proceeds_hold"] for t in trades)
    trip = sum(t["proceeds_triple"] for t in trades)
    n_yes = sum(t["settled_yes"] for t in trades)
    n_trip = sum(t["tripled"] for t in trades)
    avg_entry = np.mean([t["entry"] for t in trades])

    print(f"Positions (markets that traded < {ENTRY_CAP:.0f}c, 1 contract each): {n}")
    print(f"  settled YES: {n_yes} ({n_yes/n*100:.1f}%)   tripled before expiry: "
          f"{n_trip} ({n_trip/n*100:.1f}%)")
    print(f"  avg entry ask: {avg_entry:.2f}c   total staked: ${cost/100:,.2f}\n")

    def block(name, proceeds):
        net = proceeds - cost
        print(f"{name}")
        print(f"  total returned : ${proceeds/100:,.2f}")
        print(f"  net P&L        : ${net/100:,.2f}")
        print(f"  total ROI      : {net/cost*100:+.1f}%   "
              f"(x{proceeds/cost:.2f} on capital)")
        print(f"  avg per trade  : {net/n:+.2f}c\n")

    block("EXIT A -- hold to settlement:", hold)
    block(f"EXIT B -- sell at {TRIPLE:.0f}x entry, else settle:", trip)


if __name__ == "__main__":
    main()
