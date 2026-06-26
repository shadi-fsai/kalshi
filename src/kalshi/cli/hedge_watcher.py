"""``kalshi-hedge-watcher`` -- a single ad-hoc synthetic stop on one position.

Watches one market and, when the held side's price falls to/through the stop,
fires ONE ``reduce_only`` IOC close. This is the store-less counterpart to the
engine -- handy for protecting a single position without the Portfolio UI.

Safety defaults (per "never fail silently"):
  - Targets DEMO unless --live is passed.
  - DRY RUN unless --arm (a disarmed stop logs what it WOULD do, places nothing).
  - Production arming requires BOTH --arm and --yes-live.
  - The close is capped at ``stop - max-slippage`` so a gap-through can't fill
    catastrophically (it may not fill at all -> loud alert).

Examples:
  # Demo, dry run, auto count from your NO position, stop when NO falls to 85c:
  uv run kalshi-hedge-watcher --ticker KX...-NO

  # Demo, actually place the close on trigger:
  uv run kalshi-hedge-watcher --ticker KX...-NO --arm

  # Production (real money):
  uv run kalshi-hedge-watcher --ticker KX...-NO --arm --live --yes-live
"""

from __future__ import annotations

import argparse
import asyncio
import logging

from dotenv import load_dotenv

from kalshi.stop_engine import StopEngine, banner, log
from kalshi.stops import StopConfig

# Terminal states that mean the stop did not cleanly protect the position.
_FAILURE_STATES = {"partial_failed", "error"}


def main() -> None:
    load_dotenv()
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--ticker", required=True, help="Market ticker of the position to protect.")
    p.add_argument(
        "--held-side",
        choices=("yes", "no"),
        default="no",
        dest="held_side",
        help="Side you hold (default 'no').",
    )
    p.add_argument(
        "--stop-cents",
        type=float,
        default=85.0,
        dest="stop_cents",
        help="Fire when the held side's price falls to/through this (default 85).",
    )
    p.add_argument(
        "--count",
        type=int,
        default=None,
        help="Contracts to close. Default: auto from your position.",
    )
    p.add_argument(
        "--slippage-cents",
        type=float,
        default=2.0,
        dest="slippage_cents",
        help="How far below the stop the close may fill (default 2). Caps gap-through.",
    )
    p.add_argument(
        "--trigger-ref",
        choices=("bid", "mid", "last"),
        default="bid",
        dest="trigger_ref",
        help="Which held-side price to test (default 'bid' = price you could sell at).",
    )
    p.add_argument("--arm", action="store_true", help="Actually place the order (default: dry run).")
    p.add_argument("--live", action="store_true", help="Target PRODUCTION (default: demo).")
    p.add_argument(
        "--yes-live",
        action="store_true",
        dest="yes_live",
        help="Required acknowledgement to arm against production.",
    )
    p.add_argument("--verbose", action="store_true", help="Debug-level logging.")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    env = "prod" if args.live else "demo"
    if env == "prod" and args.arm and not args.yes_live:
        raise SystemExit(
            "Refusing to arm against PRODUCTION without --yes-live. "
            "This will place REAL orders with REAL money."
        )

    cfg = StopConfig(
        ticker=args.ticker,
        held_side=args.held_side,
        stop_cents=args.stop_cents,
        count=args.count,
        slippage_cents=args.slippage_cents,
        trigger_ref=args.trigger_ref,
        env=env,
        armed=args.arm,
    )

    mode = "ARMED (will place orders)" if args.arm else "DRY RUN (no orders)"
    banner(
        f"Synthetic stop | env={env.upper()} | mode={mode}\n"
        f"ticker={cfg.ticker} | hold {cfg.held_side.upper()} | stop@{cfg.stop_cents:.0f}c "
        f"ref={cfg.trigger_ref} | slippage<={cfg.slippage_cents:.0f}c\n"
        "NOTE: synthetic stop -- only protects while this process runs."
    )
    if env == "prod":
        log.warning("Running against PRODUCTION. Real funds are at risk.")

    engine = StopEngine(store=None)
    try:
        result = asyncio.run(engine.run_single(cfg))
    except KeyboardInterrupt:
        log.warning("Interrupted -- stop is no longer active; position UNPROTECTED.")
        raise SystemExit(130)

    state = result.get("state")
    log.info("Final state: %s -- %s", state, result.get("message", ""))
    if state in _FAILURE_STATES:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
