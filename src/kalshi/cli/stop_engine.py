"""``kalshi-stop-engine`` -- the always-on, store-backed stop-loss service.

Reads the UI-owned stop config file, watches each market, and fires protective
``reduce_only`` closes on trigger, writing status + a heartbeat the Portfolio
page reads. Defaults to whatever each stop specifies (production unless the stop
says demo). This only protects while it is running.

Run:
  uv run kalshi-stop-engine
  uv run kalshi-stop-engine --verbose
  uv run python -m kalshi.cli.stop_engine        # equivalent
"""

from __future__ import annotations

import argparse
import asyncio
import logging

from dotenv import load_dotenv

from kalshi.stop_engine import StopEngine, log
from kalshi.stops import StopStore


def main() -> None:
    load_dotenv()
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--dir",
        default=None,
        help="Stops dir (default KALSHI_STOPS_DIR or ./.kalshi_stops).",
    )
    p.add_argument("--verbose", action="store_true", help="Debug-level logging.")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    engine = StopEngine(StopStore(args.dir))
    try:
        asyncio.run(engine.run())
    except KeyboardInterrupt:
        log.warning("Interrupted -- stop engine DOWN. Positions are now UNPROTECTED.")
        raise SystemExit(130)


if __name__ == "__main__":
    main()
