"""Synthetic stop-loss engine for Kalshi.

Kalshi has NO native stop/trigger order, so a stop must be driven by a process we
keep running. :class:`StopEngine` is that process. It watches each market's live
price over the WS ``ticker`` channel (REST fallback when the socket drops) and on
trigger fires ONE ``reduce_only`` IOC order that can only flatten the held
position.

Two ways to drive it:

- :meth:`StopEngine.run` -- store-backed, multi-stop. Reconciles the UI-owned
  config file into one watcher task per stop and writes per-stop status plus a
  heartbeat so the Portfolio page can show progress and warn if the engine is
  down. This is what the ``kalshi-stop-engine`` console script runs.
- :meth:`StopEngine.run_single` -- store-less, one ad-hoc stop, returns the
  terminal status. This is what the ``kalshi-hedge-watcher`` console script runs.

IMPORTANT: This is a *synthetic* stop -- it only protects while the process is
running. If it is not running, there is no protection.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid

from kalshi.auth import KalshiCredentials
from kalshi.client import KalshiAPIError, KalshiClient
from kalshi.positions import base_url_for_env, held_contracts, rest_yes_prices
from kalshi.stops import (
    StopConfig,
    StopStore,
    exit_book_order,
    is_triggered,
    ref_price_cents,
)
from kalshi.ws import Tick, stream_ticker

log = logging.getLogger("kalshi.stop_engine")

RECONCILE_SECS = 3.0  # how often to re-read the config file
STATUS_FLUSH_SECS = 2.0  # how often to write status + heartbeat
BACKOFF_START = 1.0
BACKOFF_MAX = 30.0
MAX_FILL_RETRIES = 3
REST_FALLBACK_CYCLES = 5
REST_FALLBACK_SECS = 2.0

# States that mean "do not restart this watcher".
TERMINAL_STATES = {"filled", "partial_failed", "error", "no_position", "triggered_disarmed"}


def banner(msg: str) -> None:
    """Log a hard-to-miss banner for key state changes (loud, never silent)."""
    bar = "=" * 70
    log.info("\n%s\n%s\n%s", bar, msg, bar)


class StopEngine:
    """Watches markets and fires protective ``reduce_only`` closes on trigger.

    ``store`` is optional: pass a :class:`~kalshi.stops.StopStore` for the
    store-backed multi-stop service (:meth:`run`), or ``None`` for an ad-hoc
    single stop (:meth:`run_single`) that never touches the shared files.
    """

    def __init__(
        self,
        store: StopStore | None = None,
        *,
        credentials: KalshiCredentials | None = None,
    ):
        self.store = store
        self.creds = credentials or KalshiCredentials.from_env()
        self._clients: dict[str, KalshiClient] = {}
        self.tasks: dict[str, asyncio.Task] = {}
        self.signatures: dict[str, tuple] = {}
        # In-memory per-stop status; flushed to disk by a single writer task so
        # concurrent watchers never contend on the file.
        self.status: dict[str, dict] = {}
        self.engine_id = uuid.uuid4().hex[:8]
        self._stop = asyncio.Event()

    # --- clients -----------------------------------------------------------

    def client_for(self, env: str) -> KalshiClient:
        if env not in self._clients:
            self._clients[env] = KalshiClient(self.creds, base_url=base_url_for_env(env))
        return self._clients[env]

    # --- status helpers ----------------------------------------------------

    def set_status(self, stop_id: str, **fields) -> None:
        entry = self.status.setdefault(stop_id, {})
        entry.update(fields)
        entry["last_update_ts"] = time.time()

    @staticmethod
    def signature(cfg: StopConfig) -> tuple:
        return (
            cfg.ticker,
            cfg.held_side,
            cfg.stop_cents,
            cfg.count,
            cfg.slippage_cents,
            cfg.trigger_ref,
            cfg.env,
            cfg.armed,
        )

    def flush_status(self) -> None:
        if self.store is None:
            return
        payload = {
            "heartbeat_ts": time.time(),
            "engine_id": self.engine_id,
            "engine_pid": os.getpid(),
            "stops": self.status,
        }
        try:
            self.store.write_status(payload)
        except OSError as exc:
            log.error("failed to write status file: %s", exc)

    async def status_loop(self) -> None:
        while not self._stop.is_set():
            self.flush_status()
            await asyncio.sleep(STATUS_FLUSH_SECS)
        self.flush_status()

    # --- reconcile loop (store-backed) -------------------------------------

    async def reconcile_once(self) -> None:
        assert self.store is not None
        configs = {c.id: c for c in self.store.list_configs()}

        # Cancel watchers whose config was removed or materially changed.
        for stop_id in list(self.tasks):
            cfg = configs.get(stop_id)
            task = self.tasks[stop_id]
            changed = cfg is not None and self.signatures.get(stop_id) != self.signature(cfg)
            if cfg is None or changed:
                if not task.done():
                    task.cancel()
                self.tasks.pop(stop_id, None)
                self.signatures.pop(stop_id, None)
                if cfg is None:
                    self.status.pop(stop_id, None)  # gone from config -> drop status

        # Start watchers for new/changed/crashed (non-terminal) stops.
        for stop_id, cfg in configs.items():
            existing = self.tasks.get(stop_id)
            if existing is not None and not existing.done():
                continue
            state = self.status.get(stop_id, {}).get("state")
            if existing is not None and existing.done() and state in TERMINAL_STATES:
                continue  # finished cleanly; do not restart
            self.signatures[stop_id] = self.signature(cfg)
            self.tasks[stop_id] = asyncio.create_task(self.watch(cfg), name=f"stop-{stop_id}")

    async def reconcile_loop(self) -> None:
        while not self._stop.is_set():
            try:
                await self.reconcile_once()
            except Exception as exc:  # never let the loop die silently
                log.exception("reconcile error: %s", exc)
            await asyncio.sleep(RECONCILE_SECS)

    # --- per-stop watcher --------------------------------------------------

    async def watch(self, cfg: StopConfig) -> None:
        client = self.client_for(cfg.env)
        try:
            held = held_contracts(client, cfg.ticker, cfg.held_side)
        except KalshiAPIError as exc:
            self.set_status(cfg.id, state="error", message=f"positions lookup failed: {exc}")
            log.error("[%s] positions lookup failed: %s", cfg.ticker, exc)
            return
        count = cfg.count if cfg.count is not None else held
        if count <= 0:
            self.set_status(
                cfg.id,
                state="no_position",
                count=0,
                message=f"No {cfg.held_side.upper()} position held for {cfg.ticker}.",
            )
            log.warning("[%s] no %s position to protect.", cfg.ticker, cfg.held_side.upper())
            return

        self.set_status(
            cfg.id,
            state="active",
            count=count,
            env=cfg.env,
            armed=cfg.armed,
            ticker=cfg.ticker,
            held_side=cfg.held_side,
            stop_cents=cfg.stop_cents,
            message="watching",
        )
        log.info(
            "[%s] watching: protect %.2f %s, stop@%.0fc ref=%s env=%s armed=%s",
            cfg.ticker, count, cfg.held_side.upper(), cfg.stop_cents,
            cfg.trigger_ref, cfg.env, cfg.armed,
        )

        backoff = BACKOFF_START
        while not self._stop.is_set():
            try:
                async for tick in stream_ticker(self.creds, base_url_for_env(cfg.env), cfg.ticker):
                    backoff = BACKOFF_START
                    if self.evaluate_tick(cfg, tick):
                        await self.fire(cfg, client, count)
                        return
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.error("[%s] WS error: %s", cfg.ticker, exc)

            # WS down -> REST fallback so the stop is never blind.
            if await self.rest_fallback(cfg, client, count):
                return
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, BACKOFF_MAX)

    def evaluate_tick(self, cfg: StopConfig, tick: Tick) -> bool:
        # WS Tick prices are in DOLLARS; convert to cents for the held-side math.
        def c(v: float | None) -> float | None:
            return None if v is None else v * 100.0

        ref = ref_price_cents(c(tick.yes_bid), c(tick.yes_ask), c(tick.last), cfg.held_side, cfg.trigger_ref)
        self.set_status(
            cfg.id,
            last_ref_cents=ref,
            last_yes_bid=c(tick.yes_bid),
            last_yes_ask=c(tick.yes_ask),
        )
        return is_triggered(ref, cfg.stop_cents)

    async def rest_fallback(self, cfg: StopConfig, client: KalshiClient, count: float) -> bool:
        log.warning("[%s] WS unavailable -- REST polling so the stop is not blind.", cfg.ticker)
        for _ in range(REST_FALLBACK_CYCLES):
            if self._stop.is_set():
                return False
            try:
                yb, ya, last = rest_yes_prices(client, cfg.ticker)
                ref = ref_price_cents(yb, ya, last, cfg.held_side, cfg.trigger_ref)
                self.set_status(cfg.id, last_ref_cents=ref, last_yes_bid=yb, last_yes_ask=ya)
                if is_triggered(ref, cfg.stop_cents):
                    await self.fire(cfg, client, count)
                    return True
            except KalshiAPIError as exc:
                log.error("[%s] REST poll failed: %s", cfg.ticker, exc)
            await asyncio.sleep(REST_FALLBACK_SECS)
        return False

    async def fire(self, cfg: StopConfig, client: KalshiClient, count: float) -> None:
        bo = exit_book_order(cfg)
        price = bo.yes_price_dollars
        banner(
            f"TRIGGER {cfg.ticker}: {cfg.held_side.upper()} ref <= {cfg.stop_cents:.0f}c. "
            f"Closing {count:.2f} via {bo.book_side} YES @ {price:.2f} (IOC, reduce_only)."
        )

        if not cfg.armed:
            self.set_status(
                cfg.id,
                state="triggered_disarmed",
                fired_at=time.time(),
                message=f"Triggered but DISARMED; would sell {count:.2f} {cfg.held_side.upper()} @ stop {cfg.stop_cents:.0f}c.",
            )
            log.warning("[%s] triggered but DISARMED -- no order placed.", cfg.ticker)
            return

        self.set_status(cfg.id, state="closing", fired_at=time.time(), message="placing close order")
        remaining = count
        order_ids: list[str] = []
        for attempt in range(1, MAX_FILL_RETRIES + 1):
            coid = f"stop-{cfg.id}-{attempt}-{uuid.uuid4().hex[:6]}"
            try:
                resp = client.create_order(
                    ticker=cfg.ticker,
                    book_side=bo.book_side,
                    count=remaining,
                    price_dollars=price,
                    client_order_id=coid,
                    reduce_only=True,
                    time_in_force="immediate_or_cancel",
                )
            except KalshiAPIError as exc:
                self.set_status(cfg.id, state="error", message=f"order failed: {exc}")
                log.error("[%s] order attempt %d FAILED: %s", cfg.ticker, attempt, exc)
                return
            order = resp.get("order", resp)
            oid = order.get("order_id") or order.get("id")
            if oid:
                order_ids.append(oid)
            log.info("[%s] order attempt %d: %s", cfg.ticker, attempt, order)

            try:
                remaining = held_contracts(client, cfg.ticker, cfg.held_side)
            except KalshiAPIError as exc:
                self.set_status(cfg.id, state="error", message=f"post-fill check failed: {exc}", order_ids=order_ids)
                log.error("[%s] post-fill position check failed: %s", cfg.ticker, exc)
                return
            if remaining <= 0:
                self.set_status(cfg.id, state="filled", remaining=0, order_ids=order_ids, message="flat -- protective close complete")
                banner(f"FLAT {cfg.ticker}: protective close complete.")
                return
            self.set_status(cfg.id, state="closing", remaining=remaining, order_ids=order_ids, message=f"partial; {remaining:.2f} left")
            log.warning("[%s] partial close: %.2f remaining after attempt %d.", cfg.ticker, remaining, attempt)

        self.set_status(
            cfg.id,
            state="partial_failed",
            remaining=remaining,
            order_ids=order_ids,
            message=f"STOP DID NOT FULLY FILL: {remaining:.2f} left after {MAX_FILL_RETRIES} attempts. MANUAL ACTION REQUIRED.",
        )
        banner(
            f"!!! STOP DID NOT FULLY FILL !!! {cfg.ticker} has {remaining:.2f} left after "
            f"{MAX_FILL_RETRIES} attempts at <= {price:.2f}. MANUAL ACTION REQUIRED."
        )

    # --- lifecycle ---------------------------------------------------------

    def request_stop(self) -> None:
        self._stop.set()

    async def run(self) -> None:
        """Store-backed multi-stop service: reconcile + status loops forever."""
        if self.store is None:
            raise ValueError("run() requires a StopStore; use run_single() for an ad-hoc stop.")
        banner(
            f"Stop engine {self.engine_id} starting | dir={self.store.directory}\n"
            "Synthetic stops -- only active while this process runs."
        )
        await asyncio.gather(self.reconcile_loop(), self.status_loop())

    async def run_single(self, cfg: StopConfig) -> dict:
        """Watch ONE ad-hoc stop until it reaches a terminal state; return status."""
        flusher = asyncio.create_task(self.status_loop())
        try:
            await self.watch(cfg)
        finally:
            self._stop.set()
            await flusher
        return self.status.get(cfg.id, {})
