"""Stop-loss store and trigger math shared by the UI and the stop engine.

Kalshi has no native stop/trigger order (see :mod:`kalshi.stop_engine`), so a
stop-loss is driven by an out-of-process engine that watches prices and fires a
``reduce_only`` IOC close on trigger. This module holds the pieces both sides
need:

- :class:`StopConfig` -- one stop's static configuration (what to protect).
- Pure trigger math -- map the YES book to the held side and decide when a stop
  fires, plus the YES-book order that closes it (reusing :mod:`kalshi.orders`).
- :class:`StopStore` -- a tiny two-file JSON store. The UI owns the CONFIG file
  (adds/removes/edits); the engine owns the STATUS file (per-stop live state +
  a heartbeat). Splitting the files avoids write contention between the two
  processes. Writes are atomic (temp + ``os.replace``) and reads tolerate a
  missing or partially written file rather than failing.

Prices are handled in CENTS (1-99) to match Kalshi's tick and the rest of the
UI; the held side's price is ``100 - YES`` for a NO holding.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, field, fields
from typing import Any

from kalshi.orders import BookOrder, to_book_order

DEFAULT_STOPS_DIRNAME = ".kalshi_stops"
CONFIG_FILENAME = "config.json"
STATUS_FILENAME = "status.json"

# A stop is considered "actively managed" only if the engine wrote a heartbeat
# within this many seconds. The UI warns past this.
HEARTBEAT_STALE_SECS = 15.0

VALID_SIDES = ("yes", "no")
VALID_REFS = ("bid", "mid", "last")
VALID_ENVS = ("prod", "demo")


def stops_dir() -> str:
    """Directory holding the config/status files (``KALSHI_STOPS_DIR`` override)."""
    override = os.getenv("KALSHI_STOPS_DIR")
    if override:
        return override
    return os.path.join(os.getcwd(), DEFAULT_STOPS_DIRNAME)


@dataclass
class StopConfig:
    """One stop-loss to manage.

    ``stop_cents`` is the level on the HELD side: a NO holding with
    ``stop_cents=85`` stops out when the NO price falls to <= 85c (equivalently
    YES rises to >= 15c). ``count=None`` means "use whatever is held" (resolved
    by the engine at fire time); a numeric ``count`` may be fractional because
    Kalshi supports fixed-point contracts. ``trigger_ref`` chooses which held-side price to
    compare ("bid" = the price you could sell at right now, the safe default).
    """

    ticker: str
    held_side: str  # "yes" | "no"
    stop_cents: float
    count: float | None = None  # contracts (may be fractional); None = full position
    slippage_cents: float = 2.0
    trigger_ref: str = "bid"
    env: str = "prod"
    armed: bool = True
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        self.held_side = self.held_side.lower()
        self.trigger_ref = self.trigger_ref.lower()
        self.env = self.env.lower()
        if self.held_side not in VALID_SIDES:
            raise ValueError(f"held_side must be one of {VALID_SIDES} (got {self.held_side!r}).")
        if self.trigger_ref not in VALID_REFS:
            raise ValueError(f"trigger_ref must be one of {VALID_REFS} (got {self.trigger_ref!r}).")
        if self.env not in VALID_ENVS:
            raise ValueError(f"env must be one of {VALID_ENVS} (got {self.env!r}).")
        if not 1 <= self.stop_cents <= 99:
            raise ValueError(f"stop_cents must be in 1..99 (got {self.stop_cents}).")
        if self.slippage_cents < 0:
            raise ValueError(f"slippage_cents must be >= 0 (got {self.slippage_cents}).")
        if self.count is not None and self.count <= 0:
            raise ValueError(f"count must be positive or None (got {self.count}).")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StopConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


def held_side_prices(
    yes_bid: float | None,
    yes_ask: float | None,
    last: float | None,
    held_side: str,
) -> tuple[float | None, float | None, float | None]:
    """Map YES-book prices (cents) to the held side's (bid, ask, last) in cents.

    For a NO holding every price is ``100 - YES`` and bid/ask swap (the price you
    can SELL your NO at is ``100 - yes_ask``). For YES the prices pass through.
    Any ``None`` input stays ``None``.
    """
    held_side = held_side.lower()
    if held_side == "yes":
        return yes_bid, yes_ask, last

    def inv(v: float | None) -> float | None:
        return None if v is None else 100.0 - v

    # NO bid (sell price) comes from the YES ask; NO ask from the YES bid.
    return inv(yes_ask), inv(yes_bid), inv(last)


def ref_price_cents(
    yes_bid: float | None,
    yes_ask: float | None,
    last: float | None,
    held_side: str,
    trigger_ref: str,
) -> float | None:
    """Return the held-side reference price (cents) selected by ``trigger_ref``."""
    bid, ask, lst = held_side_prices(yes_bid, yes_ask, last, held_side)
    trigger_ref = trigger_ref.lower()
    if trigger_ref == "bid":
        return bid
    if trigger_ref == "last":
        return lst
    if trigger_ref == "mid":
        if bid is None or ask is None:
            return None
        return (bid + ask) / 2.0
    raise ValueError(f"trigger_ref must be one of {VALID_REFS} (got {trigger_ref!r}).")


def is_triggered(ref_cents: float | None, stop_cents: float) -> bool:
    """A stop fires when the held side's reference price falls to/through the stop."""
    return ref_cents is not None and ref_cents <= stop_cents


def exit_book_order(cfg: StopConfig) -> BookOrder:
    """The YES-book order that closes the held position at the slipped stop level.

    We SELL the held side at ``stop_cents - slippage_cents`` so a marketable IOC
    fills, but the slippage cap means a gap-through never fills catastrophically.
    Reuses :func:`kalshi.orders.to_book_order` (sell YES = ask; sell NO = bid at
    ``100 - price``). The result's ``book_side``/``yes_price_dollars`` feed
    ``KalshiClient.create_order`` with ``reduce_only=True``.
    """
    price = cfg.stop_cents - cfg.slippage_cents
    price = max(1.0, min(99.0, price))
    return to_book_order("sell", cfg.held_side, price)


class StopStore:
    """Two-file JSON store: UI-owned config, engine-owned status + heartbeat."""

    def __init__(self, directory: str | None = None):
        self.directory = directory or stops_dir()
        self.config_path = os.path.join(self.directory, CONFIG_FILENAME)
        self.status_path = os.path.join(self.directory, STATUS_FILENAME)

    # --- low-level io ------------------------------------------------------

    def _read_json(self, path: str) -> dict[str, Any]:
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
        except FileNotFoundError:
            return {}
        except (json.JSONDecodeError, OSError):
            # A partially written file is transient; treat as empty for this read
            # rather than crashing the caller (the next write heals it).
            return {}
        return data if isinstance(data, dict) else {}

    def _write_json(self, path: str, payload: dict[str, Any]) -> None:
        os.makedirs(self.directory, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.directory, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, sort_keys=True)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    # --- config (UI-owned) -------------------------------------------------

    def list_configs(self) -> list[StopConfig]:
        raw = self._read_json(self.config_path).get("stops", [])
        out: list[StopConfig] = []
        for item in raw:
            try:
                out.append(StopConfig.from_dict(item))
            except (TypeError, ValueError):
                # Skip malformed entries instead of failing the whole list.
                continue
        return out

    def _write_configs(self, configs: list[StopConfig]) -> None:
        self._write_json(self.config_path, {"stops": [c.to_dict() for c in configs]})

    def add(self, cfg: StopConfig) -> StopConfig:
        configs = self.list_configs()
        configs.append(cfg)
        self._write_configs(configs)
        return cfg

    def remove(self, stop_id: str) -> bool:
        configs = self.list_configs()
        kept = [c for c in configs if c.id != stop_id]
        if len(kept) == len(configs):
            return False
        self._write_configs(kept)
        return True

    def update(self, cfg: StopConfig) -> bool:
        configs = self.list_configs()
        found = False
        for i, existing in enumerate(configs):
            if existing.id == cfg.id:
                configs[i] = cfg
                found = True
                break
        if not found:
            return False
        self._write_configs(configs)
        return True

    # --- status (engine-owned) --------------------------------------------

    def read_status(self) -> dict[str, Any]:
        """Return ``{"heartbeat_ts", "engine_pid", "engine_env", "stops": {id: {...}}}``."""
        data = self._read_json(self.status_path)
        data.setdefault("stops", {})
        return data

    def write_status(self, status: dict[str, Any]) -> None:
        self._write_json(self.status_path, status)

    def heartbeat_age_secs(self, *, now: float | None = None) -> float | None:
        """Seconds since the engine last wrote a heartbeat, or None if never."""
        ts = self.read_status().get("heartbeat_ts")
        if not isinstance(ts, (int, float)):
            return None
        return (now if now is not None else time.time()) - ts

    def is_engine_live(self, *, now: float | None = None) -> bool:
        age = self.heartbeat_age_secs(now=now)
        return age is not None and age <= HEARTBEAT_STALE_SECS
