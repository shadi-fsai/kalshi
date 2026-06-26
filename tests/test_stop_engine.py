"""Unit tests for the StopEngine core (sync/async logic, no real network)."""

import asyncio

import pytest

from kalshi.client import KalshiAPIError
from kalshi.stop_engine import MAX_FILL_RETRIES, StopEngine
from kalshi.stops import StopConfig
from kalshi.ws import Tick


class FakeClient:
    """Fake KalshiClient: configurable positions + create_order behavior."""

    def __init__(self, positions=None, order_error=False):
        self.positions = positions if positions is not None else []
        self.order_error = order_error
        self.orders_placed = []

    def get_positions(self):
        return {"market_positions": self.positions}

    def create_order(self, **kwargs):
        self.orders_placed.append(kwargs)
        if self.order_error:
            raise KalshiAPIError(400, "rejected", "http://x")
        return {"order": {"order_id": f"oid-{len(self.orders_placed)}"}}


def _engine(store=None):
    # Pass dummy credentials so __init__ never reads the environment.
    return StopEngine(store=store, credentials=object())


def _cfg(**kw):
    base = dict(ticker="KXT", held_side="no", stop_cents=85.0)
    base.update(kw)
    return StopConfig(**base)


# --- pure helpers ------------------------------------------------------------


def test_signature_is_stable_and_sensitive():
    cfg = _cfg()
    eng = _engine()
    sig1 = eng.signature(cfg)
    cfg2 = _cfg(stop_cents=80.0)
    assert eng.signature(cfg2) != sig1


def test_evaluate_tick_triggers_on_held_side():
    eng = _engine()
    cfg = _cfg(stop_cents=85.0, trigger_ref="bid")
    # NO bid = 100 - yes_ask*100 = 100 - 42 = 58 <= 85 -> triggered.
    tick = Tick(market_ticker="KXT", yes_bid=0.40, yes_ask=0.42, last=0.41, ts_ms=None)
    assert eng.evaluate_tick(cfg, tick) is True
    assert eng.status[cfg.id]["last_ref_cents"] == pytest.approx(58)

    cfg_low = _cfg(stop_cents=50.0, trigger_ref="bid")
    assert eng.evaluate_tick(cfg_low, tick) is False


def test_client_for_caches():
    eng = _engine()
    c1 = eng.client_for("demo")
    c2 = eng.client_for("demo")
    assert c1 is c2


def test_flush_status_noop_without_store():
    eng = _engine()
    eng.set_status("abc", state="active")
    eng.flush_status()  # must not raise


def test_flush_status_writes_to_store():
    captured = {}

    class FakeStore:
        directory = "/tmp/x"

        def write_status(self, payload):
            captured.update(payload)

    eng = _engine(store=FakeStore())
    eng.set_status("abc", state="active")
    eng.flush_status()
    assert captured["stops"]["abc"]["state"] == "active"
    assert "heartbeat_ts" in captured


# --- fire() ------------------------------------------------------------------


def test_fire_disarmed_places_nothing():
    eng = _engine()
    cfg = _cfg(armed=False)
    client = FakeClient()
    asyncio.run(eng.fire(cfg, client, 10))
    assert eng.status[cfg.id]["state"] == "triggered_disarmed"
    assert client.orders_placed == []


def test_fire_filled_when_position_flattens():
    eng = _engine()
    cfg = _cfg(armed=True)
    # No position remains after the order -> held_contracts returns 0 -> filled.
    client = FakeClient(positions=[])
    asyncio.run(eng.fire(cfg, client, 10))
    assert eng.status[cfg.id]["state"] == "filled"
    assert len(client.orders_placed) == 1
    assert client.orders_placed[0]["reduce_only"] is True
    assert client.orders_placed[0]["time_in_force"] == "immediate_or_cancel"


def test_fire_order_error_sets_error_state():
    eng = _engine()
    cfg = _cfg(armed=True)
    client = FakeClient(order_error=True)
    asyncio.run(eng.fire(cfg, client, 10))
    assert eng.status[cfg.id]["state"] == "error"


def test_fire_partial_failed_after_retries():
    eng = _engine()
    cfg = _cfg(armed=True)
    # Position never shrinks -> retries exhaust -> partial_failed.
    client = FakeClient(positions=[{"ticker": "KXT", "position": -10}])
    asyncio.run(eng.fire(cfg, client, 10))
    assert eng.status[cfg.id]["state"] == "partial_failed"
    assert len(client.orders_placed) == MAX_FILL_RETRIES


# --- watch / run_single ------------------------------------------------------


def test_run_single_no_position(monkeypatch):
    eng = _engine()
    cfg = _cfg()
    monkeypatch.setattr(eng, "client_for", lambda env: FakeClient(positions=[]))
    result = asyncio.run(eng.run_single(cfg))
    assert result["state"] == "no_position"


def test_run_requires_store():
    eng = _engine()
    with pytest.raises(ValueError):
        asyncio.run(eng.run())


# --- reconcile ---------------------------------------------------------------


def test_reconcile_starts_and_cancels_tasks():
    class FakeStore:
        directory = "/tmp/x"

        def __init__(self):
            self.configs = []

        def list_configs(self):
            return self.configs

    cfg = _cfg()
    store = FakeStore()
    store.configs = [cfg]

    async def scenario():
        eng = _engine(store=store)

        async def fake_watch(c):
            await asyncio.sleep(3600)  # stay running so removal must cancel

        eng.watch = fake_watch
        await eng.reconcile_once()
        assert cfg.id in eng.tasks

        store.configs = []  # config removed
        await eng.reconcile_once()
        assert cfg.id not in eng.tasks

    asyncio.run(scenario())
