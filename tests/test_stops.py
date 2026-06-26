"""Unit tests for the stop-loss store and trigger math."""

import time

import pytest

from kalshi.stops import (
    StopConfig,
    StopStore,
    exit_book_order,
    held_side_prices,
    is_triggered,
    ref_price_cents,
)


# --- StopConfig --------------------------------------------------------------


def test_config_roundtrip_serialization():
    cfg = StopConfig(ticker="KXT-NO", held_side="no", stop_cents=85, count=10)
    restored = StopConfig.from_dict(cfg.to_dict())
    assert restored == cfg


def test_config_from_dict_ignores_unknown_keys():
    cfg = StopConfig.from_dict(
        {"ticker": "KXT", "held_side": "yes", "stop_cents": 50, "bogus": 1}
    )
    assert cfg.ticker == "KXT"
    assert cfg.held_side == "yes"


def test_config_normalizes_case():
    cfg = StopConfig(ticker="KXT", held_side="NO", stop_cents=80, trigger_ref="BID", env="PROD")
    assert cfg.held_side == "no"
    assert cfg.trigger_ref == "bid"
    assert cfg.env == "prod"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"held_side": "maybe"},
        {"trigger_ref": "wat"},
        {"env": "staging"},
        {"stop_cents": 0},
        {"stop_cents": 100},
        {"slippage_cents": -1},
        {"count": 0},
    ],
)
def test_config_validation(kwargs):
    base = {"ticker": "KXT", "held_side": "no", "stop_cents": 85}
    base.update(kwargs)
    with pytest.raises(ValueError):
        StopConfig(**base)


# --- trigger math ------------------------------------------------------------


def test_held_side_prices_yes_passthrough():
    assert held_side_prices(40, 42, 41, "yes") == (40, 42, 41)


def test_held_side_prices_no_inverts_and_swaps():
    # NO bid (sell price) = 100 - yes_ask; NO ask = 100 - yes_bid.
    assert held_side_prices(40, 42, 41, "no") == (58, 60, 59)


def test_held_side_prices_handles_none():
    assert held_side_prices(None, 42, None, "no") == (58, None, None)


def test_ref_price_bid_mid_last_yes():
    assert ref_price_cents(40, 42, 41, "yes", "bid") == 40
    assert ref_price_cents(40, 42, 41, "yes", "last") == 41
    assert ref_price_cents(40, 42, 41, "yes", "mid") == pytest.approx(41)


def test_ref_price_mid_none_when_missing_side():
    assert ref_price_cents(None, 42, 41, "yes", "mid") is None


def test_is_triggered_fires_at_or_below_stop():
    assert is_triggered(85, 85) is True
    assert is_triggered(84.9, 85) is True
    assert is_triggered(85.1, 85) is False
    assert is_triggered(None, 85) is False


def test_exit_book_order_no_sells_via_bid():
    # NO held: sell NO at stop-slippage = 83c -> buy YES at 17c (bid).
    cfg = StopConfig(ticker="KXT", held_side="no", stop_cents=85, slippage_cents=2)
    bo = exit_book_order(cfg)
    assert bo.book_side == "bid"
    assert bo.yes_price_dollars == pytest.approx(0.17)


def test_exit_book_order_yes_sells_via_ask():
    cfg = StopConfig(ticker="KXT", held_side="yes", stop_cents=85, slippage_cents=2)
    bo = exit_book_order(cfg)
    assert bo.book_side == "ask"
    assert bo.yes_price_dollars == pytest.approx(0.83)


def test_exit_book_order_clamps_low_stop():
    # stop 1c - slippage 2c would be negative; clamp to 1c.
    cfg = StopConfig(ticker="KXT", held_side="yes", stop_cents=1, slippage_cents=2)
    bo = exit_book_order(cfg)
    assert bo.yes_price_dollars == pytest.approx(0.01)


# --- StopStore ---------------------------------------------------------------


def test_store_empty_reads(tmp_path):
    store = StopStore(str(tmp_path))
    assert store.list_configs() == []
    assert store.read_status() == {"stops": {}}
    assert store.heartbeat_age_secs() is None
    assert store.is_engine_live() is False


def test_store_add_list_remove(tmp_path):
    store = StopStore(str(tmp_path))
    cfg = StopConfig(ticker="KXT", held_side="no", stop_cents=85)
    store.add(cfg)

    listed = store.list_configs()
    assert len(listed) == 1
    assert listed[0].id == cfg.id
    assert listed[0].ticker == "KXT"

    assert store.remove(cfg.id) is True
    assert store.list_configs() == []
    assert store.remove(cfg.id) is False


def test_store_update(tmp_path):
    store = StopStore(str(tmp_path))
    cfg = StopConfig(ticker="KXT", held_side="no", stop_cents=85)
    store.add(cfg)
    cfg.stop_cents = 80
    assert store.update(cfg) is True
    assert store.list_configs()[0].stop_cents == 80

    missing = StopConfig(ticker="OTHER", held_side="yes", stop_cents=50)
    assert store.update(missing) is False


def test_store_status_roundtrip_and_heartbeat(tmp_path):
    store = StopStore(str(tmp_path))
    now = time.time()
    store.write_status(
        {"heartbeat_ts": now, "engine_pid": 123, "stops": {"abc": {"state": "active"}}}
    )
    status = store.read_status()
    assert status["engine_pid"] == 123
    assert status["stops"]["abc"]["state"] == "active"

    assert store.heartbeat_age_secs(now=now + 5) == pytest.approx(5, abs=0.001)
    assert store.is_engine_live(now=now + 5) is True
    assert store.is_engine_live(now=now + 999) is False


def test_store_tolerates_corrupt_files(tmp_path):
    store = StopStore(str(tmp_path))
    (tmp_path / "config.json").write_text("{ not json")
    (tmp_path / "status.json").write_text("garbage")
    assert store.list_configs() == []
    assert store.read_status() == {"stops": {}}


def test_store_skips_malformed_config_entries(tmp_path):
    store = StopStore(str(tmp_path))
    # Valid entry plus a malformed one (bad held_side) -> only valid survives.
    (tmp_path / "config.json").write_text(
        '{"stops": [{"ticker": "A", "held_side": "no", "stop_cents": 85},'
        ' {"ticker": "B", "held_side": "nope", "stop_cents": 85}]}'
    )
    configs = store.list_configs()
    assert len(configs) == 1
    assert configs[0].ticker == "A"
