"""Unit tests for the stop-tool console-script entry points."""

import pytest

import kalshi.cli.hedge_watcher as hw
import kalshi.cli.stop_engine as se


class FakeEngine:
    """Captures construction + provides async run/run_single stand-ins."""

    last_instance = None
    single_result = {"state": "filled", "message": "ok"}

    def __init__(self, store=None, **kwargs):
        self.store = store
        FakeEngine.last_instance = self

    async def run(self):
        return None

    async def run_single(self, cfg):
        self.cfg = cfg
        return FakeEngine.single_result


def test_stop_engine_main_runs(monkeypatch):
    monkeypatch.setattr(se, "StopEngine", FakeEngine)
    monkeypatch.setattr(se, "StopStore", lambda directory: ("store", directory))
    monkeypatch.setattr(se, "load_dotenv", lambda: None)
    monkeypatch.setattr("sys.argv", ["kalshi-stop-engine", "--dir", "/tmp/s"])
    se.main()  # should complete without raising
    assert FakeEngine.last_instance.store == ("store", "/tmp/s")


def test_hedge_watcher_main_filled(monkeypatch):
    FakeEngine.single_result = {"state": "filled", "message": "done"}
    monkeypatch.setattr(hw, "StopEngine", FakeEngine)
    monkeypatch.setattr(hw, "load_dotenv", lambda: None)
    monkeypatch.setattr("sys.argv", ["kalshi-hedge-watcher", "--ticker", "KXT-NO"])
    hw.main()  # no SystemExit on success
    assert FakeEngine.last_instance.cfg.ticker == "KXT-NO"
    assert FakeEngine.last_instance.cfg.env == "demo"
    assert FakeEngine.last_instance.cfg.armed is False


def test_hedge_watcher_prod_requires_yes_live(monkeypatch):
    monkeypatch.setattr(hw, "StopEngine", FakeEngine)
    monkeypatch.setattr(hw, "load_dotenv", lambda: None)
    monkeypatch.setattr(
        "sys.argv", ["kalshi-hedge-watcher", "--ticker", "KXT-NO", "--arm", "--live"]
    )
    with pytest.raises(SystemExit):
        hw.main()


def test_hedge_watcher_failure_state_exits_nonzero(monkeypatch):
    FakeEngine.single_result = {"state": "error", "message": "boom"}
    monkeypatch.setattr(hw, "StopEngine", FakeEngine)
    monkeypatch.setattr(hw, "load_dotenv", lambda: None)
    monkeypatch.setattr("sys.argv", ["kalshi-hedge-watcher", "--ticker", "KXT-NO", "--arm"])
    with pytest.raises(SystemExit) as exc:
        hw.main()
    assert exc.value.code == 2
