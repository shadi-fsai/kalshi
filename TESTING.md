# Testing

The test suite covers the pure business logic in `src/kalshi/` (auth, the REST
client, fees, the Kelly sizer, order translation, the market/event/timing
helpers, realized-volatility/risk metrics, the tennis model, and the stop-loss
store/engine). Streamlit UI code is kept thin; the page-bound `ui/` modules are
exercised by import, byte-compile, and AppTest smoke tests, while all
non-trivial logic lives in importable modules so it can be unit-tested directly.

## Running the tests

```bash
uv sync              # install runtime + dev dependencies (first time only)
uv run pytest        # run everything with coverage
```

Coverage is enforced via `pyproject.toml`:

```toml
[tool.pytest.ini_options]
pythonpath = [".", "src"]
testpaths = ["tests"]
addopts = "--cov=src/kalshi --cov-report=term-missing --cov-fail-under=85"
```

The run fails if total coverage drops below **85%** (currently ~92%). To see
which lines are uncovered, read the `Missing` column in the terminal report.

Useful variations:

```bash
uv run pytest tests/test_markets.py        # one file
uv run pytest -k evaluate_in_money         # match test names
uv run pytest -q --no-header               # quieter output
uv run pytest -p no:cov                    # skip coverage gate while iterating
```

## Dev dependencies

- `pytest` — test runner.
- `pytest-cov` — coverage reporting + the `--cov-fail-under` gate.
- `requests-mock` — stubs Kalshi HTTP calls in client tests (no network).
- `freezegun` — freezes "now" for deterministic timing tests.

## Fixtures (`tests/conftest.py`)

- `rsa_private_key` / `rsa_credentials` — an in-memory 2048-bit RSA key and a
  `KalshiCredentials` built from it, so signing/headers work without env vars or
  key files.
- `winner_event` / `total_event` — sibling events for one game (used for
  grouping tests); `winner_event` is `Game`-scoped, `total_event` is not.
- `market` — a representative market with fixed-point dollar prices.
- `live_details` — a soccer live-data `details` payload (home 4 – away 1).

## What each test file covers

| File | Module under test | Focus |
| --- | --- | --- |
| `test_auth.py` | `kalshi.auth` | Key loading (PEM string, file, precedence, errors), RSA-PSS signing verified against the public key, header construction, query-string stripping, `from_env` (success + missing key id / missing key). |
| `test_client.py` | `kalshi.client` | Request signing + headers, query params, JSON body + `Content-Type` on POST, 204/empty → `{}`, error parsing (JSON message vs raw text), invalid JSON, network errors, and every endpoint method. Uses `requests_mock`. |
| `test_markets.py` | `kalshi.markets` | Price parsing (dollar vs legacy vs missing), `game_key`/`matchup_name`/`market_type_name`, competition/scope accessors, `build_game_groups` (merging + Game-scope rep), `series_ticker_for_market`, `fp_to_float`, `market_label`, `live_scores` precedence, and `evaluate_in_money` across winner/spread/total/team-total/BTTS/tie/unknown. |
| `test_timing.py` | `kalshi.markets` | `parse_ts`, `classify_timing` (live/soon/finished/later), and `classify_resolution` (resolving/ending/later) with a frozen clock. |
| `test_scan.py` | `kalshi.markets` | `scan_series_for_favorites` with a fake paginating client: price-range filtering + sort, `allowed_event_tickers`, side selection, cursor pagination, and the `max_series` truncation flag. |
| `test_kelly.py` | `kalshi.kelly` | Kelly sizing math, including fee-adjusted breakeven, no-bet on marginal edge, and contract counting. |
| `test_fees.py` | `kalshi.fees` | Quadratic fee math, rounding, multiplier scaling, per-contract vs order fees, fee-type handling, and parsing from a series payload. |
| `test_orders.py` | `kalshi.orders` | Translating human buy/sell × YES/NO into the V2 YES-book `bid`/`ask` + price. |
| `test_risk.py` | `kalshi.risk` | Realized-volatility / Sharpe metrics, the volatility stake shrink, candlestick series parsing, high-water marks, and the position correlation matrix. |
| `test_tennis.py` | `kalshi.tennis` | Best-of-3 point model and seeded Monte Carlo match pricing. |
| `test_stops.py` | `kalshi.stops` | `StopConfig` (de)serialization + validation, held-side trigger math (YES/NO), `exit_book_order` mapping, and `StopStore` CRUD / status / heartbeat with corrupt-file tolerance. |
| `test_positions.py` | `kalshi.positions` | Env base-URL mapping, fixed-point signed position parsing (fractional `position_fp` contracts, legacy `position` fallback), held-side contract counts, and the REST price snapshot. |
| `test_stop_engine.py` | `kalshi.stop_engine` | Trigger evaluation, `fire()` filled/partial/error/disarmed paths, client caching, `run_single` no-position, and config reconcile start/cancel. |
| `test_ws.py` | `kalshi.ws` | WebSocket URL derivation (prod/demo/env override), dollar parsing, and `ticker` message parsing. |
| `test_cli.py` | `kalshi.cli` | Console-script entry points: arg parsing, demo/dry-run defaults, the production-arming guard, and the failure exit code. |
| `test_tennis_ui.py` | `ui.tennis` | The tennis page's pure edge-evaluation helper (no Streamlit render). |
| `test_ui_imports.py` | `ui` | Every `ui/` module imports cleanly and exposes the callables the pages rely on. |
| `test_pages_compile.py` | `app_pages` | Every entrypoint/page script byte-compiles (syntax/indentation guard). |
| `test_app_smoke.py` | pages + `ui.data` | AppTest smoke run of each page with a fake client (no network): asserts the page renders without uncaught exceptions. |

## Conventions for adding tests

- Keep logic pure and importable. If you need to test something currently in
  `app.py`, extract it into `kalshi.markets` (or another module) first, then
  import it back into `app.py`.
- Stub HTTP with `requests_mock`; never hit the real Kalshi API in tests.
- Use `freeze_time` (and pass the frozen `now` into the classifiers) for any
  time-dependent assertions.
- Prefer explicit, small input dicts over large shared fixtures when a test
  exercises a specific branch.
- When you add a module under `src/kalshi/`, add a matching `tests/test_*.py`
  and keep total coverage at or above the configured threshold.
