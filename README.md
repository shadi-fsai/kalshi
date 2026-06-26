# Kalshi Kelly Betting App

A [Streamlit](https://streamlit.io/) app that connects to the **Kalshi**
production Trade API, lets you pull up a live sports game, and uses the
**Kelly criterion** to size bets.

The app treats the current Kalshi market price as the default fair value, then
lets you nudge your own win-probability estimate with a slider to express an
edge. It computes the Kelly-optimal stake and the number of contracts to buy.

> This app sizes bets **and** can place real limit orders (with an explicit
> confirmation step). Trade carefully.

## Features

- Four pages (shared sidebar via `st.navigation`): **Find games & size** (browse
  and size a bet), **Watch a game live** (auto-refreshing score + opportunities for
  one game), **Portfolio** (positions, correlations, and stop-losses), and
  **Tennis match pricing** (Monte Carlo vs the market). A "Watch this game live"
  button on Find hands a selected game off to the Watch page.
- RSA API-key authentication against Kalshi production (request signing per the
  [Kalshi API key docs](https://docs.kalshi.com/getting_started/api_keys)).
- Friendly game browser: filter by **Sport** then **Competition / league**
  (e.g. Soccer -> World Soccer Cup), toggle "Games only" to hide futures/props,
  and search by team name. You can also paste an `event_ticker` /
  `market_ticker` directly.
- Per market: live YES/NO price, implied probability, adjustable probability
  estimate, computed edge, full/fractional Kelly fraction, recommended stake,
  and contract count.
- Transaction-fee aware: fees are pulled live from each market's Kalshi series
  fee model (`fee_type` + `fee_multiplier` via `GET /series/{ticker}`) and the
  actual fee for the recommended contract count is shown. They are modeled into
  the Kelly edge, the fee-adjusted breakeven, and sizing, so a position that is
  only marginally +EV before fees is correctly flagged as no-bet. A "Hold to
  expiration" toggle (sidebar, on by default) controls the exit fee: when on,
  the contract settles with no sell fee (buy fee only); turn it off if you plan
  to sell before expiration and the sell fee is added round-trip into breakeven
  and sizing. A manual fallback fee (sidebar) is used only when the API fee
  can't be computed (e.g. flat-fee markets).
- Portfolio view (**Portfolio** page): cash balance, portfolio value, current
  positions (YES/NO contracts, exposure, realized P&L, fees paid), resting
  orders with per-order cancel, and a synthetic stop-loss manager (see
  "Portfolio stop-loss manager" below).
- Position correlation matrix (**Portfolio** page): a color-coded matrix of the
  empirical correlation between your held positions, computed from each
  position's mid-price returns over a selectable window (from 5m up to 7d) and
  oriented to the side you hold (a NO holding uses `1 - YES`), so a high positive value
  means those bets tend to win and lose together. Pairs at or above an
  adjustable `|correlation|` threshold are called out so concentrated risk is
  obvious at a glance.
- Order placement: a limit-order ticket (Buy/Sell x YES/NO, count, limit price)
  with a fee-inclusive cost preview and a two-step confirm before submitting a
  REAL order via Kalshi's V2 endpoint. YES/NO intent is translated to the
  YES-book bid/ask the API expects (e.g. **buy NO @ 30c rests as SELL YES @ 70c**),
  and the order carries Kalshi's canonical `outcome_side` so its UI labels it
  "buy NO" rather than the equivalent "sell YES". An order-mode selector chooses
  execution style: **Maker (post-only)** rests and never pays a taker fee
  (auto-cancelled if it would cross), **Taker (immediate)** crosses now via
  immediate-or-cancel, and **Limit GTC** is the default resting limit. Defaults
  are prefilled from the Kelly sizing.
- Live "in the money" flags: for in-progress games the app pulls the current
  score from Kalshi's live-data endpoint and flags each market 🟢 ITM / 🔴 OTM
  based on the live score (e.g. a 3:0 game makes "Over 2.5 goals" in the money).
  Supports winner, spread, totals, team totals, and both-teams-to-score; market
  types it can't evaluate from the score alone (corners, halves, player props)
  show no flag rather than guessing.
- Sharpe-aware sizing: the app estimates realized price volatility from Kalshi
  candlesticks since the game started and combines it with time-to-expiration
  to (a) show selection metrics (per-bet and time-annualized Sharpe, edge/day)
  and (b) shrink the stake on top of Kelly when a lot of time remains and the
  market is volatile (the outcome is still uncertain). Toggle and sensitivity
  live in the sidebar. See "Volatility and time" below.
- Tennis match pricing (**Tennis match pricing** page): the page leads with a
  live-edge scanner. Click "Scan live tennis for edges" and it prices every
  tennis match in play on Kalshi (up to a cap), then lists, ranked, the ones
  where a bet has a positive half-Kelly edge - evaluating BOTH the YES and NO
  side of each match-winner market. For each match the scan pulls the player
  names, match-winner ticker, pre-game odds (inverted to seed abilities, see
  below), and the full live score, builds the model, runs the Monte Carlo plus
  the ability-uncertainty sweep, and compares the model's fair value to the
  market. The opportunities table shows the match, the recommended side
  (YES/NO + player), the MC fair price vs the market ask, the edge, and the
  half-Kelly stake; a "minimum edge" filter and a "show all live matches"
  toggle let you tune what is listed. Selecting a match and clicking "Open
  match" loads the full detailed view: player names and the match-winner
  ticker, the pre-game odds (inverted to seed abilities), and the full live
  score - sets, current games, current points (game or tiebreak), and who is
  serving - read from Kalshi's `tennis_tournament_singles` live data and
  oriented to the right player via each winner market's competitor id (the
  score stays editable, and a note flags what was auto-filled). It then
  auto-runs the Monte Carlo, outputs the model's match-win probability with a
  95% CI and the set-score distribution, and re-prices/sizes the bet. You can
  also skip the scanner and enter a score manually. The sizing section prices
  the bet off the simulation: it picks the side (YES/NO) with positive edge vs
  the market, shows the MC fair price vs the market ask, and sizes it at the
  sidebar's fractional-Kelly multiplier (half-Kelly by default) using the
  account bankroll and the market's real Kalshi fee model, then offers the
  standard two-step order ticket to place it. The stake also incorporates the
  *range* of Monte Carlo outcomes: the "Ability uncertainty (+/- points)" slider
  resamples each offense/defense input across many scenarios to build a
  win-probability distribution (shown as a histogram). Because expected
  log-growth on a one-shot binary bet depends only on the mean (so parameter
  uncertainty would cancel), the bet is sized with a mean-variance Kelly that
  shrinks the fraction by the variance of that distribution - a wider range
  means a smaller bet, and the page reports the resulting shrink. "Seed abilities from pre-game
  odds" takes the selected market's implied win probability, inverts an analytic
  best-of-3 model to a per-player baseline point-win probability, and fills
  offense = baseline + 12 points and defense = baseline - 12 points (you can
  fine-tune afterward). Model assumptions: a server's point-win probability
  combines offense and the returner's defense via a normalized-odds
  (Bradley-Terry) rule; ad scoring with 7-point tiebreaks in every set;
  cross-set serve continuity is simplified for v1.
- Bankroll from manual input or pulled from your account balance.
- Errors from the API/auth are surfaced in the UI (never silently swallowed).

> Order placement and the portfolio view require an API key with trading +
> portfolio scopes. To trade against the sandbox instead of real money, point
> `KALSHI_API_BASE` at the demo endpoint (see `.env.example`).

## Prerequisites

- [uv](https://docs.astral.sh/uv/) for package management.
- A Kalshi account and an API key. Create one at
  <https://kalshi.com/account/profile> (API Keys section). You receive a
  **Key ID** and a **private key** file (RSA PEM). The private key is shown
  only once, so save it.

## Setup

1. Install dependencies (creates a `.venv`):

```bash
uv sync
```

2. Configure credentials. Copy the example env file and fill it in:

```bash
cp .env.example .env
```

Set these values in `.env`:

- `KALSHI_API_KEY_ID` - your Key ID.
- `KALSHI_PRIVATE_KEY_PATH` - path to your downloaded private key file
  (e.g. `./kalshi-private-key.key`). Alternatively set `KALSHI_PRIVATE_KEY`
  to the inline PEM contents.

The `.env` file and `*.key` / `*.pem` files are git-ignored.

## Run

```bash
uv run streamlit run app.py
```

Then open the URL Streamlit prints (default <http://localhost:8501>).

## How the Kelly sizing works

A Kalshi contract trades in cents (1-99) and pays `$1.00` on a win. Buying a
side at `price` cents costs `price/100` dollars per contract.

- Cost per contract `c = price / 100`
- Implied (breakeven) probability `= c`
- Net odds `b = (1 - c) / c`
- With your estimated win probability `q`, the full-Kelly fraction is
  `f* = q - (1 - q) / b`, clamped to `>= 0` (no bet without a positive edge).
- A fractional-Kelly multiplier (e.g. 0.5 for half-Kelly) scales `f*` down.
- Recommended stake `= f_used * bankroll`; contracts `= floor(stake / c)`.

## Volatility and time (Sharpe-aware)

For a single binary contract held to expiry, profit per contract has mean
`edge = q - breakeven` and standard deviation `sqrt(q(1-q))`, so the per-bet
Sharpe `edge / sqrt(q(1-q))` is **independent of stake size**. Volatility and
time-to-expiration therefore help in two distinct places:

- **Selection (metrics, no effect on stake).** The app shows the per-bet Sharpe
  plus a time-annualized version `sharpe_terminal * sqrt(525600 / minutes_to_expiry)`
  and `edge / day`. The horizon is measured to the minute (from each market's
  `expected_expiration_time`), so an intraday market that settles in ~45 min is
  annualized from its true remaining distance rather than a rounded day count.
  These reward shorter-dated edges that recycle capital faster and let you rank
  bets of different durations on a comparable basis.
- **Sizing (a shrink on top of Kelly).** Realized volatility is estimated from
  Kalshi candlesticks (`GET /series/{s}/markets/{t}/candlesticks`) over the
  window since kickoff, scaled to the remaining time as
  `sigma_remaining = vol_per_day * sqrt(days_to_expiry)` (capped at the terminal
  bound `sqrt(q(1-q))`). The stake is multiplied by
  `edge / (edge + sensitivity * sigma_remaining)` in `(0, 1]`: a noisy market
  with lots of time left is sized smaller because the outcome is still
  uncertain. This composes with your fractional-Kelly multiplier
  (`effective = kelly_multiplier x vol_factor`).

The adjustment is opt-out (sidebar toggle) with an adjustable sensitivity. When
price history is unavailable (pre-game, illiquid, or no expiration time), the
factor is `1.00x` and the app says why rather than silently dropping it. The
pure math lives in [src/kalshi/risk.py](src/kalshi/risk.py).

## Synthetic stop / hedge watcher

Kalshi has **no native stop or trigger order**. A resting limit order can't act
as a stop: a "sell my NO if it falls to 85c" order placed at 85c is already
marketable against the ~95c bid and fills immediately (or is cancelled with
`post_only`). The order schema only has `type`, `time_in_force`, `post_only`,
`reduce_only`, `buy_max_cost`, and `order_group_id` - no trigger field. So a stop
must run in a process *you* keep alive.

The `kalshi-hedge-watcher` console script ([src/kalshi/cli/hedge_watcher.py](src/kalshi/cli/hedge_watcher.py))
is that process for a single position. It streams the live price over the Kalshi
WebSocket `ticker` channel ([src/kalshi/ws.py](src/kalshi/ws.py)) and, when the
held side's price falls to/through your stop, fires **one** `reduce_only`
immediate-or-cancel order that can only flatten the position (it can never flip
you to the other side). The exit price is capped at `stop - max-slippage`, so a
fast gap-through won't fill at a catastrophic price - if it can't fill within the
cap it alerts loudly rather than chasing the market down. The watch/fire logic
lives in [src/kalshi/stop_engine.py](src/kalshi/stop_engine.py) and is shared
with the multi-stop engine below.

> **This is a synthetic stop: it only protects while the watcher process is
> running.** It is *not* a resting order on the exchange. If the process stops,
> there is no protection.

Safety defaults (per "never fail silently"):

- Targets the **demo** environment unless `--live` is passed.
- **Dry run by default** - it logs the order it *would* send; pass `--arm` to
  place real orders. Production requires `--arm --live --yes-live`.
- Single-fire guard, `client_order_id` idempotency, and a REST-poll fallback so
  the stop is never blind during a WebSocket outage.

```bash
# Demo, dry run (no orders), auto count, stop when a held NO falls to 85c:
uv run kalshi-hedge-watcher --ticker KX...-NO

# Demo, actually place the protective close on trigger:
uv run kalshi-hedge-watcher --ticker KX...-NO --arm
```

## Portfolio stop-loss manager

The **Portfolio** page has a "Stop losses" section to add, view, and remove
synthetic stop-losses on your open positions. Because Kalshi has no native stop
order (see above), stops are run by a **separate always-on engine process** -
the page is only the control surface:

- The page writes stop configs to a local store (`./.kalshi_stops/`, override
  with `KALSHI_STOPS_DIR`) and shows each stop's live status on a ~5s refresh.
- The `kalshi-stop-engine` console script ([src/kalshi/cli/stop_engine.py](src/kalshi/cli/stop_engine.py),
  core in [src/kalshi/stop_engine.py](src/kalshi/stop_engine.py)) reads that
  store, watches each market's price over the WebSocket `ticker` channel (REST
  fallback when it drops), and on trigger fires one `reduce_only` IOC close that
  can only flatten the position. It writes status + a heartbeat back to the store.
- If the engine is not running, the page detects the stale heartbeat and warns
  that **stops are NOT being managed** (a stop only protects while the engine
  runs).

Start the engine (each stop fires on its own environment; prod = real orders):

```bash
uv run kalshi-stop-engine
```

Each stop carries its own environment (`prod`/`demo`) and an `armed` flag, and
the add form requires an explicit real-money acknowledgement before adding an
armed production stop. A stop sells the held side at `stop - max-slippage`, so a
fast gap-through is bounded by the slippage cap rather than filling at any price
(if it can't fill within the cap it alerts loudly for manual action).

## Testing

```bash
uv run pytest
```

## Project layout

```
app.py                 Entrypoint: st.navigation router + shared sidebar
app_pages/find.py      Page: browse/search games (or enter a ticker) and size a bet
app_pages/watch.py     Page: watch one live game (auto-refreshing score + opportunities)
app_pages/portfolio.py Page: balance, positions, resting orders, stop-losses
app_pages/tennis.py    Page: tennis match Monte Carlo pricing vs the market
ui/data.py             Streamlit-cached data fetchers + client construction seam
ui/settings.py         Shared sidebar + Settings (bankroll, Kelly, volatility, fees)
ui/sizer.py            Bet sizer (Kelly + Sharpe) and the order ticket
ui/games.py            Game discovery (browse/filters/favorites) + manual ticker
ui/portfolio.py        Portfolio view (incl. position correlation matrix)
ui/stops.py            Portfolio stop-loss manager (add/list/remove + live status)
ui/tennis.py           Tennis pricing UI (inputs, simulation, market compare)
src/kalshi/auth.py     RSA request signing + signed headers
src/kalshi/client.py   REST client (events, markets, orderbook, candlesticks, balance)
src/kalshi/ws.py       WebSocket client (signed connect, ticker channel stream)
src/kalshi/stops.py    Stop store (config/status files) + held-side trigger math
src/kalshi/positions.py  Shared env base-URL + position/price helpers
src/kalshi/stop_engine.py  Synthetic stop engine core (run multi-stop + run_single)
src/kalshi/cli/stop_engine.py    Console script: kalshi-stop-engine (Portfolio engine)
src/kalshi/cli/hedge_watcher.py  Console script: kalshi-hedge-watcher (single stop)
src/kalshi/kelly.py    Pure Kelly math
src/kalshi/markets.py  Pure market/event/timing/ITM helpers
src/kalshi/fees.py     Kalshi fee-model math
src/kalshi/orders.py   Buy/sell + YES/NO -> YES-book order translation
src/kalshi/risk.py     Realized volatility + Sharpe metrics + correlation matrix
src/kalshi/tennis.py   Tennis scoring model + Monte Carlo match pricing
tests/                 Unit tests (see TESTING.md)
research/              Ad-hoc research scripts, grouped by project (not app code):
research/soccer/         Soccer favorite/correct-score backtests (+ shared helper)
research/blown_leads/    Blown-lead / comeback frequency analyses across sports
research/tennis_volatility/  Tennis price-swing / WTA-vs-ATP volatility studies
research/btc15m/         BTC 15-minute market backtest + analysis (cache/plots)
```

Research scripts under `research/` are standalone and import the installed
`kalshi` package. They need the optional research dependencies (matplotlib,
numpy): install them with `uv sync --group research`. Run each from its own
project folder so co-located helpers (e.g. `scratch_soccer_secondary.py`)
resolve (e.g. `cd research/soccer && uv run python scratch_top95_backtest.py`).
