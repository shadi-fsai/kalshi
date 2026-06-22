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

- Three pages (shared sidebar via `st.navigation`): **Find games & size** (browse
  and size a bet), **Watch a game live** (auto-refreshing score + opportunities for
  one game), and **Portfolio**. A "Watch this game live" button on Find hands a
  selected game off to the Watch page.
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
  positions (YES/NO contracts, exposure, realized P&L, fees paid), and resting
  orders with per-order cancel.
- Order placement: a limit-order ticket (Buy/Sell x YES/NO, count, limit price)
  with a fee-inclusive cost preview and a two-step confirm before submitting a
  REAL order via Kalshi's V2 endpoint. YES/NO intent is translated to the
  YES-book bid/ask the API expects. Defaults are prefilled from the Kelly sizing.
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

## Testing

```bash
uv run pytest
```

## Project layout

```
app.py                 Entrypoint: st.navigation router + shared sidebar
app_pages/find.py      Page: browse/search games (or enter a ticker) and size a bet
app_pages/watch.py     Page: watch one live game (auto-refreshing score + opportunities)
app_pages/portfolio.py Page: balance, positions, and resting orders
ui/data.py             Streamlit-cached data fetchers + client construction seam
ui/settings.py         Shared sidebar + Settings (bankroll, Kelly, volatility, fees)
ui/sizer.py            Bet sizer (Kelly + Sharpe) and the order ticket
ui/games.py            Game discovery (browse/filters/favorites) + manual ticker
ui/portfolio.py        Portfolio view
src/kalshi/auth.py     RSA request signing + signed headers
src/kalshi/client.py   REST client (events, markets, orderbook, candlesticks, balance)
src/kalshi/kelly.py    Pure Kelly math
src/kalshi/markets.py  Pure market/event/timing/ITM helpers
src/kalshi/fees.py     Kalshi fee-model math
src/kalshi/orders.py   Buy/sell + YES/NO -> YES-book order translation
src/kalshi/risk.py     Realized volatility + Sharpe metrics + vol/time shrink
tests/                 Unit tests (see TESTING.md)
```
