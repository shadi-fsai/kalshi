"""Monte Carlo pricing for a best-of-3 tennis match from a live score.

The model is point-by-point. Each point has a server and a returner; the
server's point-win probability combines the server's *offense* with the
returner's *defense* using a normalized-odds (Bradley-Terry) rule:

    P(server wins point) = o * (1 - d) / [ o * (1 - d) + (1 - o) * d ]

where ``o`` is the server's offense and ``d`` is the returner's defense (both in
``[0, 1]``). The degenerate ``0/0`` case is treated as a coin flip (0.5).

Scoring is standard best-of-3 with ad scoring: a game is first to 4 points
win-by-2 (deuce/advantage); a set is first to 6 games win-by-2 with a 7-point
win-by-2 tiebreak at 6-6 (used in every set, including the decider); the match
is first to 2 sets.

Serving alternates each game. Inside a tiebreak the player who serves the first
point is anchored to the entered/handed-off server and the standard
serve-one-then-alternate-two pattern continues from there. Serve continuity
across sets is simplified (it carries the running server rather than applying the
who-served-the-tiebreak rule) -- a documented v1 simplification.

All functions are pure (a seeded ``random.Random`` is threaded through) so the
simulation is deterministic and unit-testable.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

# Default gap (in probability units) between a player's serving (offense) and
# returning (defense) point-win ability around their baseline: offense =
# baseline + spread, defense = baseline - spread.
DEFAULT_ABILITY_SPREAD = 0.12

# Point label (0/15/30/40/AD) -> internal point count used by the game logic.
POINT_LABELS: tuple[str, ...] = ("0", "15", "30", "40", "AD")
_LABEL_TO_COUNT: dict[str, int] = {"0": 0, "15": 1, "30": 2, "40": 3, "AD": 4}


def point_count_from_label(label: str) -> int:
    """Map a tennis point label (``"0"``/``"15"``/``"30"``/``"40"``/``"AD"``) to a count."""
    try:
        return _LABEL_TO_COUNT[label.upper()]
    except (KeyError, AttributeError) as exc:
        raise ValueError(f"invalid point label {label!r}") from exc


_SCORE_TO_LABEL: dict[int, str] = {0: "0", 15: "15", 30: "30", 40: "40"}


def point_label_from_score(score: int, is_advantage: bool = False) -> str:
    """Map Kalshi's raw game score (0/15/30/40) to a point label.

    ``is_advantage`` (the player holds the deuce advantage) takes precedence and
    returns ``"AD"``. Unknown scores fall back to ``"0"``.
    """
    if is_advantage:
        return "AD"
    return _SCORE_TO_LABEL.get(int(score), "0")


def point_win_prob(offense: float, defense_opp: float) -> float:
    """Server point-win probability from server offense and returner defense.

    Uses the normalized-odds rule ``o(1-d) / [o(1-d) + (1-o)d]``. Returns ``0.5``
    for the degenerate ``0/0`` case (e.g. both inputs 0, or offense 1 with
    defense 1).

    Raises:
        ValueError: if either input is outside ``[0, 1]``.
    """
    if not 0.0 <= offense <= 1.0:
        raise ValueError(f"offense must be in [0, 1] (got {offense}).")
    if not 0.0 <= defense_opp <= 1.0:
        raise ValueError(f"defense_opp must be in [0, 1] (got {defense_opp}).")
    num = offense * (1.0 - defense_opp)
    den = num + (1.0 - offense) * defense_opp
    if den <= 0.0:
        return 0.5
    return num / den


@dataclass(frozen=True)
class MatchParams:
    """Per-player offense (serve) and defense (return) point-win abilities."""

    o1: float
    d1: float
    o2: float
    d2: float

    def __post_init__(self) -> None:
        for name in ("o1", "d1", "o2", "d2"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1] (got {value}).")


@dataclass(frozen=True)
class MatchState:
    """A live best-of-3 score snapshot.

    ``server`` is 1 or 2 (who serves the next point). ``p1_points`` /
    ``p2_points`` are internal point counts for the current game (0-4, where 4 is
    advantage) or, when ``in_tiebreak`` is set, the integer tiebreak points.
    """

    p1_sets: int = 0
    p2_sets: int = 0
    p1_games: int = 0
    p2_games: int = 0
    p1_points: int = 0
    p2_points: int = 0
    server: int = 1
    in_tiebreak: bool = False

    def __post_init__(self) -> None:
        if self.server not in (1, 2):
            raise ValueError(f"server must be 1 or 2 (got {self.server}).")

    @property
    def is_decided(self) -> bool:
        return self.p1_sets >= 2 or self.p2_sets >= 2


def params_from_baselines(
    base1: float, base2: float, *, spread: float = DEFAULT_ABILITY_SPREAD
) -> MatchParams:
    """Build :class:`MatchParams` from each player's baseline point-win ability.

    Offense = baseline + ``spread`` and defense = baseline - ``spread`` (both
    clamped to ``[0, 1]``), matching the odds-seeding the UI applies. Used to turn
    pre-game-odds baselines (see :func:`baselines_from_match_odds`) into a
    simulatable parameter set.
    """
    def _clamp(value: float) -> float:
        return min(1.0, max(0.0, value))

    return MatchParams(
        o1=_clamp(base1 + spread),
        d1=_clamp(base1 - spread),
        o2=_clamp(base2 + spread),
        d2=_clamp(base2 - spread),
    )


def match_state_from_live(parsed: dict) -> MatchState:
    """Convert a :func:`kalshi.markets.tennis_live_score` dict into a MatchState.

    The non-widget counterpart of the tennis page's score auto-fill: clamps sets
    (0-2) and games (0-6), maps the raw current-game score (0/15/30/40, plus the
    deuce ``advantage`` player) to internal point counts via
    :func:`point_label_from_score` / :func:`point_count_from_label`, and uses the
    raw tiebreak counts when ``in_tiebreak``. Missing fields default to 0-0, and a
    missing/unknown server defaults to player 1.
    """
    def _clamp(value: int, hi: int) -> int:
        return max(0, min(hi, int(value)))

    sets_pair = parsed.get("sets") or (0, 0)
    games_pair = parsed.get("games") or (0, 0)
    points_pair = parsed.get("points") or (0, 0)
    in_tiebreak = bool(parsed.get("in_tiebreak"))

    if in_tiebreak:
        p1_points = max(0, int(points_pair[0]))
        p2_points = max(0, int(points_pair[1]))
    else:
        adv = parsed.get("advantage")
        p1_points = point_count_from_label(
            point_label_from_score(points_pair[0], adv == 1)
        )
        p2_points = point_count_from_label(
            point_label_from_score(points_pair[1], adv == 2)
        )

    server = parsed.get("server")
    return MatchState(
        p1_sets=_clamp(sets_pair[0], 2),
        p2_sets=_clamp(sets_pair[1], 2),
        p1_games=_clamp(games_pair[0], 6),
        p2_games=_clamp(games_pair[1], 6),
        p1_points=p1_points,
        p2_points=p2_points,
        server=server if server in (1, 2) else 1,
        in_tiebreak=in_tiebreak,
    )


@dataclass
class SimulationResult:
    """Outcome distribution from a Monte Carlo run."""

    p1_win_prob: float
    ci95_half_width: float
    set_score_counts: dict[str, int] = field(default_factory=dict)
    n_sims: int = 0


def _other(player: int) -> int:
    return 2 if player == 1 else 1


def _server_point_prob(server: int, params: MatchParams) -> float:
    """Probability the *server* wins the point, given who is serving."""
    if server == 1:
        return point_win_prob(params.o1, params.d2)
    return point_win_prob(params.o2, params.d1)


def _tiebreak_server(starter: int, point_index: int) -> int:
    """Server for the ``point_index``-th (0-based) point of a tiebreak.

    Standard pattern: the starter serves point 0, then serve alternates every
    two points (1-2 to the other, 3-4 back, ...).
    """
    return starter if ((point_index + 1) // 2) % 2 == 0 else _other(starter)


def _play_game(
    p1_points: int, p2_points: int, server: int, params: MatchParams, rng: random.Random
) -> int:
    """Play one service game to completion; return the winner (1 or 2)."""
    p_server = _server_point_prob(server, params)
    p1, p2 = p1_points, p2_points
    while True:
        server_won = rng.random() < p_server
        winner = server if server_won else _other(server)
        if winner == 1:
            p1 += 1
        else:
            p2 += 1
        if (p1 >= 4 or p2 >= 4) and abs(p1 - p2) >= 2:
            return 1 if p1 > p2 else 2


def _play_tiebreak(
    p1_points: int, p2_points: int, starter: int, params: MatchParams, rng: random.Random
) -> int:
    """Play a 7-point win-by-2 tiebreak to completion; return the winner."""
    p1, p2 = p1_points, p2_points
    i = p1 + p2
    while True:
        server = _tiebreak_server(starter, i)
        p_server = _server_point_prob(server, params)
        server_won = rng.random() < p_server
        winner = server if server_won else _other(server)
        if winner == 1:
            p1 += 1
        else:
            p2 += 1
        i += 1
        if (p1 >= 7 or p2 >= 7) and abs(p1 - p2) >= 2:
            return 1 if p1 > p2 else 2


def _play_set(
    g1: int,
    g2: int,
    server: int,
    params: MatchParams,
    rng: random.Random,
    *,
    partial: tuple[int, int] | None = None,
    in_tiebreak: bool = False,
) -> tuple[int, int]:
    """Play one set to completion. Return ``(set_winner, next_server)``.

    ``partial`` resumes the current game (or tiebreak when ``in_tiebreak``) from
    the given points; subsequent games start fresh.
    """
    # Resume an in-progress tiebreak from the handed-off state.
    if in_tiebreak:
        start = partial or (0, 0)
        k = start[0] + start[1]
        starter = server if ((k + 1) // 2) % 2 == 0 else _other(server)
        tb_winner = _play_tiebreak(start[0], start[1], starter, params, rng)
        if tb_winner == 1:
            g1 += 1
        else:
            g2 += 1
        return (1 if g1 > g2 else 2), server

    resume = partial
    while True:
        if g1 == 6 and g2 == 6:
            tb_winner = _play_tiebreak(0, 0, server, params, rng)
            if tb_winner == 1:
                g1 += 1
            else:
                g2 += 1
            return (1 if g1 > g2 else 2), server
        if resume is not None:
            p1p, p2p = resume
            resume = None
        else:
            p1p, p2p = 0, 0
        game_winner = _play_game(p1p, p2p, server, params, rng)
        if game_winner == 1:
            g1 += 1
        else:
            g2 += 1
        server = _other(server)
        if (g1 >= 6 or g2 >= 6) and abs(g1 - g2) >= 2:
            return (1 if g1 > g2 else 2), server


def simulate_match_from_state(
    state: MatchState, params: MatchParams, rng: random.Random
) -> tuple[int, str]:
    """Simulate the rest of a best-of-3 match. Return ``(winner, "p1-p2")``."""
    p1_sets, p2_sets = state.p1_sets, state.p2_sets
    if p1_sets >= 2:
        return 1, f"{p1_sets}-{p2_sets}"
    if p2_sets >= 2:
        return 2, f"{p1_sets}-{p2_sets}"

    server = state.server
    first = True
    while p1_sets < 2 and p2_sets < 2:
        if first:
            set_winner, server = _play_set(
                state.p1_games,
                state.p2_games,
                server,
                params,
                rng,
                partial=(state.p1_points, state.p2_points),
                in_tiebreak=state.in_tiebreak,
            )
            first = False
        else:
            set_winner, server = _play_set(0, 0, server, params, rng)
        if set_winner == 1:
            p1_sets += 1
        else:
            p2_sets += 1
    winner = 1 if p1_sets > p2_sets else 2
    return winner, f"{p1_sets}-{p2_sets}"


def _game_win_prob(p: float) -> float:
    """Closed-form probability a server with point-win prob ``p`` wins a game.

    Ad scoring: counts the win-to-love/15/30 paths plus the deuce branch, where
    from deuce the server wins with ``p^2 / (p^2 + q^2)``.
    """
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    q = 1.0 - p
    pre_deuce = p**4 * (1.0 + 4.0 * q + 10.0 * q * q)
    denom = p * p + q * q
    deuce = 20.0 * p**3 * q**3 * (p * p / denom)
    return pre_deuce + deuce


def match_win_probability(params: MatchParams, *, first_server: int = 1) -> float:
    """Analytic P1 match-win probability for a best-of-3 from 0-0.

    Deterministic (no sampling): games use the closed-form
    :func:`_game_win_prob`, a set is solved by exact recursion over the game
    score with alternating serve and a 7-point tiebreak at 6-6, and the match is
    ``S^2 (3 - 2S)`` for set-win probability ``S``. Used to invert pre-game odds;
    a useful fast check on the Monte Carlo as well. Serve alternation across sets
    is approximated by reusing the same ``first_server`` each set.
    """
    pp1 = point_win_prob(params.o1, params.d2)  # P1 point win on P1 serve
    pp2 = point_win_prob(params.o2, params.d1)  # P2 point win on P2 serve
    h1 = _game_win_prob(pp1)  # P1 holds serve
    p1_break = 1.0 - _game_win_prob(pp2)  # P1 wins a game on P2's serve

    def tiebreak_p1_win(starter: int) -> float:
        memo: dict[tuple[int, int, int], float] = {}

        def rec(a: int, b: int, i: int) -> float:
            if (a >= 7 or b >= 7) and abs(a - b) >= 2:
                return 1.0 if a > b else 0.0
            if a >= 6 and b >= 6 and a == b:
                # Deuce: over the next two points serve splits one each, so the
                # win-from-deuce probability has a closed form.
                s1 = _tiebreak_server(starter, i)
                s2 = _tiebreak_server(starter, i + 1)
                q1 = pp1 if s1 == 1 else (1.0 - pp2)
                q2 = pp1 if s2 == 1 else (1.0 - pp2)
                win_pair = q1 * q2
                lose_pair = (1.0 - q1) * (1.0 - q2)
                total = win_pair + lose_pair
                return win_pair / total if total > 0.0 else 0.5
            key = (a, b, i)
            if key in memo:
                return memo[key]
            server = _tiebreak_server(starter, i)
            p1pt = pp1 if server == 1 else (1.0 - pp2)
            val = p1pt * rec(a + 1, b, i + 1) + (1.0 - p1pt) * rec(a, b + 1, i + 1)
            memo[key] = val
            return val

        return rec(0, 0, 0)

    set_memo: dict[tuple[int, int, int], float] = {}

    def set_p1_win(g1: int, g2: int, server: int) -> float:
        if g1 >= 6 and g1 - g2 >= 2:
            return 1.0
        if g2 >= 6 and g2 - g1 >= 2:
            return 0.0
        if g1 == 6 and g2 == 6:
            return tiebreak_p1_win(server)
        key = (g1, g2, server)
        if key in set_memo:
            return set_memo[key]
        if server == 1:
            val = h1 * set_p1_win(g1 + 1, g2, 2) + (1.0 - h1) * set_p1_win(g1, g2 + 1, 2)
        else:
            val = p1_break * set_p1_win(g1 + 1, g2, 1) + (1.0 - p1_break) * set_p1_win(
                g1, g2 + 1, 1
            )
        set_memo[key] = val
        return val

    s = set_p1_win(0, 0, first_server)
    return s * s * (3.0 - 2.0 * s)


def baselines_from_match_odds(
    p1_match_win: float, *, spread: float = DEFAULT_ABILITY_SPREAD
) -> tuple[float, float]:
    """Invert pre-game match odds into the two players' baseline point-win probs.

    Models the two players symmetrically around 0.5 with a single skill gap
    ``s`` (``baseline1 = 0.5 + s``, ``baseline2 = 0.5 - s``) and applies the
    fixed offense/defense ``spread`` (offense ``= baseline + spread``, defense
    ``= baseline - spread``). Bisects ``s`` so :func:`match_win_probability`
    reproduces ``p1_match_win``. The gap is bounded so all four abilities stay in
    ``[0, 1]``; odds beyond the reachable range clamp to that bound.

    Returns ``(baseline1, baseline2)``.
    """
    target = min(max(p1_match_win, 1e-6), 1.0 - 1e-6)
    s_max = max(0.0, 0.5 - spread)

    def match_win(s: float) -> float:
        b1, b2 = 0.5 + s, 0.5 - s
        params = MatchParams(
            o1=b1 + spread, d1=b1 - spread, o2=b2 + spread, d2=b2 - spread
        )
        return match_win_probability(params)

    lo, hi = -s_max, s_max
    if target <= match_win(lo):
        s = lo
    elif target >= match_win(hi):
        s = hi
    else:
        for _ in range(60):
            mid = (lo + hi) / 2.0
            if match_win(mid) < target:
                lo = mid
            else:
                hi = mid
        s = (lo + hi) / 2.0
    return 0.5 + s, 0.5 - s


def monte_carlo(
    state: MatchState,
    params: MatchParams,
    *,
    n_sims: int = 20000,
    seed: int = 0,
) -> SimulationResult:
    """Estimate P1's match-win probability and the set-score distribution.

    Runs ``n_sims`` seeded simulations from ``state``. The 95% CI half-width is
    the normal approximation ``1.96 * sqrt(p(1-p)/n)``. A match that is already
    decided short-circuits to a degenerate result.

    Raises:
        ValueError: if ``n_sims`` is not positive.
    """
    if n_sims < 1:
        raise ValueError(f"n_sims must be >= 1 (got {n_sims}).")

    if state.is_decided:
        winner = 1 if state.p1_sets > state.p2_sets else 2
        score = f"{state.p1_sets}-{state.p2_sets}"
        return SimulationResult(
            p1_win_prob=1.0 if winner == 1 else 0.0,
            ci95_half_width=0.0,
            set_score_counts={score: n_sims},
            n_sims=n_sims,
        )

    rng = random.Random(seed)
    counts: dict[str, int] = {}
    p1_wins = 0
    for _ in range(n_sims):
        winner, score = simulate_match_from_state(state, params, rng)
        counts[score] = counts.get(score, 0) + 1
        if winner == 1:
            p1_wins += 1
    p = p1_wins / n_sims
    half = 1.96 * math.sqrt(p * (1.0 - p) / n_sims)
    return SimulationResult(
        p1_win_prob=p,
        ci95_half_width=half,
        set_score_counts=counts,
        n_sims=n_sims,
    )


def win_prob_distribution(
    state: MatchState,
    params: MatchParams,
    *,
    ability_sd: float,
    n_scenarios: int = 200,
    n_sims: int = 400,
    seed: int = 0,
) -> list[float]:
    """P1 match-win probabilities across scenarios that perturb the abilities.

    Captures parameter (model input) uncertainty: each scenario adds independent
    Gaussian noise with standard deviation ``ability_sd`` (in probability units)
    to each of the four abilities (clamped to ``[0, 1]``), then runs a Monte
    Carlo of ``n_sims`` matches. The returned list of per-scenario win
    probabilities is the "range of outcomes" used for uncertainty-aware sizing.

    With ``ability_sd <= 0`` there is no parameter uncertainty, so a single
    scenario at the given ``params`` is returned. Deterministic for a fixed
    ``seed``.

    Raises:
        ValueError: if ``n_scenarios`` or ``n_sims`` is not positive, or
            ``ability_sd`` is negative.
    """
    if n_scenarios < 1:
        raise ValueError(f"n_scenarios must be >= 1 (got {n_scenarios}).")
    if n_sims < 1:
        raise ValueError(f"n_sims must be >= 1 (got {n_sims}).")
    if ability_sd < 0:
        raise ValueError(f"ability_sd must be >= 0 (got {ability_sd}).")

    if ability_sd == 0:
        return [monte_carlo(state, params, n_sims=n_sims, seed=seed).p1_win_prob]

    out: list[float] = []
    for s in range(n_scenarios):
        rng = random.Random(seed * 100_003 + s)

        def jitter(value: float) -> float:
            return min(1.0, max(0.0, value + rng.gauss(0.0, ability_sd)))

        scenario = MatchParams(
            o1=jitter(params.o1),
            d1=jitter(params.d1),
            o2=jitter(params.o2),
            d2=jitter(params.d2),
        )
        result = monte_carlo(state, scenario, n_sims=n_sims, seed=seed + 1 + s)
        out.append(result.p1_win_prob)
    return out
