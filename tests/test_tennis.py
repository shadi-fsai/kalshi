"""Tests for kalshi.tennis: point model and seeded Monte Carlo pricing."""

from __future__ import annotations

import random
import statistics

import pytest

from kalshi.tennis import (
    DEFAULT_ABILITY_SPREAD,
    MatchParams,
    MatchState,
    baselines_from_match_odds,
    match_win_probability,
    match_state_from_live,
    monte_carlo,
    params_from_baselines,
    point_count_from_label,
    point_label_from_score,
    point_win_prob,
    simulate_match_from_state,
    win_prob_distribution,
)


def test_params_from_baselines_applies_spread():
    p = params_from_baselines(0.50, 0.40, spread=0.12)
    assert p.o1 == pytest.approx(0.62)
    assert p.d1 == pytest.approx(0.38)
    assert p.o2 == pytest.approx(0.52)
    assert p.d2 == pytest.approx(0.28)


def test_params_from_baselines_clamps_to_unit_interval():
    p = params_from_baselines(0.95, 0.05, spread=0.12)
    assert p.o1 == 1.0
    assert p.d1 == pytest.approx(0.83)
    assert p.o2 == pytest.approx(0.17)
    assert p.d2 == 0.0


def test_match_state_from_live_regular_points():
    parsed = {
        "sets": (1, 0),
        "games": (3, 4),
        "points": (30, 40),
        "in_tiebreak": False,
        "advantage": None,
        "server": 2,
    }
    state = match_state_from_live(parsed)
    assert (state.p1_sets, state.p2_sets) == (1, 0)
    assert (state.p1_games, state.p2_games) == (3, 4)
    assert (state.p1_points, state.p2_points) == (2, 3)
    assert state.server == 2
    assert state.in_tiebreak is False


def test_match_state_from_live_advantage():
    parsed = {
        "sets": (0, 0),
        "games": (2, 2),
        "points": (40, 40),
        "in_tiebreak": False,
        "advantage": 1,
        "server": 1,
    }
    state = match_state_from_live(parsed)
    assert state.p1_points == point_count_from_label("AD")
    assert state.p2_points == point_count_from_label("40")


def test_match_state_from_live_tiebreak_uses_raw_counts():
    parsed = {
        "sets": (1, 1),
        "games": (6, 6),
        "points": (5, 3),
        "in_tiebreak": True,
        "advantage": None,
        "server": 2,
    }
    state = match_state_from_live(parsed)
    assert state.in_tiebreak is True
    assert (state.p1_points, state.p2_points) == (5, 3)


def test_match_state_from_live_defaults_when_missing():
    state = match_state_from_live({})
    assert (state.p1_sets, state.p2_sets, state.p1_games, state.p2_games) == (0, 0, 0, 0)
    assert (state.p1_points, state.p2_points) == (0, 0)
    assert state.server == 1
    assert state.in_tiebreak is False


def _even_state_params():
    return MatchState(server=1), MatchParams(o1=0.62, d1=0.38, o2=0.62, d2=0.38)


def test_win_prob_distribution_zero_sd_is_single_point():
    state, params = _even_state_params()
    out = win_prob_distribution(state, params, ability_sd=0.0, n_sims=2000, seed=3)
    point = monte_carlo(state, params, n_sims=2000, seed=3).p1_win_prob
    assert out == [point]


def test_win_prob_distribution_length_and_range():
    state, params = _even_state_params()
    out = win_prob_distribution(
        state, params, ability_sd=0.05, n_scenarios=40, n_sims=200, seed=1
    )
    assert len(out) == 40
    assert all(0.0 <= p <= 1.0 for p in out)


def test_win_prob_distribution_reproducible():
    state, params = _even_state_params()
    kw = dict(ability_sd=0.05, n_scenarios=30, n_sims=200, seed=7)
    assert win_prob_distribution(state, params, **kw) == win_prob_distribution(
        state, params, **kw
    )


def test_win_prob_distribution_spreads_with_uncertainty():
    state, params = _even_state_params()
    small = win_prob_distribution(
        state, params, ability_sd=0.02, n_scenarios=80, n_sims=300, seed=5
    )
    big = win_prob_distribution(
        state, params, ability_sd=0.12, n_scenarios=80, n_sims=300, seed=5
    )
    assert statistics.pstdev(big) > statistics.pstdev(small)


def test_win_prob_distribution_rejects_bad_args():
    state, params = _even_state_params()
    with pytest.raises(ValueError):
        win_prob_distribution(state, params, ability_sd=-0.1)
    with pytest.raises(ValueError):
        win_prob_distribution(state, params, ability_sd=0.05, n_scenarios=0)
    with pytest.raises(ValueError):
        win_prob_distribution(state, params, ability_sd=0.05, n_sims=0)


def test_point_label_from_score_maps_game_points():
    assert point_label_from_score(0) == "0"
    assert point_label_from_score(15) == "15"
    assert point_label_from_score(30) == "30"
    assert point_label_from_score(40) == "40"


def test_point_label_from_score_advantage_wins():
    assert point_label_from_score(40, is_advantage=True) == "AD"


def test_point_label_from_score_unknown_falls_back_to_zero():
    assert point_label_from_score(99) == "0"

# --- point_win_prob ------------------------------------------------------


def test_point_win_prob_balanced_is_half():
    assert point_win_prob(0.5, 0.5) == pytest.approx(0.5)
    assert point_win_prob(0.7, 0.7) == pytest.approx(0.7 * 0.3 / (0.7 * 0.3 + 0.3 * 0.7))


def test_point_win_prob_monotonic_in_offense_and_defense():
    base = point_win_prob(0.6, 0.4)
    assert point_win_prob(0.7, 0.4) > base  # more offense -> more likely
    assert point_win_prob(0.6, 0.5) < base  # stronger returner -> less likely


def test_point_win_prob_degenerate_guard_is_half():
    assert point_win_prob(1.0, 1.0) == pytest.approx(0.5)
    assert point_win_prob(0.0, 0.0) == pytest.approx(0.5)


def test_point_win_prob_extremes():
    assert point_win_prob(1.0, 0.0) == pytest.approx(1.0)
    assert point_win_prob(0.0, 1.0) == pytest.approx(0.0)


@pytest.mark.parametrize("offense,defense", [(-0.1, 0.5), (1.1, 0.5), (0.5, -0.1), (0.5, 1.1)])
def test_point_win_prob_validates(offense, defense):
    with pytest.raises(ValueError):
        point_win_prob(offense, defense)


# --- point_count_from_label ----------------------------------------------


def test_point_count_from_label():
    assert point_count_from_label("0") == 0
    assert point_count_from_label("15") == 1
    assert point_count_from_label("30") == 2
    assert point_count_from_label("40") == 3
    assert point_count_from_label("ad") == 4
    with pytest.raises(ValueError):
        point_count_from_label("50")


# --- MatchParams / MatchState validation ---------------------------------


def test_match_params_validates_range():
    with pytest.raises(ValueError):
        MatchParams(o1=1.5, d1=0.4, o2=0.6, d2=0.4)


def test_match_state_validates_server():
    with pytest.raises(ValueError):
        MatchState(server=3)


# --- deterministic scoring -----------------------------------------------


def test_server_with_certain_offense_always_holds():
    # P1 serves with offense 1.0 vs P2 defense 0.0 -> wins every point it serves.
    params = MatchParams(o1=1.0, d1=1.0, o2=0.0, d2=0.0)
    state = MatchState(server=1)
    rng = random.Random(123)
    # With P1 unbeatable on serve and P2 unable to win on serve, P1 wins the match.
    winner, score = simulate_match_from_state(state, params, rng)
    assert winner == 1
    assert score == "2-0"


def test_simulate_from_decided_state_returns_winner_directly():
    params = MatchParams(o1=0.6, d1=0.4, o2=0.6, d2=0.4)
    rng = random.Random(0)
    assert simulate_match_from_state(MatchState(p1_sets=2), params, rng) == (1, "2-0")
    assert simulate_match_from_state(
        MatchState(p1_sets=1, p2_sets=2), params, rng
    ) == (2, "1-2")


def test_already_decided_match_short_circuits():
    params = MatchParams(o1=0.6, d1=0.4, o2=0.6, d2=0.4)
    state = MatchState(p1_sets=2, p2_sets=0, server=1)
    result = monte_carlo(state, params, n_sims=5000, seed=1)
    assert result.p1_win_prob == 1.0
    assert result.ci95_half_width == 0.0
    assert result.set_score_counts == {"2-0": 5000}


def test_already_decided_match_for_p2():
    params = MatchParams(o1=0.6, d1=0.4, o2=0.6, d2=0.4)
    state = MatchState(p1_sets=1, p2_sets=2, server=1)
    result = monte_carlo(state, params, n_sims=10, seed=1)
    assert result.p1_win_prob == 0.0
    assert result.set_score_counts == {"1-2": 10}


# --- monte_carlo ---------------------------------------------------------


def test_monte_carlo_symmetric_is_roughly_even():
    params = MatchParams(o1=0.62, d1=0.38, o2=0.62, d2=0.38)
    state = MatchState(server=1)
    result = monte_carlo(state, params, n_sims=20000, seed=7)
    # Symmetric abilities from 0-0 -> near 50/50 (small serve-first asymmetry).
    assert result.p1_win_prob == pytest.approx(0.5, abs=0.03)
    assert sum(result.set_score_counts.values()) == 20000


def test_monte_carlo_lopsided_favours_stronger_player():
    params = MatchParams(o1=0.85, d1=0.7, o2=0.5, d2=0.2)
    state = MatchState(server=1)
    result = monte_carlo(state, params, n_sims=10000, seed=3)
    assert result.p1_win_prob > 0.95


def test_monte_carlo_is_reproducible_with_seed():
    params = MatchParams(o1=0.65, d1=0.4, o2=0.6, d2=0.42)
    state = MatchState(server=2)
    a = monte_carlo(state, params, n_sims=4000, seed=42)
    b = monte_carlo(state, params, n_sims=4000, seed=42)
    assert a.p1_win_prob == b.p1_win_prob
    assert a.set_score_counts == b.set_score_counts


def test_monte_carlo_rejects_nonpositive_n():
    params = MatchParams(o1=0.6, d1=0.4, o2=0.6, d2=0.4)
    with pytest.raises(ValueError):
        monte_carlo(MatchState(), params, n_sims=0)


# --- analytic match_win_probability --------------------------------------


def test_game_win_prob_edges():
    from kalshi.tennis import _game_win_prob

    assert _game_win_prob(0.0) == 0.0
    assert _game_win_prob(1.0) == 1.0
    assert _game_win_prob(0.5) == pytest.approx(0.5)


def test_match_win_probability_symmetric_is_half():
    params = MatchParams(o1=0.62, d1=0.38, o2=0.62, d2=0.38)
    assert match_win_probability(params) == pytest.approx(0.5, abs=1e-9)


def test_match_win_probability_monotonic_and_bounded():
    weak = match_win_probability(MatchParams(o1=0.60, d1=0.40, o2=0.62, d2=0.38))
    strong = match_win_probability(MatchParams(o1=0.70, d1=0.50, o2=0.62, d2=0.38))
    assert 0.0 <= weak < 0.5 < strong <= 1.0


def test_match_win_probability_matches_monte_carlo():
    # The closed-form set/match model should track the simulation closely.
    params = MatchParams(o1=0.68, d1=0.45, o2=0.60, d2=0.36)
    analytic = match_win_probability(params)
    sim = monte_carlo(MatchState(server=1), params, n_sims=40000, seed=11)
    assert analytic == pytest.approx(sim.p1_win_prob, abs=0.02)


# --- baselines_from_match_odds (inversion) -------------------------------


def test_baselines_from_even_odds_are_balanced():
    b1, b2 = baselines_from_match_odds(0.5)
    assert b1 == pytest.approx(0.5, abs=1e-3)
    assert b2 == pytest.approx(0.5, abs=1e-3)


def test_baselines_reproduce_target_odds():
    target = 0.75
    b1, b2 = baselines_from_match_odds(target)
    assert b1 > 0.5 > b2
    spread = DEFAULT_ABILITY_SPREAD
    params = MatchParams(o1=b1 + spread, d1=b1 - spread, o2=b2 + spread, d2=b2 - spread)
    assert match_win_probability(params) == pytest.approx(target, abs=0.01)


def test_baselines_are_symmetric_for_mirror_odds():
    b1_fav, b2_fav = baselines_from_match_odds(0.8)
    b1_dog, b2_dog = baselines_from_match_odds(0.2)
    assert b1_fav == pytest.approx(b2_dog, abs=1e-3)
    assert b2_fav == pytest.approx(b1_dog, abs=1e-3)


def test_baselines_keep_abilities_in_range_for_extreme_odds():
    b1, b2 = baselines_from_match_odds(0.999)
    spread = DEFAULT_ABILITY_SPREAD
    for value in (b1 + spread, b1 - spread, b2 + spread, b2 - spread):
        assert 0.0 <= value <= 1.0


def test_monte_carlo_resumes_from_tiebreak_state():
    # 6-6, P1 leads the tiebreak 6-0 -> overwhelmingly P1 takes the set, and
    # combined with a set in hand, the match.
    params = MatchParams(o1=0.62, d1=0.38, o2=0.62, d2=0.38)
    state = MatchState(
        p1_sets=1,
        p2_sets=0,
        p1_games=6,
        p2_games=6,
        p1_points=6,
        p2_points=0,
        server=1,
        in_tiebreak=True,
    )
    result = monte_carlo(state, params, n_sims=4000, seed=5)
    assert result.p1_win_prob > 0.95
