"""Tests for kalshi.markets pure helpers."""

from __future__ import annotations

from kalshi.markets import (
    build_game_groups,
    evaluate_in_money,
    event_competition,
    event_scope,
    fp_to_float,
    game_key,
    in_money_badge,
    live_scores,
    market_label,
    market_type_name,
    matchup_name,
    price_cents_for_side,
    series_ticker_for_market,
    tennis_live_score,
)

HOME = "home-1"
AWAY = "away-2"


# --- price_cents_for_side ------------------------------------------------


def test_price_prefers_dollar_field():
    market = {"yes_ask_dollars": "0.5600", "yes_ask": 99}
    assert price_cents_for_side(market, "yes", "ask") == 56.0


def test_price_falls_back_to_legacy_cents():
    market = {"no_bid": 44}
    assert price_cents_for_side(market, "no", "bid") == 44.0


def test_price_zero_dollar_returns_none():
    assert price_cents_for_side({"yes_ask_dollars": "0.0000"}, "yes", "ask") is None


def test_price_missing_returns_none():
    assert price_cents_for_side({}, "yes", "ask") is None


def test_price_bad_value_returns_none():
    assert price_cents_for_side({"yes_ask_dollars": "abc"}, "yes", "ask") is None


# --- game_key ------------------------------------------------------------


def test_game_key_strips_series_prefix_and_suffix():
    ev = {"event_ticker": "KXWCGAME-26JUN20NEDSWE-NED", "series_ticker": "KXWCGAME"}
    assert game_key(ev) == "26JUN20NEDSWE"


def test_game_key_no_series_uses_first_dash_split():
    ev = {"event_ticker": "ABC-26JUN20NEDSWE"}
    assert game_key(ev) == "26JUN20NEDSWE"


def test_game_key_no_dash_returns_ticker():
    ev = {"event_ticker": "PLAIN"}
    assert game_key(ev) == "PLAIN"


# --- matchup / market type ----------------------------------------------


def test_matchup_name_drops_market_suffix():
    ev = {"title": "Netherlands vs Sweden: Spread", "sub_title": "x"}
    assert matchup_name(ev) == "Netherlands vs Sweden"


def test_matchup_name_falls_back_to_sub_title():
    assert matchup_name({"sub_title": "Fallback"}) == "Fallback"


def test_market_type_name_from_title():
    assert market_type_name({"title": "A vs B: Total Goals"}) == "Total Goals"


def test_market_type_name_defaults_to_winner():
    assert market_type_name({"title": "A vs B"}) == "Winner"


# --- competition / scope -------------------------------------------------


def test_event_competition_and_scope():
    ev = {"product_metadata": {"competition": "WC", "competition_scope": "Game"}}
    assert event_competition(ev) == "WC"
    assert event_scope(ev) == "Game"


def test_event_competition_missing():
    assert event_competition({}) == ""
    assert event_scope({}) == ""


# --- build_game_groups ---------------------------------------------------


def test_build_game_groups_merges_siblings_and_picks_game_rep(winner_event, total_event):
    groups = build_game_groups([total_event, winner_event])
    assert len(groups) == 1
    g = groups[0]
    assert g["competition"] == "World Soccer Cup"
    assert g["has_game"] is True
    # The Game-scope event is chosen as the representative.
    assert g["rep_ticker"] == "KXWCGAME-26JUN20NEDSWE"
    assert g["matchup"] == "Netherlands vs Sweden"
    assert set(g["series"]) == {"KXWCGAME", "KXWCTOTAL"}
    assert g["event_tickers"] == (
        "KXWCGAME-26JUN20NEDSWE",
        "KXWCTOTAL-26JUN20NEDSWE",
    )


def test_build_game_groups_separate_games(winner_event):
    other = dict(winner_event)
    other["event_ticker"] = "KXWCGAME-26JUN21BRAARG"
    groups = build_game_groups([winner_event, other])
    assert len(groups) == 2


def test_build_game_groups_no_game_scope_uses_first(total_event):
    groups = build_game_groups([total_event])
    assert groups[0]["has_game"] is False
    assert groups[0]["rep_ticker"] == "KXWCTOTAL-26JUN20NEDSWE"


# --- series_ticker_for_market / fp_to_float / market_label ---------------


def test_series_ticker_prefers_explicit_field():
    assert series_ticker_for_market({"series_ticker": "KXWCGAME"}) == "KXWCGAME"


def test_series_ticker_from_event_ticker():
    assert (
        series_ticker_for_market({"event_ticker": "KXWCGAME-26JUN20NEDSWE"})
        == "KXWCGAME"
    )


def test_series_ticker_none_when_unknown():
    assert series_ticker_for_market({"ticker": "PLAIN"}) is None


def test_fp_to_float_parses_and_defaults():
    assert fp_to_float("1.50") == 1.5
    assert fp_to_float(None) == 0.0
    assert fp_to_float("nope") == 0.0


def test_market_label_includes_ticker_and_ask():
    label = market_label(
        {"yes_sub_title": "Netherlands", "ticker": "T1", "yes_ask_dollars": "0.5600"}
    )
    assert "Netherlands" in label
    assert "[T1]" in label
    assert "56c" in label


# --- live_scores ---------------------------------------------------------


def test_live_scores_prefers_same_game():
    details = {
        "home_same_game_score": 4,
        "away_same_game_score": 1,
        "home_aggregate_score": 9,
        "away_aggregate_score": 9,
    }
    assert live_scores(details) == (4.0, 1.0)


def test_live_scores_falls_back_to_aggregate():
    details = {"home_aggregate_score": 2, "away_aggregate_score": 0}
    assert live_scores(details) == (2.0, 0.0)


def test_live_scores_none_when_missing():
    assert live_scores({"status": "live"}) is None


# --- tennis_live_score ---------------------------------------------------

# A real-shaped tennis_tournament_singles live-data payload (Proietti vs
# Trevisan): comp1 leads sets 1-1, current set comp1 0 games / comp2 2 games,
# comp1 at 15 / comp2 at 0, comp2 serving.
_TENNIS_DETAILS = {
    "competitor1_id": "c1",
    "competitor2_id": "c2",
    "competitor1_overall_score": 1,
    "competitor2_overall_score": 1,
    "competitor1_round_scores": [
        {"outcome": "loser", "score": 2},
        {"outcome": "winner", "score": 6},
        {"outcome": "ongoing", "score": 0},
    ],
    "competitor2_round_scores": [
        {"outcome": "winner", "score": 6},
        {"outcome": "loser", "score": 1},
        {"outcome": "ongoing", "score": 2},
    ],
    "competitor1_current_round_score": 15,
    "competitor2_current_round_score": 0,
    "advantage": "",
    "server": "c2",
    "winner": "",
}


def test_tennis_live_score_oriented_by_competitor_id():
    out = tennis_live_score(_TENNIS_DETAILS, "c1", "c2")
    assert out["sets"] == (1, 1)
    assert out["games"] == (0, 2)
    assert out["points"] == (15, 0)
    assert out["server"] == 2
    assert out["winner"] is None
    assert out["in_tiebreak"] is False
    assert out["oriented"] is True


def test_tennis_live_score_swaps_when_p1_is_competitor2():
    out = tennis_live_score(_TENNIS_DETAILS, "c2", "c1")
    assert out["sets"] == (1, 1)
    assert out["games"] == (2, 0)  # P1 is now competitor2
    assert out["points"] == (0, 15)
    assert out["server"] == 1  # competitor2 (the server) is now P1
    assert out["oriented"] is True


def test_tennis_live_score_assumes_competitor1_when_ids_missing():
    out = tennis_live_score(_TENNIS_DETAILS)
    assert out["games"] == (0, 2)
    assert out["oriented"] is False


def test_tennis_live_score_tiebreak_and_advantage_and_winner():
    details = {
        "competitor1_id": "c1",
        "competitor2_id": "c2",
        "competitor1_overall_score": 1,
        "competitor2_overall_score": 1,
        "competitor1_round_scores": [{"outcome": "ongoing", "score": 6}],
        "competitor2_round_scores": [{"outcome": "ongoing", "score": 6}],
        "competitor1_current_round_score": 5,
        "competitor2_current_round_score": 3,
        "advantage": "c1",
        "server": "c1",
        "winner": "c2",
    }
    out = tennis_live_score(details, "c1", "c2")
    assert out["in_tiebreak"] is True
    assert out["points"] == (5, 3)
    assert out["advantage"] == 1
    assert out["winner"] == 2


def test_tennis_live_score_none_when_not_tennis():
    assert tennis_live_score({"status": "live"}) is None


# --- evaluate_in_money ---------------------------------------------------

DETAILS = {"home_same_game_score": 3, "away_same_game_score": 0}


def test_itm_total_over():
    market = {"yes_sub_title": "Over 2.5 goals scored"}
    assert evaluate_in_money(market, DETAILS, HOME, AWAY) is True


def test_otm_total_over():
    market = {"yes_sub_title": "Over 3.5 goals scored"}
    assert evaluate_in_money(market, DETAILS, HOME, AWAY) is False


def test_total_under():
    market = {"yes_sub_title": "Under 3.5 goals scored"}
    assert evaluate_in_money(market, DETAILS, HOME, AWAY) is True


def test_total_no_threshold_returns_none():
    market = {"yes_sub_title": "Over many goals"}
    assert evaluate_in_money(market, DETAILS, HOME, AWAY) is None


def test_both_teams_to_score_false():
    market = {"yes_sub_title": "Both teams to score"}
    assert evaluate_in_money(market, DETAILS, HOME, AWAY) is False


def test_both_teams_to_score_true():
    details = {"home_same_game_score": 1, "away_same_game_score": 2}
    market = {"yes_sub_title": "Both teams to score"}
    assert evaluate_in_money(market, details, HOME, AWAY) is True


def test_winner_home_itm():
    market = {"yes_sub_title": "Netherlands", "custom_strike": {"soccer_team": HOME}}
    assert evaluate_in_money(market, DETAILS, HOME, AWAY) is True


def test_winner_away_otm():
    market = {"yes_sub_title": "Sweden", "custom_strike": {"soccer_team": AWAY}}
    assert evaluate_in_money(market, DETAILS, HOME, AWAY) is False


def test_spread_more_than():
    market = {
        "yes_sub_title": "Netherlands wins by more than 2.5 goals",
        "custom_strike": {"soccer_team": HOME},
    }
    assert evaluate_in_money(market, DETAILS, HOME, AWAY) is True


def test_team_total_over():
    market = {
        "yes_sub_title": "Netherlands over 1.5 goals",
        "custom_strike": {"soccer_team": HOME},
    }
    assert evaluate_in_money(market, DETAILS, HOME, AWAY) is True


def test_tie_sentinel_team():
    market = {"yes_sub_title": "Tie", "custom_strike": {"soccer_team": "tie-id"}}
    draw = {"home_same_game_score": 1, "away_same_game_score": 1}
    assert evaluate_in_money(market, draw, HOME, AWAY) is True
    assert evaluate_in_money(market, DETAILS, HOME, AWAY) is False


def test_no_scores_returns_none():
    market = {"yes_sub_title": "Over 2.5 goals scored"}
    assert evaluate_in_money(market, {"status": "live"}, HOME, AWAY) is None


def test_unknown_market_type_returns_none():
    # No team, no over/under, no "both teams" -> not evaluable from score alone.
    market = {"yes_sub_title": "Red card shown in match", "custom_strike": {}}
    assert evaluate_in_money(market, DETAILS, HOME, AWAY) is None


# --- in_money_badge ------------------------------------------------------


def test_in_money_badge_labels():
    assert "ITM" in in_money_badge(True)
    assert "OTM" in in_money_badge(False)
    assert in_money_badge(None) == ""
