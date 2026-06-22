"""Shared pytest fixtures for the Kalshi Kelly test suite."""

from __future__ import annotations

import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from kalshi.auth import KalshiCredentials

# A fixed, throwaway team id used as the "tie" sentinel in fixtures.
TIE_TEAM_ID = "tie-sentinel-0000"
HOME_TEAM_ID = "home-team-1111"
AWAY_TEAM_ID = "away-team-2222"


@pytest.fixture(scope="session")
def rsa_private_key() -> rsa.RSAPrivateKey:
    """A small in-memory RSA key (2048-bit) for signing tests."""
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


@pytest.fixture
def rsa_credentials(rsa_private_key) -> KalshiCredentials:
    """Credentials backed by the in-memory RSA key (no network/env needed)."""
    return KalshiCredentials(api_key_id="test-key-id", private_key=rsa_private_key)


@pytest.fixture
def winner_event() -> dict:
    """A head-to-head (Game-scope) winner event."""
    return {
        "event_ticker": "KXWCGAME-26JUN20NEDSWE",
        "series_ticker": "KXWCGAME",
        "title": "Netherlands vs Sweden",
        "sub_title": "Who wins?",
        "product_metadata": {
            "competition": "World Soccer Cup",
            "competition_scope": "Game",
        },
    }


@pytest.fixture
def total_event() -> dict:
    """A sibling totals event for the same game (non-Game scope)."""
    return {
        "event_ticker": "KXWCTOTAL-26JUN20NEDSWE",
        "series_ticker": "KXWCTOTAL",
        "title": "Netherlands vs Sweden: Total Goals",
        "sub_title": "Total goals",
        "product_metadata": {
            "competition": "World Soccer Cup",
            "competition_scope": "Total",
        },
    }


@pytest.fixture
def market() -> dict:
    """A representative market with fixed-point dollar prices."""
    return {
        "ticker": "KXWCGAME-26JUN20NEDSWE-NED",
        "event_ticker": "KXWCGAME-26JUN20NEDSWE",
        "series_ticker": "KXWCGAME",
        "yes_sub_title": "Netherlands",
        "yes_ask_dollars": "0.5600",
        "no_ask_dollars": "0.4500",
        "status": "active",
        "custom_strike": {"soccer_team": HOME_TEAM_ID},
    }


@pytest.fixture
def live_details() -> dict:
    """Soccer live-data details mirroring the real NED (home) 4-1 SWE (away)."""
    return {
        "home_same_game_score": 4,
        "away_same_game_score": 1,
        "home_aggregate_score": 4,
        "away_aggregate_score": 1,
        "status": "live",
        "status_text": "2nd - 69'",
        "match_status": "2nd half",
    }
