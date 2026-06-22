"""Tests for kalshi.auth: key loading, signing, and signed headers."""

from __future__ import annotations

import base64

import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, padding

from kalshi.auth import (
    KalshiAuthError,
    KalshiCredentials,
    load_private_key,
    sign_pss_text,
)


def _pem_string(private_key) -> str:
    return private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("utf-8")


def test_load_private_key_from_pem_string(rsa_private_key):
    pem = _pem_string(rsa_private_key)
    loaded = load_private_key(key_pem=pem)
    assert loaded.key_size == rsa_private_key.key_size


def test_load_private_key_from_file(rsa_private_key, tmp_path):
    pem = _pem_string(rsa_private_key)
    key_file = tmp_path / "key.pem"
    key_file.write_text(pem)
    loaded = load_private_key(key_path=str(key_file))
    assert loaded.key_size == rsa_private_key.key_size


def test_load_private_key_path_precedence(rsa_private_key, tmp_path):
    """key_path is used even when key_pem is also supplied."""
    pem = _pem_string(rsa_private_key)
    key_file = tmp_path / "key.pem"
    key_file.write_text(pem)
    loaded = load_private_key(key_path=str(key_file), key_pem="garbage-not-a-key")
    assert loaded.key_size == rsa_private_key.key_size


def test_load_private_key_missing_file_raises(tmp_path):
    with pytest.raises(KalshiAuthError, match="not found"):
        load_private_key(key_path=str(tmp_path / "nope.pem"))


def test_load_private_key_no_source_raises():
    with pytest.raises(KalshiAuthError, match="No private key provided"):
        load_private_key()


def test_load_non_rsa_key_raises():
    ec_key = ec.generate_private_key(ec.SECP256R1())
    pem = ec_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("utf-8")
    with pytest.raises(KalshiAuthError, match="not an RSA private key"):
        load_private_key(key_pem=pem)


def test_sign_pss_text_verifies_against_public_key(rsa_private_key):
    text = "1700000000000GET/trade-api/v2/markets"
    signature_b64 = sign_pss_text(rsa_private_key, text)
    signature = base64.b64decode(signature_b64)
    # Should not raise -> signature is valid for the public key.
    rsa_private_key.public_key().verify(
        signature,
        text.encode("utf-8"),
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.DIGEST_LENGTH,
        ),
        hashes.SHA256(),
    )


def test_headers_sign_timestamp_method_path(rsa_credentials):
    headers = rsa_credentials.headers("get", "/trade-api/v2/markets")
    assert headers["KALSHI-ACCESS-KEY"] == "test-key-id"
    assert set(headers) == {
        "KALSHI-ACCESS-KEY",
        "KALSHI-ACCESS-SIGNATURE",
        "KALSHI-ACCESS-TIMESTAMP",
    }
    ts = headers["KALSHI-ACCESS-TIMESTAMP"]
    # Method is upper-cased and signed message is ts + METHOD + path.
    message = ts + "GET" + "/trade-api/v2/markets"
    signature = base64.b64decode(headers["KALSHI-ACCESS-SIGNATURE"])
    rsa_credentials.private_key.public_key().verify(
        signature,
        message.encode("utf-8"),
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.DIGEST_LENGTH,
        ),
        hashes.SHA256(),
    )


def test_headers_strip_query_string(rsa_credentials):
    """The signature must cover the path without the query string."""
    headers = rsa_credentials.headers("GET", "/trade-api/v2/markets?limit=10&x=1")
    ts = headers["KALSHI-ACCESS-TIMESTAMP"]
    message = ts + "GET" + "/trade-api/v2/markets"
    signature = base64.b64decode(headers["KALSHI-ACCESS-SIGNATURE"])
    rsa_credentials.private_key.public_key().verify(
        signature,
        message.encode("utf-8"),
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.DIGEST_LENGTH,
        ),
        hashes.SHA256(),
    )


def test_from_env_with_inline_key(monkeypatch, rsa_private_key):
    monkeypatch.setenv("KALSHI_API_KEY_ID", "env-key-id")
    monkeypatch.setenv("KALSHI_PRIVATE_KEY", _pem_string(rsa_private_key))
    monkeypatch.delenv("KALSHI_PRIVATE_KEY_PATH", raising=False)
    creds = KalshiCredentials.from_env()
    assert creds.api_key_id == "env-key-id"
    assert creds.private_key.key_size == rsa_private_key.key_size


def test_from_env_missing_key_id_raises(monkeypatch):
    monkeypatch.delenv("KALSHI_API_KEY_ID", raising=False)
    with pytest.raises(KalshiAuthError, match="KALSHI_API_KEY_ID is not set"):
        KalshiCredentials.from_env()


def test_from_env_missing_private_key_raises(monkeypatch):
    monkeypatch.setenv("KALSHI_API_KEY_ID", "env-key-id")
    monkeypatch.delenv("KALSHI_PRIVATE_KEY", raising=False)
    monkeypatch.delenv("KALSHI_PRIVATE_KEY_PATH", raising=False)
    with pytest.raises(KalshiAuthError, match="No private key provided"):
        KalshiCredentials.from_env()
