"""Kalshi API key authentication.

Builds the signed request headers required by the Kalshi Trade API. Each
request is signed by concatenating the millisecond timestamp, the HTTP method,
and the request path (without the query string), then signing that string with
the account's RSA private key using RSA-PSS / SHA-256.

See: https://docs.kalshi.com/getting_started/api_keys
"""

from __future__ import annotations

import base64
import datetime
import os
from dataclasses import dataclass

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa


class KalshiAuthError(Exception):
    """Raised when credentials are missing or the private key cannot be loaded."""


def _load_private_key_from_pem(pem_bytes: bytes) -> rsa.RSAPrivateKey:
    key = serialization.load_pem_private_key(
        pem_bytes,
        password=None,
        backend=default_backend(),
    )
    if not isinstance(key, rsa.RSAPrivateKey):
        raise KalshiAuthError(
            "The provided private key is not an RSA private key, which Kalshi requires."
        )
    return key


def load_private_key(
    *, key_path: str | None = None, key_pem: str | None = None
) -> rsa.RSAPrivateKey:
    """Load an RSA private key from a file path or an inline PEM string.

    Exactly one source is used; ``key_path`` takes precedence when both are set.
    """
    if key_path:
        if not os.path.exists(key_path):
            raise KalshiAuthError(
                f"Private key file not found at '{key_path}'. "
                "Set KALSHI_PRIVATE_KEY_PATH to the downloaded Kalshi key file."
            )
        with open(key_path, "rb") as fh:
            return _load_private_key_from_pem(fh.read())

    if key_pem:
        return _load_private_key_from_pem(key_pem.encode("utf-8"))

    raise KalshiAuthError(
        "No private key provided. Set KALSHI_PRIVATE_KEY_PATH or KALSHI_PRIVATE_KEY."
    )


def sign_pss_text(private_key: rsa.RSAPrivateKey, text: str) -> str:
    """Sign ``text`` with RSA-PSS / SHA-256 and return a base64 signature."""
    signature = private_key.sign(
        text.encode("utf-8"),
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.DIGEST_LENGTH,
        ),
        hashes.SHA256(),
    )
    return base64.b64encode(signature).decode("utf-8")


@dataclass
class KalshiCredentials:
    """Holds the loaded key id and private key, and builds signed headers."""

    api_key_id: str
    private_key: rsa.RSAPrivateKey

    @classmethod
    def from_env(cls) -> "KalshiCredentials":
        """Build credentials from environment variables.

        Expects ``KALSHI_API_KEY_ID`` plus one of ``KALSHI_PRIVATE_KEY_PATH``
        or ``KALSHI_PRIVATE_KEY``.
        """
        api_key_id = os.getenv("KALSHI_API_KEY_ID", "").strip()
        if not api_key_id:
            raise KalshiAuthError(
                "KALSHI_API_KEY_ID is not set. Add it to your .env file."
            )

        key_path = os.getenv("KALSHI_PRIVATE_KEY_PATH", "").strip() or None
        key_pem = os.getenv("KALSHI_PRIVATE_KEY", "").strip() or None
        private_key = load_private_key(key_path=key_path, key_pem=key_pem)
        return cls(api_key_id=api_key_id, private_key=private_key)

    def headers(self, method: str, path: str) -> dict[str, str]:
        """Build signed request headers for ``method`` and ``path``.

        ``path`` must be the full API path (e.g. ``/trade-api/v2/markets``)
        WITHOUT a query string; the query string must be stripped before signing.
        """
        path_without_query = path.split("?", 1)[0]
        timestamp_ms = str(
            int(datetime.datetime.now(datetime.timezone.utc).timestamp() * 1000)
        )
        message = timestamp_ms + method.upper() + path_without_query
        signature = sign_pss_text(self.private_key, message)
        return {
            "KALSHI-ACCESS-KEY": self.api_key_id,
            "KALSHI-ACCESS-SIGNATURE": signature,
            "KALSHI-ACCESS-TIMESTAMP": timestamp_ms,
        }
