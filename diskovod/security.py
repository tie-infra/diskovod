from __future__ import annotations

import base64
import hashlib
import hmac
import os

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


class SecretBox:
    def __init__(self, secret: str):
        if len(secret) < 32:
            raise ValueError("The secret key file must contain at least 32 characters")
        self._key = hashlib.sha256(secret.encode()).digest()

    def seal(self, value: str) -> str:
        nonce = os.urandom(12)
        body = AESGCM(self._key).encrypt(nonce, value.encode(), b"diskovod-v1")
        return "v1." + base64.urlsafe_b64encode(nonce + body).decode()

    def open(self, value: str) -> str:
        version, encoded = value.split(".", 1)
        if version != "v1":
            raise ValueError("Unsupported encrypted value")
        data = base64.urlsafe_b64decode(encoded)
        return AESGCM(self._key).decrypt(data[:12], data[12:], b"diskovod-v1").decode()


def password_matches(candidate: str, expected: str) -> bool:
    return hmac.compare_digest(candidate.encode(), expected.encode())
