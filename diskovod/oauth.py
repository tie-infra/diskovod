from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import secrets
import time
from dataclasses import dataclass
from urllib.parse import urlencode

import aiohttp

from .models import ChatCredentials
from .store import Store

AUTHORIZE_URL = "https://auth.openai.com/oauth/authorize"
TOKEN_URL = "https://auth.openai.com/oauth/token"
CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
CALLBACK_URL = "http://localhost:1455/auth/callback"


@dataclass(slots=True)
class OAuthAttempt:
    state: str
    verifier: str
    redirect_uri: str


class ChatGPTAccount:
    """Own ChatGPT OAuth lifecycle without owning any model invocation logic."""

    def __init__(self, store: Store):
        self.store = store
        self.session: aiohttp.ClientSession | None = None
        self.oauth: OAuthAttempt | None = None
        self.last_error: str | None = None
        self._refresh_lock = asyncio.Lock()

    async def start(self) -> None:
        if self.session is None:
            self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60))

    async def close(self) -> None:
        self.oauth = None
        if self.session is not None:
            await self.session.close()
            self.session = None

    @property
    def connected(self) -> bool:
        return self.store.chat_credentials() is not None

    @property
    def email(self) -> str | None:
        credentials = self.store.chat_credentials()
        return credentials.email if credentials else None

    async def begin_oauth(self) -> str:
        verifier = base64.urlsafe_b64encode(secrets.token_bytes(32)).rstrip(b"=").decode()
        challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
        state = secrets.token_hex(16)
        self.oauth = OAuthAttempt(state, verifier, CALLBACK_URL)
        return (
            AUTHORIZE_URL
            + "?"
            + urlencode(
                {
                    "client_id": CLIENT_ID,
                    "redirect_uri": CALLBACK_URL,
                    "scope": "openid profile email offline_access",
                    "response_type": "code",
                    "code_challenge": challenge,
                    "code_challenge_method": "S256",
                    "id_token_add_organizations": "true",
                    "state": state,
                    "codex_cli_simplified_flow": "true",
                    "originator": "diskovod",
                }
            )
        )

    async def finish_oauth(
        self,
        *,
        code: str | None,
        state: str | None,
        error: str | None,
    ) -> None:
        attempt = self.oauth
        if not attempt or not state or not secrets.compare_digest(state, attempt.state):
            raise RuntimeError("Invalid or expired ChatGPT OAuth state")
        self.oauth = None
        if error:
            self.last_error = error
            raise RuntimeError(f"OpenAI sign-in failed: {error}")
        if not code:
            raise RuntimeError("ChatGPT OAuth callback did not include an authorization code")
        try:
            await self._exchange(
                {
                    "grant_type": "authorization_code",
                    "client_id": CLIENT_ID,
                    "code": code,
                    "redirect_uri": attempt.redirect_uri,
                    "code_verifier": attempt.verifier,
                }
            )
        except Exception as exc:
            self.last_error = str(exc)
            raise

    async def credentials(self) -> ChatCredentials:
        credentials = self.store.chat_credentials()
        if credentials is None:
            raise RuntimeError("ChatGPT Subscription is not connected")
        if credentials.expires_at > time.time() + 300:
            return credentials
        async with self._refresh_lock:
            credentials = self.store.chat_credentials()
            if credentials and credentials.expires_at > time.time() + 300:
                return credentials
            if credentials is None:
                raise RuntimeError("ChatGPT Subscription is not connected")
            return await self._exchange(
                {
                    "grant_type": "refresh_token",
                    "client_id": CLIENT_ID,
                    "refresh_token": credentials.refresh_token,
                },
                clear_on_auth_error=True,
            )

    async def _exchange(
        self,
        data: dict[str, str],
        *,
        clear_on_auth_error: bool = False,
    ) -> ChatCredentials:
        if self.session is None:
            raise RuntimeError("ChatGPT OAuth service has not started")
        async with self.session.post(TOKEN_URL, data=data) as response:
            payload = await response.json(content_type=None)
            if response.status >= 400:
                if clear_on_auth_error and response.status in {400, 401, 403}:
                    self.store.clear_chat_credentials()
                detail = payload.get("error_description") or payload.get("message") or payload.get("error")
                raise RuntimeError(
                    f"OpenAI token exchange returned HTTP {response.status}: {detail or 'unknown error'}"
                )
        return self._save_tokens(payload)

    def _save_tokens(self, payload: dict) -> ChatCredentials:
        claims = self._jwt_claims(payload.get("id_token") or payload["access_token"])
        auth = claims.get("https://api.openai.com/auth") or {}
        organizations = claims.get("organizations") or []
        account_id = claims.get("chatgpt_account_id") or auth.get("chatgpt_account_id")
        if not account_id and organizations:
            account_id = organizations[0].get("id")
        previous = self.store.chat_credentials()
        credentials = ChatCredentials(
            access_token=payload["access_token"],
            refresh_token=payload.get("refresh_token") or (previous.refresh_token if previous else ""),
            expires_at=time.time() + float(payload.get("expires_in", 3600)),
            account_id=account_id or (previous.account_id if previous else None),
            email=claims.get("email") or payload.get("email") or (previous.email if previous else None),
        )
        self.store.set_chat_credentials(credentials)
        self.last_error = None
        return credentials

    @staticmethod
    def _jwt_claims(token: str) -> dict:
        try:
            encoded = token.split(".")[1]
            encoded += "=" * (-len(encoded) % 4)
            return json.loads(base64.urlsafe_b64decode(encoded))
        except Exception:
            return {}
