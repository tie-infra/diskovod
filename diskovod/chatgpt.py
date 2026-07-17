from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import secrets
import time
from dataclasses import dataclass
from urllib.parse import urlencode, urlparse

import aiohttp

from .models import ChatCredentials
from .store import Store

log = logging.getLogger(__name__)
AUTHORIZE_URL = "https://auth.openai.com/oauth/authorize"
TOKEN_URL = "https://auth.openai.com/oauth/token"
BACKEND_URL = "https://chatgpt.com/backend-api/codex"
CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
CALLBACK_URL = "http://localhost:1455/auth/callback"
ORIGINATOR = "zed"
PROVIDERS = frozenset({"chatgpt", "custom"})


def normalize_custom_base_url(value: str) -> str:
    base_url = value.strip().rstrip("/")
    try:
        parsed = urlparse(base_url)
        hostname = parsed.hostname
        parsed.port
    except ValueError as exc:
        raise ValueError(
            "API base URL must be an absolute HTTP(S) URL without credentials or a query"
        ) from exc
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or not hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or any(character.isspace() for character in base_url)
    ):
        raise ValueError("API base URL must be an absolute HTTP(S) URL without credentials or a query")
    return base_url


@dataclass(slots=True)
class OAuthAttempt:
    state: str
    verifier: str
    redirect_uri: str


class ChatGPTClient:
    def __init__(self, store: Store):
        self.store = store
        self.session: aiohttp.ClientSession | None = None
        self.oauth: OAuthAttempt | None = None
        self.last_error: str | None = None
        self._refresh_lock = asyncio.Lock()

    async def start(self) -> None:
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=180))

    async def close(self) -> None:
        self.oauth = None
        if self.session:
            await self.session.close()

    @property
    def connected(self) -> bool:
        provider = self.store.app_settings().provider
        if provider == "custom":
            return self.store.custom_provider() is not None
        return self.subscription_connected

    @property
    def subscription_connected(self) -> bool:
        return self.store.chat_credentials() is not None

    @property
    def custom_connected(self) -> bool:
        return self.store.custom_provider() is not None

    @property
    def active_provider(self) -> str:
        provider = self.store.app_settings().provider
        return provider if provider in PROVIDERS else "chatgpt"

    @property
    def provider_label(self) -> str:
        if self.active_provider == "custom":
            provider = self.store.custom_provider()
            return provider.name if provider else "Custom API"
        return "ChatGPT subscription"

    @property
    def email(self) -> str | None:
        creds = self.store.chat_credentials()
        return creds.email if creds else None

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
                    "originator": ORIGINATOR,
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
            await self._exchange(code, attempt.verifier, attempt.redirect_uri)
        except Exception as exc:
            log.exception("OAuth callback failed")
            self.last_error = str(exc)
            raise

    async def _exchange(self, code: str, verifier: str, redirect_uri: str) -> None:
        assert self.session
        async with self.session.post(
            TOKEN_URL,
            data={
                "grant_type": "authorization_code",
                "client_id": CLIENT_ID,
                "code": code,
                "redirect_uri": redirect_uri,
                "code_verifier": verifier,
            },
        ) as response:
            payload = await response.json(content_type=None)
            if response.status >= 400:
                detail = payload.get("error_description") or payload.get("message") or payload.get("error")
                raise RuntimeError(
                    f"OpenAI token exchange returned HTTP {response.status}: {detail or 'unknown error'}"
                )
        self._save_tokens(payload)

    def _save_tokens(self, payload: dict) -> ChatCredentials:
        claims = self._jwt_claims(payload.get("id_token") or payload["access_token"])
        auth = claims.get("https://api.openai.com/auth") or {}
        orgs = claims.get("organizations") or []
        account_id = claims.get("chatgpt_account_id") or auth.get("chatgpt_account_id")
        if not account_id and orgs:
            account_id = orgs[0].get("id")
        creds = ChatCredentials(
            payload["access_token"],
            payload["refresh_token"],
            time.time() + float(payload.get("expires_in", 3600)),
            account_id,
            claims.get("email") or payload.get("email"),
        )
        self.store.set_chat_credentials(creds)
        self.last_error = None
        return creds

    @staticmethod
    def _jwt_claims(token: str) -> dict:
        try:
            encoded = token.split(".")[1]
            encoded += "=" * (-len(encoded) % 4)
            return json.loads(base64.urlsafe_b64decode(encoded))
        except Exception:
            return {}

    async def credentials(self) -> ChatCredentials:
        creds = self.store.chat_credentials()
        if not creds:
            raise RuntimeError("ChatGPT is not connected")
        if creds.expires_at > time.time() + 300:
            return creds
        async with self._refresh_lock:
            creds = self.store.chat_credentials()
            if creds and creds.expires_at > time.time() + 300:
                return creds
            assert creds and self.session
            async with self.session.post(
                TOKEN_URL,
                data={
                    "grant_type": "refresh_token",
                    "client_id": CLIENT_ID,
                    "refresh_token": creds.refresh_token,
                },
            ) as response:
                payload = await response.json(content_type=None)
                if response.status >= 400:
                    if response.status in (400, 401, 403):
                        self.store.clear_chat_credentials()
                    raise RuntimeError(f"OpenAI token refresh returned HTTP {response.status}")
            return self._save_tokens(payload)

    async def complete(
        self,
        messages: list[dict[str, str]],
        instructions: str,
        model: str,
        effort: str,
        *,
        purpose: str = "conversation",
        max_output_tokens: int | None = None,
    ) -> str:
        try:
            if self.active_provider == "custom":
                result = await self._complete_custom(
                    messages, instructions, model, purpose, max_output_tokens
                )
            else:
                result = await self._complete_subscription(
                    messages, instructions, model, effort, purpose, max_output_tokens
                )
        except Exception as exc:
            self.last_error = str(exc)
            raise
        self.last_error = None
        return result

    async def _complete_subscription(
        self,
        messages: list[dict[str, str]],
        instructions: str,
        model: str,
        effort: str,
        purpose: str,
        max_output_tokens: int | None,
    ) -> str:
        creds = await self.credentials()
        assert self.session
        input_items = [
            {
                "type": "message",
                "role": m["role"],
                "content": [
                    {
                        "type": "output_text" if m["role"] == "assistant" else "input_text",
                        "text": m["content"],
                        **({"annotations": []} if m["role"] == "assistant" else {}),
                    }
                ],
            }
            for m in messages
        ]
        headers = {
            "Authorization": f"Bearer {creds.access_token}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "OpenAI-Beta": "responses=experimental",
            "Originator": ORIGINATOR,
        }
        if creds.account_id:
            headers["ChatGPT-Account-Id"] = creds.account_id
        body = {
            "model": model,
            "instructions": instructions,
            "input": input_items,
            "stream": True,
            "store": False,
            "reasoning": {"effort": effort, "summary": "auto"},
        }
        if max_output_tokens is not None:
            # The ChatGPT Codex backend rejects the public Responses API's
            # max_output_tokens field. Preserve the requested budget as a
            # best-effort instruction while omitting the incompatible field.
            body["instructions"] += (
                f"\n\nLength budget: keep the final response within approximately "
                f"{max(1, max_output_tokens)} tokens."
            )
        chunks: list[str] = []
        usage_record: dict | None = None
        async with self.session.post(f"{BACKEND_URL}/responses", headers=headers, json=body) as response:
            if response.status >= 400:
                detail = (await response.text())[:1000]
                raise RuntimeError(f"ChatGPT returned HTTP {response.status}: {detail}")
            buffer = ""
            async for raw in response.content.iter_any():
                buffer += raw.decode(errors="replace").replace("\r\n", "\n")
                while "\n\n" in buffer:
                    event, buffer = buffer.split("\n\n", 1)
                    for line in event.splitlines():
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            continue
                        payload = json.loads(data)
                        if payload.get("type") == "response.output_text.delta":
                            chunks.append(payload.get("delta", ""))
                        if payload.get("type") == "response.completed":
                            usage_record = self._usage_from_response(payload.get("response"))
                        if payload.get("type") in ("error", "response.failed"):
                            raise RuntimeError(str(payload.get("error") or payload))
        if usage_record:
            self.store.record_chatgpt_usage(
                response_id=usage_record["response_id"],
                model=usage_record["model"] or model,
                purpose=purpose,
                input_tokens=usage_record["input_tokens"],
                cached_input_tokens=usage_record["cached_input_tokens"],
                output_tokens=usage_record["output_tokens"],
                reasoning_tokens=usage_record["reasoning_tokens"],
                total_tokens=usage_record["total_tokens"],
            )
        result = "".join(chunks).strip()
        if not result:
            raise RuntimeError("ChatGPT returned an empty response")
        return result

    async def _complete_custom(
        self,
        messages: list[dict[str, str]],
        instructions: str,
        model: str,
        purpose: str,
        max_output_tokens: int | None,
    ) -> str:
        provider = self.store.custom_provider()
        if provider is None:
            raise RuntimeError("The custom OpenAI-compatible provider is not configured")
        assert self.session
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if provider.api_key:
            headers["Authorization"] = f"Bearer {provider.api_key}"
        body = {
            "model": model,
            "messages": [{"role": "system", "content": instructions}, *messages],
            "stream": False,
        }
        if max_output_tokens is not None:
            body["max_completion_tokens"] = max(1, max_output_tokens)
        url = normalize_custom_base_url(provider.base_url) + "/chat/completions"
        async with self.session.post(url, headers=headers, json=body) as response:
            try:
                payload = await response.json(content_type=None)
            except (aiohttp.ContentTypeError, json.JSONDecodeError) as exc:
                detail = (await response.text())[:1000].strip()
                raise RuntimeError(
                    f"{provider.name} returned HTTP {response.status} with invalid JSON"
                    + (f": {detail}" if detail else "")
                ) from exc
            if response.status >= 400:
                error = payload.get("error") if isinstance(payload, dict) else None
                if isinstance(error, dict):
                    detail = error.get("message") or error.get("type")
                else:
                    detail = error or (payload.get("message") if isinstance(payload, dict) else None)
                raise RuntimeError(
                    f"{provider.name} returned HTTP {response.status}: {detail or 'unknown error'}"
                )
        if not isinstance(payload, dict):
            raise RuntimeError(f"{provider.name} returned an invalid JSON response")
        result = self._chat_completion_text(payload)
        if not result:
            raise RuntimeError(f"{provider.name} returned an empty response")
        usage_record = self._usage_from_chat_completion(payload)
        if usage_record:
            response_id = usage_record["response_id"]
            if response_id:
                provider_id = hashlib.sha256(provider.base_url.encode()).hexdigest()[:12]
                response_id = f"custom:{provider_id}:{response_id}"
            self.store.record_chatgpt_usage(
                response_id=response_id,
                model=usage_record["model"] or model,
                purpose=purpose,
                input_tokens=usage_record["input_tokens"],
                cached_input_tokens=usage_record["cached_input_tokens"],
                output_tokens=usage_record["output_tokens"],
                reasoning_tokens=usage_record["reasoning_tokens"],
                total_tokens=usage_record["total_tokens"],
            )
        return result

    @staticmethod
    def _chat_completion_text(payload: dict) -> str:
        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            return ""
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            return "".join(
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("type") in {"text", "output_text"}
            ).strip()
        return ""

    @classmethod
    def _usage_from_chat_completion(cls, payload: object) -> dict | None:
        if not isinstance(payload, dict) or not isinstance(payload.get("usage"), dict):
            return None
        usage = payload["usage"]
        prompt_details = usage.get("prompt_tokens_details")
        completion_details = usage.get("completion_tokens_details")
        prompt_details = prompt_details if isinstance(prompt_details, dict) else {}
        completion_details = completion_details if isinstance(completion_details, dict) else {}
        input_tokens = cls._token_count(usage.get("prompt_tokens"))
        output_tokens = cls._token_count(usage.get("completion_tokens"))
        total_tokens = cls._token_count(usage.get("total_tokens")) or input_tokens + output_tokens
        return {
            "response_id": payload.get("id"),
            "model": payload.get("model"),
            "input_tokens": input_tokens,
            "cached_input_tokens": cls._token_count(prompt_details.get("cached_tokens")),
            "output_tokens": output_tokens,
            "reasoning_tokens": cls._token_count(completion_details.get("reasoning_tokens")),
            "total_tokens": total_tokens,
        }

    @staticmethod
    def _token_count(value: object) -> int:
        try:
            return max(0, int(value or 0))
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _usage_from_response(response: object) -> dict | None:
        if not isinstance(response, dict) or not isinstance(response.get("usage"), dict):
            return None
        usage = response["usage"]
        input_details = usage.get("input_tokens_details")
        output_details = usage.get("output_tokens_details")
        input_details = input_details if isinstance(input_details, dict) else {}
        output_details = output_details if isinstance(output_details, dict) else {}

        return {
            "response_id": response.get("id"),
            "model": response.get("model"),
            "input_tokens": ChatGPTClient._token_count(usage.get("input_tokens")),
            "cached_input_tokens": ChatGPTClient._token_count(input_details.get("cached_tokens")),
            "output_tokens": ChatGPTClient._token_count(usage.get("output_tokens")),
            "reasoning_tokens": ChatGPTClient._token_count(output_details.get("reasoning_tokens")),
            "total_tokens": ChatGPTClient._token_count(usage.get("total_tokens")),
        }
