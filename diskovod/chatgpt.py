from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import secrets
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode, urlparse

import aiohttp

from .localization import prompts_for
from .models import (
    ChatCredentials,
    CustomProvider,
    FunctionCall,
    HostedToolCall,
    ModelResult,
    TextOutput,
    attachment_context,
    can_send_file,
    can_send_image,
)
from .store import Store
from .tooling import WEB_SEARCH_TOOL

log = logging.getLogger(__name__)
AUTHORIZE_URL = "https://auth.openai.com/oauth/authorize"
TOKEN_URL = "https://auth.openai.com/oauth/token"
BACKEND_URL = "https://chatgpt.com/backend-api/codex"
CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
CALLBACK_URL = "http://localhost:1455/auth/callback"
ORIGINATOR = "zed"
PROVIDERS = frozenset({"chatgpt", "custom"})
CUSTOM_PROTOCOLS = frozenset({"responses", "chat_completions"})


class ProviderHTTPError(RuntimeError):
    def __init__(self, provider: str, status: int, detail: str):
        self.provider = provider
        self.status = status
        self.detail = detail
        super().__init__(f"{provider} returned HTTP {status}: {detail or 'unknown error'}")


@dataclass(frozen=True, slots=True)
class ProtocolDetection:
    protocol: str
    native_function_calls: bool
    hosted_web_search: bool


def make_prompt_cache_key(scope: str, identity: str) -> str:
    digest = hashlib.sha256(f"{scope}\0{identity}".encode()).hexdigest()[:32]
    return f"diskovod:{scope}:{digest}"


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
    def automation_ready(self) -> bool:
        if self.active_provider != "custom":
            return self.subscription_connected
        provider = self.store.custom_provider()
        return bool(provider and provider.supports("native_function_calls"))

    @property
    def automation_error(self) -> str | None:
        if self.active_provider == "custom" and self.custom_connected and not self.automation_ready:
            return "The custom provider must pass native function-call detection before automation."
        return None

    @property
    def hosted_web_search_available(self) -> bool:
        if self.active_provider == "custom":
            provider = self.store.custom_provider()
            return bool(
                provider
                and provider.protocol == "responses"
                and provider.supports("native_function_calls")
                and provider.supports("hosted_web_search")
            )
        settings = self.store.app_settings()
        return self.store.subscription_web_search_capability(settings.model) is True

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
        messages: list[dict[str, Any]],
        instructions: str,
        model: str,
        effort: str,
        *,
        purpose: str = "conversation",
        max_output_tokens: int | None = None,
        cache_key: str | None = None,
        locale: str = "en",
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        continuation_items: list[dict[str, Any]] | None = None,
    ) -> str:
        result = await self.complete_result(
            messages,
            instructions,
            model,
            effort,
            purpose=purpose,
            max_output_tokens=max_output_tokens,
            cache_key=cache_key,
            locale=locale,
            tools=tools,
            tool_choice=tool_choice,
            continuation_items=continuation_items,
        )
        text = result.text
        if not text:
            raise RuntimeError(f"{self.provider_label} returned an empty response")
        return text

    async def complete_result(
        self,
        messages: list[dict[str, Any]],
        instructions: str,
        model: str,
        effort: str,
        *,
        purpose: str = "conversation",
        max_output_tokens: int | None = None,
        cache_key: str | None = None,
        locale: str = "en",
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        continuation_items: list[dict[str, Any]] | None = None,
    ) -> ModelResult:
        try:
            if self.active_provider == "custom":
                result = await self._complete_custom(
                    messages,
                    instructions,
                    model,
                    purpose,
                    max_output_tokens,
                    cache_key,
                    tools,
                    tool_choice,
                    continuation_items,
                )
            else:
                result = await self._complete_subscription(
                    messages,
                    instructions,
                    model,
                    effort,
                    purpose,
                    max_output_tokens,
                    cache_key,
                    locale,
                    tools,
                    tool_choice,
                    continuation_items,
                )
        except Exception as exc:
            self.last_error = str(exc)
            raise
        self.last_error = None
        return result

    async def _complete_subscription(
        self,
        messages: list[dict[str, Any]],
        instructions: str,
        model: str,
        effort: str,
        purpose: str,
        max_output_tokens: int | None,
        cache_key: str | None,
        locale: str,
        tools: list[dict[str, Any]] | None,
        tool_choice: str | dict[str, Any] | None,
        continuation_items: list[dict[str, Any]] | None,
    ) -> ModelResult:
        creds = await self.credentials()
        assert self.session
        input_items = [self._responses_message(message, model) for message in messages]
        input_items.extend(continuation_items or [])
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
            body["instructions"] += "\n\n" + prompts_for(locale).length_budget.format(
                tokens=max(1, max_output_tokens)
            )
        if cache_key:
            body["prompt_cache_key"] = cache_key[:64]
        if tools:
            body["tools"] = tools
            body["tool_choice"] = tool_choice or "required"
            body["parallel_tool_calls"] = False
        chunks: list[str] = []
        completed_response: dict[str, Any] | None = None
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
                            candidate = payload.get("response")
                            if isinstance(candidate, dict):
                                completed_response = candidate
                        if payload.get("type") in ("error", "response.failed"):
                            raise RuntimeError(str(payload.get("error") or payload))
        result = self._model_result_from_response(completed_response or {})
        if not result.text and chunks:
            result.text_outputs.append(TextOutput("".join(chunks).strip(), []))
        self._record_usage(result, model=model, purpose=purpose)
        return result

    async def _complete_custom(
        self,
        messages: list[dict[str, Any]],
        instructions: str,
        model: str,
        purpose: str,
        max_output_tokens: int | None,
        cache_key: str | None,
        tools: list[dict[str, Any]] | None,
        tool_choice: str | dict[str, Any] | None,
        continuation_items: list[dict[str, Any]] | None,
    ) -> ModelResult:
        provider = self.store.custom_provider()
        if provider is None:
            raise RuntimeError("The custom OpenAI-compatible provider is not configured")
        if provider.protocol not in CUSTOM_PROTOCOLS:
            raise RuntimeError(f"Unsupported custom provider protocol: {provider.protocol}")
        if provider.protocol == "responses":
            return await self._complete_custom_responses(
                provider,
                messages,
                instructions,
                model,
                purpose,
                max_output_tokens,
                cache_key,
                tools,
                tool_choice,
                continuation_items,
            )
        return await self._complete_custom_chat_completions(
            provider,
            messages,
            instructions,
            model,
            purpose,
            max_output_tokens,
            cache_key,
            tools,
            tool_choice,
            continuation_items,
        )

    async def _complete_custom_responses(
        self,
        provider: CustomProvider,
        messages: list[dict[str, Any]],
        instructions: str,
        model: str,
        purpose: str,
        max_output_tokens: int | None,
        cache_key: str | None,
        tools: list[dict[str, Any]] | None,
        tool_choice: str | dict[str, Any] | None,
        continuation_items: list[dict[str, Any]] | None,
    ) -> ModelResult:
        body: dict[str, Any] = {
            "model": model,
            "instructions": instructions,
            "input": [self._responses_message(message, model, provider="custom") for message in messages]
            + (continuation_items or []),
            "stream": False,
            "store": False,
        }
        if max_output_tokens is not None:
            body["max_output_tokens"] = max(1, max_output_tokens)
        if tools:
            body["tools"] = self._custom_tools(provider, tools)
            body["tool_choice"] = tool_choice or "required"
            if provider.supports("parallel_tool_control"):
                body["parallel_tool_calls"] = False
        if cache_key and provider.supports("prompt_cache_key"):
            body["prompt_cache_key"] = cache_key[:64]
        payload = await self._post_custom_json(provider, "/responses", body)
        result = self._model_result_from_response(payload)
        self._record_usage(result, model=model, purpose=purpose, provider_base_url=provider.base_url)
        return result

    async def _complete_custom_chat_completions(
        self,
        provider: CustomProvider,
        messages: list[dict[str, Any]],
        instructions: str,
        model: str,
        purpose: str,
        max_output_tokens: int | None,
        cache_key: str | None,
        tools: list[dict[str, Any]] | None,
        tool_choice: str | dict[str, Any] | None,
        continuation_items: list[dict[str, Any]] | None,
    ) -> ModelResult:
        assert self.session
        provider_messages = [self._custom_message(message, model) for message in messages]
        chat_messages = [{"role": "system", "content": instructions}, *provider_messages]
        chat_messages.extend(self._chat_continuation_messages(continuation_items or []))
        body = {
            "model": model,
            "messages": chat_messages,
            "stream": False,
        }
        if max_output_tokens is not None:
            body["max_completion_tokens"] = max(1, max_output_tokens)
        if cache_key and provider.supports("prompt_cache_key"):
            body["prompt_cache_key"] = cache_key[:64]
        if tools:
            body["tools"] = [self._chat_tool(tool) for tool in self._custom_tools(provider, tools)]
            body["tool_choice"] = self._chat_tool_choice(tool_choice or "required")
            if provider.supports("parallel_tool_control"):
                body["parallel_tool_calls"] = False
        payload = await self._post_custom_json(provider, "/chat/completions", body)
        result = self._model_result_from_chat_completion(payload)
        self._record_usage(result, model=model, purpose=purpose, provider_base_url=provider.base_url)
        return result

    async def _post_custom_json(
        self, provider: CustomProvider, path: str, body: dict[str, Any]
    ) -> dict[str, Any]:
        assert self.session
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if provider.api_key:
            headers["Authorization"] = f"Bearer {provider.api_key}"
        url = normalize_custom_base_url(provider.base_url) + path
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
                raise ProviderHTTPError(provider.name, response.status, self._error_detail(payload))
        if not isinstance(payload, dict):
            raise RuntimeError(f"{provider.name} returned an invalid JSON response")
        return payload

    async def detect_custom_protocol(
        self,
        *,
        name: str,
        base_url: str,
        api_key: str,
        model: str,
    ) -> ProtocolDetection:
        """Probe setup endpoints without changing or consulting the active provider."""
        provider = CustomProvider(name, normalize_custom_base_url(base_url), api_key, "responses")
        responses_body = {
            "model": model,
            "instructions": "This is a connection test. Reply with OK.",
            "input": [{"role": "user", "content": "Connection test"}],
            "max_output_tokens": 16,
            "stream": False,
            "store": False,
        }
        try:
            payload = await self._post_custom_json(provider, "/responses", responses_body)
        except ProviderHTTPError as exc:
            if not self._responses_conclusively_unsupported(exc.status, exc.detail):
                raise
        else:
            result = self._model_result_from_response(payload)
            if result.text or result.function_calls or result.hosted_tool_calls:
                native = await self._probe_native_function_calls(provider, "responses", model)
                web_search = await self._probe_custom_hosted_web_search(provider, model) if native else False
                return ProtocolDetection("responses", native, web_search)
            raise RuntimeError(f"{name} returned an invalid Responses API probe result")

        chat_body = {
            "model": model,
            "messages": [
                {"role": "system", "content": "This is a connection test. Reply with OK."},
                {"role": "user", "content": "Connection test"},
            ],
            "max_completion_tokens": 16,
            "stream": False,
        }
        payload = await self._post_custom_json(provider, "/chat/completions", chat_body)
        result = self._model_result_from_chat_completion(payload)
        if not result.text and not result.function_calls:
            raise RuntimeError(f"{name} returned an invalid Chat Completions probe result")
        native = await self._probe_native_function_calls(provider, "chat_completions", model)
        return ProtocolDetection("chat_completions", native, False)

    async def detect_subscription_web_search(self, model: str, effort: str) -> bool:
        """Probe the private subscription transport without assuming public API parity."""
        result = await self._complete_subscription(
            [
                {
                    "role": "user",
                    "content": "Find the official OpenAI homepage, then complete the test.",
                }
            ],
            (
                "This is a capability test. Use web search once, then call connection_test with "
                "ok=true. Do not return ordinary text."
            ),
            model,
            effort,
            "web_search_capability_probe",
            64,
            None,
            "en",
            [WEB_SEARCH_TOOL, self._connection_test_tool()],
            "required",
            None,
        )
        supported = self._is_successful_web_search_probe(result)
        self.store.set_subscription_web_search_capability(model, supported)
        return supported

    async def _probe_native_function_calls(self, provider: CustomProvider, protocol: str, model: str) -> bool:
        tool = self._connection_test_tool()
        if protocol == "responses":
            body = {
                "model": model,
                "input": [{"role": "user", "content": "Complete the connection test."}],
                "tools": [tool],
                "tool_choice": {"type": "function", "name": "connection_test"},
                "parallel_tool_calls": False,
                "max_output_tokens": 32,
                "stream": False,
                "store": False,
            }
            path = "/responses"
            parser = self._model_result_from_response
        else:
            body = {
                "model": model,
                "messages": [{"role": "user", "content": "Complete the connection test."}],
                "tools": [self._chat_tool(tool)],
                "tool_choice": self._chat_tool_choice({"type": "function", "name": "connection_test"}),
                "parallel_tool_calls": False,
                "max_completion_tokens": 32,
                "stream": False,
            }
            path = "/chat/completions"
            parser = self._model_result_from_chat_completion
        try:
            result = parser(await self._post_custom_json(provider, path, body))
        except Exception as exc:
            log.info("Native function-call capability probe failed for %s: %s", provider.name, exc)
            return False
        return (
            len(result.function_calls) == 1
            and result.function_calls[0].name == "connection_test"
            and result.function_calls[0].parsed_arguments is not None
        )

    async def _probe_custom_hosted_web_search(
        self,
        provider: CustomProvider,
        model: str,
    ) -> bool:
        body = {
            "model": model,
            "instructions": (
                "Use web search once to find the official OpenAI homepage, then call "
                "connection_test with ok=true. Do not return ordinary text."
            ),
            "input": [{"role": "user", "content": "Complete the web search capability test."}],
            "tools": [WEB_SEARCH_TOOL, self._connection_test_tool()],
            "tool_choice": "required",
            "parallel_tool_calls": False,
            "max_output_tokens": 64,
            "stream": False,
            "store": False,
        }
        try:
            result = self._model_result_from_response(
                await self._post_custom_json(provider, "/responses", body)
            )
        except Exception as exc:
            log.info("Hosted web-search capability probe failed for %s: %s", provider.name, exc)
            return False
        return self._is_successful_web_search_probe(result)

    @staticmethod
    def _connection_test_tool() -> dict[str, Any]:
        return {
            "type": "function",
            "name": "connection_test",
            "description": "Complete the connection test.",
            "parameters": {
                "type": "object",
                "properties": {"ok": {"type": "boolean"}},
                "required": ["ok"],
                "additionalProperties": False,
            },
            "strict": True,
        }

    @staticmethod
    def _is_successful_web_search_probe(result: ModelResult) -> bool:
        return (
            not result.text
            and len(result.function_calls) == 1
            and result.function_calls[0].name == "connection_test"
            and result.function_calls[0].parsed_arguments == {"ok": True}
            and 1 <= len(result.hosted_tool_calls) <= 2
            and all(
                call.kind == "web_search_call" and call.status == "completed"
                for call in result.hosted_tool_calls
            )
        )

    @staticmethod
    def _responses_conclusively_unsupported(status: int, detail: str) -> bool:
        if status in {404, 405, 501}:
            return True
        if status != 400:
            return False
        normalized = detail.casefold()
        return "responses" in normalized and any(
            marker in normalized
            for marker in ("unsupported", "not supported", "unknown endpoint", "unimplemented")
        )

    @staticmethod
    def _error_detail(payload: object) -> str:
        if not isinstance(payload, dict):
            return "unknown error"
        error = payload.get("error")
        if isinstance(error, dict):
            return str(error.get("message") or error.get("type") or "unknown error")
        return str(error or payload.get("message") or "unknown error")

    @staticmethod
    def _chat_tool(tool: dict[str, Any]) -> dict[str, Any]:
        function = {key: tool[key] for key in ("name", "description", "parameters", "strict") if key in tool}
        return {"type": "function", "function": function}

    @staticmethod
    def _custom_tools(provider: CustomProvider, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if provider.supports("strict_function_schemas"):
            return tools
        return [{key: value for key, value in tool.items() if key != "strict"} for tool in tools]

    @staticmethod
    def _chat_tool_choice(choice: str | dict[str, Any]) -> str | dict[str, Any]:
        if isinstance(choice, str):
            return choice
        if choice.get("type") == "function" and isinstance(choice.get("name"), str):
            return {"type": "function", "function": {"name": choice["name"]}}
        return choice

    @staticmethod
    def _chat_continuation_messages(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        for item in items:
            if item.get("type") == "function_call":
                messages.append(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": str(item.get("call_id") or item.get("id") or ""),
                                "type": "function",
                                "function": {
                                    "name": str(item.get("name") or ""),
                                    "arguments": str(item.get("arguments") or ""),
                                },
                            }
                        ],
                    }
                )
            elif item.get("type") == "function_call_output":
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": str(item.get("call_id") or ""),
                        "content": str(item.get("output") or ""),
                    }
                )
        return messages

    @staticmethod
    def _responses_message(
        message: dict[str, Any], model: str, *, provider: str = "chatgpt"
    ) -> dict[str, Any]:
        role = message["role"]
        attachments = message.get("attachments") or []
        text = attachment_context(
            str(message.get("content") or ""),
            attachments,
            provider=provider,
            model=model,
            locale=str(message.get("locale") or "en"),
        )
        if role == "assistant":
            content = [{"type": "output_text", "text": text, "annotations": []}]
        else:
            content = [{"type": "input_text", "text": text}]
            for attachment in attachments:
                if can_send_image(attachment, model):
                    content.append(
                        {
                            "type": "input_image",
                            "image_url": attachment["url"],
                            "detail": "auto",
                        }
                    )
                elif can_send_file(attachment, provider, model):
                    content.append(
                        {
                            "type": "input_file",
                            "file_url": attachment["url"],
                        }
                    )
        return {"type": "message", "role": role, "content": content}

    @staticmethod
    def _custom_message(message: dict[str, Any], model: str) -> dict[str, Any]:
        role = message["role"]
        attachments = message.get("attachments") or []
        text = attachment_context(
            str(message.get("content") or ""),
            attachments,
            provider="custom",
            model=model,
            locale=str(message.get("locale") or "en"),
        )
        images = (
            [attachment for attachment in attachments if can_send_image(attachment, model)]
            if role == "user"
            else []
        )
        if not images:
            return {"role": role, "content": text}
        content: list[dict[str, Any]] = [{"type": "text", "text": text}]
        content.extend(
            {
                "type": "image_url",
                "image_url": {"url": attachment["url"], "detail": "auto"},
            }
            for attachment in images
        )
        return {"role": role, "content": content}

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
    def _model_result_from_response(cls, payload: object) -> ModelResult:
        if not isinstance(payload, dict):
            return ModelResult([], [], [])
        text_outputs: list[TextOutput] = []
        function_calls: list[FunctionCall] = []
        hosted_tool_calls: list[HostedToolCall] = []
        output = payload.get("output")
        for item in output if isinstance(output, list) else []:
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type") or "")
            if item_type == "message":
                content = item.get("content")
                for part in content if isinstance(content, list) else []:
                    if not isinstance(part, dict) or part.get("type") != "output_text":
                        continue
                    annotations = part.get("annotations")
                    text_outputs.append(
                        TextOutput(
                            str(part.get("text") or "").strip(),
                            [value for value in annotations if isinstance(value, dict)]
                            if isinstance(annotations, list)
                            else [],
                        )
                    )
            elif item_type == "function_call":
                arguments = str(item.get("arguments") or "")
                function_calls.append(
                    FunctionCall(
                        call_id=str(item.get("call_id") or item.get("id") or ""),
                        name=str(item.get("name") or ""),
                        arguments=arguments,
                        parsed_arguments=cls._parse_function_arguments(arguments),
                    )
                )
            elif item_type.endswith("_call"):
                metadata: dict[str, Any] = {}
                action = item.get("action")
                if isinstance(action, dict):
                    metadata["action"] = {
                        str(key): value[:500] if isinstance(value, str) else value
                        for key, value in list(action.items())[:10]
                        if isinstance(value, (str, int, float, bool)) or value is None
                    }
                hosted_tool_calls.append(HostedToolCall(item_type, str(item.get("status") or ""), metadata))
        return ModelResult(
            text_outputs,
            function_calls,
            hosted_tool_calls,
            cls._usage_from_response(payload),
            str(payload.get("id")) if payload.get("id") else None,
        )

    @classmethod
    def _model_result_from_chat_completion(cls, payload: object) -> ModelResult:
        if not isinstance(payload, dict):
            return ModelResult([], [], [])
        try:
            message = payload["choices"][0]["message"]
        except (KeyError, IndexError, TypeError):
            message = {}
        text = cls._chat_completion_text(payload)
        text_outputs = [TextOutput(text, [])] if text else []
        function_calls: list[FunctionCall] = []
        calls = message.get("tool_calls") if isinstance(message, dict) else None
        for item in calls if isinstance(calls, list) else []:
            if not isinstance(item, dict) or item.get("type") not in {None, "function"}:
                continue
            function = item.get("function")
            if not isinstance(function, dict):
                continue
            arguments = str(function.get("arguments") or "")
            function_calls.append(
                FunctionCall(
                    call_id=str(item.get("id") or ""),
                    name=str(function.get("name") or ""),
                    arguments=arguments,
                    parsed_arguments=cls._parse_function_arguments(arguments),
                )
            )
        return ModelResult(
            text_outputs,
            function_calls,
            [],
            cls._usage_from_chat_completion(payload),
            str(payload.get("id")) if payload.get("id") else None,
        )

    @staticmethod
    def _parse_function_arguments(arguments: str) -> dict[str, Any] | None:
        try:
            value = json.loads(arguments)
        except (TypeError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None

    def _record_usage(
        self,
        result: ModelResult,
        *,
        model: str,
        purpose: str,
        provider_base_url: str | None = None,
    ) -> None:
        usage = result.usage
        if not usage:
            return
        response_id = usage.get("response_id") or result.provider_response_id
        if response_id and provider_base_url:
            provider_id = hashlib.sha256(provider_base_url.encode()).hexdigest()[:12]
            response_id = f"custom:{provider_id}:{response_id}"
        self.store.record_chatgpt_usage(
            response_id=response_id,
            model=usage.get("model") or model,
            purpose=purpose,
            input_tokens=usage["input_tokens"],
            cached_input_tokens=usage["cached_input_tokens"],
            output_tokens=usage["output_tokens"],
            reasoning_tokens=usage["reasoning_tokens"],
            total_tokens=usage["total_tokens"],
        )

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
