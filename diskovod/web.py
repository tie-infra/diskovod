from __future__ import annotations

import hashlib
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from urllib.parse import urlencode, urlparse

from fastapi import Depends, FastAPI, Form, HTTPException, Request, status
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.templating import Jinja2Templates

from .automation import Automation
from .chatgpt import PROVIDERS, ChatGPTClient, make_prompt_cache_key, normalize_custom_base_url
from .discord import DiscordService
from .localization import SUPPORTED_LOCALES, normalize_locale, prompts_for
from .models import AppSettings, CustomProvider
from .security import password_matches
from .store import Store
from .ui_localization import ui_text

PERSONALITY_PROMPT_VERSION = "style-base-rates-examples-and-sequences-v4"
PERSONALITY_MAX_OUTPUT_TOKENS = 2000
PERSONALITY_INSTRUCTIONS = prompts_for("en").personality


def personality_source_hash(samples: str, locale: str = "en") -> str:
    return hashlib.sha256(f"{PERSONALITY_PROMPT_VERSION}\0{locale}\0{samples}".encode()).hexdigest()


def localized_base_instructions(previous_locale: str, new_locale: str, submitted: str) -> str:
    """Translate only the stock prompt; never overwrite user-customized instructions."""
    submitted = submitted.strip()
    if submitted == prompts_for(previous_locale).base:
        return prompts_for(new_locale).base
    return submitted


class WebApp:
    def __init__(
        self,
        store: Store,
        chatgpt: ChatGPTClient,
        discord: DiscordService,
        automation: Automation,
        admin_password: str,
        public_url: str,
    ):
        self.store, self.chatgpt, self.discord, self.automation = store, chatgpt, discord, automation
        self.admin_password = admin_password
        self.public_url = public_url.rstrip("/")
        self.public_origin = self._normalized_origin(self.public_url)
        self.security = HTTPBasic()
        base = Path(__file__).parent
        self.templates = Jinja2Templates(directory=base / "templates")
        self.app = FastAPI(title="Diskovod", docs_url=None, redoc_url=None, openapi_url=None)
        self._security_headers()
        self._routes()

    def _security_headers(self) -> None:
        @self.app.middleware("http")
        async def headers(request: Request, call_next):
            response = await call_next(request)
            response.headers["Content-Security-Policy"] = (
                "default-src 'none'; style-src 'self' https://cdn.jsdelivr.net; "
                "form-action 'self'; base-uri 'none'; frame-ancestors 'none'"
            )
            response.headers["X-Content-Type-Options"] = "nosniff"
            # `no-referrer` serializes the Origin of ordinary form POSTs as `null`.
            response.headers["Referrer-Policy"] = "same-origin"
            response.headers["Cache-Control"] = "no-store"
            return response

    def require_admin(
        self, request: Request, credentials: HTTPBasicCredentials = Depends(HTTPBasic())
    ) -> str:
        if credentials.username != "admin" or not password_matches(credentials.password, self.admin_password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, headers={"WWW-Authenticate": "Basic"}
            )
        if origin := request.headers.get("origin"):
            if self._normalized_origin(origin) != self.public_origin:
                raise HTTPException(
                    status_code=403,
                    detail=f"Cross-origin form submission rejected; expected {self.public_url}",
                )
        return credentials.username

    @staticmethod
    def _normalized_origin(url: str) -> tuple[str, str, int] | None:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            return None
        try:
            port = parsed.port
        except ValueError:
            return None
        return (
            parsed.scheme.lower(),
            parsed.hostname.lower(),
            port or (443 if parsed.scheme.lower() == "https" else 80),
        )

    def _routes(self) -> None:
        auth = self.require_admin

        @self.app.get("/")
        async def dashboard(
            request: Request,
            db_table: str = "messages",
            db_page: int = 1,
            db_query: str = "",
            _: str = Depends(auth),
        ):
            custom_provider = self.store.custom_provider()
            app_settings = self.store.app_settings()
            return self.templates.TemplateResponse(
                request,
                "index.html",
                {
                    "app_settings": app_settings,
                    "locale": app_settings.admin_locale,
                    "locales": SUPPORTED_LOCALES,
                    "t": lambda key, **values: ui_text(app_settings.admin_locale, key, **values),
                    "public_url": self.public_url,
                    "chat_connected": self.chatgpt.subscription_connected,
                    "chat_email": self.chatgpt.email,
                    "chat_error": self.chatgpt.last_error,
                    "model_connected": self.chatgpt.connected,
                    "active_provider": self.chatgpt.active_provider,
                    "provider_label": self.chatgpt.provider_label,
                    "custom_provider": (
                        {
                            "name": custom_provider.name,
                            "base_url": custom_provider.base_url,
                            "has_api_key": bool(custom_provider.api_key),
                        }
                        if custom_provider
                        else None
                    ),
                    "discord_connected": self.discord.connected,
                    "discord_identity": self.discord.identity,
                    "discord_error": self.discord.error,
                    "has_discord_token": self.store.discord_token() is not None,
                    "captcha_requests": self.discord.captcha_requests(),
                    "personality": self.store.personality(),
                    "conversations": self._conversation_views(),
                    "usage_stats": self._usage_views(),
                    "database": self._database_view(db_table, db_page, db_query),
                    "message": request.query_params.get("message"),
                    "error": request.query_params.get("error"),
                },
            )

        @self.app.get("/static/style.css")
        async def css():
            return FileResponse(Path(__file__).parent / "static" / "style.css", media_type="text/css")

        @self.app.post("/chatgpt/connect")
        async def chat_connect(_: str = Depends(auth)):
            try:
                return RedirectResponse(await self.chatgpt.begin_oauth(), status_code=303)
            except Exception as exc:
                return self._back(error=str(exc))

        @self.app.get("/chatgpt/oauth/callback")
        async def chat_callback(
            code: str | None = None,
            state: str | None = None,
            error: str | None = None,
        ):
            try:
                await self.chatgpt.finish_oauth(code=code, state=state, error=error)
            except Exception as exc:
                return self._back(error=str(exc))
            self._set_provider("chatgpt")
            return self._back(message="ChatGPT connected and selected")

        @self.app.post("/chatgpt/disconnect")
        async def chat_disconnect(_: str = Depends(auth)):
            self.store.clear_chat_credentials()
            self.chatgpt.last_error = None
            return self._back(message="ChatGPT disconnected")

        @self.app.post("/provider/custom")
        async def custom_provider_save(
            name: str = Form(...),
            base_url: str = Form(...),
            api_key: str = Form(""),
            clear_api_key: str | None = Form(None),
            _: str = Depends(auth),
        ):
            name = name.strip()
            if not name or len(name) > 80:
                return self._back(error="Provider name must contain between 1 and 80 characters")
            try:
                base_url = normalize_custom_base_url(base_url)
            except ValueError as exc:
                return self._back(error=str(exc))
            existing = self.store.custom_provider()
            api_key = api_key.strip()
            if not api_key and clear_api_key is None and existing:
                api_key = existing.api_key
            self.store.set_custom_provider(CustomProvider(name, base_url, api_key))
            self._set_provider("custom")
            self.chatgpt.last_error = None
            return self._back(message=f"{name} saved and selected")

        @self.app.post("/provider/custom/remove")
        async def custom_provider_remove(_: str = Depends(auth)):
            self.store.clear_custom_provider()
            if self.chatgpt.active_provider == "custom":
                self._set_provider("chatgpt")
            self.chatgpt.last_error = None
            return self._back(message="Custom provider removed")

        @self.app.post("/provider/select")
        async def provider_select(provider: str = Form(...), _: str = Depends(auth)):
            if provider not in PROVIDERS:
                return self._back(error="Unknown model provider")
            if provider == "chatgpt" and not self.chatgpt.subscription_connected:
                return self._back(error="Connect ChatGPT before selecting it")
            if provider == "custom" and not self.chatgpt.custom_connected:
                return self._back(error="Configure a custom provider before selecting it")
            self._set_provider(provider)
            self.chatgpt.last_error = None
            return self._back(message=f"{self.chatgpt.provider_label} selected")

        @self.app.post("/discord/connect")
        async def discord_connect(token: str = Form(...), _: str = Depends(auth)):
            token = token.strip()
            if len(token) < 20:
                return self._back(error="Discord token appears to be incomplete")
            self.store.set_discord_token(token)
            self.discord.error = None
            await self.discord.restart()
            return self._back(message="Discord connection started")

        @self.app.post("/discord/disconnect")
        async def discord_disconnect(_: str = Depends(auth)):
            await self.discord.stop()
            self.store.clear_discord_token()
            return self._back(message="Discord disconnected")

        @self.app.post("/discord/captcha/{request_id}")
        async def discord_captcha(
            request_id: str,
            solution: str = Form(...),
            _: str = Depends(auth),
        ):
            solution = solution.strip()
            if not solution:
                return self._back(error="Enter the CAPTCHA solution token")
            if not self.discord.solve_captcha(request_id, solution):
                return self._back(error="The CAPTCHA request expired or was already answered")
            return self._back(message="CAPTCHA solution submitted")

        @self.app.post("/settings")
        async def settings(
            enabled: str | None = Form(None),
            silent_replies: str | None = Form(None),
            robot_prefix: str | None = Form(None),
            admin_locale: str = Form("en"),
            prompt_locale: str = Form("en"),
            multi_message_replies: str | None = Form(None),
            multi_message_chance: float = Form(12.0),
            max_reply_messages: int = Form(3),
            min_message_gap_seconds: float = Form(0.7),
            max_message_gap_seconds: float = Form(2.0),
            conversation_default: str = Form("opt_in"),
            provider: str = Form("chatgpt"),
            model: str = Form(...),
            reasoning_effort: str = Form("low"),
            max_reply_tokens: int = Form(256),
            debounce_seconds: float = Form(...),
            min_delay_seconds: float = Form(...),
            max_delay_seconds: float = Form(...),
            min_typing_cps: float = Form(...),
            max_typing_cps: float = Form(...),
            min_human_quiet_minutes: float = Form(...),
            max_human_quiet_minutes: float = Form(...),
            history_limit: int = Form(...),
            owner_details: str = Form(""),
            base_instructions: str = Form(...),
            _: str = Depends(auth),
        ):
            if admin_locale not in SUPPORTED_LOCALES or prompt_locale not in SUPPORTED_LOCALES:
                return self._back(error=ui_text(admin_locale, "unknown_locale"))
            if provider not in PROVIDERS:
                return self._back(error="Unknown model provider")
            if conversation_default not in {"opt_in", "opt_out"}:
                return self._back(error="Unknown new-conversation default")
            if provider == "custom" and not self.chatgpt.custom_connected:
                return self._back(error="Configure a custom provider before selecting it")
            model = model.strip()
            if not model:
                return self._back(error="Model name cannot be empty")
            owner_details = owner_details.strip()
            if len(owner_details) > 20_000:
                return self._back(error="Owner details cannot exceed 20,000 characters")
            if (
                min_delay_seconds > max_delay_seconds
                or min_message_gap_seconds > max_message_gap_seconds
                or min_typing_cps > max_typing_cps
                or min_human_quiet_minutes > max_human_quiet_minutes
            ):
                return self._back(error="Minimum values cannot exceed maximum values")
            previous = self.store.app_settings()
            normalized_prompt_locale = normalize_locale(prompt_locale)
            base_instructions = localized_base_instructions(
                previous.prompt_locale,
                normalized_prompt_locale,
                base_instructions,
            )
            value = AppSettings(
                enabled=enabled is not None,
                silent_replies=silent_replies is not None,
                robot_prefix=robot_prefix is not None,
                admin_locale=normalize_locale(admin_locale),
                prompt_locale=normalized_prompt_locale,
                multi_message_replies=multi_message_replies is not None,
                multi_message_chance=max(0.0, min(multi_message_chance, 100.0)),
                max_reply_messages=max(2, min(max_reply_messages, 5)),
                min_message_gap_seconds=max(0.0, min(min_message_gap_seconds, 30.0)),
                max_message_gap_seconds=max(0.0, min(max_message_gap_seconds, 30.0)),
                default_conversation_enabled=conversation_default == "opt_in",
                provider=provider,
                model=model,
                reasoning_effort=reasoning_effort,
                max_reply_tokens=max(32, min(max_reply_tokens, 2048)),
                debounce_seconds=debounce_seconds,
                min_delay_seconds=min_delay_seconds,
                max_delay_seconds=max_delay_seconds,
                min_typing_cps=min_typing_cps,
                max_typing_cps=max_typing_cps,
                min_human_quiet_minutes=max(0.0, min_human_quiet_minutes),
                max_human_quiet_minutes=max(0.0, max_human_quiet_minutes),
                history_limit=max(4, min(history_limit, 100)),
                owner_details=owner_details,
                base_instructions=base_instructions,
            )
            self.store.set_app_settings(value)
            return self._back(message=ui_text(value.admin_locale, "settings_saved"))

        @self.app.post("/personality/infer")
        async def personality_infer(samples: str = Form(...), _: str = Depends(auth)):
            samples = samples.strip()
            if len(samples) < 200:
                return self._back(error="Provide at least 200 characters of message history")
            return await self._infer_personality(samples, source="pasted history")

        @self.app.post("/personality/infer-history")
        async def personality_infer_history(
            history_limit: int = Form(100),
            _: str = Depends(auth),
        ):
            try:
                messages = await self.discord.personality_history(max(20, min(history_limit, 500)))
            except Exception as exc:
                return self._back(error=str(exc))
            samples = "\n\n---\n\n".join(messages)
            if len(samples) < 200:
                return self._back(
                    error="The selected Discord history did not contain enough human-authored text"
                )
            return await self._infer_personality(samples, source="Discord history")

        @self.app.post("/personality/save")
        async def personality_save(profile: str = Form(...), _: str = Depends(auth)):
            profile = profile.strip()
            if len(profile) < 50:
                return self._back(error="The personality description must be at least 50 characters")
            source_hash = hashlib.sha256(("edited\n" + profile).encode()).hexdigest()
            self.store.set_personality(profile, source_hash, source="edited")
            return self._back(message="Personality updated")

        @self.app.post("/conversations/{channel_id}/pause")
        async def pause(channel_id: str, _: str = Depends(auth)):
            self.automation.permanently_pause(channel_id)
            return self._back(message="Automation disabled for this conversation")

        @self.app.post("/conversations/{channel_id}/resume")
        async def resume(channel_id: str, _: str = Depends(auth)):
            self.automation.cancel(channel_id)
            self.store.set_permanent_pause(channel_id, False)
            self.store.clear_snooze(channel_id)
            return self._back(message="Automation enabled; the next incoming DM may receive a reply")

        @self.app.post("/conversations/{channel_id}/force-reply")
        async def force_reply(channel_id: str, _: str = Depends(auth)):
            if not self.chatgpt.connected:
                return self._back(error="Connect the active model provider before forcing a reply")
            try:
                await self.discord.force_reply(channel_id)
            except Exception as exc:
                return self._back(error=str(exc))
            return self._back(message="Forced reply scheduled")

        @self.app.post("/database/delete")
        async def database_delete(
            table: str = Form(...),
            row_key: str = Form(...),
            confirm: str | None = Form(None),
            db_query: str = Form(""),
            _: str = Depends(auth),
        ):
            if confirm != "delete":
                return self._database_back(
                    table,
                    db_query,
                    error="Confirm the row deletion before submitting",
                )
            try:
                deleted = self.store.delete_database_row(table, row_key)
            except ValueError as exc:
                return self._database_back(table, db_query, error=str(exc))
            if not deleted:
                return self._database_back(table, db_query, error="Database row was not found")
            return self._database_back(table, db_query, message=f"Deleted row {row_key!r} from {table}")

    def _conversation_views(self) -> list[dict]:
        now = time.time()
        result = self.store.conversations()
        for conversation in result:
            until = conversation["snoozed_until"]
            conversation["snoozed"] = bool(until and until > now)
            conversation["quiet_minutes_remaining"] = (
                max(1, int((until - now + 59) // 60)) if until and until > now else 0
            )
        return result

    def _usage_views(self) -> dict:
        stats = self.store.chatgpt_usage_stats()
        for record in stats["recent"]:
            record["recorded_at_label"] = (
                datetime.fromtimestamp(record["recorded_at"]).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
            )
        return stats

    def _database_view(self, table: str, page: int, query: str) -> dict:
        tables = self.store.database_tables()
        table_names = {item["name"] for item in tables}
        selected = table if table in table_names else "messages"
        search = query.strip()[:200]
        current_page = max(1, page)
        data = self.store.database_rows(
            selected,
            limit=50,
            offset=(current_page - 1) * 50,
            query=search,
        )
        if data["offset"] >= data["total"] and data["total"]:
            current_page = max(1, (data["total"] - 1) // data["limit"] + 1)
            data = self.store.database_rows(
                selected,
                limit=50,
                offset=(current_page - 1) * 50,
                query=search,
            )
        for item in tables:
            item["url"] = self._database_url(item["name"], 1, "")
            item["selected"] = item["name"] == selected
        rows = []
        for row in data["rows"]:
            cells = []
            for column in data["columns"]:
                raw = row.get(column)
                value = "NULL" if raw is None else str(raw)
                cells.append(value if len(value) <= 500 else value[:497] + "…")
            rows.append({"key": str(row[data["primary_key"]]), "cells": cells})
        data.update(
            tables=tables,
            rows=rows,
            page=current_page,
            pages=max(1, (data["total"] + data["limit"] - 1) // data["limit"]),
            previous_url=(
                self._database_url(selected, current_page - 1, search) if current_page > 1 else None
            ),
            next_url=(
                self._database_url(selected, current_page + 1, search)
                if data["offset"] + data["limit"] < data["total"]
                else None
            ),
        )
        return data

    def _set_provider(self, provider: str) -> None:
        if provider not in PROVIDERS:
            raise ValueError("Unknown model provider")
        self.store.set_app_settings(replace(self.store.app_settings(), provider=provider))

    async def _infer_personality(self, samples: str, *, source: str) -> RedirectResponse:
        cfg = self.store.app_settings()
        source_hash = personality_source_hash(samples, cfg.prompt_locale)
        cached = self.store.personality()
        if cached and cached["source_hash"] == source_hash:
            return self._back(message="This message history is already cached; no model call was made")
        try:
            profile = await self.chatgpt.complete(
                [{"role": "user", "content": samples}],
                prompts_for(cfg.prompt_locale).personality,
                cfg.model,
                cfg.reasoning_effort,
                purpose="personality_inference",
                max_output_tokens=PERSONALITY_MAX_OUTPUT_TOKENS,
                cache_key=make_prompt_cache_key("personality", cfg.model),
                locale=cfg.prompt_locale,
            )
        except Exception as exc:
            return self._back(error=str(exc))
        self.store.set_personality(profile, source_hash, source=source)
        return self._back(message="Personality inferred and cached")

    def _url(self, path: str) -> str:
        return self.public_url + "/" + path.lstrip("/")

    def _database_url(self, table: str, page: int, query: str) -> str:
        parameters = {"db_table": table, "db_page": max(1, page)}
        if query:
            parameters["db_query"] = query
        return self._url("/") + "?" + urlencode(parameters) + "#database"

    def _database_back(
        self,
        table: str,
        query: str,
        *,
        message: str | None = None,
        error: str | None = None,
    ) -> RedirectResponse:
        parameters = {"db_table": table, "db_page": 1}
        if query:
            parameters["db_query"] = query
        if message:
            parameters["message"] = message
        if error:
            parameters["error"] = error
        return RedirectResponse(
            self._url("/") + "?" + urlencode(parameters) + "#database",
            status_code=303,
        )

    def _back(self, *, message: str | None = None, error: str | None = None) -> RedirectResponse:
        query = urlencode({k: v for k, v in {"message": message, "error": error}.items() if v})
        return RedirectResponse(self._url("/") + ("?" + query if query else ""), status_code=303)
