from __future__ import annotations

import hashlib
import json
import secrets
import time
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlparse
from zoneinfo import ZoneInfo

from fastapi import Depends, FastAPI, Form, HTTPException, Request, status
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.templating import Jinja2Templates
from langchain_core.messages import HumanMessage, SystemMessage

from .discord import DiscordService
from .localization import (
    SUPPORTED_LOCALES,
    assistant_name_for,
    normalize_locale,
    prompts_for,
    ui_text,
)
from .models import ADMIN_THEMES, REASONING_EFFORTS, AppSettings, CustomProvider
from .oauth import ChatGPTAccount
from .providers import (
    ModelConfiguration,
    ModelService,
    ProviderCapabilities,
    ProviderCredentials,
    ProviderSetup,
    normalize_custom_base_url,
)
from .runtime import AgentService
from .security import password_matches
from .store import Store

PERSONALITY_PROMPT_VERSION = "style-base-rates-examples-and-sequences-v4"
PERSONALITY_INSTRUCTIONS = prompts_for("en").personality
ADMIN_TABS = frozenset({"connections", "assistant", "conversations", "usage", "database"})
PROVIDERS = frozenset({"chatgpt", "custom"})
CUSTOM_PROTOCOLS = frozenset({"responses", "chat_completions"})


@dataclass(slots=True)
class ProviderDraft:
    name: str
    base_url: str
    api_key: str
    protocol: str
    capabilities: dict[str, bool]
    probe_model: str
    expires_at: float


def personality_source_hash(samples: str, locale: str = "en") -> str:
    return hashlib.sha256(f"{PERSONALITY_PROMPT_VERSION}\0{locale}\0{samples}".encode()).hexdigest()


def localized_base_instructions(previous_locale: str, new_locale: str, submitted: str) -> str:
    """Translate only the stock prompt; never overwrite user-customized instructions."""
    submitted = submitted.strip()
    if submitted == prompts_for(previous_locale).base:
        return prompts_for(new_locale).base
    return submitted


def assistant_settings_defaults(current: AppSettings) -> AppSettings:
    """Reset assistant behavior without changing admin appearance preferences."""
    return replace(
        AppSettings(),
        admin_locale=current.admin_locale,
        admin_theme=current.admin_theme,
    )


class WebApp:
    def __init__(
        self,
        store: Store,
        account: ChatGPTAccount,
        models: ModelService,
        provider_setup: ProviderSetup,
        discord: DiscordService,
        runtime: AgentService,
        admin_password: str,
        public_url: str,
    ):
        self.store = store
        self.account = account
        self.models = models
        self.provider_setup = provider_setup
        self.discord = discord
        self.runtime = runtime
        self.admin_password = admin_password
        self.public_url = public_url.rstrip("/")
        self.public_origin = self._normalized_origin(self.public_url)
        self.security = HTTPBasic()
        self.provider_drafts: dict[str, ProviderDraft] = {}
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
                    detail=self._t("cross_origin_rejected", url=self.public_url),
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
            tab: str = "connections",
            db_table: str = "messages",
            db_page: int = 1,
            db_query: str = "",
            provider_draft: str = "",
            _: str = Depends(auth),
        ):
            custom_provider = self.store.custom_provider()
            app_settings = self.store.app_settings()
            model_view = self._model_view()
            active_tab = tab if tab in ADMIN_TABS else "connections"
            draft = self._provider_draft(provider_draft)
            provider_view = (
                {
                    "name": draft.name,
                    "base_url": draft.base_url,
                    "protocol": draft.protocol,
                    "capabilities": draft.capabilities,
                    "probe_model": draft.probe_model,
                    "has_api_key": bool(draft.api_key),
                    "draft_token": provider_draft,
                }
                if draft
                else (
                    {
                        "name": custom_provider.name,
                        "base_url": custom_provider.base_url,
                        "protocol": custom_provider.protocol,
                        "capabilities": {
                            **custom_provider.capabilities,
                            "output_token_limit": custom_provider.supports("output_token_limit"),
                        },
                        "probe_model": model_view["model"],
                        "has_api_key": bool(custom_provider.api_key),
                        "draft_token": "",
                    }
                    if custom_provider
                    else None
                )
            )
            return self.templates.TemplateResponse(
                request,
                "index.html",
                {
                    "app_settings": app_settings,
                    "model_view": model_view,
                    "active_tab": active_tab,
                    "assistant_display_name": assistant_name_for(
                        app_settings.prompt_locale,
                        app_settings.assistant_name,
                    ),
                    "default_assistant_name": assistant_name_for(app_settings.prompt_locale),
                    "locale": app_settings.admin_locale,
                    "locales": SUPPORTED_LOCALES,
                    "t": lambda key, **values: ui_text(app_settings.admin_locale, key, **values),
                    "public_url": self.public_url,
                    "chat_connected": self.account.connected,
                    "chat_email": self.account.email,
                    "chat_error": self.account.last_error,
                    "subscription_web_search": self._hosted_search_capability(),
                    "subscription_web_search_probe": await self._subscription_web_search_probe_view(
                        model_view["model"]
                    ),
                    "model_connected": self.models.ready,
                    "automation_ready": self.runtime.ready,
                    "automation_error": None,
                    "active_provider": self._active_provider(),
                    "provider_label": self.models.provider_label,
                    "custom_provider": provider_view,
                    "discord_connected": self.discord.connected,
                    "discord_identity": self.discord.identity,
                    "discord_error": self.discord.error,
                    "has_discord_token": self.store.discord_token() is not None,
                    "captcha_requests": self.discord.captcha_requests(),
                    "personality": self._personality_view(),
                    "escalations": await self._escalation_views(),
                    "conversations": await self._conversation_views(),
                    "agent_runs": await self._agent_run_views() if active_tab == "usage" else [],
                    "capability_probes": (
                        await self._capability_probe_views() if active_tab == "usage" else []
                    ),
                    "checkpoint_threads": (
                        await self.runtime.checkpoint_views() if active_tab == "usage" else []
                    ),
                    "runtime_records": (await self._runtime_record_views() if active_tab == "usage" else {}),
                    "database": await self._database_view(db_table, db_page, db_query),
                    "message": request.query_params.get("message"),
                    "error": request.query_params.get("error"),
                },
            )

        @self.app.get("/static/style.css")
        async def css():
            return FileResponse(Path(__file__).parent / "static" / "style.css", media_type="text/css")

        @self.app.post("/settings/theme")
        async def theme_save(
            admin_theme: str = Form(...),
            tab: str = Form("connections"),
            _: str = Depends(auth),
        ):
            if admin_theme not in ADMIN_THEMES:
                return self._back(tab=tab, error=self._t("unknown_theme"))
            current = self.store.app_settings()
            await self.store.aset_app_settings(replace(current, admin_theme=admin_theme))
            return self._back(tab=tab, message=self._t("theme_saved"))

        @self.app.post("/chatgpt/connect")
        async def chat_connect(_: str = Depends(auth)):
            try:
                return RedirectResponse(await self.account.begin_oauth(), status_code=303)
            except Exception as exc:
                return self._back(error=str(exc))

        @self.app.get("/chatgpt/oauth/callback")
        async def chat_callback(
            code: str | None = None,
            state: str | None = None,
            error: str | None = None,
        ):
            try:
                await self.account.finish_oauth(code=code, state=state, error=error)
            except Exception as exc:
                return self._back(error=str(exc))
            selected = self._model_view()
            try:
                await self._save_provider_selection(
                    "chatgpt",
                    model=selected["model"],
                    reasoning_effort=selected["reasoning_effort"],
                    max_output_tokens=selected["max_output_tokens"],
                )
            except RuntimeError as exc:
                return self._back(error=str(exc))
            return self._back(message=self._t("chatgpt_connected"))

        @self.app.post("/chatgpt/disconnect")
        async def chat_disconnect(_: str = Depends(auth)):
            await self.store.aclear_chat_credentials()
            self.account.last_error = None
            return self._back(message=self._t("chatgpt_disconnected"))

        @self.app.post("/chatgpt/web-search/detect")
        async def chat_web_search_detect(_: str = Depends(auth)):
            try:
                configuration = self.models.configuration
                if configuration is None or configuration.provider_id != "chatgpt_subscription":
                    raise RuntimeError("ChatGPT Subscription is not the selected model provider")
                probe = await self.provider_setup.probe_hosted_web_search(
                    configuration,
                    self.models.credentials_for(configuration),
                )
            except Exception as exc:
                return self._back(error=self._t("web_search_probe_failed", detail=exc))
            await self.store.asave_agent_configuration(
                self.provider_setup.configuration_with_capability(
                    configuration,
                    "hosted_web_search",
                    probe.supported,
                    probe.id,
                ),
            )
            return self._back(
                message=(
                    self._t("web_search_probe_available")
                    if probe.supported
                    else self._t("web_search_probe_unavailable")
                )
            )

        @self.app.post("/provider/custom")
        async def custom_provider_save(
            name: str = Form(...),
            base_url: str = Form(...),
            api_key: str = Form(""),
            protocol: str = Form("responses"),
            draft_token: str = Form(""),
            native_function_calls: str | None = Form(None),
            strict_function_schemas: str | None = Form(None),
            parallel_tool_control: str | None = Form(None),
            prompt_cache_key: str | None = Form(None),
            output_token_limit: str | None = Form(None),
            hosted_web_search: str | None = Form(None),
            clear_api_key: str | None = Form(None),
            _: str = Depends(auth),
        ):
            name = name.strip()
            if not name or len(name) > 80:
                return self._back(error=self._t("provider_name_invalid"))
            try:
                base_url = normalize_custom_base_url(base_url)
            except ValueError:
                return self._back(error=self._t("api_base_url_invalid"))
            if protocol not in CUSTOM_PROTOCOLS:
                return self._back(error=self._t("unknown_api_protocol"))
            existing = self.store.custom_provider()
            draft = self._provider_draft(draft_token)
            api_key = api_key.strip()
            if clear_api_key is not None:
                api_key = ""
            elif not api_key and draft and draft.name == name and draft.base_url == base_url:
                api_key = draft.api_key
            elif not api_key and existing:
                api_key = existing.api_key
            capabilities = {
                "native_function_calls": native_function_calls is not None,
                "strict_function_schemas": strict_function_schemas is not None,
                "parallel_tool_control": parallel_tool_control is not None,
                "prompt_cache_key": prompt_cache_key is not None,
                "output_token_limit": output_token_limit is not None,
                "hosted_web_search": hosted_web_search is not None and protocol == "responses",
            }
            if self.store.app_settings().enabled and not capabilities["native_function_calls"]:
                return self._back(error=self._t("disable_automation_for_provider"))
            saved_provider = CustomProvider(name, base_url, api_key, protocol, capabilities)
            await self.store.aset_custom_provider(saved_provider)
            if draft_token:
                self.provider_drafts.pop(draft_token, None)
            selected = self._model_view()
            try:
                await self._save_provider_selection(
                    "custom",
                    model=selected["model"],
                    reasoning_effort=selected["reasoning_effort"],
                    max_output_tokens=selected["max_output_tokens"],
                )
            except RuntimeError as exc:
                return self._back(error=str(exc))
            return self._back(message=self._t("provider_saved", name=name))

        @self.app.post("/provider/custom/detect")
        async def custom_provider_detect(
            name: str = Form(...),
            base_url: str = Form(...),
            api_key: str = Form(""),
            protocol: str = Form("responses"),
            probe_model: str = Form(...),
            draft_token: str = Form(""),
            native_function_calls: str | None = Form(None),
            strict_function_schemas: str | None = Form(None),
            parallel_tool_control: str | None = Form(None),
            prompt_cache_key: str | None = Form(None),
            output_token_limit: str | None = Form(None),
            hosted_web_search: str | None = Form(None),
            clear_api_key: str | None = Form(None),
            _: str = Depends(auth),
        ):
            name = name.strip()
            probe_model = probe_model.strip()
            if not name or len(name) > 80:
                return self._back(error=self._t("provider_name_invalid"))
            if not probe_model:
                return self._back(error=self._t("detection_model_required"))
            try:
                base_url = normalize_custom_base_url(base_url)
            except ValueError:
                return self._back(error=self._t("api_base_url_invalid"))
            if protocol not in CUSTOM_PROTOCOLS:
                protocol = "responses"
            existing = self.store.custom_provider()
            previous_draft = self._provider_draft(draft_token)
            api_key = api_key.strip()
            if clear_api_key is not None:
                api_key = ""
            elif not api_key and previous_draft:
                api_key = previous_draft.api_key
            elif not api_key and existing:
                api_key = existing.api_key
            token = secrets.token_urlsafe(24)
            draft = ProviderDraft(
                name,
                base_url,
                api_key,
                protocol,
                {
                    "native_function_calls": native_function_calls is not None,
                    "strict_function_schemas": strict_function_schemas is not None,
                    "parallel_tool_control": parallel_tool_control is not None,
                    "prompt_cache_key": prompt_cache_key is not None,
                    "output_token_limit": output_token_limit is not None,
                    "hosted_web_search": hosted_web_search is not None and protocol == "responses",
                },
                probe_model,
                time.time() + 15 * 60,
            )
            self.provider_drafts[token] = draft
            try:
                configuration = ModelConfiguration(
                    provider_id="custom_openai",
                    model_id=probe_model,
                    transport_profile=protocol,
                    credential_profile="setup_probe",
                    endpoint=base_url,
                    capabilities=ProviderCapabilities(),
                )
                native_probe = await self.provider_setup.probe_client_tools(
                    configuration,
                    ProviderCredentials(api_key=api_key),
                )
                draft.capabilities["native_function_calls"] = native_probe.supported
                draft.capabilities["strict_function_schemas"] = False
                draft.capabilities["parallel_tool_control"] = native_probe.supported
                if protocol == "responses":
                    web_probe = await self.provider_setup.probe_hosted_web_search(
                        configuration,
                        ProviderCredentials(api_key=api_key),
                    )
                    draft.capabilities["hosted_web_search"] = web_probe.supported
                else:
                    draft.capabilities["hosted_web_search"] = False
            except Exception as exc:
                return self._provider_draft_back(token, error=str(exc))
            label = "Responses" if draft.protocol == "responses" else "Chat Completions"
            native_label = (
                self._t("native_calls_available")
                if draft.capabilities["native_function_calls"]
                else self._t("native_calls_unavailable")
            )
            web_label = (
                self._t("hosted_search_available")
                if draft.capabilities["hosted_web_search"]
                else self._t("hosted_search_unavailable")
            )
            return self._provider_draft_back(
                token,
                message=self._t(
                    "provider_detection_result",
                    protocol=label,
                    native=native_label,
                    web=web_label,
                ),
            )

        @self.app.post("/provider/custom/remove")
        async def custom_provider_remove(_: str = Depends(auth)):
            await self.store.aclear_custom_provider()
            await self.store.aclear_provider_credentials("custom_openai_default")
            return self._back(message=self._t("custom_provider_removed"))

        @self.app.post("/provider/select")
        async def provider_select(provider: str = Form(...), _: str = Depends(auth)):
            if provider not in PROVIDERS:
                return self._back(error=self._t("unknown_model_provider"))
            if provider == "chatgpt" and not self.account.connected:
                return self._back(error=self._t("connect_chatgpt_first"))
            if provider == "custom" and self.store.custom_provider() is None:
                return self._back(error=self._t("configure_custom_provider_first"))
            custom = self.store.custom_provider()
            if (
                provider == "custom"
                and self.store.app_settings().enabled
                and custom
                and not custom.supports("native_function_calls")
            ):
                return self._back(error=self._t("detect_native_calls_before_select"))
            selected = self._model_view()
            await self._save_provider_selection(
                provider,
                model=selected["model"],
                reasoning_effort=selected["reasoning_effort"],
                max_output_tokens=selected["max_output_tokens"],
            )
            return self._back(message=self._t("provider_selected", name=self.models.provider_label))

        @self.app.post("/discord/connect")
        async def discord_connect(token: str = Form(...), _: str = Depends(auth)):
            token = token.strip()
            if len(token) < 20:
                return self._back(error=self._t("discord_token_incomplete"))
            await self.store.aset_discord_token(token)
            self.discord.error = None
            await self.discord.restart()
            return self._back(message=self._t("discord_connection_started"))

        @self.app.post("/discord/disconnect")
        async def discord_disconnect(_: str = Depends(auth)):
            await self.discord.stop()
            await self.store.aclear_discord_token()
            return self._back(message=self._t("discord_disconnected"))

        @self.app.post("/discord/captcha/{request_id}")
        async def discord_captcha(
            request_id: str,
            solution: str = Form(...),
            _: str = Depends(auth),
        ):
            solution = solution.strip()
            if not solution:
                return self._back(error=self._t("captcha_solution_required"))
            if not self.discord.solve_captcha(request_id, solution):
                return self._back(error=self._t("captcha_expired"))
            return self._back(message=self._t("captcha_submitted"))

        @self.app.post("/settings")
        async def settings(
            enabled: str | None = Form(None),
            silent_replies: str | None = Form(None),
            robot_prefix: str | None = Form(None),
            admin_locale: str = Form("en"),
            prompt_locale: str = Form("en"),
            assistant_name: str = Form(""),
            owner_timezone: str = Form("UTC"),
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
            owner_details: str = Form(""),
            base_instructions: str = Form(...),
            _: str = Depends(auth),
        ):
            if admin_locale not in SUPPORTED_LOCALES or prompt_locale not in SUPPORTED_LOCALES:
                return self._back(tab="assistant", error=ui_text(admin_locale, "unknown_locale"))
            try:
                ZoneInfo(owner_timezone)
            except (KeyError, ValueError):
                return self._back(tab="assistant", error=self._t("timezone_invalid"))
            if provider not in PROVIDERS:
                return self._back(tab="assistant", error=self._t("unknown_model_provider"))
            if reasoning_effort not in REASONING_EFFORTS:
                return self._back(tab="assistant", error=self._t("unknown_reasoning_effort"))
            if conversation_default not in {"opt_in", "opt_out"}:
                return self._back(tab="assistant", error=self._t("conversation_default_invalid"))
            if provider == "custom" and self.store.custom_provider() is None:
                return self._back(tab="assistant", error=self._t("configure_custom_provider_first"))
            custom = self.store.custom_provider()
            if (
                enabled is not None
                and provider == "custom"
                and (not custom or not custom.supports("native_function_calls"))
            ):
                return self._back(tab="assistant", error=self._t("detect_native_calls_before_enable"))
            model = model.strip()
            if not model:
                return self._back(tab="assistant", error=self._t("model_name_required"))
            assistant_name = assistant_name.strip()
            if len(assistant_name) > 80 or not all(character.isprintable() for character in assistant_name):
                return self._back(tab="assistant", error=self._t("assistant_name_invalid"))
            owner_details = owner_details.strip()
            if len(owner_details) > 20_000:
                return self._back(tab="assistant", error=self._t("owner_details_too_long"))
            if (
                min_delay_seconds > max_delay_seconds
                or min_message_gap_seconds > max_message_gap_seconds
                or min_typing_cps > max_typing_cps
                or min_human_quiet_minutes > max_human_quiet_minutes
            ):
                return self._back(tab="assistant", error=self._t("minimum_exceeds_maximum"))
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
                admin_theme=previous.admin_theme,
                prompt_locale=normalized_prompt_locale,
                assistant_name=assistant_name,
                owner_timezone=owner_timezone,
                min_message_gap_seconds=max(0.0, min(min_message_gap_seconds, 30.0)),
                max_message_gap_seconds=max(0.0, min(max_message_gap_seconds, 30.0)),
                default_conversation_enabled=conversation_default == "opt_in",
                debounce_seconds=debounce_seconds,
                min_delay_seconds=min_delay_seconds,
                max_delay_seconds=max_delay_seconds,
                min_typing_cps=min_typing_cps,
                max_typing_cps=max_typing_cps,
                min_human_quiet_minutes=max(0.0, min_human_quiet_minutes),
                max_human_quiet_minutes=max(0.0, max_human_quiet_minutes),
                owner_details=owner_details,
                base_instructions=base_instructions,
            )
            await self.store.aset_app_settings(value)
            await self._save_provider_selection(
                provider,
                model=model,
                reasoning_effort=reasoning_effort,
                max_output_tokens=max(32, min(max_reply_tokens, 2048)),
            )
            return self._back(
                tab="assistant",
                message=ui_text(value.admin_locale, "settings_saved"),
            )

        @self.app.post("/settings/reset")
        async def settings_reset(
            confirm: str = Form(""),
            _: str = Depends(auth),
        ):
            current = self.store.app_settings()
            if confirm != "reset":
                return self._back(
                    tab="assistant",
                    error=ui_text(current.admin_locale, "assistant_settings_reset_confirm"),
                )
            value = assistant_settings_defaults(current)
            await self.store.aset_app_settings(value)
            await self.models.arefresh_prompt_cache_identity()
            return self._back(
                tab="assistant",
                message=ui_text(value.admin_locale, "assistant_settings_reset"),
            )

        @self.app.post("/personality/infer")
        async def personality_infer(samples: str = Form(...), _: str = Depends(auth)):
            samples = samples.strip()
            if len(samples) < 200:
                return self._back(tab="assistant", error=self._t("history_too_short"))
            return await self._infer_personality(samples, source="pasted_history")

        @self.app.post("/personality/infer-history")
        async def personality_infer_history(
            history_limit: int = Form(100),
            _: str = Depends(auth),
        ):
            try:
                messages = await self.discord.personality_history(max(20, min(history_limit, 500)))
            except Exception as exc:
                return self._back(tab="assistant", error=self._localized_known_error(exc))
            samples = "\n\n---\n\n".join(messages)
            if len(samples) < 200:
                return self._back(tab="assistant", error=self._t("discord_history_too_short"))
            return await self._infer_personality(samples, source="discord_history")

        @self.app.post("/personality/save")
        async def personality_save(profile: str = Form(...), _: str = Depends(auth)):
            profile = profile.strip()
            if len(profile) < 50:
                return self._back(tab="assistant", error=self._t("personality_too_short"))
            source_hash = hashlib.sha256(("edited\n" + profile).encode()).hexdigest()
            await self.store.aset_personality(
                profile,
                source_hash,
                source="edited",
            )
            await self.models.arefresh_prompt_cache_identity()
            return self._back(tab="assistant", message=self._t("personality_updated"))

        @self.app.post("/conversations/{channel_id}/pause")
        async def pause(channel_id: str, _: str = Depends(auth)):
            await self.runtime.permanently_pause(channel_id)
            return self._back(tab="conversations", message=self._t("conversation_paused"))

        @self.app.post("/conversations/{channel_id}/resume")
        async def resume(channel_id: str, _: str = Depends(auth)):
            self.runtime.cancel(channel_id)
            escalation = await self.store.aactive_interrupt_for_channel(channel_id)
            if escalation is not None:
                await self.runtime.resume_escalation(str(escalation["id"]), action="resolved")
            await self.store.aset_permanent_pause(channel_id, False)
            await self.store.aclear_snooze(channel_id)
            return self._back(tab="conversations", message=self._t("conversation_resumed"))

        @self.app.post("/conversations/{channel_id}/mode")
        async def conversation_mode(
            channel_id: str,
            mode: str = Form(...),
            _: str = Depends(auth),
        ):
            if mode not in {"automatic", "inline", "paused"}:
                return self._back(
                    tab="conversations",
                    error=self._t("conversation_mode_invalid"),
                )
            if not await self.store.aset_conversation_mode(channel_id, mode):
                return self._back(
                    tab="conversations",
                    error=self._t("conversation_not_found"),
                )
            self.runtime.cancel(channel_id)
            return self._back(
                tab="conversations",
                message=self._t("conversation_mode_saved"),
            )

        @self.app.post("/conversations/{channel_id}/force-reply")
        async def force_reply(channel_id: str, _: str = Depends(auth)):
            if not self.runtime.ready:
                return self._back(
                    tab="conversations",
                    error=self._t("connect_provider_before_force"),
                )
            try:
                await self.discord.force_reply(channel_id)
            except Exception as exc:
                return self._back(tab="conversations", error=self._localized_known_error(exc))
            return self._back(tab="conversations", message=self._t("forced_reply_scheduled"))

        @self.app.post("/conversations/{channel_id}/steering")
        async def live_steering(
            channel_id: str,
            enabled: str | None = Form(None),
            _: str = Depends(auth),
        ):
            if await self.store.aconversation(channel_id) is None:
                return self._back(tab="conversations", error=self._t("conversation_not_found"))
            await self.runtime.set_live_steering(channel_id, enabled is not None)
            return self._back(tab="conversations", message=self._t("settings_saved"))

        @self.app.post("/escalations/{escalation_id}/claim")
        async def escalation_claim(escalation_id: str, _: str = Depends(auth)):
            if not await self.runtime.claim_escalation(escalation_id):
                return self._back(tab="conversations", error=self._t("escalation_not_found"))
            return self._back(tab="conversations", message=self._t("escalation_claimed"))

        @self.app.post("/escalations/{escalation_id}/resolve")
        async def escalation_resolve(
            escalation_id: str,
            resume: str | None = Form(None),
            _: str = Depends(auth),
        ):
            if not await self.runtime.resume_escalation(escalation_id, action="resolved"):
                return self._back(tab="conversations", error=self._t("escalation_not_found"))
            if resume is not None:
                escalation = await self.store.aescalation_interrupt(escalation_id)
                if escalation:
                    channel_id = str(escalation["channel_id"])
                    await self.store.aset_permanent_pause(channel_id, False)
                    await self.store.aclear_snooze(channel_id)
            return self._back(
                tab="conversations",
                message=self._t("escalation_resolved")
                + (self._t("automation_resumed_suffix") if resume is not None else ""),
            )

        @self.app.post("/escalations/{escalation_id}/dismiss")
        async def escalation_dismiss(
            escalation_id: str,
            resume: str | None = Form(None),
            _: str = Depends(auth),
        ):
            if not await self.runtime.resume_escalation(escalation_id, action="dismissed"):
                return self._back(tab="conversations", error=self._t("escalation_not_found"))
            return self._back(
                tab="conversations",
                message=self._t("escalation_dismissed")
                + (self._t("automation_resumed_suffix") if resume is not None else ""),
            )

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
                    error=self._t("delete_confirmation_required"),
                )
            try:
                deleted = await self.store.adelete_database_row(table, row_key)
            except ValueError as exc:
                key = (
                    "database_table_read_only"
                    if str(exc) == "This database table is read-only"
                    else "database_table_unknown"
                )
                return self._database_back(table, db_query, error=self._t(key))
            if not deleted:
                return self._database_back(table, db_query, error=self._t("database_row_not_found"))
            return self._database_back(
                table,
                db_query,
                message=self._t("database_row_deleted", row=repr(row_key), table=table),
            )

        @self.app.post("/diagnostics/replay")
        async def diagnostics_replay(
            thread_id: str = Form(...),
            checkpoint_id: str = Form(...),
            confirm: str = Form(""),
            _: str = Depends(auth),
        ):
            if confirm != "emulate":
                return self._back(tab="usage", error=self._t("replay_confirmation_required"))
            try:
                run_id = await self.runtime.replay_checkpoint(thread_id, checkpoint_id)
            except (RuntimeError, ValueError) as error:
                return self._back(tab="usage", error=str(error))
            return self._back(tab="usage", message=self._t("replay_completed", id=run_id))

        @self.app.post("/memories/delete")
        async def memory_delete(
            namespace: str = Form(...),
            key: str = Form(...),
            confirm: str = Form(""),
            _: str = Depends(auth),
        ):
            if confirm != "delete":
                return self._back(tab="usage", error=self._t("delete_confirmation_required"))
            try:
                labels = tuple(json.loads(namespace))
                if not labels or not all(isinstance(label, str) for label in labels):
                    raise ValueError
                await self.runtime.memory.adelete(labels, key)
            except (TypeError, ValueError, json.JSONDecodeError):
                return self._back(tab="usage", error=self._t("memory_identity_invalid"))
            return self._back(tab="usage", message=self._t("memory_deleted"))

    async def _conversation_views(self) -> list[dict]:
        now = time.time()
        result = await self.store.aconversations_with_steering()
        for conversation in result:
            until = conversation["snoozed_until"]
            conversation["snoozed"] = bool(until and until > now)
            conversation["quiet_minutes_remaining"] = (
                max(1, int((until - now + 59) // 60)) if until and until > now else 0
            )
        return result

    async def _agent_run_views(self) -> list[dict[str, Any]]:
        runs, traces = await self.store.aagent_diagnostics()
        by_run: dict[str, list[dict[str, Any]]] = {}
        for trace in traces:
            item = dict(trace)
            item["payload_json"] = json.dumps(
                json.loads(item["payload"]), ensure_ascii=False, indent=2, sort_keys=True
            )
            by_run.setdefault(str(item["run_id"]), []).append(item)
        result = []
        for row in runs:
            item = dict(row)
            item["started_at_label"] = (
                datetime.fromtimestamp(item["started_at"]).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
            )
            item["traces"] = by_run.get(str(item["id"]), [])
            result.append(item)
        return result

    async def _capability_probe_views(self) -> list[dict[str, Any]]:
        rows = await self.store.acapability_probes()
        result = []
        for row in rows:
            item = dict(row)
            item["completed_at_label"] = (
                datetime.fromtimestamp(item["completed_at"]).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
            )
            for field in ("configuration", "request_payload", "response_payload"):
                value = json.loads(item[field]) if item[field] else None
                item[f"{field}_json"] = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
            result.append(item)
        return result

    async def _runtime_record_views(self) -> dict[str, list[dict[str, Any]]]:
        records = await self.store.aruntime_diagnostics()
        result: dict[str, list[dict[str, Any]]] = {}
        for name, rows in records.items():
            values = []
            for row in rows:
                item = dict(row)
                item["json"] = json.dumps(item, ensure_ascii=False, indent=2, default=str)
                values.append(item)
            result[name] = values
        return result

    async def _subscription_web_search_probe_view(self, model: str) -> dict[str, Any] | None:
        report = await self.store.alatest_capability_probe("hosted_web_search")
        if report is None:
            return None
        configuration = json.loads(report["configuration"])
        status = str(report["status"])
        outcome = (
            "verified"
            if status == "supported"
            else ("request_error" if status == "error" else "response_mismatch")
        )
        response = json.loads(report["response_payload"]) if report["response_payload"] else {}
        return {
            "model": str(configuration.get("model_id") or model),
            "effort": str(configuration.get("options", {}).get("reasoning_effort") or "—"),
            "checked_at_label": datetime.fromtimestamp(report["completed_at"])
            .astimezone()
            .strftime("%Y-%m-%d %H:%M:%S %Z"),
            "result_label": self._t(f"web_search_probe_result_{outcome}"),
            "response_id": str(report["id"]),
            "observed": json.dumps(response, ensure_ascii=False)[:2000] or "—",
            "error": str(report["conclusion"] if status == "error" else ""),
            "request_log_id": None,
            "request_log_url": None,
        }

    def _personality_view(self) -> dict[str, Any] | None:
        personality = self.store.personality()
        if not personality:
            return None
        source = str(personality.get("source") or "").lower().replace(" ", "_")
        personality["source_label"] = self._t(f"source_{source}") if source else self._t("inferred_history")
        return personality

    async def _escalation_views(self) -> list[dict[str, Any]]:
        result = await self.store.aactive_interrupts()
        for escalation in result:
            payload = escalation["payload"]
            escalation["reason"] = str(payload.get("reason") or "other_explicit_request")
            escalation["delivery_error"] = None
            escalation["requested_at"] = escalation["created_at"]
            escalation["reason_label"] = ui_text(
                self.store.app_settings().admin_locale,
                f"reason_{escalation['reason']}",
            )
            escalation["requested_at_label"] = (
                datetime.fromtimestamp(escalation["requested_at"])
                .astimezone()
                .strftime("%Y-%m-%d %H:%M:%S %Z")
            )
        return result

    async def _database_view(self, table: str, page: int, query: str) -> dict:
        tables = await self.store.adatabase_tables()
        locale = self.store.app_settings().admin_locale
        for item in tables:
            item["label"] = ui_text(locale, f"table_{item['name']}")
        table_names = {item["name"] for item in tables}
        selected = table if table in table_names else "messages"
        search = query.strip()[:200]
        current_page = max(1, page)
        data = await self.store.adatabase_rows(
            selected,
            limit=50,
            offset=(current_page - 1) * 50,
            query=search,
        )
        if data["offset"] >= data["total"] and data["total"]:
            current_page = max(1, (data["total"] - 1) // data["limit"] + 1)
            data = await self.store.adatabase_rows(
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

    async def _save_provider_selection(
        self,
        provider: str,
        *,
        model: str,
        reasoning_effort: str,
        max_output_tokens: int,
    ) -> None:
        previous = self.models.configuration
        target_transport = "responses"
        if provider == "custom" and (custom := self.store.custom_provider()) is not None:
            target_transport = custom.protocol
        target_provider = "custom_openai" if provider == "custom" else "chatgpt_subscription"
        if previous and self.runtime._affinity(previous) != (target_provider, model, target_transport):
            await self.runtime.ensure_configuration_transition_allowed()
        if provider == "chatgpt":
            await self.models.asave_subscription(
                model_id=model,
                reasoning_effort=reasoning_effort,
                max_output_tokens=max_output_tokens,
            )
            await self.runtime.apply_configuration_transition(previous)
            return
        custom = self.store.custom_provider()
        if provider != "custom" or custom is None:
            raise ValueError("Unknown or unavailable model provider")
        await self.models.asave_custom_openai(
            custom,
            model_id=model,
            reasoning_effort=reasoning_effort,
            max_output_tokens=max_output_tokens,
        )
        await self.runtime.apply_configuration_transition(previous)

    def _model_view(self) -> dict[str, Any]:
        configuration = self.models.configuration
        if configuration is None:
            return {
                "model": "gpt-5.4-mini",
                "reasoning_effort": "low",
                "max_output_tokens": 256,
            }
        options = configuration.options
        return {
            "model": configuration.model_id,
            "reasoning_effort": str(options.get("reasoning_effort") or "low"),
            "max_output_tokens": int(options.get("max_completion_tokens") or 256),
        }

    def _active_provider(self) -> str:
        configuration = self.models.configuration
        return "custom" if configuration and configuration.provider_id == "custom_openai" else "chatgpt"

    def _hosted_search_capability(self) -> bool | None:
        configuration = self.models.configuration
        return configuration.capabilities.hosted_web_search if configuration else None

    async def _infer_personality(self, samples: str, *, source: str) -> RedirectResponse:
        cfg = self.store.app_settings()
        source_hash = personality_source_hash(samples, cfg.prompt_locale)
        cached = self.store.personality()
        if cached and cached["source_hash"] == source_hash:
            return self._back(tab="assistant", message=self._t("personality_cached"))
        try:
            response = await self.models.build_model().ainvoke(
                [
                    SystemMessage(prompts_for(cfg.prompt_locale).personality),
                    HumanMessage(samples),
                ]
            )
            profile = response.text.strip()
        except Exception as exc:
            return self._back(tab="assistant", error=str(exc))
        await self.store.aset_personality(profile, source_hash, source=source)
        await self.models.arefresh_prompt_cache_identity()
        return self._back(tab="assistant", message=self._t("personality_inferred"))

    def _url(self, path: str) -> str:
        return self.public_url + "/" + path.lstrip("/")

    def _t(self, key: str, **values: object) -> str:
        locale = self.store.app_settings().admin_locale if self.store is not None else "en"
        return ui_text(locale, key, **values)

    def _localized_known_error(self, error: Exception) -> str:
        key = {
            "Discord must be connected before forcing a reply": "discord_required_force",
            "Invalid Discord channel ID": "discord_channel_invalid",
            "Discord conversation is not available": "discord_conversation_unavailable",
            "This conversation has no incoming message to answer": "discord_no_incoming_message",
            "The latest incoming Discord message ID is invalid": "discord_message_id_invalid",
            "Discord must be connected before loading message history": "discord_required_history",
        }.get(str(error))
        return self._t(key) if key else str(error)

    def _database_url(self, table: str, page: int, query: str) -> str:
        parameters = {"tab": "database", "db_table": table, "db_page": max(1, page)}
        if query:
            parameters["db_query"] = query
        return self._url("/") + "?" + urlencode(parameters)

    def _database_back(
        self,
        table: str,
        query: str,
        *,
        message: str | None = None,
        error: str | None = None,
    ) -> RedirectResponse:
        parameters = {"tab": "database", "db_table": table, "db_page": 1}
        if query:
            parameters["db_query"] = query
        if message:
            parameters["message"] = message
        if error:
            parameters["error"] = error
        return RedirectResponse(
            self._url("/") + "?" + urlencode(parameters),
            status_code=303,
        )

    def _provider_draft(self, token: str) -> ProviderDraft | None:
        now = time.time()
        for key, value in list(self.provider_drafts.items()):
            if value.expires_at <= now:
                self.provider_drafts.pop(key, None)
        return self.provider_drafts.get(token)

    def _provider_draft_back(
        self,
        token: str,
        *,
        message: str | None = None,
        error: str | None = None,
    ) -> RedirectResponse:
        values = {
            "tab": "connections",
            "provider_draft": token,
            "message": message,
            "error": error,
        }
        query = urlencode({key: value for key, value in values.items() if value})
        return RedirectResponse(self._url("/") + "?" + query, status_code=303)

    def _back(
        self,
        *,
        tab: str = "connections",
        message: str | None = None,
        error: str | None = None,
    ) -> RedirectResponse:
        selected_tab = tab if tab in ADMIN_TABS else "connections"
        query = urlencode(
            {
                key: value
                for key, value in {"tab": selected_tab, "message": message, "error": error}.items()
                if value
            }
        )
        return RedirectResponse(self._url("/") + ("?" + query if query else ""), status_code=303)
