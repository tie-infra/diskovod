from __future__ import annotations

import asyncio
import hashlib
import importlib.metadata
import json
import platform
import re
import secrets
import time
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlparse
from zoneinfo import ZoneInfo

from fastapi import Depends, FastAPI, Form, HTTPException, Request, status
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response, StreamingResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.templating import Jinja2Templates
from .discord import DiscordService
from .admin_jobs import AdminJobService, AdminJobWorker
from .admin_queries import AdminQueryService
from .localization import (
    SUPPORTED_LOCALES,
    assistant_name_for,
    invocation_attention_words,
    prompts_for,
    ui_text,
)
from .interaction import (
    ActiveTurnInput,
    AvailabilitySchedule,
    InteractionPolicy,
    InvocationAlias,
    OwnerHandoff,
    TriggerRule,
    TypoTolerance,
    evaluate_trigger,
    normalize_invocation_text,
    preset_policy,
)
from .models import (
    ADMIN_THEMES,
    REASONING_EFFORTS,
    AssistantProfile,
    AutomationSettings,
    CustomProvider,
    InterfaceSettings,
)
from .oauth import ChatGPTAccount
from .personality import assistant_profile_fingerprint, personality_source_hash
from .redaction import redact_sensitive
from .providers import (
    ModelConfiguration,
    ModelService,
    ProviderCapabilities,
    ProviderSetup,
    normalize_custom_base_url,
)
from .runtime import AgentService
from .security import password_matches
from .store import Store

PERSONALITY_INSTRUCTIONS = prompts_for("en").personality
PROVIDERS = frozenset({"chatgpt", "custom"})
CUSTOM_PROTOCOLS = frozenset({"responses", "chat_completions"})
LIVE_TOPIC_PATTERN = re.compile(r"^(?:jobs|inbox|(?:chat|run|job):[A-Za-z0-9_.:-]{1,500})$")
AUTOMATION_TIMING_FIELDS = (
    "debounce_seconds",
    "min_typing_cps",
    "max_typing_cps",
    "min_human_quiet_minutes",
    "max_human_quiet_minutes",
    "min_message_gap_seconds",
    "max_message_gap_seconds",
)
AUTOMATION_PRESETS = {
    "responsive": (0.8, 24.0, 40.0, 5.0, 12.0, 0.3, 0.9),
    "natural": (1.8, 18.0, 32.0, 15.0, 30.0, 0.7, 2.0),
    "reserved": (3.0, 14.0, 24.0, 30.0, 60.0, 1.2, 3.0),
}


@dataclass(slots=True)
class ProviderDraft:
    name: str
    base_url: str
    api_key: str
    protocol: str
    capabilities: dict[str, bool]
    probe_model: str
    expires_at: float


def localized_base_instructions(previous_locale: str, new_locale: str, submitted: str) -> str:
    """Translate only the stock prompt; never overwrite user-customized instructions."""
    submitted = submitted.strip()
    if submitted == prompts_for(previous_locale).base:
        return prompts_for(new_locale).base
    return submitted


def assistant_settings_defaults() -> AssistantProfile:
    return AssistantProfile()


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
        jobs: AdminJobService | None = None,
        job_worker: AdminJobWorker | None = None,
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
        self.jobs = jobs
        self.job_worker = job_worker
        self.queries = AdminQueryService(store) if store is not None else None
        self.server_epoch = secrets.token_urlsafe(12)
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
                "default-src 'none'; style-src 'self'; script-src 'self'; font-src 'self'; "
                "img-src 'self' data:; connect-src 'self'; form-action 'self'; base-uri 'none'; "
                "frame-ancestors 'none'"
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
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=self._t("authentication_required"),
                headers={"WWW-Authenticate": "Basic"},
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

    def _interaction_policy_from_form(self, form: Any) -> InteractionPolicy:
        preset = str(form.get("preset") or "")
        if preset not in {"autonomous", "shared", "on_invocation", "manual", "draft"}:
            raise ValueError("Unknown interaction preset")
        profile = self.store.assistant_profile()
        timing = (
            "queue_for_next_turn"
            if form.get("active_turn_timing") == "queue_for_next_turn"
            else "inject_at_safe_points"
        )
        policy = preset_policy(
            preset,  # type: ignore[arg-type]
            prompt_locale=profile.prompt_locale,
            inject_active_input=timing == "inject_at_safe_points",
        )
        policy = replace(
            policy,
            trigger_participants=frozenset(
                item for item in form.getlist("trigger_participants") if item in {"owner", "peer"}
            ),
            active_turn_input=ActiveTurnInput(
                timing=timing,
                participants=frozenset(
                    item for item in form.getlist("active_turn_participants") if item in {"owner", "peer"}
                ),
            ),
            invocation_snooze_behavior=(
                "respect" if form.get("invocation_snooze_behavior") == "respect" else "bypass"
            ),
            availability_schedule=AvailabilitySchedule(
                enabled=form.get("schedule_enabled") is not None,
                weekdays=frozenset(
                    int(item)
                    for item in form.getlist("schedule_weekdays")
                    if str(item).isdigit() and 0 <= int(item) <= 6
                ),
                start_minute=self._time_input_minutes(str(form.get("schedule_start") or "09:00")),
                end_minute=self._time_input_minutes(str(form.get("schedule_end") or "17:00")),
                timezone=str(form.get("schedule_timezone") or "").strip(),
            ),
            owner_handoff=OwnerHandoff(
                availability_transition=(
                    str(form.get("owner_handoff_transition"))
                    if form.get("owner_handoff_transition") in {"none", "snooze", "pause"}
                    else policy.owner_handoff.availability_transition
                ),
                active_run_action=(
                    "cancel" if form.get("owner_handoff_active_action") == "cancel" else "keep_or_inject"
                ),
            ),
            conversation_role=(
                str(form.get("conversation_role"))
                if form.get("conversation_role") in {"owner_delegate", "shared_assistant", "owner_copilot"}
                else policy.conversation_role
            ),
            identity_marker=(
                str(form.get("identity_marker"))
                if form.get("identity_marker") in {"configurable", "forced"}
                else policy.identity_marker
            ),
            delivery=(
                str(form.get("delivery"))
                if form.get("delivery") in {"immediate", "owner_approval", "dashboard_only"}
                else policy.delivery
            ),
        )
        rules: list[TriggerRule] = []
        if form.get("trigger_every_message") is not None:
            rules.append(TriggerRule("every_message", id="every-message"))
        if form.get("trigger_direct_address") is not None:
            aliases = [InvocationAlias()] if form.get("use_assistant_name_alias") is not None else []
            aliases.extend(
                InvocationAlias("literal", clean[:80])
                for item in str(form.get("invocation_aliases") or "").splitlines()
                if (clean := item.strip())
            )
        if form.get("trigger_reply_to_assistant") is not None:
            rules.append(TriggerRule("reply_to_assistant", id="reply-to-assistant"))
        reactions = tuple(
            clean[:64]
            for item in str(form.get("reaction_emojis") or "").splitlines()
            if (clean := item.strip())
        )[:16]
        if reactions:
            rules.append(
                TriggerRule(
                    "reaction_invocation",
                    id="reaction-invocation",
                    reactions=reactions,
                )
            )
        if form.get("trigger_direct_address") is not None:
            rules.append(
                TriggerRule(
                    "direct_address",
                    id="direct-address",
                    aliases=tuple(aliases[:16]),
                    attention_locales=(
                        ()
                        if form.get("attention_mode") == "replace"
                        else tuple(
                            item for item in form.getlist("attention_locales") if item in SUPPORTED_LOCALES
                        )
                        or (profile.prompt_locale,)
                    ),
                    additional_attention_words=tuple(
                        clean[:80]
                        for item in str(form.get("additional_attention_words") or "").splitlines()
                        if (clean := item.strip())
                    )[:32],
                    allow_bare_alias=form.get("allow_bare_alias") is not None,
                    typo_tolerance=TypoTolerance(enabled=form.get("typo_tolerance") is not None),
                )
            )
        rules.extend(
            TriggerRule("literal_prefix", id=f"literal-{index}", literal=clean[:80])
            for index, item in enumerate(
                str(form.get("literal_prefixes") or "").splitlines(),
                1,
            )
            if (clean := item.strip())
        )
        return replace(policy, trigger_rules=tuple(rules[:16]))

    @staticmethod
    def _time_input_minutes(value: str) -> int:
        match = re.fullmatch(r"([01]\d|2[0-3]):([0-5]\d)", value)
        if match is None:
            raise ValueError("Invalid schedule time")
        return int(match.group(1)) * 60 + int(match.group(2))

    def _routes(self) -> None:
        auth = self.require_admin

        @self.app.get("/")
        async def overview(request: Request, _: str = Depends(auth)):
            view = await self.queries.overview()
            view.update(
                model_ready=self.models.ready,
                model_provider=self.models.provider_label,
                model_id=self._model_view()["model"],
                discord_connected=self.discord.connected,
                discord_identity=self.discord.identity,
                discord_error=self._localized_error(self.discord.error),
                chat_connected=self.account.connected,
                automation_enabled=self.store.automation_settings().enabled,
            )
            return await self._render(
                request,
                "overview.html",
                "overview",
                "overview",
                overview=view,
            )

        @self.app.get("/inbox")
        async def inbox(request: Request, offset: int = 0, _: str = Depends(auth)):
            return await self._render(
                request,
                "inbox.html",
                "inbox",
                "inbox",
                escalations=await self.queries.inbox(offset=offset),
                drafts=[
                    item
                    for item in await self.runtime.publisher.drafts(limit=100)
                    if item["state"] in {"pending", "recorded", "failed"}
                ],
                failed_runs=self._localize_run_items(await self.queries.actionable_runs()),
                captcha_requests=self.discord.captcha_requests(),
                connection_errors=[
                    {
                        "kind": "chatgpt",
                        "kind_label": self._t("service_chatgpt"),
                        "summary": self._localized_error(self.account.last_error),
                    }
                    for _ in (0,)
                    if self.account.last_error
                ]
                + [
                    {
                        "kind": "discord",
                        "kind_label": self._t("discord"),
                        "summary": self._localized_error(self.discord.error),
                    }
                    for _ in (0,)
                    if self.discord.error
                ],
                live_topic="inbox",
            )

        @self.app.get("/inbox/escalations/{escalation_id}")
        async def escalation_detail(request: Request, escalation_id: str, _: str = Depends(auth)):
            view = await self.queries.escalation(escalation_id)
            if view is None:
                raise HTTPException(404, self._t("escalation_not_found"))
            return await self._render(
                request,
                "escalation.html",
                "inbox",
                "escalation_detail",
                detail=view,
                live_topic="inbox",
            )

        @self.app.get("/search")
        async def search(request: Request, q: str = "", _: str = Depends(auth)):
            return await self._render(
                request,
                "search.html",
                "",
                "search_results",
                query=q,
                results=await self.queries.search(q),
            )

        @self.app.get("/chats")
        async def chats(
            request: Request,
            q: str = "",
            state: str = "",
            offset: int = 0,
            _: str = Depends(auth),
        ):
            return await self._render(
                request,
                "chats.html",
                "chats",
                "chats",
                chats=await self.queries.chats(query=q, state=state, offset=offset),
            )

        @self.app.get("/chats/{channel_id}")
        async def chat(
            request: Request,
            channel_id: str,
            test_input: str = "",
            test_participant: str = "peer",
            _: str = Depends(auth),
        ):
            view = await self.queries.chat(channel_id)
            if view is None:
                raise HTTPException(404, self._t("conversation_not_found"))
            await self._prepare_chat_view(view)
            profile = self.store.assistant_profile()
            view["assistant_display_name"] = assistant_name_for(profile.prompt_locale, profile.assistant_name)
            view["owner_timezone"] = profile.owner_timezone
            if test_input:
                policy, _, _ = await self.store.ainteraction_policy(channel_id)
                participant = "owner" if test_participant == "owner" else "peer"
                view["invocation_test_input"] = test_input[:1000]
                view["invocation_test_participant"] = participant
                view["invocation_test_result"] = evaluate_trigger(
                    policy,
                    participant=participant,
                    content=test_input[:1000],
                    assistant_name=assistant_name_for(profile.prompt_locale, profile.assistant_name),
                    attention_words=invocation_attention_words(),
                ).to_dict() | {"normalized_input": normalize_invocation_text(test_input[:1000])}
            return await self._render(
                request, "chat.html", "chats", "chat", chat=view, live_topic=f"chat:{channel_id}"
            )

        @self.app.get("/chats/{channel_id}/generations/{generation}")
        async def chat_generation(request: Request, channel_id: str, generation: int, _: str = Depends(auth)):
            view = await self.queries.chat(channel_id, generation=generation)
            if view is None:
                raise HTTPException(404, self._t("conversation_not_found"))
            await self._prepare_chat_view(view)
            return await self._render(request, "chat.html", "chats", "chat", chat=view)

        @self.app.get("/chats/{channel_id}/generations/{generation}/checkpoints/{checkpoint_id}")
        async def checkpoint_detail(
            request: Request,
            channel_id: str,
            generation: int,
            checkpoint_id: str,
            _: str = Depends(auth),
        ):
            metadata = await self.queries.checkpoint(channel_id, generation, checkpoint_id)
            if metadata is None or self.runtime.checkpointer is None:
                raise HTTPException(404, self._t("checkpoint_not_found"))
            config = {
                "configurable": {
                    "thread_id": metadata["thread_id"],
                    "checkpoint_id": checkpoint_id,
                }
            }
            snapshot = await self.runtime.checkpointer.aget_tuple(config)
            if snapshot is None:
                raise HTTPException(404, self._t("checkpoint_not_found"))
            parent = (
                await self.runtime.checkpointer.aget_tuple(snapshot.parent_config)
                if snapshot.parent_config
                else None
            )
            return await self._render(
                request,
                "checkpoint.html",
                "chats",
                "checkpoint_detail",
                checkpoint=self._checkpoint_view(metadata, snapshot, parent),
            )

        @self.app.get("/activity/runs")
        async def runs(
            request: Request,
            run_status: str = "",
            channel_id: str = "",
            offset: int = 0,
            _: str = Depends(auth),
        ):
            return await self._render(
                request,
                "runs.html",
                "activity",
                "activity",
                runs=self._localize_run_page(
                    await self.queries.runs(
                        status=run_status,
                        channel_id=channel_id,
                        offset=offset,
                    )
                ),
            )

        @self.app.get("/activity/runs/{run_id}")
        async def run_detail(
            request: Request,
            run_id: str,
            panel: str = "timeline",
            offset: int = 0,
            _: str = Depends(auth),
        ):
            selected_panel = (
                panel
                if panel in {"summary", "timeline", "conversation", "model_io", "state", "raw"}
                else "timeline"
            )
            view = await self.queries.run(run_id, event_offset=offset)
            if view is None:
                raise HTTPException(404, self._t("run_not_found"))
            self._localize_run_view(view)
            return await self._render(
                request,
                "run.html",
                "activity",
                "run_detail",
                run=view,
                panel=selected_panel,
                live_topic=f"run:{run_id}",
            )

        @self.app.get("/activity/runs/{run_id}/diagnostic.json")
        async def run_diagnostic(run_id: str, _: str = Depends(auth)):
            payload = await self.queries.run_diagnostic(run_id)
            if payload is None:
                raise HTTPException(404, self._t("run_not_found"))
            return Response(
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
                media_type="application/json",
                headers={"Content-Disposition": 'attachment; filename="diskovod-run-diagnostic.json"'},
            )

        @self.app.post("/activity/runs/{run_id}/deliveries/{action_id}/{operation}")
        async def resolve_run_delivery(
            run_id: str,
            action_id: str,
            operation: str,
            remote_id: str = Form(""),
            _: str = Depends(auth),
        ):
            if operation not in {"retry", "confirmed_succeeded", "confirmed_failed"}:
                raise HTTPException(404, self._t("outbound_action_not_found"))
            result = await self.runtime.resolve_outbound_action(
                run_id,
                action_id,
                operation,
                remote_id=remote_id,
            )
            if result is None:
                return self._redirect(
                    f"/activity/runs/{run_id}?panel=raw",
                    error=self._t("outbound_action_not_actionable"),
                )
            return self._redirect(
                f"/activity/runs/{run_id}?panel=raw",
                message=self._t("outbound_action_updated"),
            )

        @self.app.get("/activity/jobs")
        async def jobs(
            request: Request,
            job_status: str = "",
            offset: int = 0,
            _: str = Depends(auth),
        ):
            page_limit = 50
            page_offset = max(0, offset)
            items = (
                await self.jobs.repository.list(
                    limit=page_limit,
                    offset=page_offset,
                    status=job_status or None,
                )
                if self.jobs
                else []
            )
            total = await self.jobs.repository.count(status=job_status or None) if self.jobs else 0
            return await self._render(
                request,
                "jobs.html",
                "activity",
                "jobs",
                jobs={
                    "items": [self._job_view(item) for item in items],
                    "status": job_status,
                    "offset": page_offset,
                    "limit": page_limit,
                    "total": total,
                    "previous_offset": max(0, page_offset - page_limit) if page_offset else None,
                    "next_offset": page_offset + page_limit if page_offset + page_limit < total else None,
                },
            )

        @self.app.get("/activity/jobs/{job_id}")
        async def job_detail(request: Request, job_id: str, _: str = Depends(auth)):
            job = await self.jobs.get(job_id) if self.jobs else None
            if job is None:
                raise HTTPException(404, self._t("job_not_found"))
            events = [self._job_event_view(event) for event in await self.jobs.repository.events(job_id)]
            return await self._render(
                request,
                "job.html",
                "activity",
                "job_detail",
                job=self._job_view(job),
                job_events=events,
                live_topic=f"job:{job_id}",
            )

        @self.app.get("/knowledge/memories")
        async def memories(
            request: Request,
            q: str = "",
            scope: str = "",
            offset: int = 0,
            _: str = Depends(auth),
        ):
            return await self._render(
                request,
                "memories.html",
                "knowledge",
                "memories",
                memories=await self.queries.memories(query=q, scope=scope, offset=offset),
            )

        @self.app.get("/knowledge/attachments")
        async def attachments(
            request: Request,
            q: str = "",
            channel_id: str = "",
            media_type: str = "",
            offset: int = 0,
            _: str = Depends(auth),
        ):
            return await self._render(
                request,
                "attachments.html",
                "knowledge",
                "attachments",
                attachments=await self.queries.attachments(
                    query=q,
                    channel_id=channel_id,
                    media_type=media_type,
                    offset=offset,
                ),
            )

        @self.app.get("/settings/connections")
        async def connections(request: Request, provider_draft: str = "", _: str = Depends(auth)):
            return await self._render_connections(request, provider_draft)

        @self.app.get("/settings/model")
        async def model_settings(request: Request, _: str = Depends(auth)):
            probe_jobs = []
            if self.jobs:
                probe_jobs = [
                    self._job_view(job)
                    for job in await self.jobs.repository.list(limit=20)
                    if job["type"] == "provider.capability_probe"
                ][:5]
            return await self._render(
                request,
                "settings_model.html",
                "settings",
                "model_settings",
                model_view=self._model_view(),
                active_provider=self._active_provider(),
                provider_label=self.models.provider_label,
                model_capabilities=(
                    {
                        "native_tools": configuration.capabilities.native_tools,
                        "hosted_web_search": configuration.capabilities.hosted_web_search,
                        "output_token_limit": configuration.capabilities.output_token_limit,
                    }
                    if (configuration := self.models.configuration) is not None
                    else {}
                ),
                subscription_web_search=self._hosted_search_capability(),
                latest_probe=await self._subscription_web_search_probe_view(self._model_view()["model"]),
                configuration_versions=self._localize_configuration_versions(
                    await self.queries.configuration_versions()
                ),
                probe_jobs=probe_jobs,
                live_topic="jobs",
            )

        @self.app.get("/settings/assistant")
        async def assistant_settings(request: Request, _: str = Depends(auth)):
            profile = self.store.assistant_profile()
            return await self._render(
                request,
                "settings_assistant.html",
                "settings",
                "assistant_settings",
                assistant_profile=profile,
                personality=self._personality_view(),
                assistant_display_name=assistant_name_for(profile.prompt_locale, profile.assistant_name),
                default_assistant_name=assistant_name_for(profile.prompt_locale),
            )

        @self.app.get("/settings/automation")
        async def automation_settings(request: Request, _: str = Depends(auth)):
            settings = self.store.automation_settings()
            return await self._render(
                request,
                "settings_automation.html",
                "settings",
                "automation_settings",
                automation_settings=settings,
                automation_preset=self._automation_preset(settings),
                automation_presets={
                    name: dict(zip(AUTOMATION_TIMING_FIELDS, values, strict=True))
                    for name, values in AUTOMATION_PRESETS.items()
                },
            )

        @self.app.get("/settings/interaction")
        async def interaction_settings(
            request: Request,
            test_input: str = "",
            test_participant: str = "peer",
            _: str = Depends(auth),
        ):
            policy = self.store.default_interaction_policy()
            profile = self.store.assistant_profile()
            test_result = None
            participant = "owner" if test_participant == "owner" else "peer"
            if test_input:
                test_result = evaluate_trigger(
                    policy,
                    participant=participant,
                    content=test_input[:1000],
                    assistant_name=assistant_name_for(profile.prompt_locale, profile.assistant_name),
                    attention_words=invocation_attention_words(),
                ).to_dict() | {"normalized_input": normalize_invocation_text(test_input[:1000])}
            return await self._render(
                request,
                "settings_interaction.html",
                "settings",
                "interaction_settings",
                interaction_policy=policy.to_dict(),
                assistant_display_name=assistant_name_for(
                    profile.prompt_locale,
                    profile.assistant_name,
                ),
                owner_timezone=profile.owner_timezone,
                invocation_test_input=test_input[:1000],
                invocation_test_participant=participant,
                invocation_test_result=test_result,
            )

        @self.app.get("/settings/interface")
        async def interface_settings(request: Request, _: str = Depends(auth)):
            return await self._render(
                request,
                "settings_interface.html",
                "settings",
                "interface_settings",
                interface_settings=self.store.interface_settings(),
                owner_timezone=self.store.assistant_profile().owner_timezone,
            )

        @self.app.get("/system/diagnostics")
        async def diagnostics(request: Request, offset: int = 0, _: str = Depends(auth)):
            counts = await self.queries.diagnostic_counts()
            return await self._render(
                request,
                "diagnostics.html",
                "system",
                "diagnostics",
                capability_probes=self._localize_probe_page(
                    await self.queries.capability_probes(offset=offset)
                ),
                diagnostic_counts=counts,
                diagnostic_metrics=[
                    {"label": self._t(name), "value": value}
                    for name, value in counts.items()
                    if name != "sqlite_version"
                ],
                health=[
                    {"label": self._t("discord"), "ok": self.discord.connected},
                    {"label": self._t("model"), "ok": self.models.ready},
                    {"label": self._t("service_chatgpt"), "ok": self.account.connected},
                ],
                versions=self._diagnostic_versions(),
            )

        @self.app.get("/system/diagnostics.json")
        async def diagnostic_bundle(_: str = Depends(auth)):
            probes = await self.queries.capability_probes(limit=50)
            payload = redact_sensitive(
                {
                    "generated_at": time.time(),
                    "health": {
                        "discord": self.discord.connected,
                        "model": self.models.ready,
                        "chatgpt": self.account.connected,
                    },
                    "counts": await self.queries.diagnostic_counts(),
                    "versions": self._diagnostic_versions(),
                    "capability_probes": probes["items"],
                }
            )
            return Response(
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
                media_type="application/json",
                headers={"Content-Disposition": 'attachment; filename="diskovod-diagnostic.json"'},
            )

        @self.app.get("/system/database")
        async def database(
            request: Request,
            table: str = "messages",
            page: int = 1,
            q: str = "",
            _: str = Depends(auth),
        ):
            return await self._render(
                request,
                "database.html",
                "system",
                "database",
                database=await self._database_view(table, page, q),
            )

        @self.app.get("/static/style.css")
        async def css():
            return FileResponse(Path(__file__).parent / "static" / "style.css", media_type="text/css")

        @self.app.get("/static/bootstrap.min.css")
        async def bootstrap_css():
            return FileResponse(Path(__file__).parent / "static" / "bootstrap.min.css", media_type="text/css")

        @self.app.get("/static/bootstrap.bundle.min.js")
        async def bootstrap_javascript():
            return FileResponse(
                Path(__file__).parent / "static" / "bootstrap.bundle.min.js",
                media_type="text/javascript",
            )

        @self.app.get("/static/app.js")
        async def javascript():
            return FileResponse(Path(__file__).parent / "static" / "app.js", media_type="text/javascript")

        @self.app.get("/api/jobs/{job_id}")
        async def job_api(job_id: str, _: str = Depends(auth)):
            job = await self.jobs.get(job_id) if self.jobs else None
            if job is None:
                raise HTTPException(404, self._t("job_not_found"))
            return JSONResponse(self._job_view(job))

        @self.app.get("/api/inbox")
        async def inbox_api(offset: int = 0, limit: int = 50, _: str = Depends(auth)):
            return JSONResponse(await self.queries.inbox(offset=offset, limit=limit))

        @self.app.get("/api/jobs")
        async def jobs_api(
            limit: int = 20,
            offset: int = 0,
            job_status: str = "",
            _: str = Depends(auth),
        ):
            if self.jobs is None:
                return JSONResponse({"items": [], "active_count": 0, "next_offset": None})
            try:
                items = await self.jobs.repository.list(
                    limit=limit,
                    offset=offset,
                    status=job_status or None,
                )
                total = await self.jobs.repository.count(status=job_status or None)
            except ValueError as error:
                raise HTTPException(400, self._localized_error(error)) from error
            bounded_limit = max(1, min(limit, 500))
            bounded_offset = max(0, offset)
            return JSONResponse(
                {
                    "items": [self._job_view(item) for item in items],
                    "active_count": await self.jobs.repository.active_count(),
                    "next_offset": (
                        bounded_offset + bounded_limit if bounded_offset + bounded_limit < total else None
                    ),
                }
            )

        @self.app.get("/api/chats")
        async def chats_api(
            q: str = "",
            state: str = "",
            limit: int = 50,
            offset: int = 0,
            _: str = Depends(auth),
        ):
            return JSONResponse(await self.queries.chats(query=q, state=state, limit=limit, offset=offset))

        @self.app.get("/api/chats/{channel_id}/messages")
        async def chat_messages_api(
            channel_id: str,
            before: float | None = None,
            limit: int = 50,
            _: str = Depends(auth),
        ):
            result = await self.queries.messages(channel_id, before=before, limit=limit)
            if result is None:
                raise HTTPException(404, self._t("conversation_not_found"))
            return JSONResponse(result)

        @self.app.get("/api/chats/{channel_id}/timeline")
        async def chat_timeline_api(
            channel_id: str,
            after: int = 0,
            limit: int = 100,
            _: str = Depends(auth),
        ):
            return JSONResponse(await self.queries.chat_timeline(channel_id, after=after, limit=limit))

        @self.app.get("/api/runs")
        async def runs_api(
            run_status: str = "",
            channel_id: str = "",
            limit: int = 50,
            offset: int = 0,
            _: str = Depends(auth),
        ):
            return JSONResponse(
                self._localize_run_page(
                    await self.queries.runs(
                        status=run_status,
                        channel_id=channel_id,
                        limit=limit,
                        offset=offset,
                    )
                )
            )

        @self.app.get("/api/runs/{run_id}/events")
        async def run_events_api(
            run_id: str,
            after: int = 0,
            limit: int = 100,
            _: str = Depends(auth),
        ):
            result = await self.queries.run_events(run_id, after=after, limit=limit)
            if result is None:
                raise HTTPException(404, self._t("run_not_found"))
            return JSONResponse(result)

        @self.app.get("/api/runs/{run_id}/events/{sequence}")
        async def run_event_api(run_id: str, sequence: int, _: str = Depends(auth)):
            result = await self.queries.run_event(run_id, sequence)
            if result is None:
                raise HTTPException(404, self._t("run_not_found"))
            return JSONResponse(result)

        @self.app.get("/api/runs/{run_id}/delivery")
        async def run_delivery_api(
            run_id: str,
            action_id: str,
            _: str = Depends(auth),
        ):
            result = await self.queries.run_delivery(run_id, action_id)
            if result is None:
                raise HTTPException(404, self._t("run_not_found"))
            return JSONResponse(result)

        @self.app.get("/api/runs/{run_id}")
        async def run_api(run_id: str, _: str = Depends(auth)):
            result = await self.queries.run(run_id)
            if result is None:
                raise HTTPException(404, self._t("run_not_found"))
            self._localize_run_view(result)
            return JSONResponse(result)

        @self.app.get("/api/search")
        async def search_api(q: str = "", limit: int = 10, _: str = Depends(auth)):
            return JSONResponse(await self.queries.search(q, limit=limit))

        @self.app.get("/api/diagnostics/probes/{probe_id}")
        async def capability_probe_api(probe_id: str, _: str = Depends(auth)):
            result = await self.queries.capability_probe(probe_id)
            if result is None:
                raise HTTPException(404, self._t("probe_not_found"))
            return JSONResponse(result)

        @self.app.get("/api/events/stream")
        async def event_stream(request: Request, topics: str = "jobs", _: str = Depends(auth)):
            if request.headers.get("sec-fetch-site") not in {None, "same-origin"}:
                raise HTTPException(403, self._t("cross_origin_rejected", url=self.public_url))
            selected = {value for value in topics.split(",") if value}
            if not selected:
                selected = {"jobs"}
            if len(selected) > 8 or any(not LIVE_TOPIC_PATTERN.fullmatch(value) for value in selected):
                raise HTTPException(400, self._t("invalid_live_topics"))

            async def records():
                previous: dict[str, str] = {}
                deadline = time.monotonic() + 25
                heartbeat_at = 0.0
                yield (
                    json.dumps(
                        {"type": "hello", "server_epoch": self.server_epoch},
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                while time.monotonic() < deadline and not await request.is_disconnected():
                    versions = await self.queries.resource_versions(selected)
                    for topic, version in versions.items():
                        if previous.get(topic) == version:
                            continue
                        previous[topic] = version
                        kind, _, identifier = topic.partition(":")
                        payload: dict[str, Any] = {
                            "type": f"{kind}.updated",
                            "version": version,
                        }
                        if identifier:
                            payload["id"] = identifier
                        if topic == "jobs" and self.jobs:
                            payload["active_count"] = await self.jobs.repository.active_count()
                        yield json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
                    if time.monotonic() >= heartbeat_at:
                        heartbeat_at = time.monotonic() + 10
                        yield '{"type":"heartbeat"}\n'
                    await asyncio.sleep(1)

            return StreamingResponse(
                records(),
                media_type="application/x-ndjson",
                headers={"X-Accel-Buffering": "no"},
            )

        @self.app.post("/activity/jobs/{job_id}/cancel")
        async def cancel_job(job_id: str, _: str = Depends(auth)):
            job = await self.jobs.cancel(job_id) if self.jobs else None
            if job is None:
                raise HTTPException(404, self._t("job_not_found"))
            return RedirectResponse(self._url(f"/activity/jobs/{job_id}"), status_code=303)

        @self.app.post("/settings/interface")
        async def interface_save(
            locale: str = Form("en"),
            theme: str = Form("system"),
            density: str = Form("comfortable"),
            display_timezone_mode: str = Form("browser"),
            named_timezone: str = Form("UTC"),
            show_advanced_ids: str | None = Form(None),
            _: str = Depends(auth),
        ):
            if locale not in SUPPORTED_LOCALES:
                return self._redirect("/settings/interface", error=self._t("unknown_locale"))
            if theme not in ADMIN_THEMES:
                return self._redirect("/settings/interface", error=self._t("unknown_theme"))
            if density not in {"comfortable", "compact"}:
                return self._redirect("/settings/interface", error=self._t("unknown_density"))
            if display_timezone_mode not in {"browser", "owner", "named"}:
                return self._redirect("/settings/interface", error=self._t("timezone_invalid"))
            display_timezone = display_timezone_mode
            if display_timezone_mode == "named":
                display_timezone = named_timezone.strip()
                try:
                    ZoneInfo(display_timezone)
                except (KeyError, ValueError):
                    return self._redirect("/settings/interface", error=self._t("timezone_invalid"))
            await self.store.aset_interface_settings(
                InterfaceSettings(
                    locale=locale,
                    theme=theme,
                    density=density,
                    display_timezone=display_timezone,
                    show_advanced_ids=show_advanced_ids is not None,
                )
            )
            return self._redirect("/settings/interface", message=ui_text(locale, "settings_saved"))

        @self.app.post("/settings/assistant")
        async def assistant_save(
            prompt_locale: str = Form("en"),
            assistant_name: str = Form(""),
            owner_timezone: str = Form("UTC"),
            owner_details: str = Form(""),
            base_instructions: str = Form(...),
            allow_conversational_followups: str | None = Form(None),
            _: str = Depends(auth),
        ):
            if prompt_locale not in SUPPORTED_LOCALES:
                return self._redirect("/settings/assistant", error=self._t("unknown_locale"))
            try:
                ZoneInfo(owner_timezone)
            except (KeyError, ValueError):
                return self._redirect("/settings/assistant", error=self._t("timezone_invalid"))
            assistant_name = assistant_name.strip()
            owner_details = owner_details.strip()
            if len(assistant_name) > 80 or not all(character.isprintable() for character in assistant_name):
                return self._redirect("/settings/assistant", error=self._t("assistant_name_invalid"))
            if len(owner_details) > 20_000:
                return self._redirect("/settings/assistant", error=self._t("owner_details_too_long"))
            previous = self.store.assistant_profile()
            await self.store.aset_assistant_profile(
                AssistantProfile(
                    prompt_locale=prompt_locale,
                    assistant_name=assistant_name,
                    owner_timezone=owner_timezone,
                    owner_details=owner_details,
                    base_instructions=localized_base_instructions(
                        previous.prompt_locale, prompt_locale, base_instructions
                    ),
                    allow_conversational_followups=allow_conversational_followups is not None,
                )
            )
            await self.models.arefresh_prompt_cache_identity()
            return self._redirect("/settings/assistant", message=self._t("settings_saved"))

        @self.app.post("/settings/automation")
        async def automation_save(
            preset: str = Form("natural"),
            enabled: str | None = Form(None),
            silent_replies: str | None = Form(None),
            robot_prefix: str | None = Form(None),
            default_conversation_enabled: str | None = Form(None),
            default_interaction_preset: str = Form("autonomous"),
            debounce_seconds: float = Form(1.8),
            min_typing_cps: float = Form(18),
            max_typing_cps: float = Form(32),
            min_human_quiet_minutes: float = Form(15),
            max_human_quiet_minutes: float = Form(30),
            min_message_gap_seconds: float = Form(0.7),
            max_message_gap_seconds: float = Form(2),
            _: str = Depends(auth),
        ):
            if preset in AUTOMATION_PRESETS:
                (
                    debounce_seconds,
                    min_typing_cps,
                    max_typing_cps,
                    min_human_quiet_minutes,
                    max_human_quiet_minutes,
                    min_message_gap_seconds,
                    max_message_gap_seconds,
                ) = AUTOMATION_PRESETS[preset]
            elif preset != "custom":
                return self._redirect("/settings/automation", error=self._t("unknown_automation_preset"))
            if (
                min_typing_cps > max_typing_cps
                or min_human_quiet_minutes > max_human_quiet_minutes
                or min_message_gap_seconds > max_message_gap_seconds
            ):
                return self._redirect("/settings/automation", error=self._t("minimum_exceeds_maximum"))
            custom = self.store.custom_provider()
            if (
                enabled is not None
                and self._active_provider() == "custom"
                and (custom is None or not custom.supports("native_function_calls"))
            ):
                return self._redirect(
                    "/settings/automation", error=self._t("detect_native_calls_before_enable")
                )
            previous_automation = self.store.automation_settings()
            previous_default_policy = self.store.default_interaction_policy()
            if default_interaction_preset not in {
                "autonomous",
                "shared",
                "on_invocation",
                "manual",
                "draft",
            }:
                return self._redirect("/settings/automation", error=self._t("interaction_policy_invalid"))
            updated_automation = AutomationSettings(
                enabled=enabled is not None,
                silent_replies=silent_replies is not None,
                robot_prefix=robot_prefix is not None,
                default_conversation_enabled=default_conversation_enabled is not None,
                default_interaction_preset=default_interaction_preset,
                debounce_seconds=max(0, debounce_seconds),
                min_typing_cps=max(1, min_typing_cps),
                max_typing_cps=max(1, max_typing_cps),
                min_human_quiet_minutes=max(0, min_human_quiet_minutes),
                max_human_quiet_minutes=max(0, max_human_quiet_minutes),
                min_message_gap_seconds=max(0, min_message_gap_seconds),
                max_message_gap_seconds=max(0, max_message_gap_seconds),
            )
            await self.store.aset_automation_settings(updated_automation)
            if previous_default_policy.preset != default_interaction_preset:
                await self.store.aset_default_interaction_policy(
                    preset_policy(
                        default_interaction_preset,  # type: ignore[arg-type]
                        prompt_locale=self.store.assistant_profile().prompt_locale,
                    )
                )
            if previous_automation.enabled and not updated_automation.enabled:
                await self.runtime.cancel_all_followups("automation_disabled")
            return self._redirect("/settings/automation", message=self._t("settings_saved"))

        @self.app.post("/settings/interaction")
        async def interaction_save(request: Request, _: str = Depends(auth)):
            try:
                policy = self._interaction_policy_from_form(await request.form())
                await self.store.aset_default_interaction_policy(policy)
            except (KeyError, TypeError, ValueError):
                return self._redirect(
                    "/settings/interaction",
                    error=self._t("interaction_policy_invalid"),
                )
            return self._redirect(
                "/settings/interaction",
                message=self._t("interaction_policy_saved"),
            )

        @self.app.post("/settings/interaction/reset")
        async def interaction_reset(_: str = Depends(auth)):
            await self.store.areset_default_interaction_policy()
            return self._redirect(
                "/settings/interaction",
                message=self._t("interaction_policy_saved"),
            )

        @self.app.post("/settings/model")
        async def model_save(
            provider: str = Form("chatgpt"),
            model: str = Form(...),
            reasoning_effort: str = Form("low"),
            max_reply_tokens: int = Form(256),
            _: str = Depends(auth),
        ):
            if provider not in PROVIDERS:
                return self._redirect("/settings/model", error=self._t("unknown_model_provider"))
            if reasoning_effort not in REASONING_EFFORTS:
                return self._redirect("/settings/model", error=self._t("unknown_reasoning_effort"))
            if not model.strip():
                return self._redirect("/settings/model", error=self._t("model_name_required"))
            try:
                await self._save_provider_selection(
                    provider,
                    model=model.strip(),
                    reasoning_effort=reasoning_effort,
                    max_output_tokens=max(32, min(max_reply_tokens, 2048)),
                )
            except (RuntimeError, ValueError) as error:
                return self._redirect("/settings/model", error=self._localized_error(error))
            return self._redirect("/settings/model", message=self._t("settings_saved"))

        @self.app.post("/chatgpt/connect")
        async def chat_connect(_: str = Depends(auth)):
            try:
                return RedirectResponse(await self.account.begin_oauth(), status_code=303)
            except Exception as exc:
                return self._redirect("/settings/connections", error=self._localized_error(exc))

        @self.app.get("/chatgpt/oauth/callback")
        async def chat_callback(
            code: str | None = None,
            state: str | None = None,
            error: str | None = None,
        ):
            try:
                await self.account.finish_oauth(code=code, state=state, error=error)
            except Exception as exc:
                return self._redirect("/settings/connections", error=self._localized_error(exc))
            selected = self._model_view()
            try:
                await self._save_provider_selection(
                    "chatgpt",
                    model=selected["model"],
                    reasoning_effort=selected["reasoning_effort"],
                    max_output_tokens=selected["max_output_tokens"],
                )
            except RuntimeError as exc:
                return self._redirect("/settings/connections", error=self._localized_error(exc))
            return self._redirect("/settings/connections", message=self._t("chatgpt_connected"))

        @self.app.post("/chatgpt/disconnect")
        async def chat_disconnect(_: str = Depends(auth)):
            await self.store.aclear_chat_credentials()
            self.account.last_error = None
            return self._redirect("/settings/connections", message=self._t("chatgpt_disconnected"))

        @self.app.post("/chatgpt/web-search/detect")
        async def chat_web_search_detect(_: str = Depends(auth)):
            try:
                configuration = self.models.configuration
                if configuration is None or configuration.provider_id != "chatgpt_subscription":
                    raise RuntimeError("ChatGPT Subscription is not the selected model provider")
                configuration_id = await self.store.aactive_configuration_id()
                if configuration_id is None or self.jobs is None:
                    raise RuntimeError("Administrative job service is not ready")
                job, _ = await self.jobs.enqueue(
                    "provider.capability_probe",
                    {
                        "configuration_id": configuration_id,
                        "capability": "hosted_web_search",
                        "apply_result": True,
                    },
                    idempotency_key=f"provider-capability:{configuration_id}:hosted_web_search",
                    target_kind="model_configuration",
                    target_id=str(configuration_id),
                )
            except Exception as exc:
                return self._redirect("/settings/model", error=self._t("web_search_probe_failed", detail=exc))
            return RedirectResponse(
                self._url(f"/activity/jobs/{job['id']}"),
                status_code=303,
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
                return self._redirect("/settings/connections", error=self._t("provider_name_invalid"))
            try:
                base_url = normalize_custom_base_url(base_url)
            except ValueError:
                return self._redirect("/settings/connections", error=self._t("api_base_url_invalid"))
            if protocol not in CUSTOM_PROTOCOLS:
                return self._redirect("/settings/connections", error=self._t("unknown_api_protocol"))
            existing = self.store.custom_provider()
            draft = await self._provider_draft(draft_token)
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
            if self.store.automation_settings().enabled and not capabilities["native_function_calls"]:
                return self._redirect(
                    "/settings/connections", error=self._t("disable_automation_for_provider")
                )
            saved_provider = CustomProvider(name, base_url, api_key, protocol, capabilities)
            await self.store.aset_custom_provider(saved_provider)
            if draft_token:
                await self.store.adelete_provider_setup_draft(draft_token)
            selected = self._model_view()
            try:
                await self._save_provider_selection(
                    "custom",
                    model=selected["model"],
                    reasoning_effort=selected["reasoning_effort"],
                    max_output_tokens=selected["max_output_tokens"],
                )
            except RuntimeError as exc:
                return self._redirect("/settings/connections", error=self._localized_error(exc))
            return self._redirect("/settings/connections", message=self._t("provider_saved", name=name))

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
                return self._redirect("/settings/connections", error=self._t("provider_name_invalid"))
            if not probe_model:
                return self._redirect("/settings/connections", error=self._t("detection_model_required"))
            try:
                base_url = normalize_custom_base_url(base_url)
            except ValueError:
                return self._redirect("/settings/connections", error=self._t("api_base_url_invalid"))
            if protocol not in CUSTOM_PROTOCOLS:
                protocol = "responses"
            existing = self.store.custom_provider()
            previous_draft = await self._provider_draft(draft_token)
            api_key = api_key.strip()
            if clear_api_key is not None:
                api_key = ""
            elif not api_key and previous_draft:
                api_key = previous_draft.api_key
            elif not api_key and existing:
                api_key = existing.api_key
            token = secrets.token_urlsafe(24)
            try:
                if self.jobs is None:
                    raise RuntimeError("Administrative job service is not ready")
                configuration = ModelConfiguration(
                    provider_id="custom_openai",
                    model_id=probe_model,
                    transport_profile=protocol,
                    credential_profile="setup_probe",
                    endpoint=base_url,
                    capabilities=ProviderCapabilities(),
                )
                capabilities = {
                    "native_tools": native_function_calls is not None,
                    "strict_function_schemas": strict_function_schemas is not None,
                    "parallel_tool_control": parallel_tool_control is not None,
                    "prompt_cache_key": prompt_cache_key is not None,
                    "output_token_limit": output_token_limit is not None,
                    "hosted_web_search": hosted_web_search is not None and protocol == "responses",
                }
                expires_at = time.time() + 15 * 60
                fingerprint = hashlib.sha256(
                    json.dumps(
                        {
                            "name": name,
                            "base_url": base_url,
                            "protocol": protocol,
                            "model": probe_model,
                        },
                        sort_keys=True,
                    ).encode()
                ).hexdigest()
                await self.store.acreate_provider_setup_draft(
                    token,
                    {
                        "name": name,
                        "base_url": base_url,
                        "api_key": api_key,
                        "protocol": protocol,
                        "probe_model": probe_model,
                        "capabilities": capabilities,
                        "configuration": configuration.to_dict(),
                        "credentials": {"api_key": api_key},
                    },
                    fingerprint,
                    expires_at=expires_at,
                )
                native_job, native_created = await self.jobs.enqueue(
                    "provider.setup_draft_probe",
                    {"draft_id": token, "capability": "native_tools"},
                    idempotency_key=f"provider-draft:{fingerprint}:native_tools",
                    target_kind="provider_setup_draft",
                    target_id=token,
                )
                if not native_created and native_job.get("target_id"):
                    existing_token = str(native_job["target_id"])
                    if existing_token != token:
                        await self.store.adelete_provider_setup_draft(token)
                        token = existing_token
                if protocol == "responses":
                    await self.jobs.enqueue(
                        "provider.setup_draft_probe",
                        {"draft_id": token, "capability": "hosted_web_search"},
                        idempotency_key=f"provider-draft:{fingerprint}:hosted_web_search",
                        target_kind="provider_setup_draft",
                        target_id=token,
                    )
            except Exception as exc:
                return self._provider_draft_back(token, error=self._localized_error(exc))
            return self._provider_draft_back(
                token,
                message=self._t("connecting"),
            )

        @self.app.post("/provider/custom/remove")
        async def custom_provider_remove(_: str = Depends(auth)):
            await self.store.aclear_custom_provider()
            await self.store.aclear_provider_credentials("custom_openai_default")
            return self._redirect("/settings/connections", message=self._t("custom_provider_removed"))

        @self.app.post("/provider/select")
        async def provider_select(provider: str = Form(...), _: str = Depends(auth)):
            if provider not in PROVIDERS:
                return self._redirect("/settings/connections", error=self._t("unknown_model_provider"))
            if provider == "chatgpt" and not self.account.connected:
                return self._redirect("/settings/connections", error=self._t("connect_chatgpt_first"))
            if provider == "custom" and self.store.custom_provider() is None:
                return self._redirect(
                    "/settings/connections", error=self._t("configure_custom_provider_first")
                )
            custom = self.store.custom_provider()
            if (
                provider == "custom"
                and self.store.automation_settings().enabled
                and custom
                and not custom.supports("native_function_calls")
            ):
                return self._redirect(
                    "/settings/connections", error=self._t("detect_native_calls_before_select")
                )
            selected = self._model_view()
            await self._save_provider_selection(
                provider,
                model=selected["model"],
                reasoning_effort=selected["reasoning_effort"],
                max_output_tokens=selected["max_output_tokens"],
            )
            return self._redirect(
                "/settings/connections", message=self._t("provider_selected", name=self.models.provider_label)
            )

        @self.app.post("/discord/connect")
        async def discord_connect(token: str = Form(...), _: str = Depends(auth)):
            token = token.strip()
            if len(token) < 20:
                return self._redirect("/settings/connections", error=self._t("discord_token_incomplete"))
            await self.store.aset_discord_token(token)
            self.discord.error = None
            await self.discord.restart()
            return self._redirect("/settings/connections", message=self._t("discord_connection_started"))

        @self.app.post("/discord/disconnect")
        async def discord_disconnect(_: str = Depends(auth)):
            await self.discord.stop()
            await self.store.aclear_discord_token()
            return self._redirect("/settings/connections", message=self._t("discord_disconnected"))

        @self.app.post("/discord/captcha/{request_id}")
        async def discord_captcha(
            request_id: str,
            solution: str = Form(...),
            _: str = Depends(auth),
        ):
            solution = solution.strip()
            if not solution:
                return self._redirect("/inbox", error=self._t("captcha_solution_required"))
            if not self.discord.solve_captcha(request_id, solution):
                return self._redirect("/inbox", error=self._t("captcha_expired"))
            return self._redirect("/inbox", message=self._t("captcha_submitted"))

        @self.app.post("/settings/assistant/reset")
        async def assistant_reset(
            confirm: str = Form(""),
            _: str = Depends(auth),
        ):
            interface = self.store.interface_settings()
            if confirm != "reset":
                return self._redirect(
                    "/settings/assistant",
                    error=ui_text(interface.locale, "assistant_settings_reset_confirm"),
                )
            await self.store.aset_assistant_profile(assistant_settings_defaults())
            await self.models.arefresh_prompt_cache_identity()
            return self._redirect(
                "/settings/assistant",
                message=ui_text(interface.locale, "assistant_settings_reset"),
            )

        @self.app.post("/settings/automation/reset")
        async def automation_reset(confirm: str = Form(""), _: str = Depends(auth)):
            if confirm != "reset":
                return self._redirect("/settings/automation", error=self._t("settings_reset_confirm"))
            previous = self.store.automation_settings()
            await self.store.aset_automation_settings(AutomationSettings())
            if previous.enabled:
                await self.runtime.cancel_all_followups("automation_disabled")
            return self._redirect("/settings/automation", message=self._t("automation_settings_reset"))

        @self.app.post("/settings/interface/reset")
        async def interface_reset(confirm: str = Form(""), _: str = Depends(auth)):
            if confirm != "reset":
                return self._redirect("/settings/interface", error=self._t("settings_reset_confirm"))
            await self.store.aset_interface_settings(InterfaceSettings())
            return self._redirect("/settings/interface", message=ui_text("en", "interface_settings_reset"))

        @self.app.post("/personality/infer")
        async def personality_infer(samples: str = Form(...), _: str = Depends(auth)):
            samples = samples.strip()
            if len(samples) < 200:
                return self._redirect("/settings/assistant", error=self._t("history_too_short"))
            profile = self.store.assistant_profile()
            source_hash = personality_source_hash(samples, profile.prompt_locale)
            cached = self.store.personality()
            if cached and cached["source_hash"] == source_hash:
                return self._redirect("/settings/assistant", message=self._t("personality_cached"))
            configuration_id = await self.store.aactive_configuration_id()
            if configuration_id is None or self.jobs is None:
                return self._redirect("/settings/assistant", error=self._t("connect_provider_before_force"))
            input_id = secrets.token_urlsafe(24)
            await self.store.acreate_admin_job_input(
                input_id,
                {"samples": samples},
                expires_at=time.time() + 60 * 60,
            )
            try:
                job, created = await self.jobs.enqueue(
                    "assistant.personality_inference",
                    {
                        "configuration_id": configuration_id,
                        "prompt_locale": profile.prompt_locale,
                        "profile_fingerprint": assistant_profile_fingerprint(profile),
                        "source": "pasted_history",
                        "input_id": input_id,
                    },
                    idempotency_key=f"personality:{source_hash}",
                    target_kind="assistant_profile",
                    target_id=assistant_profile_fingerprint(profile),
                )
            except Exception:
                await self.store.adelete_admin_job_input(input_id)
                raise
            if not created:
                await self.store.adelete_admin_job_input(input_id)
            return RedirectResponse(self._url(f"/activity/jobs/{job['id']}"), status_code=303)

        @self.app.post("/personality/infer-history")
        async def personality_infer_history(
            history_limit: int = Form(100),
            _: str = Depends(auth),
        ):
            profile = self.store.assistant_profile()
            configuration_id = await self.store.aactive_configuration_id()
            if configuration_id is None or self.jobs is None:
                return self._redirect("/settings/assistant", error=self._t("connect_provider_before_force"))
            fingerprint = assistant_profile_fingerprint(profile)
            job, _ = await self.jobs.enqueue(
                "assistant.personality_inference",
                {
                    "configuration_id": configuration_id,
                    "prompt_locale": profile.prompt_locale,
                    "profile_fingerprint": fingerprint,
                    "source": "discord_history",
                    "history_limit": max(20, min(history_limit, 500)),
                },
                idempotency_key=f"personality-history:{configuration_id}:{fingerprint}",
                target_kind="assistant_profile",
                target_id=fingerprint,
            )
            return RedirectResponse(self._url(f"/activity/jobs/{job['id']}"), status_code=303)

        @self.app.post("/personality/save")
        async def personality_save(profile: str = Form(...), _: str = Depends(auth)):
            profile = profile.strip()
            if len(profile) < 50:
                return self._redirect("/settings/assistant", error=self._t("personality_too_short"))
            source_hash = hashlib.sha256(("edited\n" + profile).encode()).hexdigest()
            await self.store.aset_personality(
                profile,
                source_hash,
                source="edited",
            )
            await self.models.arefresh_prompt_cache_identity()
            return self._redirect("/settings/assistant", message=self._t("personality_updated"))

        @self.app.post("/chats/{channel_id}/pause")
        async def pause(channel_id: str, _: str = Depends(auth)):
            await self.runtime.permanently_pause(channel_id)
            return self._redirect(f"/chats/{channel_id}", message=self._t("conversation_paused"))

        @self.app.post("/chats/{channel_id}/snooze")
        async def snooze(channel_id: str, minutes: int = Form(...), _: str = Depends(auth)):
            if await self.store.aconversation(channel_id) is None:
                return self._redirect(f"/chats/{channel_id}", error=self._t("conversation_not_found"))
            until = await self.store.asnooze(channel_id, max(1, min(minutes, 10_080)) * 60)
            owner_timezone = self.store.assistant_profile().owner_timezone
            until_label = datetime.fromtimestamp(until, ZoneInfo(owner_timezone)).strftime(
                "%Y-%m-%d %H:%M %Z"
            )
            return self._redirect(
                f"/chats/{channel_id}",
                message=self._t("conversation_snoozed_until", time=until_label),
            )

        @self.app.post("/chats/{channel_id}/snooze/clear")
        async def clear_snooze(channel_id: str, _: str = Depends(auth)):
            await self.store.aclear_snooze(channel_id)
            return self._redirect(f"/chats/{channel_id}", message=self._t("conversation_resumed"))

        @self.app.post("/chats/{channel_id}/resume")
        async def resume(channel_id: str, _: str = Depends(auth)):
            self.runtime.cancel(channel_id)
            escalation = await self.store.aactive_interrupt_for_channel(channel_id)
            if escalation is not None:
                await self.runtime.resume_escalation(str(escalation["id"]), action="resolved")
            await self.store.aset_permanent_pause(channel_id, False)
            await self.store.aclear_snooze(channel_id)
            return self._redirect(f"/chats/{channel_id}", message=self._t("conversation_resumed"))

        @self.app.post("/chats/{channel_id}/interaction")
        async def conversation_interaction(
            request: Request,
            channel_id: str,
            _: str = Depends(auth),
        ):
            try:
                policy = self._interaction_policy_from_form(await request.form())
                saved = await self.store.aset_interaction_policy(channel_id, policy)
            except (KeyError, TypeError, ValueError):
                return self._redirect(
                    f"/chats/{channel_id}",
                    error=self._t("interaction_policy_invalid"),
                )
            if not saved:
                return self._redirect(
                    f"/chats/{channel_id}",
                    error=self._t("conversation_not_found"),
                )
            return self._redirect(
                f"/chats/{channel_id}",
                message=self._t("interaction_policy_saved"),
            )

        @self.app.post("/chats/{channel_id}/interaction/reset")
        async def conversation_interaction_reset(channel_id: str, _: str = Depends(auth)):
            await self.store.areset_interaction_policy(channel_id)
            return self._redirect(
                f"/chats/{channel_id}",
                message=self._t("interaction_policy_saved"),
            )

        @self.app.post("/chats/{channel_id}/force-reply")
        async def force_reply(channel_id: str, _: str = Depends(auth)):
            if not self.runtime.ready:
                return self._redirect(
                    f"/chats/{channel_id}",
                    error=self._t("connect_provider_before_force"),
                )
            try:
                await self.discord.force_reply(channel_id)
            except Exception as exc:
                return self._redirect(f"/chats/{channel_id}", error=self._localized_error(exc))
            return self._redirect(f"/chats/{channel_id}", message=self._t("forced_reply_scheduled"))

        @self.app.post("/chats/{channel_id}/followup/cancel")
        async def cancel_chat_followup(channel_id: str, _: str = Depends(auth)):
            if not await self.runtime.cancel_followup(channel_id):
                return self._redirect(f"/chats/{channel_id}", error=self._t("followup_not_active"))
            return self._redirect(f"/chats/{channel_id}", message=self._t("followup_cancelled"))

        @self.app.post("/inbox/escalations/{escalation_id}/claim")
        async def escalation_claim(escalation_id: str, _: str = Depends(auth)):
            if not await self.runtime.claim_escalation(escalation_id):
                return self._redirect("/inbox", error=self._t("escalation_not_found"))
            return self._redirect(
                f"/inbox/escalations/{escalation_id}", message=self._t("escalation_claimed")
            )

        @self.app.post("/inbox/drafts/{draft_id}/approve")
        async def approve_draft(
            draft_id: str,
            message: str | None = Form(None),
            _: str = Depends(auth),
        ):
            try:
                result = await self.runtime.publisher.approve_draft(draft_id, message=message)
            except ValueError:
                return self._redirect("/inbox", error=self._t("draft_message_required"))
            if result is None:
                return self._redirect("/inbox", error=self._t("draft_not_actionable"))
            if not result.accepted:
                return self._redirect("/inbox", error=self._t("draft_delivery_failed"))
            return self._redirect("/inbox", message=self._t("draft_approved"))

        @self.app.post("/inbox/drafts/{draft_id}/reject")
        async def reject_draft(draft_id: str, _: str = Depends(auth)):
            if not await self.runtime.publisher.reject_draft(draft_id):
                return self._redirect("/inbox", error=self._t("draft_not_actionable"))
            return self._redirect("/inbox", message=self._t("draft_rejected"))

        @self.app.post("/inbox/escalations/{escalation_id}/resolve")
        async def escalation_resolve(
            escalation_id: str,
            resume: str | None = Form(None),
            _: str = Depends(auth),
        ):
            if not await self.runtime.resume_escalation(escalation_id, action="resolved"):
                return self._redirect("/inbox", error=self._t("escalation_not_found"))
            if resume is not None:
                escalation = await self.store.aescalation_interrupt(escalation_id)
                if escalation:
                    channel_id = str(escalation["channel_id"])
                    await self.store.aset_permanent_pause(channel_id, False)
                    await self.store.aclear_snooze(channel_id)
            return self._redirect(
                "/inbox",
                message=self._t("escalation_resolved")
                + (self._t("automation_resumed_suffix") if resume is not None else ""),
            )

        @self.app.post("/inbox/escalations/{escalation_id}/dismiss")
        async def escalation_dismiss(
            escalation_id: str,
            resume: str | None = Form(None),
            _: str = Depends(auth),
        ):
            if not await self.runtime.resume_escalation(escalation_id, action="dismissed"):
                return self._redirect("/inbox", error=self._t("escalation_not_found"))
            if resume is not None:
                escalation = await self.store.aescalation_interrupt(escalation_id)
                if escalation:
                    channel_id = str(escalation["channel_id"])
                    await self.store.aset_permanent_pause(channel_id, False)
                    await self.store.aclear_snooze(channel_id)
            return self._redirect(
                "/inbox",
                message=self._t("escalation_dismissed")
                + (self._t("automation_resumed_suffix") if resume is not None else ""),
            )

        @self.app.post("/system/database/delete")
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

        @self.app.post("/activity/checkpoints/{thread_id}/{checkpoint_id}/replay")
        async def diagnostics_replay(
            thread_id: str,
            checkpoint_id: str,
            confirm: str = Form(""),
            _: str = Depends(auth),
        ):
            if confirm != "emulate":
                return self._redirect(
                    f"/activity/checkpoints/{thread_id}/{checkpoint_id}",
                    error=self._t("replay_confirmation_required"),
                )
            configuration_id = await self.store.aactive_configuration_id()
            if configuration_id is None or self.jobs is None:
                return self._redirect(
                    f"/activity/checkpoints/{thread_id}/{checkpoint_id}",
                    error=self._t("connect_provider_before_force"),
                )
            job, _ = await self.jobs.enqueue(
                "runtime.checkpoint_replay",
                {
                    "thread_id": thread_id,
                    "checkpoint_id": checkpoint_id,
                    "configuration_id": configuration_id,
                },
                idempotency_key=f"checkpoint-replay:{thread_id}:{checkpoint_id}:{configuration_id}",
                target_kind="checkpoint",
                target_id=f"{thread_id}:{checkpoint_id}",
            )
            return RedirectResponse(self._url(f"/activity/jobs/{job['id']}"), status_code=303)

        @self.app.post("/knowledge/memories/delete")
        async def memory_delete(
            namespace: str = Form(...),
            key: str = Form(...),
            confirm: str = Form(""),
            _: str = Depends(auth),
        ):
            if confirm != "delete":
                return self._redirect("/knowledge/memories", error=self._t("delete_confirmation_required"))
            try:
                labels = tuple(json.loads(namespace))
                if not labels or not all(isinstance(label, str) for label in labels):
                    raise ValueError
                await self.runtime.memory.adelete(labels, key)
            except (TypeError, ValueError, json.JSONDecodeError):
                return self._redirect("/knowledge/memories", error=self._t("memory_identity_invalid"))
            return self._redirect("/knowledge/memories", message=self._t("memory_deleted"))

    async def _render(
        self,
        request: Request,
        template: str,
        active_section: str,
        title_key: str,
        **context: Any,
    ):
        interface = self.store.interface_settings()
        display_timezone = interface.display_timezone
        if display_timezone == "owner":
            display_timezone = self.store.assistant_profile().owner_timezone
        jobs = await self.jobs.repository.list(limit=5) if self.jobs else []
        active_jobs = await self.jobs.repository.active_count() if self.jobs else 0
        base = {
            "active_section": active_section,
            "page_title": ui_text(interface.locale, title_key),
            "interface_settings": interface,
            "resolved_display_timezone": display_timezone,
            "automation_settings": self.store.automation_settings(),
            "locale": interface.locale,
            "locales": SUPPORTED_LOCALES,
            "t": lambda key, **values: ui_text(interface.locale, key, **values),
            "format_minutes": lambda value: f"{int(value) // 60:02d}:{int(value) % 60:02d}",
            "public_url": self.public_url,
            "message": request.query_params.get("message"),
            "error": request.query_params.get("error"),
            "active_job_count": active_jobs,
            "recent_jobs": jobs,
            "inbox_count": await self.queries.inbox_count(),
        }
        return self.templates.TemplateResponse(request, template, base | context)

    async def _render_connections(self, request: Request, provider_draft: str = ""):
        custom_provider = self.store.custom_provider()
        model_view = self._model_view()
        draft = await self._provider_draft(provider_draft)
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
                    "capabilities": custom_provider.capabilities,
                    "probe_model": model_view["model"],
                    "has_api_key": bool(custom_provider.api_key),
                    "draft_token": "",
                }
                if custom_provider
                else None
            )
        )
        return await self._render(
            request,
            "settings_connections.html",
            "settings",
            "connections",
            chat_connected=self.account.connected,
            chat_email=self.account.email,
            chat_error=self._localized_error(self.account.last_error),
            active_provider=self._active_provider(),
            provider_label=self.models.provider_label,
            custom_provider=provider_view,
            discord_connected=self.discord.connected,
            discord_identity=self.discord.identity,
            discord_error=self._localized_error(self.discord.error),
            has_discord_token=self.store.discord_token() is not None,
            captcha_requests=self.discord.captcha_requests(),
        )

    @staticmethod
    def _diagnostic_versions() -> dict[str, str]:
        versions = {"Python": platform.python_version()}
        for label, distribution in (
            ("Diskovod", "diskovod"),
            ("LangChain", "langchain"),
            ("LangGraph", "langgraph"),
            ("langchain-openai", "langchain-openai"),
            ("discord.py-self", "discord.py-self"),
        ):
            try:
                versions[label] = importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError:
                versions[label] = "—"
        return versions

    @staticmethod
    def _automation_preset(settings: AutomationSettings) -> str:
        current = tuple(float(getattr(settings, field)) for field in AUTOMATION_TIMING_FIELDS)
        return next(
            (
                name
                for name, values in AUTOMATION_PRESETS.items()
                if all(
                    abs(actual - expected) < 1e-9 for actual, expected in zip(current, values, strict=True)
                )
            ),
            "custom",
        )

    def _checkpoint_view(self, metadata: dict[str, Any], snapshot, parent) -> dict[str, Any]:
        values = snapshot.checkpoint.get("channel_values", {})
        messages = list(values.get("messages") or [])
        parent_messages = (
            list(parent.checkpoint.get("channel_values", {}).get("messages") or [])
            if parent is not None
            else []
        )
        current_ids = {str(getattr(message, "id", "")) for message in messages}
        parent_ids = {str(getattr(message, "id", "")) for message in parent_messages}
        state = {
            str(key): redact_sensitive(self._serializable(value))
            for key, value in values.items()
            if key != "messages"
        }
        return {
            "metadata": metadata | {"source_label": self._checkpoint_source_label(metadata.get("source"))},
            "messages": [self._checkpoint_message(message) for message in messages],
            "state": state,
            "added_messages": len(current_ids - parent_ids),
            "removed_messages": len(parent_ids - current_ids),
            "checkpoint_metadata": redact_sensitive(self._serializable(snapshot.metadata)),
        }

    async def _prepare_chat_view(self, view: dict[str, Any]) -> None:
        view["runtime_ready"] = self.runtime.ready
        view["drafts"] = await self.runtime.publisher.drafts(
            channel_id=str(view["conversation"]["channel_id"]),
            limit=20,
        )
        configuration = view.get("configuration") or {}
        configuration["provider_label"] = self._provider_label(configuration.get("provider_id"))
        for generation in view.get("generations") or []:
            reason = str(generation.get("close_reason") or "")
            generation["close_reason_label"] = self._t(f"generation_reason_{reason}") if reason else ""
        for checkpoint in view.get("checkpoints") or []:
            checkpoint["source_label"] = self._checkpoint_source_label(checkpoint.get("source"))
        if not view["historical"] or not view["checkpoints"] or self.runtime.checkpointer is None:
            return
        checkpoint = view["checkpoints"][0]
        snapshot = await self.runtime.checkpointer.aget_tuple(
            {
                "configurable": {
                    "thread_id": checkpoint["thread_id"],
                    "checkpoint_id": checkpoint["checkpoint_id"],
                }
            }
        )
        if snapshot is None:
            return
        values = snapshot.checkpoint.get("channel_values", {})
        view["messages"] = [self._checkpoint_message(message) for message in values.get("messages") or []]
        view["older_messages_before"] = None

    def _checkpoint_message(self, message: Any) -> dict[str, Any]:
        additional = getattr(message, "additional_kwargs", {}) or {}
        participant = additional.get("diskovod_participant") or {}
        message_type = str(getattr(message, "type", type(message).__name__)).lower()
        role = str(
            participant.get("role") or {"human": "peer", "ai": "assistant"}.get(message_type, message_type)
        )
        content = getattr(message, "content", "")
        if not isinstance(content, str):
            content = json.dumps(
                redact_sensitive(self._serializable(content)),
                ensure_ascii=False,
                indent=2,
            )
        author = str(participant.get("name") or "")
        if not author:
            role_key = f"role_{role}"
            author = self._t(role_key) if self._t(role_key) != role_key else self._t("unknown_peer")
        return {
            "id": str(getattr(message, "id", "") or ""),
            "role": role,
            "author": author,
            "author_name": author,
            "content": content,
            "direction": "in" if role == "peer" else "out",
            "timestamp": None,
            "timestamp_label": "",
            "edited_at": None,
            "deleted_at": None,
            "attachments": [],
            "assistant_reaction": None,
        }

    @staticmethod
    def _serializable(value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if hasattr(value, "model_dump"):
            return value.model_dump(mode="json")
        if isinstance(value, dict):
            return {str(key): WebApp._serializable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [WebApp._serializable(item) for item in value]
        return str(value)

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
        return {
            "model": str(configuration.get("model_id") or model),
            "effort": str(configuration.get("options", {}).get("reasoning_effort") or "—"),
            "result_label": self._t(f"web_search_probe_result_{outcome}"),
            "response_id": str(report["id"]),
            "error": self._localized_error(report["conclusion"] if status == "error" else ""),
        }

    def _personality_view(self) -> dict[str, Any] | None:
        personality = self.store.personality()
        if not personality:
            return None
        source = str(personality.get("source") or "").lower().replace(" ", "_")
        personality["source_label"] = self._t(f"source_{source}") if source else self._t("inferred_history")
        return personality

    async def _database_view(self, table: str, page: int, query: str) -> dict:
        tables = await self.store.adatabase_tables()
        locale = self.store.interface_settings().locale
        for item in tables:
            label_key = f"table_{item['name']}"
            label = ui_text(locale, label_key)
            item["label"] = (
                ui_text(locale, "database_table_label", name=item["name"]) if label == label_key else label
            )
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

    def _job_view(self, job: dict[str, Any]) -> dict[str, Any]:
        item = dict(job)
        job_type = str(item.get("type") or "")
        progress_stage = str(item.get("progress_stage") or "")
        type_key = "job_type_" + job_type.replace(".", "_")
        stage_key = "job_stage_" + progress_stage
        item["type_label"] = self._t(type_key) if self._t(type_key) != type_key else job_type
        item["stage_label"] = (
            self._t(stage_key) if progress_stage and self._t(stage_key) != stage_key else progress_stage
        )
        item["status_label"] = self._t(f"status_{item.get('status')}")
        item["error_summary"] = self._localized_error(item.get("error_summary"))
        result_kind = str(item.get("result_kind") or "")
        result_id = str(item.get("result_id") or "")
        item["result_kind_label"] = self._t(f"result_kind_{result_kind}") if result_kind else ""
        item["result_url"] = (
            self._url(f"/activity/runs/{result_id}")
            if result_kind == "agent_run" and result_id
            else (
                self._url("/settings/assistant")
                if result_kind == "assistant_personality"
                else (
                    self._url("/system/diagnostics") if result_kind == "provider_capability_probe" else None
                )
            )
        )
        if item.get("target_kind") == "provider_setup_draft" and item.get("target_id"):
            item["target_url"] = (
                self._url("/settings/connections")
                + "?"
                + urlencode({"provider_draft": str(item["target_id"])})
            )
        else:
            item["target_url"] = None
        return item

    def _job_event_view(self, event: dict[str, Any]) -> dict[str, Any]:
        item = dict(event)
        kind = str(item.get("kind") or "")
        key = (
            f"status_{kind}"
            if kind in {"queued", "running", "cancellation_requested", "succeeded", "failed", "cancelled"}
            else f"job_event_{kind}"
        )
        item["kind_label"] = self._t(key) if self._t(key) != key else kind
        payload = dict(item.get("payload") or {})
        stage = str(payload.get("stage") or "")
        stage_key = f"job_stage_{stage}"
        if stage and self._t(stage_key) != stage_key:
            payload["stage"] = self._t(stage_key)
        item["payload"] = payload
        return item

    def _active_provider(self) -> str:
        configuration = self.models.configuration
        return "custom" if configuration and configuration.provider_id == "custom_openai" else "chatgpt"

    def _hosted_search_capability(self) -> bool | None:
        configuration = self.models.configuration
        return configuration.capabilities.hosted_web_search if configuration else None

    def _url(self, path: str) -> str:
        return self.public_url + "/" + path.lstrip("/")

    def _t(self, key: str, **values: object) -> str:
        locale = self.store.interface_settings().locale if self.store is not None else "en"
        return ui_text(locale, key, **values)

    def _localized_error(self, error: object) -> str:
        if not error:
            return ""
        detail = str(error)
        key = {
            "Discord must be connected before forcing a reply": "discord_required_force",
            "Invalid Discord channel ID": "discord_channel_invalid",
            "Discord conversation is not available": "discord_conversation_unavailable",
            "This conversation has no incoming message to answer": "discord_no_incoming_message",
            "The latest incoming Discord message ID is invalid": "discord_message_id_invalid",
            "Discord must be connected before loading message history": "discord_required_history",
            "Invalid or expired ChatGPT OAuth state": "oauth_state_invalid",
            "ChatGPT OAuth callback did not include an authorization code": "oauth_callback_code_missing",
            "ChatGPT OAuth service has not started": "oauth_service_not_started",
            "ChatGPT Subscription is not connected": "chatgpt_subscription_not_connected",
            "Model configuration cannot change while an agent run is active": "model_change_run_active",
            "Resolve active owner escalations before changing the model": "model_change_escalation_active",
            "Administrative job service is not ready": "admin_job_service_unavailable",
            "ChatGPT Subscription is not the selected model provider": "subscription_provider_not_selected",
            "Unknown or unavailable model provider": "unknown_model_provider",
            "Discord connection closed; retrying": "discord_connection_retrying",
            "The model configuration no longer exists": "job_error_configuration_missing",
            "The provider setup draft expired or does not exist": "job_error_setup_draft_expired",
            "The provider setup draft expired before the result could be saved": (
                "job_error_setup_draft_save_expired"
            ),
            "The encrypted personality input expired or does not exist": (
                "job_error_personality_input_expired"
            ),
            "Not enough representative message history was available": (
                "job_error_personality_history_insufficient"
            ),
            "Assistant settings changed while personality inference was running": (
                "job_error_assistant_profile_changed"
            ),
            "Administrative job lease was lost": "job_error_lease_expired",
            "Replay cancelled by owner": "replay_cancelled_by_owner",
        }.get(detail)
        return self._t(key) if key else self._t("operation_failed", detail=detail)

    def _provider_label(self, provider_id: object) -> str:
        provider = str(provider_id or "")
        key = {
            "chatgpt_subscription": "chatgpt_subscription",
            "custom_openai": "provider_custom_openai",
        }.get(provider)
        return self._t(key) if key else provider

    def _checkpoint_source_label(self, source: object) -> str:
        value = str(source or "")
        return self._t(f"checkpoint_source_{value}") if value else ""

    def _localize_configuration_versions(self, versions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for item in versions:
            configuration = item.get("configuration") or {}
            configuration["provider_label"] = self._provider_label(configuration.get("provider_id"))
        return versions

    def _localize_probe_page(self, page: dict[str, Any]) -> dict[str, Any]:
        for probe in page.get("items") or []:
            conclusion = str(probe.get("conclusion") or "")
            key = f"probe_conclusion_{conclusion}"
            probe["conclusion_label"] = (
                self._t(key)
                if conclusion and self._t(key) != key
                else self._t("probe_conclusion_request_error")
            )
        return page

    def _localize_run_view(self, view: dict[str, Any]) -> None:
        run = view["run"]
        run["status_label"] = self._t(f"status_{run['status']}")
        run["trigger_label"] = (
            self._t("force_reply")
            if run.get("trigger_kind") == "force_reply"
            else self._t(f"trigger_{run.get('trigger_kind')}")
        )
        run["error"] = self._localized_error(run.get("error"))
        configuration = run.get("configuration") or {}
        configuration["provider_label"] = self._provider_label(configuration.get("provider_id"))
        kind_keys = {
            "model_request": "model_request",
            "model_response": "model_response",
        }
        for event in view.get("timeline") or []:
            event["category_label"] = self._t(f"trace_category_{event['category']}")
            event["kind_label"] = self._t(kind_keys.get(event["kind"], f"trace_kind_{event['kind']}"))
            summary_key = str(event.get("summary_key") or "")
            event["summary_label"] = (
                self._t(summary_key, **(event.get("summary_values") or {}))
                if summary_key
                else str(event.get("summary") or "")
            )
        for checkpoint in view.get("checkpoints") or []:
            checkpoint["source_label"] = self._checkpoint_source_label(checkpoint.get("source"))
        for wait in view.get("waits") or []:
            wait["state_label"] = self._t(f"wait_state_{wait['state']}")
        for delivery in view.get("deliveries") or []:
            action = str(delivery.get("action") or "")
            state = str(delivery.get("state") or "")
            delivery["action_label"] = self._t(f"delivery_action_{action}")
            delivery["state_label"] = self._t(
                {
                    "failed": "status_failed",
                    "pending": "status_queued",
                    "dispatching": "status_running",
                    "succeeded": "delivery_state_completed",
                }.get(state, f"delivery_state_{state}")
            )

    def _localize_run_items(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for run in items:
            run["error"] = self._localized_error(run.get("error"))
        return items

    def _localize_run_page(self, page: dict[str, Any]) -> dict[str, Any]:
        self._localize_run_items(page.get("items") or [])
        return page

    def _database_url(self, table: str, page: int, query: str) -> str:
        parameters = {"table": table, "page": max(1, page)}
        if query:
            parameters["q"] = query
        return self._url("/system/database") + "?" + urlencode(parameters)

    def _database_back(
        self,
        table: str,
        query: str,
        *,
        message: str | None = None,
        error: str | None = None,
    ) -> RedirectResponse:
        parameters = {"table": table, "page": 1}
        if query:
            parameters["q"] = query
        if message:
            parameters["message"] = message
        if error:
            parameters["error"] = error
        return RedirectResponse(
            self._url("/system/database") + "?" + urlencode(parameters),
            status_code=303,
        )

    async def _provider_draft(self, token: str) -> ProviderDraft | None:
        if not token:
            return None
        stored = await self.store.aprovider_setup_draft(token)
        if stored is None:
            return None
        value = stored["payload"]
        capabilities = dict(value.get("capabilities") or {})
        capabilities["native_function_calls"] = bool(capabilities.pop("native_tools", False))
        return ProviderDraft(
            name=str(value.get("name") or ""),
            base_url=str(value.get("base_url") or ""),
            api_key=str(value.get("api_key") or ""),
            protocol=str(value.get("protocol") or "responses"),
            capabilities=capabilities,
            probe_model=str(value.get("probe_model") or ""),
            expires_at=float(stored["expires_at"]),
        )

    def _provider_draft_back(
        self,
        token: str,
        *,
        message: str | None = None,
        error: str | None = None,
    ) -> RedirectResponse:
        values = {
            "provider_draft": token,
            "message": message,
            "error": error,
        }
        query = urlencode({key: value for key, value in values.items() if value})
        return RedirectResponse(self._url("/settings/connections") + "?" + query, status_code=303)

    def _redirect(
        self,
        path: str,
        *,
        message: str | None = None,
        error: str | None = None,
    ) -> RedirectResponse:
        query = urlencode(
            {key: value for key, value in {"message": message, "error": error}.items() if value}
        )
        return RedirectResponse(self._url(path) + ("?" + query if query else ""), status_code=303)
