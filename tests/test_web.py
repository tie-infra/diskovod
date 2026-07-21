import hashlib
import json
import time
from typing import cast

import pytest
from fastapi import HTTPException
from fastapi.security import HTTPBasicCredentials
from starlette.requests import Request
from starlette.datastructures import FormData

from diskovod.discord import DiscordService
from diskovod.store import Store
from diskovod.models import AssistantProfile, AutomationSettings
from diskovod.web import (
    PERSONALITY_INSTRUCTIONS,
    WebApp,
    assistant_settings_defaults,
    personality_source_hash,
)


def make_web(public_url: str = "https://diskovod.example/base") -> WebApp:
    return WebApp(
        cast(Store, None),
        cast(object, None),
        cast(object, None),
        cast(object, None),
        cast(DiscordService, None),
        cast(object, None),
        "a-long-admin-password",
        public_url,
    )


def request_with_origin(origin: str, host: str) -> Request:
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/discord/connect",
            "headers": [
                (b"origin", origin.encode()),
                (b"host", host.encode()),
            ],
        }
    )


def test_auth_callbacks_and_redirects_use_public_url():
    web = make_web()

    route_paths = {route.path for route in web.app.routes}
    assert "/chatgpt/oauth/callback" in route_paths
    assert "/chatgpt/web-search/detect" in route_paths
    assert "/provider/custom" in route_paths
    assert "/provider/custom/detect" in route_paths
    assert "/provider/custom/remove" in route_paths
    assert "/provider/select" in route_paths
    assert "/discord/connect" in route_paths
    assert "/settings/theme" not in route_paths
    assert "/settings/reset" not in route_paths
    assert "/settings/assistant/reset" in route_paths
    assert "/settings/automation/reset" in route_paths
    assert "/settings/interface/reset" in route_paths
    assert "/inbox" in route_paths
    assert "/inbox/escalations/{escalation_id}" in route_paths
    assert "/search" in route_paths
    assert "/chats" in route_paths
    assert "/chats/{channel_id}" in route_paths
    assert "/chats/{channel_id}/generations/{generation}/checkpoints/{checkpoint_id}" in route_paths
    assert "/activity/runs" in route_paths
    assert "/activity/runs/{run_id}" in route_paths
    assert "/activity/runs/{run_id}/diagnostic.json" in route_paths
    assert "/activity/runs/{run_id}/deliveries/{action_id}/{operation}" in route_paths
    assert "/activity/jobs" in route_paths
    assert "/activity/jobs/{job_id}" in route_paths
    assert "/knowledge/memories" in route_paths
    assert "/knowledge/attachments" in route_paths
    assert "/settings/connections" in route_paths
    assert "/settings/model" in route_paths
    assert "/settings/assistant" in route_paths
    assert "/settings/automation" in route_paths
    assert "/settings/interaction" in route_paths
    assert "/settings/interaction/reset" in route_paths
    assert "/settings/interface" in route_paths
    assert "/system/diagnostics" in route_paths
    assert "/system/diagnostics.json" in route_paths
    assert "/system/database" in route_paths
    assert "/api/events/stream" in route_paths
    assert "/api/inbox" in route_paths
    assert "/api/chats" in route_paths
    assert "/api/chats/{channel_id}/messages" in route_paths
    assert "/api/chats/{channel_id}/timeline" in route_paths
    assert "/api/runs" in route_paths
    assert "/api/runs/{run_id}" in route_paths
    assert "/api/runs/{run_id}/events" in route_paths
    assert "/api/runs/{run_id}/events/{sequence}" in route_paths
    assert "/api/runs/{run_id}/delivery" in route_paths
    assert "/api/diagnostics/probes/{probe_id}" in route_paths
    assert "/api/jobs" in route_paths
    assert "/api/jobs/{job_id}" in route_paths
    assert "/api/search" in route_paths
    assert "/static/bootstrap.bundle.min.js" in route_paths
    assert "/discord/settings" not in route_paths
    assert "/discord/captcha/{request_id}" in route_paths
    assert "/system/database/delete" in route_paths
    assert "/chats/{channel_id}/force-reply" in route_paths
    assert "/chats/{channel_id}/interaction" in route_paths
    assert "/chats/{channel_id}/interaction/reset" in route_paths
    assert "/chats/{channel_id}/snooze" in route_paths
    assert "/chats/{channel_id}/snooze/clear" in route_paths
    assert "/inbox/escalations/{escalation_id}/claim" in route_paths
    assert "/inbox/escalations/{escalation_id}/resolve" in route_paths
    assert "/inbox/escalations/{escalation_id}/dismiss" in route_paths
    assert "/inbox/drafts/{draft_id}/approve" in route_paths
    assert "/inbox/drafts/{draft_id}/reject" in route_paths
    assert web._url("/chatgpt/oauth/callback") == ("https://diskovod.example/base/chatgpt/oauth/callback")
    assert (
        web._redirect("/settings/connections", message="connected")
        .headers["location"]
        .startswith("https://diskovod.example/base/settings/connections")
    )
    assert web._database_url("messages", 2, "hello world") == (
        "https://diskovod.example/base/system/database?table=messages&page=2&q=hello+world"
    )


def test_assistant_settings_defaults_reset_the_assistant_domain():
    reset = assistant_settings_defaults()

    assert reset == AssistantProfile()


def test_automation_presets_are_explicit_values_and_custom_changes_remain_custom():
    assert WebApp._automation_preset(AutomationSettings()) == "natural"
    assert WebApp._automation_preset(AutomationSettings(debounce_seconds=9)) == "custom"


async def test_interaction_form_saves_direct_address_without_requiring_reactions(tmp_path):
    store = await Store.open(tmp_path / "state.sqlite3", "x" * 32)
    web = make_web()
    web.store = store
    policy = web._interaction_policy_from_form(
        FormData(
            [
                ("preset", "on_invocation"),
                ("trigger_direct_address", "on"),
                ("use_assistant_name_alias", "on"),
                ("allow_bare_alias", "on"),
                ("trigger_participants", "peer"),
                ("active_turn_participants", "peer"),
                ("schedule_start", "09:00"),
                ("schedule_end", "17:00"),
            ]
        )
    )
    assert [rule.kind for rule in policy.trigger_rules] == ["direct_address"]
    await store.aclose()


async def test_subscription_probe_summary_defers_payload_to_diagnostics(tmp_path):
    store = await Store.open(tmp_path / "state.sqlite3", "x" * 32)
    async with store.database.transaction() as connection:
        await connection.execute(
            """INSERT INTO provider_capability_probes
               VALUES(?, ?, 'hosted_web_search', 'unsupported', ?, ?, ?, ?, ?)""",
            (
                "probe-1",
                json.dumps({"model_id": "gpt-model", "options": {"reasoning_effort": "low"}}),
                json.dumps({"tools": [{"type": "web_search"}]}),
                json.dumps({"content_blocks": [{"type": "web_search_call", "status": "failed"}]}),
                "no_hosted_search_blocks_in_normalized_response",
                time.time(),
                time.time(),
            ),
        )
    web = WebApp(
        store,
        cast(object, None),
        cast(object, None),
        cast(object, None),
        cast(DiscordService, None),
        cast(object, None),
        "a-long-admin-password",
        "https://diskovod.example",
    )

    view = await web._subscription_web_search_probe_view("gpt-model")

    assert view["result_label"].startswith("Inconclusive")
    assert view["response_id"] == "probe-1"
    assert "observed" not in view
    assert "access" not in str(view)
    detail = await web.queries.capability_probe("probe-1")
    assert detail["response_payload"]["content_blocks"][0]["type"] == "web_search_call"
    await store.aclose()


@pytest.mark.asyncio
async def test_security_policy_allows_only_the_pinned_bootstrap_stylesheet_origin():
    web = make_web()
    sent: list[dict] = []
    received = False

    async def receive():
        nonlocal received
        if not received:
            received = True
            return {"type": "http.request", "body": b"", "more_body": False}
        return {"type": "http.disconnect"}

    async def send(message: dict):
        sent.append(message)

    await web.app(
        {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "GET",
            "scheme": "https",
            "path": "/static/style.css",
            "raw_path": b"/static/style.css",
            "query_string": b"",
            "headers": [],
            "client": ("::1", 1234),
            "server": ("diskovod.example", 443),
        },
        receive,
        send,
    )

    response_start = next(message for message in sent if message["type"] == "http.response.start")
    headers = {name.decode(): value.decode() for name, value in response_start["headers"]}
    assert response_start["status"] == 200
    assert "style-src 'self'" in headers["content-security-policy"]
    assert "script-src 'self'" in headers["content-security-policy"]
    assert "cdn.jsdelivr.net" not in headers["content-security-policy"]


def test_public_origin_is_accepted_behind_reverse_proxy():
    web = make_web()
    request = request_with_origin("https://diskovod.example", "[::1]:3090")

    assert (
        web.require_admin(
            request,
            HTTPBasicCredentials(username="admin", password="a-long-admin-password"),
        )
        == "admin"
    )


def test_foreign_origin_is_rejected():
    web = make_web()
    request = request_with_origin("https://attacker.example", "diskovod.example")

    with pytest.raises(HTTPException, match="Cross-origin") as error:
        web.require_admin(
            request,
            HTTPBasicCredentials(username="admin", password="a-long-admin-password"),
        )

    assert error.value.status_code == 403


def test_origin_normalization_handles_default_ports_and_ipv6():
    assert WebApp._normalized_origin("https://EXAMPLE.com:443/path") == (
        "https",
        "example.com",
        443,
    )
    assert WebApp._normalized_origin("http://[::1]:3090") == ("http", "::1", 3090)


def test_personality_inference_requests_a_full_profile():
    for topic in (
        "base rates",
        "single-line",
        "Message sequencing",
        "consecutive-message bursts",
        "frequency and density",
        "languages",
        "preferences",
        "temperament",
        "Rare or context-dependent",
        "Representative examples",
        "synthetic examples, not samples",
    ):
        assert topic in PERSONALITY_INSTRUCTIONS


def test_personality_prompt_revision_invalidates_legacy_cache_key():
    samples = "representative message history"

    assert personality_source_hash(samples) != hashlib.sha256(samples.encode()).hexdigest()
    assert personality_source_hash(samples) == personality_source_hash(samples)
