import hashlib
import json
import time
from typing import cast

import pytest
from fastapi import HTTPException
from fastapi.security import HTTPBasicCredentials
from starlette.requests import Request

from diskovod.discord import DiscordService
from diskovod.store import Store
from diskovod.models import AppSettings
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
    assert "/settings/theme" in route_paths
    assert "/settings/reset" in route_paths
    assert "/discord/settings" not in route_paths
    assert "/discord/captcha/{request_id}" in route_paths
    assert "/database/delete" in route_paths
    assert "/conversations/{channel_id}/force-reply" in route_paths
    assert "/conversations/{channel_id}/mode" in route_paths
    assert "/escalations/{escalation_id}/claim" in route_paths
    assert "/escalations/{escalation_id}/resolve" in route_paths
    assert "/escalations/{escalation_id}/dismiss" in route_paths
    assert web._url("/chatgpt/oauth/callback") == ("https://diskovod.example/base/chatgpt/oauth/callback")
    assert web._back(message="connected").headers["location"].startswith("https://diskovod.example/base/")
    assert web._database_url("messages", 2, "hello world") == (
        "https://diskovod.example/base/?tab=database&db_table=messages&db_page=2&db_query=hello+world"
    )


def test_assistant_settings_defaults_preserve_only_admin_appearance():
    current = AppSettings(
        enabled=True,
        admin_locale="fr",
        admin_theme="black",
        prompt_locale="ja",
        assistant_name="Helper",
        provider="custom",
        model="custom-model",
        owner_details="Private details",
        base_instructions="Custom instructions",
    )

    reset = assistant_settings_defaults(current)

    assert reset == AppSettings(admin_locale="fr", admin_theme="black")


def test_subscription_web_search_probe_view_exposes_safe_debug_metadata(tmp_path):
    store = Store(tmp_path / "state.sqlite3", "x" * 32)
    with store._db:
        store._db.execute(
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

    view = web._subscription_web_search_probe_view("gpt-model")

    assert view["result_label"].startswith("Inconclusive")
    assert view["response_id"] == "probe-1"
    assert "web_search_call" in view["observed"]
    assert "access" not in str(view)
    store.close()


def test_model_request_log_view_correlates_validation_with_conversation(tmp_path):
    store = Store(tmp_path / "state.sqlite3", "x" * 32)
    store.upsert_conversation("dm-1", "peer", "Peer")
    request_id = store.start_model_request(
        provider="ChatGPT subscription",
        protocol="responses",
        model="gpt-model",
        purpose="dm_reply_tool_continuation",
        request_summary={"messages": [{"role": "user", "content_characters": 20}]},
        channel_id="dm-1",
        attempt=2,
        repair=True,
    )
    store.finish_model_request(
        request_id,
        status="completed",
        duration_ms=500,
        response_summary={"text_outputs": [{"characters": 10}]},
        request_payload={"instructions": "private prompt"},
        response_payload={"output": [{"type": "message", "text": "model output"}]},
    )
    store.annotate_model_request(
        request_id,
        "rejected",
        "non_terminal_or_ambiguous_output_after_repair",
        {"observed": {"response_text_present": True}},
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

    view = web._model_request_log_views()[0]

    assert view["conversation_label"] == "Peer"
    assert view["purpose_label"] == "DM tool continuation"
    assert view["validation_label"] == "Rejected"
    assert view["is_problem"] is True
    assert "content_characters" in view["request_json"]
    assert "characters" in view["response_json"]
    assert "private prompt" in view["request_payload_json"]
    assert "model output" in view["response_payload_json"]
    assert "response_text_present" in view["validation_summary_json"]
    store.close()


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
            "client": ("127.0.0.1", 1234),
            "server": ("diskovod.example", 443),
        },
        receive,
        send,
    )

    response_start = next(message for message in sent if message["type"] == "http.response.start")
    headers = {name.decode(): value.decode() for name, value in response_start["headers"]}
    assert response_start["status"] == 200
    assert "style-src 'self' https://cdn.jsdelivr.net" in headers["content-security-policy"]
    assert "script-src" not in headers["content-security-policy"]


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
