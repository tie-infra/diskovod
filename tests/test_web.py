import hashlib
from typing import cast

import pytest
from fastapi import HTTPException
from fastapi.security import HTTPBasicCredentials
from starlette.requests import Request

from diskovod.automation import Automation
from diskovod.chatgpt import ChatGPTClient
from diskovod.discord import DiscordService
from diskovod.store import Store
from diskovod.web import PERSONALITY_INSTRUCTIONS, WebApp, personality_source_hash


def make_web(public_url: str = "https://diskovod.example/base") -> WebApp:
    return WebApp(
        cast(Store, None),
        cast(ChatGPTClient, None),
        cast(DiscordService, None),
        cast(Automation, None),
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
