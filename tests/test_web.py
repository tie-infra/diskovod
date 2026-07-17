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
    assert "/provider/custom" in route_paths
    assert "/provider/custom/remove" in route_paths
    assert "/provider/select" in route_paths
    assert "/discord/connect" in route_paths
    assert "/discord/settings" not in route_paths
    assert "/discord/captcha/{request_id}" in route_paths
    assert web._url("/chatgpt/oauth/callback") == ("https://diskovod.example/base/chatgpt/oauth/callback")
    assert web._back(message="connected").headers["location"].startswith("https://diskovod.example/base/")


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
