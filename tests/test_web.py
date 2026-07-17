from typing import cast

from diskovod.automation import Automation
from diskovod.chatgpt import ChatGPTClient
from diskovod.discord import DiscordService
from diskovod.store import Store
from diskovod.web import PERSONALITY_INSTRUCTIONS, WebApp


def test_auth_callbacks_and_redirects_use_public_url():
    web = WebApp(
        cast(Store, None),
        cast(ChatGPTClient, None),
        cast(DiscordService, None),
        cast(Automation, None),
        "a-long-admin-password",
        "https://diskovod.example/base",
    )

    route_paths = {route.path for route in web.app.routes}
    assert "/chatgpt/oauth/callback" in route_paths
    assert "/discord/connect" in route_paths
    assert "/discord/settings" not in route_paths
    assert "/discord/captcha/{request_id}" in route_paths
    assert web._url("/chatgpt/oauth/callback") == ("https://diskovod.example/base/chatgpt/oauth/callback")
    assert web._back(message="connected").headers["location"].startswith("https://diskovod.example/base/")


def test_personality_inference_requests_a_full_profile():
    for topic in ("communication habits", "languages", "preferences", "temperament", "stable traits"):
        assert topic in PERSONALITY_INSTRUCTIONS
