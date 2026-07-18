from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from diskovod.localization import SUPPORTED_LOCALES
from diskovod.models import AppSettings
from diskovod.ui_localization import ui_text


def test_admin_template_is_script_free_and_contains_human_quiet_controls():
    template_dir = Path(__file__).parents[1] / "diskovod" / "templates"
    environment = Environment(
        loader=FileSystemLoader(template_dir),
        autoescape=select_autoescape(["html"]),
    )
    rendered = environment.get_template("index.html").render(
        app_settings=AppSettings(),
        locale="en",
        locales=SUPPORTED_LOCALES,
        t=lambda key, **values: ui_text("en", key, **values),
        public_url="http://localhost:3090",
        chat_connected=False,
        chat_email=None,
        chat_error=None,
        subscription_web_search=None,
        model_connected=True,
        automation_ready=True,
        automation_error=None,
        active_provider="custom",
        provider_label="Local model",
        custom_provider={
            "name": "Local model",
            "base_url": "http://localhost:8000/v1",
            "has_api_key": True,
            "protocol": "chat_completions",
            "capabilities": {
                "native_function_calls": True,
                "strict_function_schemas": True,
                "parallel_tool_control": True,
                "prompt_cache_key": True,
                "hosted_web_search": False,
            },
            "probe_model": "local-model",
            "draft_token": "",
        },
        discord_connected=False,
        discord_identity=None,
        discord_error=None,
        has_discord_token=False,
        captcha_requests=[
            {
                "id": "captcha-id",
                "service": "hcaptcha",
                "sitekey": "site-key",
                "rqdata": "request-data",
                "errors": [],
                "should_serve_invisible": False,
                "expires_in_seconds": 500,
            }
        ],
        personality=None,
        escalations=[
            {
                "id": 3,
                "channel_id": "dm-1",
                "peer_name": "Peer",
                "state": "pending",
                "reason": "peer_requested_owner",
                "reason_label": "peer requested the owner",
                "requested_at_label": "2026-07-19 12:00:00 MSK",
                "delivery_error": None,
            }
        ],
        conversations=[
            {
                "channel_id": "dm-1",
                "peer_name": "Peer",
                "paused": False,
                "snoozed": True,
                "quiet_minutes_remaining": 12,
            }
        ],
        usage_stats={
            "all_time": {
                "requests": 2,
                "input_tokens": 1234,
                "cached_input_tokens": 900,
                "output_tokens": 200,
                "reasoning_tokens": 50,
                "total_tokens": 1434,
                "average_tokens": 717,
                "cache_rate": 72.9,
            },
            "windows": [
                {
                    "label": "All time",
                    "requests": 2,
                    "input_tokens": 1234,
                    "cached_input_tokens": 900,
                    "output_tokens": 200,
                    "reasoning_tokens": 50,
                    "total_tokens": 1434,
                    "average_tokens": 717,
                    "cache_rate": 72.9,
                }
            ],
            "by_model": [
                {
                    "name": "gpt-5",
                    "requests": 2,
                    "input_tokens": 1234,
                    "cached_input_tokens": 900,
                    "output_tokens": 200,
                    "reasoning_tokens": 50,
                    "total_tokens": 1434,
                }
            ],
            "by_purpose": [
                {
                    "name": "dm_reply",
                    "requests": 2,
                    "input_tokens": 1234,
                    "cached_input_tokens": 900,
                    "output_tokens": 200,
                    "reasoning_tokens": 50,
                    "total_tokens": 1434,
                }
            ],
            "recent": [
                {
                    "recorded_at_label": "2026-07-17 12:00:00 MSK",
                    "model": "gpt-5",
                    "purpose": "dm_reply",
                    "input_tokens": 1234,
                    "cached_input_tokens": 900,
                    "output_tokens": 200,
                    "reasoning_tokens": 50,
                    "total_tokens": 1434,
                }
            ],
        },
        database={
            "name": "messages",
            "label": "Messages",
            "primary_key": "id",
            "read_only": False,
            "columns": ["id", "content"],
            "rows": [{"key": "message-1", "cells": ["message-1", "hello"]}],
            "total": 1,
            "query": "",
            "page": 1,
            "pages": 1,
            "previous_url": None,
            "next_url": None,
            "tables": [
                {
                    "name": "messages",
                    "label": "Messages",
                    "count": 1,
                    "selected": True,
                    "url": "/?db_table=messages#database",
                },
                {
                    "name": "config",
                    "label": "Configuration",
                    "count": 2,
                    "selected": False,
                    "url": "/?db_table=config#database",
                },
            ],
        },
        message=None,
        error=None,
    )

    assert "<script" not in rendered
    assert "bootstrap@5.3.8/dist/css/bootstrap.min.css" in rendered
    assert 'aria-label="Admin navigation"' in rendered
    assert 'href="#connections"' in rendered
    assert 'href="#personality"' in rendered
    assert 'href="#behavior"' in rendered
    assert 'href="#usage"' in rendered
    assert 'href="#escalations"' in rendered
    assert 'href="#conversations"' in rendered
    assert 'href="#database"' in rendered
    assert 'id="main-content"' in rendered
    assert 'name="min_human_quiet_minutes"' in rendered
    assert 'name="max_human_quiet_minutes"' in rendered
    assert 'name="admin_locale"' in rendered
    assert 'name="prompt_locale"' in rendered
    assert 'name="owner_timezone"' in rendered
    assert 'value="uk"' in rendered
    assert 'name="history_limit"' in rendered
    assert 'name="max_reply_tokens"' in rendered
    assert "Reply token budget" in rendered
    assert "does not accept <code>max_output_tokens</code>" in rendered
    assert 'name="silent_replies"' in rendered
    assert 'name="robot_prefix"' in rendered
    assert 'name="multi_message_replies"' in rendered
    assert 'name="max_reply_messages"' in rendered
    assert 'name="min_message_gap_seconds"' in rendered
    assert 'name="max_message_gap_seconds"' in rendered
    assert 'name="conversation_default"' in rendered
    assert 'name="owner_details"' in rendered
    assert 'maxlength="20000"' in rendered
    assert 'value="opt_in"' in rendered
    assert 'value="opt_out"' in rendered
    assert "Send generated replies without notifications" in rendered
    assert "suppress-notifications message option" in rendered
    assert "Prefix generated replies with 🤖" in rendered
    assert 'action="/personality/save"' in rendered
    assert 'action="/personality/infer-history"' in rendered
    assert 'action="/discord/connect"' in rendered
    assert 'action="/provider/custom"' in rendered
    assert 'formaction="/provider/custom/detect"' in rendered
    assert 'name="protocol"' in rendered
    assert 'name="native_function_calls"' in rendered
    assert "Detect API support" in rendered
    assert 'name="provider"' in rendered
    assert "http://localhost:8000/v1/chat/completions" in rendered
    assert "Local model" in rendered
    assert 'action="/discord/settings"' not in rendered
    assert 'name="api_base"' not in rendered
    assert 'action="/discord/captcha/captcha-id"' in rendered
    assert 'type="password"' in rendered
    assert "site-key" in rendered
    assert "Human active · 12 min" in rendered
    assert "http://localhost:3090/chatgpt/oauth/callback" in rendered
    assert "http://localhost:1455/auth/callback" in rendered
    assert "keep the complete query string" in rendered
    assert "Model token usage" in rendered
    assert "1,434" in rendered
    assert "Dm Reply" in rendered
    assert "2026-07-17 12:00:00 MSK" in rendered
    assert "Database explorer" in rendered
    assert 'action="/database/delete"' in rendered
    assert 'name="row_key"' in rendered
    assert 'action="/conversations/dm-1/force-reply"' in rendered
    assert "Force reply" in rendered
    assert "Owner escalations" in rendered
    assert 'action="/escalations/3/claim"' in rendered
    assert 'action="/escalations/3/resolve"' in rendered
    assert 'action="/escalations/3/dismiss"' in rendered
