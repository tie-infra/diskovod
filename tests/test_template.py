import re
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from diskovod.localization import SUPPORTED_LOCALES, ui_text
from diskovod.models import AppSettings


def test_admin_template_is_script_free_and_contains_human_quiet_controls():
    template_dir = Path(__file__).parents[1] / "diskovod" / "templates"
    environment = Environment(
        loader=FileSystemLoader(template_dir),
        autoescape=select_autoescape(["html"]),
    )
    template = environment.get_template("index.html")
    context = dict(
        app_settings=AppSettings(),
        model_view={"model": "gpt-5.4-mini", "reasoning_effort": "low", "max_output_tokens": 256},
        assistant_display_name="Diskovod",
        default_assistant_name="Diskovod",
        locale="en",
        locales=SUPPORTED_LOCALES,
        t=lambda key, **values: ui_text("en", key, **values),
        public_url="http://localhost:3090",
        chat_connected=True,
        chat_email=None,
        chat_error=None,
        subscription_web_search=False,
        subscription_web_search_probe={
            "model": "gpt-5.4-mini",
            "effort": "low",
            "checked_at_label": "2026-07-19 12:00:00 MSK",
            "result_label": "Inconclusive: response did not match the probe contract",
            "response_id": "resp-probe",
            "observed": "text_output=false; function_calls=0 []; connection_test_ok=false; hosted_calls=0 []",
            "error": "",
            "request_log_id": 42,
            "request_log_url": "http://localhost:3090/?tab=usage#model-request-42",
        },
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
                "mode": "automatic",
                "paused": False,
                "snoozed": True,
                "quiet_minutes_remaining": 12,
            }
        ],
        agent_runs=[
            {
                "id": "run-1",
                "started_at_label": "2026-07-19 12:00:00 MSK",
                "status": "failed",
                "channel_id": "dm-1",
                "error": "provider error",
                "traces": [
                    {
                        "sequence": 1,
                        "kind": "model_request",
                        "payload_json": '{"messages":[{"content":"hello"}]}',
                    }
                ],
            }
        ],
        capability_probes=[
            {
                "completed_at_label": "2026-07-19 11:00:00 MSK",
                "capability": "native_tools",
                "status": "supported",
                "conclusion": "client_tool_call_verified",
                "configuration_json": '{"model_id":"gpt-5"}',
                "request_payload_json": '{"tool_choice":"required"}',
                "response_payload_json": '{"tool_calls":["probe"]}',
            }
        ],
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
    rendered_by_tab = {
        tab: template.render(active_tab=tab, **context)
        for tab in ("connections", "assistant", "conversations", "usage", "database")
    }
    rendered = "\n".join(rendered_by_tab.values())

    assert "<script" not in rendered
    assert "bootstrap@5.3.8/dist/css/bootstrap.min.css" in rendered
    assert 'aria-label="Admin navigation"' in rendered
    assert 'href="/?tab=connections"' in rendered
    assert 'href="/?tab=assistant"' in rendered
    assert 'href="/?tab=usage"' in rendered
    assert 'href="/?tab=conversations"' in rendered
    assert 'href="/?tab=database"' in rendered
    assert 'name="admin_theme"' in rendered
    assert 'value="system" selected' in rendered
    assert 'value="light"' in rendered
    assert 'value="dark"' in rendered
    assert 'value="black"' in rendered
    assert "Black (OLED)" in rendered
    assert 'data-admin-theme="system"' in rendered
    assert "table table-dark" not in rendered
    assert 'action="/discord/connect"' in rendered_by_tab["connections"]
    assert 'action="/discord/connect"' not in rendered_by_tab["assistant"]
    assert 'action="/personality/save"' in rendered_by_tab["assistant"]
    assert 'action="/personality/save"' not in rendered_by_tab["connections"]
    assert 'class="d-grid gap-4 page-section"' in rendered_by_tab["assistant"]
    assert rendered_by_tab["assistant"].index('id="personality"') < rendered_by_tab["assistant"].index(
        'id="behavior"'
    )
    assert 'action="/conversations/dm-1/force-reply"' in rendered_by_tab["conversations"]
    assert 'action="/conversations/dm-1/mode"' in rendered_by_tab["conversations"]
    assert "Inline collaboration" in rendered_by_tab["conversations"]
    assert "Agent runs and exchanges" in rendered_by_tab["usage"]
    assert 'action="/database/delete"' in rendered_by_tab["database"]
    assert 'id="main-content"' in rendered
    assert 'name="min_human_quiet_minutes"' in rendered
    assert 'name="max_human_quiet_minutes"' in rendered
    assert 'name="admin_locale"' in rendered
    assert 'name="prompt_locale"' in rendered
    assert 'name="assistant_name"' in rendered
    assert 'placeholder="Diskovod"' in rendered
    assert "Leave blank to use the localized name Diskovod." in rendered
    assert "<title>Diskovod</title>" in rendered
    assert 'name="owner_timezone"' in rendered
    assert 'value="uk"' in rendered
    assert rendered.count('name="history_limit"') == 1
    assert 'name="max_reply_tokens"' in rendered
    assert re.search(r'<option\s+value="low"\s+selected', rendered)
    assert 'value="medium"' in rendered
    assert 'value="high"' in rendered
    assert "Reply token budget" in rendered
    assert "custom APIs receive a hard token limit" in rendered
    assert 'name="silent_replies"' in rendered
    assert 'name="robot_prefix"' in rendered
    assert 'name="min_message_gap_seconds"' in rendered
    assert 'name="max_message_gap_seconds"' in rendered
    assert 'name="conversation_default"' in rendered
    assert 'name="owner_details"' in rendered
    assert 'maxlength="20000"' in rendered
    assert 'value="opt_in"' in rendered
    assert 'value="opt_out"' in rendered
    assert "Send generated replies without notifications" in rendered
    assert "Uses Discord suppress-notifications" in rendered
    assert "Prefix generated replies with 🤖" in rendered
    assert 'action="/personality/save"' in rendered
    assert 'action="/personality/infer-history"' in rendered
    assert 'action="/settings/reset"' in rendered
    assert 'name="confirm" value="reset" required' in rendered
    assert "Reset assistant settings" in rendered
    assert 'action="/discord/connect"' in rendered
    assert 'action="/provider/custom"' in rendered
    assert 'formaction="/provider/custom/detect"' in rendered
    assert 'name="protocol"' in rendered
    assert 'name="native_function_calls"' in rendered
    assert "Detect API support" in rendered
    assert 'name="provider"' in rendered
    assert "http://localhost:8000/v1" in rendered
    assert "Local model" in rendered
    assert 'action="/discord/settings"' not in rendered
    assert 'name="api_base"' not in rendered
    assert 'action="/discord/captcha/captcha-id"' in rendered
    assert 'type="password"' in rendered
    assert "site-key" in rendered
    assert "Human active · 12 min" in rendered
    assert "http://localhost:3090/chatgpt/oauth/callback" in rendered
    assert "localhost:1455" in rendered
    assert "keep the complete query string" in rendered
    assert "Probe diagnostics" in rendered
    assert "resp-probe" in rendered
    assert "connection_test_ok=false" in rendered
    assert "fixed probe prompt and raw response" in rendered
    assert "Agent runs and exchanges" in rendered
    assert "provider error" in rendered
    assert "model_request" in rendered
    assert "Provider capability probes" in rendered
    assert "client_tool_call_verified" in rendered
    assert "Database explorer" in rendered
    assert 'action="/database/delete"' in rendered
    assert 'name="row_key"' in rendered
    assert 'action="/conversations/dm-1/force-reply"' in rendered
    assert "Force reply" in rendered
    assert "Owner escalations" in rendered
    assert 'action="/escalations/3/claim"' in rendered
    assert 'action="/escalations/3/resolve"' in rendered
    assert 'action="/escalations/3/dismiss"' in rendered


def test_oled_theme_uses_true_black_bootstrap_overrides():
    stylesheet = (Path(__file__).parents[1] / "diskovod" / "static" / "style.css").read_text()

    assert '[data-admin-theme="black"]' in stylesheet
    assert "--bs-body-bg: #000" in stylesheet
    assert "--bs-secondary-bg: #070707" in stylesheet


def test_system_theme_tracks_the_dark_color_scheme_without_javascript():
    stylesheet = (Path(__file__).parents[1] / "diskovod" / "static" / "style.css").read_text()

    assert "@media (prefers-color-scheme: dark)" in stylesheet
    assert '[data-admin-theme="system"]' in stylesheet
    assert "--bs-body-bg: #212529" in stylesheet
