from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from diskovod.models import AppSettings


def test_admin_template_is_script_free_and_contains_human_quiet_controls():
    template_dir = Path(__file__).parents[1] / "diskovod" / "templates"
    environment = Environment(
        loader=FileSystemLoader(template_dir),
        autoescape=select_autoescape(["html"]),
    )
    rendered = environment.get_template("index.html").render(
        app_settings=AppSettings(),
        public_url="http://localhost:3090",
        chat_connected=False,
        chat_email=None,
        chat_error=None,
        model_connected=True,
        active_provider="custom",
        provider_label="Local model",
        custom_provider={
            "name": "Local model",
            "base_url": "http://localhost:8000/v1",
            "has_api_key": True,
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
        message=None,
        error=None,
    )

    assert "<script" not in rendered
    assert 'name="min_human_quiet_minutes"' in rendered
    assert 'name="max_human_quiet_minutes"' in rendered
    assert 'name="history_limit"' in rendered
    assert 'name="max_reply_tokens"' in rendered
    assert "Reply token budget" in rendered
    assert "does not accept <code>max_output_tokens</code>" in rendered
    assert 'name="silent_replies"' in rendered
    assert 'name="conversation_default"' in rendered
    assert 'name="owner_details"' in rendered
    assert 'maxlength="20000"' in rendered
    assert 'value="opt_in"' in rendered
    assert 'value="opt_out"' in rendered
    assert "Prefix generated replies with <code>@silent</code>" in rendered
    assert 'action="/personality/save"' in rendered
    assert 'action="/personality/infer-history"' in rendered
    assert 'action="/discord/connect"' in rendered
    assert 'action="/provider/custom"' in rendered
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
