from pathlib import Path
import re

from jinja2 import Environment, FileSystemLoader, select_autoescape

from diskovod.localization import SUPPORTED_LOCALES, ui_text
from diskovod.models import AutomationSettings, InterfaceSettings


def test_multipage_admin_templates_parse_and_use_the_shared_shell():
    template_dir = Path(__file__).parents[1] / "diskovod" / "templates"
    environment = Environment(
        loader=FileSystemLoader(template_dir),
        autoescape=select_autoescape(["html"]),
    )
    expected = {
        "overview.html",
        "inbox.html",
        "escalation.html",
        "search.html",
        "chats.html",
        "chat.html",
        "checkpoint.html",
        "runs.html",
        "run.html",
        "jobs.html",
        "job.html",
        "memories.html",
        "attachments.html",
        "settings_connections.html",
        "settings_model.html",
        "settings_assistant.html",
        "settings_automation.html",
        "settings_interface.html",
        "diagnostics.html",
        "database.html",
    }
    assert expected <= set(environment.list_templates())
    assert "index.html" not in environment.list_templates()
    for name in expected:
        environment.get_template(name)

    rendered = environment.get_template("base.html").render(
        locale="en",
        page_title="Overview",
        active_section="overview",
        interface_settings=InterfaceSettings(),
        automation_settings=AutomationSettings(),
        active_job_count=0,
        inbox_count=0,
        t=lambda key, **values: ui_text("en", key, **values),
    )
    assert 'href="/static/bootstrap.min.css"' in rendered
    assert 'src="/static/bootstrap.bundle.min.js"' in rendered
    assert 'src="/static/app.js"' in rendered
    assert 'href="/inbox"' in rendered
    assert 'href="/chats"' in rendered
    assert 'href="/activity/runs"' in rendered
    assert 'href="/knowledge/memories"' in rendered
    assert 'href="/settings/connections"' in rendered
    assert 'href="/system/diagnostics"' in rendered
    assert 'action="/search"' in rendered
    assert 'data-admin-theme="system"' in rendered
    assert 'id="main-content"' in rendered
    assert 'aria-current="page"' in rendered


def test_admin_pages_keep_settings_in_their_owned_domains():
    template_dir = Path(__file__).parents[1] / "diskovod" / "templates"
    assistant = (template_dir / "settings_assistant.html").read_text()
    automation = (template_dir / "settings_automation.html").read_text()
    interface = (template_dir / "settings_interface.html").read_text()
    model = (template_dir / "settings_model.html").read_text()

    assert 'name="prompt_locale"' in assistant
    assert 'name="assistant_name"' in assistant
    assert 'name="admin_locale"' not in assistant
    assert 'name="model"' not in assistant
    assert 'name="enabled"' in automation
    assert 'name="locale"' in interface
    assert 'name="theme"' in interface
    assert 'name="density"' in interface
    assert 'name="display_timezone_mode"' in interface
    assert 'name="named_timezone"' in interface
    assert 'name="preset"' in automation
    assert 'action="/personality/infer"' in assistant
    assert 'name="model"' in model
    assert 'name="reasoning_effort"' in model
    assert set(SUPPORTED_LOCALES) == {"en", "ru", "uk", "ja", "zh", "de", "fr"}


def test_live_updates_use_fetch_readable_stream_not_websockets_or_eventsource():
    script = (Path(__file__).parents[1] / "diskovod" / "static" / "app.js").read_text()

    assert "response.body.pipeThrough" in script
    assert "TextDecoderStream" in script
    assert "application/x-ndjson" in script
    assert "WebSocket" not in script
    assert "EventSource" not in script
    assert 'details[data-json-url]' in script
    assert "JSON.stringify(payload, null, 2)" in script
    assert "beforeunload" in script


def test_admin_templates_do_not_depend_on_csp_blocked_inline_javascript():
    template_dir = Path(__file__).parents[1] / "diskovod" / "templates"
    inline_handler = re.compile(r"\son(?:click|change|submit|input|load)\s*=", re.IGNORECASE)

    for template in template_dir.glob("*.html"):
        assert inline_handler.search(template.read_text()) is None, template.name


def test_admin_assets_are_self_hosted_and_old_dashboard_css_is_removed():
    static_dir = Path(__file__).parents[1] / "diskovod" / "static"
    bootstrap_bundle = (static_dir / "bootstrap.bundle.min.js").read_text()
    stylesheet = (static_dir / "style.css").read_text()

    assert "Bootstrap v5.3.8" in bootstrap_bundle
    assert ".admin-layout" in stylesheet
    assert ".chat-layout" in stylesheet
    assert ".tab-navigation" not in stylesheet
    assert ".usage-breakdown" not in stylesheet
    assert ".connection-grid" not in stylesheet


def test_diagnostic_pages_do_not_eagerly_render_raw_payloads():
    template_dir = Path(__file__).parents[1] / "diskovod" / "templates"
    run = (template_dir / "run.html").read_text()
    diagnostics = (template_dir / "diagnostics.html").read_text()

    assert "payload_pretty" not in run
    assert "request_pretty" not in run
    assert "request_payload_json" not in diagnostics
    assert "data-json-url" in run
    assert "data-json-url" in diagnostics
