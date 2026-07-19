from __future__ import annotations

from io import StringIO
import logging
from pathlib import Path

import pytest

from diskovod.config import RuntimeConfig
from diskovod.main import _SuccessfulAccessLogFilter, build


def _access_record(status_code: int) -> logging.LogRecord:
    return logging.LogRecord(
        "uvicorn.access",
        logging.INFO,
        __file__,
        1,
        '%s - "%s %s HTTP/%s" %d',
        ("127.0.0.1", "GET", "/", "1.1", status_code),
        None,
    )


def test_successful_access_logs_are_debug_but_errors_remain_info() -> None:
    success = _access_record(204)
    error = _access_record(404)

    log_filter = _SuccessfulAccessLogFilter()
    assert log_filter.filter(success)
    assert log_filter.filter(error)

    assert success.levelno == logging.DEBUG
    assert success.levelname == "DEBUG"
    assert error.levelno == logging.INFO


def test_info_access_handler_suppresses_success_after_filtering() -> None:
    output = StringIO()
    logger = logging.Logger("test.access", logging.INFO)
    logger.addFilter(_SuccessfulAccessLogFilter())
    handler = logging.StreamHandler(output)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    logger.handle(_access_record(200))
    logger.handle(_access_record(401))

    assert output.getvalue().splitlines() == ['127.0.0.1 - "GET / HTTP/1.1" 401']


@pytest.mark.asyncio
async def test_build_shares_one_public_http_client_across_untrusted_consumers(tmp_path: Path) -> None:
    password = tmp_path / "password"
    password.write_text("a-long-admin-password")
    secret = tmp_path / "secret"
    secret.write_text("x" * 32)
    config = RuntimeConfig(
        data_dir=tmp_path / "data",
        admin_password_file=password,
        secret_key_file=secret,
    )

    web, store, account, discord, runtime, public_http = build(config)

    assert web.runtime is runtime
    assert discord.runtime is runtime
    assert runtime.http is public_http
    assert runtime.attachments.http is public_http

    await runtime.close()
    await account.close()
    await public_http.close()
    store.close()
