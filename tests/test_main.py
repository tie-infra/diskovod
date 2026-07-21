from __future__ import annotations

from pathlib import Path

import pytest

from diskovod.config import RuntimeConfig
from diskovod.main import _logging_config, build


def test_logging_config_applies_root_and_component_levels() -> None:
    config = _logging_config(
        RuntimeConfig(
            log_level="INFO",
            log_levels={
                "uvicorn": "DEBUG",
                "diskovod.runtime": "DEBUG",
            },
        )
    )
    assert config["root"]["level"] == "INFO"
    assert config["loggers"]["uvicorn"]["level"] == "DEBUG"
    assert config["loggers"]["uvicorn.error"]["level"] == "NOTSET"
    assert config["loggers"]["uvicorn.access"]["level"] == "WARNING"
    assert config["loggers"]["diskovod.runtime"]["level"] == "DEBUG"
    assert "level" not in config["handlers"]["access"]
    assert "filters" not in config["loggers"]["uvicorn.access"]


def test_access_log_level_can_be_overridden() -> None:
    config = _logging_config(RuntimeConfig(log_levels={"uvicorn.access": "INFO"}))
    assert config["loggers"]["uvicorn.access"]["level"] == "INFO"


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
    await store.aclose()
