from __future__ import annotations

from pathlib import Path

import pytest

from diskovod.config import RuntimeConfig
from diskovod.main import build


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
