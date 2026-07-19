from __future__ import annotations

import argparse
from copy import deepcopy
import logging
from pathlib import Path
from typing import Any

import uvicorn
from uvicorn.config import LOGGING_CONFIG

from .config import RuntimeConfig
from .discord import DiscordService
from .http_client import PublicHTTPClient
from .migration import LegacyMigrator
from .oauth import ChatGPTAccount
from .providers import ModelService, ProviderSetup
from .runtime import AgentService
from .store import Store
from .web import WebApp


class _SuccessfulAccessLogFilter(logging.Filter):
    """Expose successful Uvicorn access records at DEBUG instead of INFO."""

    def filter(self, record: logging.LogRecord) -> bool:
        args = record.args
        if isinstance(args, tuple) and len(args) >= 5:
            status_code = args[4]
            if isinstance(status_code, int) and 200 <= status_code < 400:
                record.levelno = logging.DEBUG
                record.levelname = logging.getLevelName(logging.DEBUG)
        return True


def _uvicorn_log_config(log_level: str) -> dict[str, Any]:
    config = deepcopy(LOGGING_CONFIG)
    config["filters"] = {
        "successful_access": {"()": _SuccessfulAccessLogFilter},
    }
    config["loggers"]["uvicorn.access"]["filters"] = ["successful_access"]
    config["handlers"]["access"]["level"] = log_level
    return config


def build(
    config: RuntimeConfig,
) -> tuple[WebApp, Store, ChatGPTAccount, DiscordService, AgentService, PublicHTTPClient]:
    password = RuntimeConfig.read_secret(config.admin_password_file, "admin password", 12)
    secret = RuntimeConfig.read_secret(config.secret_key_file, "secret key", 32)
    store = Store(config.data_dir / "diskovod.sqlite3", secret)
    account = ChatGPTAccount(store)
    models = ModelService(store, account)
    provider_setup = ProviderSetup(store, models)
    discord = DiscordService(store)
    public_http = PublicHTTPClient()
    runtime = AgentService(store, models, discord, secret, public_http)
    discord.attach_runtime(runtime)
    return (
        WebApp(
            store,
            account,
            models,
            provider_setup,
            discord,
            runtime,
            password,
            config.public_url,
        ),
        store,
        account,
        discord,
        runtime,
        public_http,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Diskovod DM assistant")
    parser.add_argument("--config", type=Path, help="Path to the JSON configuration file")
    args = parser.parse_args()
    config = RuntimeConfig.load(args.config)
    logging.basicConfig(
        level=config.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    web, store, account, discord, runtime, public_http = build(config)

    @web.app.on_event("startup")
    async def startup() -> None:
        await store.start()
        await account.start()
        await web.models.migrate_legacy_selection()
        await runtime.start()
        await LegacyMigrator(store, runtime).run()
        await store.aprune()
        await discord.start()

    @web.app.on_event("shutdown")
    async def shutdown() -> None:
        await discord.stop()
        await runtime.close()
        await account.close()
        await public_http.close()
        await store.aclose()

    uvicorn.run(
        web.app,
        host=config.host,
        port=config.port,
        log_config=_uvicorn_log_config(config.log_level),
        proxy_headers=False,
    )


if __name__ == "__main__":
    main()
