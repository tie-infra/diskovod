from __future__ import annotations

import argparse
import logging
from pathlib import Path

import uvicorn

from .config import RuntimeConfig
from .discord import DiscordService
from .migration import LegacyMigrator
from .oauth import ChatGPTAccount
from .providers import ModelService, ProviderSetup
from .runtime import AgentService
from .store import Store
from .web import WebApp


def build(
    config: RuntimeConfig,
) -> tuple[WebApp, Store, ChatGPTAccount, DiscordService, AgentService]:
    password = RuntimeConfig.read_secret(config.admin_password_file, "admin password", 12)
    secret = RuntimeConfig.read_secret(config.secret_key_file, "secret key", 32)
    store = Store(config.data_dir / "diskovod.sqlite3", secret)
    account = ChatGPTAccount(store)
    models = ModelService(store, account)
    provider_setup = ProviderSetup(store, models)
    discord = DiscordService(store)
    runtime = AgentService(store, models, discord, secret)
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
    web, store, account, discord, runtime = build(config)

    @web.app.on_event("startup")
    async def startup() -> None:
        await account.start()
        web.models.migrate_legacy_selection()
        await runtime.start()
        await LegacyMigrator(store, runtime).run()
        store.prune()
        await discord.start()

    @web.app.on_event("shutdown")
    async def shutdown() -> None:
        await discord.stop()
        await runtime.close()
        await account.close()
        store.close()

    uvicorn.run(
        web.app,
        host=config.host,
        port=config.port,
        proxy_headers=False,
    )


if __name__ == "__main__":
    main()
