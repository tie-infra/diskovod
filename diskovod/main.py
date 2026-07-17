from __future__ import annotations

import argparse
import logging
from pathlib import Path

import uvicorn

from .automation import Automation
from .chatgpt import ChatGPTClient
from .config import RuntimeConfig
from .discord import DiscordService
from .store import Store
from .web import WebApp


def build(config: RuntimeConfig) -> tuple[WebApp, Store, ChatGPTClient, DiscordService, Automation]:
    password = RuntimeConfig.read_secret(config.admin_password_file, "admin password", 12)
    secret = RuntimeConfig.read_secret(config.secret_key_file, "secret key", 32)
    store = Store(config.data_dir / "diskovod.sqlite3", secret)
    chatgpt = ChatGPTClient(store)
    automation = Automation(store, chatgpt)
    discord = DiscordService(store, automation)
    return (
        WebApp(store, chatgpt, discord, automation, password, config.public_url),
        store,
        chatgpt,
        discord,
        automation,
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
    web, store, chatgpt, discord, automation = build(config)

    @web.app.on_event("startup")
    async def startup() -> None:
        await chatgpt.start()
        store.prune()
        await discord.start()

    @web.app.on_event("shutdown")
    async def shutdown() -> None:
        await automation.close()
        await discord.stop()
        await chatgpt.close()
        store.close()

    uvicorn.run(
        web.app,
        host=config.host,
        port=config.port,
        proxy_headers=False,
    )


if __name__ == "__main__":
    main()
