from __future__ import annotations

import argparse
from copy import deepcopy
from logging.config import dictConfig
from pathlib import Path
from typing import Any

import uvicorn
from uvicorn.config import LOGGING_CONFIG

from .config import RuntimeConfig
from .admin_job_handlers import register_provider_jobs, register_runtime_jobs
from .admin_jobs import AdminJobRepository, AdminJobService, AdminJobWorker
from .discord import DiscordService
from .http_client import PublicHTTPClient
from .migration import LegacyMigrator
from .oauth import ChatGPTAccount
from .providers import ModelService, ProviderSetup
from .runtime import AgentService
from .store import Store
from .web import WebApp


def _logging_config(runtime_config: RuntimeConfig) -> dict[str, Any]:
    config = deepcopy(LOGGING_CONFIG)
    config["formatters"]["application"] = {
        "format": "%(asctime)s %(levelname)s %(name)s: %(message)s",
    }
    config["handlers"]["application"] = {
        "class": "logging.StreamHandler",
        "formatter": "application",
        "stream": "ext://sys.stderr",
    }
    config["root"] = {
        "handlers": ["application"],
        "level": runtime_config.log_level,
    }

    # Uvicorn supplies explicit INFO levels for these loggers. Make its child
    # loggers inherit the configured Uvicorn level unless specifically overridden.
    config["loggers"]["uvicorn"]["level"] = runtime_config.log_level
    config["loggers"]["uvicorn.error"]["level"] = "NOTSET"
    config["loggers"]["uvicorn.asgi"] = {"level": "NOTSET"}

    for logger_name, level in runtime_config.log_levels.items():
        logger_config = config["loggers"].setdefault(logger_name, {})
        logger_config["level"] = level

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
    jobs = AdminJobService(AdminJobRepository(store.database))
    register_provider_jobs(jobs, store, models, provider_setup)
    job_worker = AdminJobWorker(jobs)
    discord = DiscordService(store)
    public_http = PublicHTTPClient()
    runtime = AgentService(store, models, discord, secret, public_http)
    discord.attach_runtime(runtime)
    register_runtime_jobs(jobs, store, models, discord, runtime)
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
            jobs,
            job_worker,
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
    dictConfig(_logging_config(config))
    web, store, account, discord, runtime, public_http = build(config)

    @web.app.on_event("startup")
    async def startup() -> None:
        await store.start()
        await account.start()
        await web.models.migrate_legacy_selection()
        assert web.job_worker is not None
        await web.job_worker.start()
        await runtime.start()
        await LegacyMigrator(store, runtime).run()
        await store.aprune()
        await discord.start()

    @web.app.on_event("shutdown")
    async def shutdown() -> None:
        assert web.job_worker is not None
        await web.job_worker.close()
        await discord.stop()
        await runtime.close()
        await account.close()
        await public_http.close()
        await store.aclose()

    uvicorn.run(
        web.app,
        host=config.host,
        port=config.port,
        log_config=None,
        proxy_headers=False,
    )


if __name__ == "__main__":
    main()
