from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


_LOG_LEVELS = frozenset({"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"})
_DEFAULT_COMPONENT_LOG_LEVELS = {"uvicorn.access": "WARNING"}


@dataclass(slots=True)
class RuntimeConfig:
    host: str = "::1"
    port: int = 3090
    public_url: str = "http://localhost:3090"
    data_dir: Path = Path("./data")
    log_level: str = "INFO"
    log_levels: dict[str, str] = field(default_factory=lambda: dict(_DEFAULT_COMPONENT_LOG_LEVELS))
    admin_password_file: Path | None = None
    secret_key_file: Path | None = None

    def __post_init__(self) -> None:
        self.log_level = self._log_level(self.log_level, "log_level")
        self.log_levels = self._component_log_levels(self.log_levels)

    @classmethod
    def load(cls, path: Path | None) -> "RuntimeConfig":
        payload: dict[str, Any] = {}
        if path is not None:
            payload = json.loads(path.read_text())
            if not isinstance(payload, dict):
                raise ValueError("The Diskovod JSON configuration must be an object")

        public_url = str(payload.get("public_url", "http://localhost:3090")).rstrip("/")
        parsed_public_url = urlparse(public_url)
        if parsed_public_url.scheme not in ("http", "https") or not parsed_public_url.netloc:
            raise ValueError("public_url must be an absolute HTTP(S) URL")
        if parsed_public_url.query or parsed_public_url.fragment:
            raise ValueError("public_url must not contain a query or fragment")

        return cls(
            host=str(payload.get("host", "::1")),
            port=int(payload.get("port", 3090)),
            public_url=public_url,
            data_dir=Path(payload.get("data_dir", "./data")),
            log_level=payload.get("log_level", "INFO"),
            log_levels=payload.get("log_levels", {}),
            admin_password_file=cls._path(payload.get("admin_password_file")),
            secret_key_file=cls._path(payload.get("secret_key_file")),
        )

    @staticmethod
    def _log_level(value: Any, setting: str) -> str:
        if not isinstance(value, str) or value.upper() not in _LOG_LEVELS:
            choices = ", ".join(sorted(_LOG_LEVELS))
            raise ValueError(f"{setting} must be one of: {choices}")
        return value.upper()

    @classmethod
    def _component_log_levels(cls, value: Any) -> dict[str, str]:
        if not isinstance(value, dict):
            raise ValueError("log_levels must be an object mapping logger names to levels")

        levels = dict(_DEFAULT_COMPONENT_LOG_LEVELS)
        for logger_name, level in value.items():
            if not isinstance(logger_name, str) or not logger_name.strip():
                raise ValueError("log_levels keys must be non-empty logger names")
            if logger_name == "root":
                raise ValueError("use log_level to configure the root logger")
            levels[logger_name] = cls._log_level(level, f"log_levels.{logger_name}")
        return levels

    @staticmethod
    def _path(value: Any) -> Path | None:
        return Path(value) if value else None

    @staticmethod
    def read_secret(path: Path | None, name: str, minimum_length: int = 1) -> str:
        if path is None:
            raise RuntimeError(f"{name} must be passed as a file path")
        value = path.read_text().strip()
        if len(value) < minimum_length:
            raise RuntimeError(f"{name} file must contain at least {minimum_length} characters")
        return value
