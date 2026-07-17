from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


@dataclass(slots=True)
class RuntimeConfig:
    host: str = "::1"
    port: int = 3090
    public_url: str = "http://localhost:3090"
    data_dir: Path = Path("./data")
    log_level: str = "INFO"
    admin_password_file: Path | None = None
    secret_key_file: Path | None = None

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
            log_level=str(payload.get("log_level", "INFO")),
            admin_password_file=cls._path(payload.get("admin_password_file")),
            secret_key_file=cls._path(payload.get("secret_key_file")),
        )

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
