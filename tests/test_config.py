import json
from pathlib import Path

import pytest

from diskovod.config import RuntimeConfig


def test_ipv6_first_defaults():
    config = RuntimeConfig.load(None)
    assert config.host == "::1"
    assert config.port == 3090
    assert config.public_url == "http://localhost:3090"
    assert config.log_level == "INFO"
    assert config.log_levels == {"uvicorn.access": "WARNING"}


def test_json_configuration_and_secret_paths(tmp_path: Path):
    password_file = tmp_path / "password"
    password_file.write_text("a-long-admin-password\n")
    secret_file = tmp_path / "secret"
    secret_file.write_text("x" * 32)
    config_file = tmp_path / "diskovod.json"
    config_file.write_text(
        json.dumps(
            {
                "host": "::",
                "port": 8443,
                "public_url": "https://diskovod.example/base/",
                "data_dir": str(tmp_path / "state"),
                "log_level": "warning",
                "log_levels": {"uvicorn": "debug", "diskovod.runtime": "ERROR"},
                "admin_password_file": str(password_file),
                "secret_key_file": str(secret_file),
            }
        )
    )

    config = RuntimeConfig.load(config_file)
    assert config.host == "::"
    assert config.port == 8443
    assert config.public_url == "https://diskovod.example/base"
    assert config.log_level == "WARNING"
    assert config.log_levels == {
        "uvicorn.access": "WARNING",
        "uvicorn": "DEBUG",
        "diskovod.runtime": "ERROR",
    }
    assert config.admin_password_file == password_file
    assert config.secret_key_file == secret_file
    assert RuntimeConfig.read_secret(config.admin_password_file, "password", 12) == "a-long-admin-password"


def test_secret_file_environment_variables_are_ignored(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DISKOVOD_ADMIN_PASSWORD_FILE", "/run/secrets/ignored")
    config = RuntimeConfig.load(None)
    assert config.admin_password_file is None


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"log_level": "verbose"}, "log_level"),
        ({"log_levels": []}, "log_levels"),
        ({"log_levels": {"uvicorn": "verbose"}}, "log_levels.uvicorn"),
        ({"log_levels": {"root": "DEBUG"}}, "use log_level"),
    ],
)
def test_logging_configuration_is_validated(tmp_path: Path, payload: object, message: str):
    path = tmp_path / "diskovod.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=message):
        RuntimeConfig.load(path)


@pytest.mark.parametrize("public_url", ["localhost:3090", "ftp://localhost/app", "https://host/app?q=1"])
def test_public_url_must_be_an_absolute_http_url(tmp_path: Path, public_url: str):
    path = tmp_path / "diskovod.json"
    path.write_text(json.dumps({"public_url": public_url}))
    with pytest.raises(ValueError, match="public_url"):
        RuntimeConfig.load(path)
