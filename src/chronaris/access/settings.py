"""Runtime settings for Chronaris store access."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _get_env(name: str, *, default: str | None = None, required: bool = False) -> str | None:
    value = os.getenv(name, default)
    if required and not value:
        raise RuntimeError(f"Required environment variable is missing: {name}")
    return value


@dataclass(frozen=True, slots=True)
class InfluxSettings:
    """Connection settings for InfluxDB access."""

    url: str
    org: str
    token_env: str | None = "CHRONARIS_INFLUX_TOKEN"
    token_value: str | None = None
    physiology_bucket: str | None = None
    vehicle_bucket: str | None = None
    timeout_seconds: int = 30

    @property
    def token(self) -> str:
        if self.token_value is not None:
            return self.token_value
        if self.token_env is None:
            raise RuntimeError("Influx token is not configured.")
        value = _get_env(self.token_env, required=True)
        assert value is not None
        return value

    @classmethod
    def from_env(cls, prefix: str = "CHRONARIS_INFLUX_") -> "InfluxSettings":
        return cls(
            url=_get_env(f"{prefix}URL", required=True) or "",
            org=_get_env(f"{prefix}ORG", required=True) or "",
            token_env=_get_env(f"{prefix}TOKEN_ENV", default="CHRONARIS_INFLUX_TOKEN") or "CHRONARIS_INFLUX_TOKEN",
            physiology_bucket=_get_env(f"{prefix}PHYSIOLOGY_BUCKET"),
            vehicle_bucket=_get_env(f"{prefix}VEHICLE_BUCKET"),
            timeout_seconds=int(_get_env(f"{prefix}TIMEOUT_SECONDS", default="30") or "30"),
        )

    @classmethod
    def from_java_properties(cls, properties_path: str | Path) -> "InfluxSettings":
        properties = _load_java_properties(properties_path)
        return cls(
            url=properties["influxdb.url"],
            org=properties["influxdb.org"],
            token_env=None,
            token_value=properties["influxdb.token"],
        )


@dataclass(frozen=True, slots=True)
class MySQLSettings:
    """Connection settings for MySQL metadata access."""

    host: str
    port: int
    database: str
    user: str
    password_env: str | None = "CHRONARIS_MYSQL_PASSWORD"
    password_value: str | None = None

    @property
    def password(self) -> str:
        if self.password_value is not None:
            return self.password_value
        if self.password_env is None:
            raise RuntimeError("MySQL password is not configured.")
        value = _get_env(self.password_env, required=True)
        assert value is not None
        return value

    @classmethod
    def from_env(cls, prefix: str = "CHRONARIS_MYSQL_") -> "MySQLSettings":
        return cls(
            host=_get_env(f"{prefix}HOST", required=True) or "",
            port=int(_get_env(f"{prefix}PORT", default="3306") or "3306"),
            database=_get_env(f"{prefix}DATABASE", required=True) or "",
            user=_get_env(f"{prefix}USER", required=True) or "",
            password_env=_get_env(f"{prefix}PASSWORD_ENV", default="CHRONARIS_MYSQL_PASSWORD")
            or "CHRONARIS_MYSQL_PASSWORD",
        )

    @classmethod
    def from_java_properties(
        cls,
        properties_path: str | Path,
        *,
        database: str,
    ) -> "MySQLSettings":
        properties = _load_java_properties(properties_path)
        jdbc_prefix = properties["mysql.url.prefix"]
        host_port = jdbc_prefix.removeprefix("jdbc:mysql://").rstrip("/")
        host, port = host_port.split(":", maxsplit=1)
        return cls(
            host=host,
            port=int(port),
            database=database,
            user=properties["mysql.username"],
            password_env=None,
            password_value=properties["mysql.password"],
        )


@dataclass(frozen=True, slots=True)
class AppSettings:
    """Top-level runtime settings for Chronaris data access."""

    influx: InfluxSettings
    mysql: MySQLSettings


def _load_java_properties(properties_path: str | Path) -> dict[str, str]:
    path = Path(properties_path)
    properties: dict[str, str] = {}

    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", maxsplit=1)
        properties[key.strip()] = value.strip()

    return properties
