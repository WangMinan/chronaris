"""Shared CLI configuration retains overrides and missing-field failures."""

import pytest

from chronaris.access.settings import resolve_influx_settings, resolve_mysql_settings


def test_local_settings_precedence_and_address_policies(tmp_path, monkeypatch):
    for name in ("URL", "ORG", "TOKEN"):
        monkeypatch.delenv(f"CHRONARIS_INFLUX_{name}", raising=False)
    for name in ("HOST", "PORT", "USER", "PASSWORD"):
        monkeypatch.delenv(f"CHRONARIS_MYSQL_{name}", raising=False)
    path = tmp_path / "settings.md"
    path.write_text(
        "influxdb.url: http://file-host:8086\n+ influxdb.org: test-org\n"
        "influxdb.token: test-token\nhost: file-host\nport: 3307\n"
        "username: file-user\npassword: test-password\n"
    )
    assert resolve_influx_settings(path).url == "http://file-host:8086"
    assert resolve_mysql_settings("db", path).port == 3307
    local = resolve_mysql_settings("db", path, default_host="127.0.0.1", default_port=3306)
    assert (local.host, local.port, local.password) == ("127.0.0.1", 3306, "test-password")
    assert resolve_influx_settings(path, default_url="http://127.0.0.1:8086").url == "http://127.0.0.1:8086"
    for name, value in {"URL": "http://env:8086", "ORG": "env-org", "TOKEN": "env-token"}.items():
        monkeypatch.setenv(f"CHRONARIS_INFLUX_{name}", value)
    for name, value in {"HOST": "env", "PORT": "3308", "USER": "env-user", "PASSWORD": "env-password"}.items():
        monkeypatch.setenv(f"CHRONARIS_MYSQL_{name}", value)
    path.unlink()
    assert resolve_influx_settings(path).token == "env-token"
    assert resolve_mysql_settings("db", path).host == "env"
    assert resolve_mysql_settings("db", path).port == 3308
    monkeypatch.setenv("CHRONARIS_MYSQL_PORT", "invalid")
    with pytest.raises(ValueError):
        resolve_mysql_settings("db", path)
    monkeypatch.delenv("CHRONARIS_INFLUX_ORG")
    path.write_text("influxdb.token: test-token\n")
    with pytest.raises(RuntimeError, match="Missing secret key.*influxdb.org"):
        resolve_influx_settings(path)
