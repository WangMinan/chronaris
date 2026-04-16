"""MySQL access helpers backed by the local mysql CLI."""

from __future__ import annotations

import csv
import io
import os
import subprocess
from dataclasses import dataclass
from typing import Mapping, Protocol

from chronaris.access.settings import MySQLSettings


RowMapping = Mapping[str, str | None]


class SQLQueryRunner(Protocol):
    """A minimal query-runner protocol used by metadata readers."""

    def query(self, sql: str) -> tuple[RowMapping, ...]:
        """Run a SQL query and return zero or more row mappings."""

    def query_one(self, sql: str) -> RowMapping | None:
        """Run a SQL query and return at most one row mapping."""


@dataclass(frozen=True, slots=True)
class MySQLCliRunner:
    """Runs SQL against MySQL via the local mysql executable."""

    settings: MySQLSettings
    mysql_binary: str = "mysql"
    default_charset: str = "utf8mb4"

    def query(self, sql: str) -> tuple[RowMapping, ...]:
        completed = subprocess.run(
            self._build_command(sql),
            capture_output=True,
            check=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=self._build_env(),
        )
        return _parse_mysql_batch_output(completed.stdout)

    def query_one(self, sql: str) -> RowMapping | None:
        rows = self.query(sql)
        if not rows:
            return None
        return rows[0]

    def _build_command(self, sql: str) -> list[str]:
        return [
            self.mysql_binary,
            f"--host={self.settings.host}",
            f"--port={self.settings.port}",
            f"--user={self.settings.user}",
            f"--database={self.settings.database}",
            f"--default-character-set={self.default_charset}",
            "--batch",
            "--raw",
            "--execute",
            sql,
        ]

    def _build_env(self) -> dict[str, str]:
        env = os.environ.copy()
        env["MYSQL_PWD"] = self.settings.password
        return env


def parse_tsv_rowset(content: str) -> tuple[RowMapping, ...]:
    """Parse mysql --batch output into row mappings."""

    stripped = content.strip()
    if not stripped:
        return ()

    reader = csv.reader(io.StringIO(stripped), delimiter="\t")
    rows = list(reader)
    if not rows:
        return ()

    header = rows[0]
    result: list[RowMapping] = []

    for values in rows[1:]:
        padded_values = values + [""] * max(0, len(header) - len(values))
        row = {
            key: _normalize_mysql_value(value)
            for key, value in zip(header, padded_values)
        }
        result.append(row)

    return tuple(result)


def _parse_mysql_batch_output(content: str) -> tuple[RowMapping, ...]:
    return parse_tsv_rowset(content)


def _normalize_mysql_value(value: str) -> str | None:
    if value == "NULL" or value == "":
        return None
    return value
