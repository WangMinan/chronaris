"""Audit fixed Dingxin data and build fold-fitted application-task labels."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import asdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chronaris.access import MySQLCliRunner, MySQLSettings  # noqa: E402
from chronaris.evaluation.application_tasks.fixed_data_audit import (  # noqa: E402
    DEFAULT_E_MANIFEST,
    DEFAULT_F_MANIFEST,
    DEFAULT_RUN_ID,
    FixedDataAuditConfig,
    run_fixed_data_audit,
)
from chronaris.modeling.common.run_observer import (  # noqa: E402
    configure_task_eval_cli_logging,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--output-root", default=str(REPO_ROOT / "docs/artifacts/runs"))
    parser.add_argument("--e-run-manifest", default=str(REPO_ROOT / DEFAULT_E_MANIFEST))
    parser.add_argument("--f-run-manifest", default=str(REPO_ROOT / DEFAULT_F_MANIFEST))
    parser.add_argument("--bus-access-rule-id", type=int, default=6000019510066)
    parser.add_argument("--mysql-host", default=os.environ.get("CHRONARIS_MYSQL_HOST", "127.0.0.1"))
    parser.add_argument("--mysql-port", type=int, default=int(os.environ.get("CHRONARIS_MYSQL_PORT", "3306")))
    parser.add_argument("--mysql-database", default=os.environ.get("CHRONARIS_MYSQL_DATABASE", "rjgx_backend"))
    parser.add_argument("--mysql-user", default=os.environ.get("CHRONARIS_MYSQL_USER"))
    parser.add_argument("--mysql-password", default=os.environ.get("CHRONARIS_MYSQL_PASSWORD"))
    parser.add_argument("--mysql-binary", default="mysql")
    parser.add_argument("--secrets-path", default=str(REPO_ROOT / "docs/SECRETS.md"))
    parser.add_argument(
        "--allow-metadata-unavailable",
        action="store_true",
        help="Allow an incomplete smoke audit when MySQL field semantics are unavailable.",
    )
    return parser.parse_args()


def main() -> int:
    configure_task_eval_cli_logging(sys.stderr)
    args = parse_args()
    mysql_runner = _build_mysql_runner(args)
    result = run_fixed_data_audit(
        FixedDataAuditConfig(
            run_id=args.run_id,
            output_root=args.output_root,
            e_run_manifest_path=args.e_run_manifest,
            f_run_manifest_path=args.f_run_manifest,
            bus_access_rule_id=args.bus_access_rule_id,
            strict_mysql_field_labels=not args.allow_metadata_unavailable,
        ),
        mysql_runner=mysql_runner,
    )
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    return 0


def _build_mysql_runner(args: argparse.Namespace) -> MySQLCliRunner | None:
    user = args.mysql_user
    password = args.mysql_password
    if not user or not password:
        secrets_text = Path(args.secrets_path).read_text(encoding="utf-8")
        user = user or _extract_secret(secrets_text, "username")
        password = password or _extract_secret(secrets_text, "password")
    if not user or not password:
        if args.allow_metadata_unavailable:
            return None
        raise RuntimeError("MySQL user/password are required for strict field-semantic audit.")
    return MySQLCliRunner(
        settings=MySQLSettings(
            host=args.mysql_host,
            port=args.mysql_port,
            database=args.mysql_database,
            user=user,
            password_env=None,
            password_value=password,
        ),
        mysql_binary=args.mysql_binary,
    )


def _extract_secret(md_text: str, key: str) -> str | None:
    matched = re.search(rf"^\+?\s*{re.escape(key)}:\s*(.+)$", md_text, re.MULTILINE)
    return None if matched is None else matched.group(1).strip()


if __name__ == "__main__":
    raise SystemExit(main())
