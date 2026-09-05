"""Moved command entrypoints must still resolve this checkout."""

import ast
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def test_script_repository_roots():
    checked = 0
    for path in (REPO / "scripts").rglob("*.py"):
        for node in ast.parse(path.read_text()).body:
            if not isinstance(node, ast.Assign):
                continue
            if not any(isinstance(t, ast.Name) and t.id == "REPO_ROOT" for t in node.targets):
                continue
            value = eval(
                compile(ast.Expression(node.value), str(path), "eval"),
                {"Path": Path, "__file__": str(path)},
            )
            assert value == REPO, path
            checked += 1
    assert checked >= 24


def test_moved_cli_help_outside_checkout(tmp_path):
    for name in ("evidence/build_support.py", "runtime/run_demo.py"):
        result = subprocess.run(
            [sys.executable, "-I", str(REPO / "scripts" / name), "--help"],
            cwd=tmp_path, capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, result.stderr
        assert "usage:" in result.stdout
