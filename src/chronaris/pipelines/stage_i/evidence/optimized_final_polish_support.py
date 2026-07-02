"""Small support helpers for P37 optimized final polish."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping


def infer_existing_nested_roots(run_root: Path, run_id: str) -> dict[str, str | None]:
    candidates = {
        "t3_screen_root": run_root / "nested_private" / f"{run_id}-t3-screen",
        "t1_screen_root": run_root / "nested_private" / f"{run_id}-t1-screen",
        "private_confirm_root": run_root / "nested_private" / f"{run_id}-private-confirm",
        "public_screen_root": run_root / "nested_public" / f"{run_id}-public-screen",
        "public_confirm_root": run_root / "nested_public" / f"{run_id}-public-confirm",
    }
    return {key: str(path) for key, path in candidates.items() if path.exists()}


def status_from_outputs(config: object, roots: Mapping[str, str | None]) -> str:
    if roots.get("private_confirm_root") and roots.get("public_confirm_root"):
        return "completed"
    if getattr(config, "run_private_confirm") and "private_confirm_root" not in roots:
        return "partial"
    if getattr(config, "run_public_confirm") and "public_confirm_root" not in roots:
        return "partial"
    return "completed" if getattr(config, "run_private_confirm") and getattr(config, "run_public_confirm") else "partial"
