"""Small run-observer utilities for long Stage I jobs."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import IO, Mapping


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(slots=True)
class StageIRunProgress:
    """Persist lightweight progress snapshots for disconnect-safe runs."""

    run_root: Path
    run_id: str
    stage_name: str
    progress_path: Path = field(init=False)
    state: dict[str, object] = field(default_factory=dict)
    events: list[dict[str, object]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.progress_path = self.run_root / "progress.json"

    def start(self, **fields: object) -> None:
        self.update("start", **fields)

    def update(self, event: str, **fields: object) -> None:
        record = {
            "timestamp_utc": _utc_now(),
            "event": event,
            **fields,
        }
        self.events.append(record)
        self.state.update(fields)
        self.state["stage_name"] = self.stage_name
        self.state["run_id"] = self.run_id
        self.state["last_event"] = event
        self.state["updated_at_utc"] = record["timestamp_utc"]
        self.state["events"] = self.events[-200:]
        self.progress_path.write_text(
            json.dumps(self.state, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def finish(self, **fields: object) -> None:
        self.update("finished", **fields)

    def fail(self, error: BaseException) -> None:
        self.update(
            "failed",
            error_type=type(error).__name__,
            error_message=str(error),
        )


class StageIRunObserver:
    """Context manager that writes `run.log` and `progress.json` under a run root."""

    def __init__(
        self,
        *,
        run_root: str | Path,
        run_id: str,
        stage_name: str,
        logger: logging.Logger,
        initial_progress: Mapping[str, object] | None = None,
    ) -> None:
        self.run_root = Path(run_root)
        self.run_id = run_id
        self.stage_name = stage_name
        self.logger = logger
        self.initial_progress = dict(initial_progress or {})
        self.progress = StageIRunProgress(
            run_root=self.run_root,
            run_id=run_id,
            stage_name=stage_name,
        )
        self._handler: logging.Handler | None = None
        self._namespace_logger = logging.getLogger("chronaris.pipelines.stage_i")
        self._namespace_level: int | None = None
        self._namespace_propagate: bool | None = None

    def __enter__(self) -> StageIRunProgress:
        self.run_root.mkdir(parents=True, exist_ok=True)
        self._handler = logging.FileHandler(
            self.run_root / "run.log",
            mode="a",
            encoding="utf-8",
        )
        self._handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        self._namespace_level = self._namespace_logger.level
        self._namespace_propagate = self._namespace_logger.propagate
        self._namespace_logger.setLevel(logging.INFO)
        self._namespace_logger.propagate = False
        self._namespace_logger.addHandler(self._handler)
        self.progress.start(**self.initial_progress)
        self.logger.info(
            "%s observer start run_id=%s run_root=%s",
            self.stage_name,
            self.run_id,
            self.run_root,
        )
        return self.progress

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        del exc_type, tb
        if exc is not None:
            self.progress.fail(exc)
            self.logger.exception("%s failed run_id=%s", self.stage_name, self.run_id)
        if self._handler is not None:
            self._namespace_logger.removeHandler(self._handler)
            self._handler.close()
        if self._namespace_level is not None:
            self._namespace_logger.setLevel(self._namespace_level)
        if self._namespace_propagate is not None:
            self._namespace_logger.propagate = self._namespace_propagate
        return False


def open_stage_i_run_observer(
    *,
    run_root: str | Path,
    run_id: str,
    stage_name: str,
    logger: logging.Logger,
    initial_progress: Mapping[str, object] | None = None,
) -> StageIRunObserver:
    return StageIRunObserver(
        run_root=run_root,
        run_id=run_id,
        stage_name=stage_name,
        logger=logger,
        initial_progress=initial_progress,
    )


def configure_stage_i_cli_logging(stream: IO[str]) -> None:
    """Mirror Stage I INFO logs to a CLI stream without relying on root logging."""

    namespace_logger = logging.getLogger("chronaris.pipelines.stage_i")
    namespace_logger.setLevel(logging.INFO)
    for handler in namespace_logger.handlers:
        if getattr(handler, "_chronaris_cli_handler", False):
            return
    handler = logging.StreamHandler(stream)
    handler._chronaris_cli_handler = True  # type: ignore[attr-defined]
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    namespace_logger.addHandler(handler)
