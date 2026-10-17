"""Portable observed-window packages; no database or training dependencies."""
from pathlib import Path
import json

import numpy as np

from chronaris.representation.collation import ObservedDualStreamSample
from chronaris.representation.contracts import ObservationSchema
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file
from chronaris.serving.fusion_workbench.presentation import display_text

MAX_WINDOWS = 512

STREAMS = ("physiology", "vehicle")
ARRAYS = tuple(f"{s}_{f}" for s in STREAMS for f in ("values", "timestamps_s", "feature_mask"))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def package_path(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("案例文件必须位于案例包内")
    return path


class Case:
    def __init__(self, path):
        self.root = Path(path).resolve().parent
        self.manifest = read_json(path)
        m = self.manifest
        if m.get("format") not in ("chronaris.demo_case.v1", "chronaris.observations.v1") or not 1 <= len(m["windows"]) <= MAX_WINDOWS:
            raise ValueError("观测记录格式或窗口数量不合法（1—512）")
        self.schema = ObservationSchema(**{
            k: tuple(v) if isinstance(v, list) else v for k, v in m["schema"].items()
        })
        self.checkpoint = package_path(self.root, m["checkpoint"])
        if sha256_file(self.checkpoint) != m["checkpoint_sha256"]:
            raise ValueError("冻结模型校验失败")
        ids = [w["sample_id"] for w in m["windows"]]
        if len(ids) != len(set(ids)):
            raise ValueError("案例窗口标识重复")
        for w in m["windows"]:
            if not np.isfinite(w["start_s"]) or w["start_s"] < 0:
                raise ValueError("窗口起点必须是有限非负秒数")

    def sample(self, index):
        w = self.manifest["windows"][index]
        path = package_path(self.root, w["file"])
        if sha256_file(path) != w["sha256"]:
            raise ValueError("输入窗口校验失败")
        with np.load(path, allow_pickle=False) as archive:
            arrays = {k: archive[k] for k in ARRAYS}
        if not np.isfinite(w["duration_s"]):
            raise ValueError("窗口时长不合法")
        return ObservedDualStreamSample(
            sample_id=w["sample_id"], group_id=w["group_id"], schema=self.schema,
            source_sample_hash=w["source_sample_hash"], context_duration_s=w["duration_s"], **arrays,
        )

    def summary(self):
        m = self.manifest
        return {"id": m["id"], "title": display_text(m["title"]), "description": display_text(m["description"]),
                "model_label": display_text(m["model_label"]), "checkpoint_sha256": m["checkpoint_sha256"],
                "source_label": display_text(m["source_label"]), "appendable": m.get("origin") == "uploaded", "field_counts": [len(self.schema.physiology_feature_names),
                len(self.schema.vehicle_feature_names)], "windows": [
                    {"label": f"窗口 {i + 1:02}", "start_s": w["start_s"], "duration_s": w["duration_s"]}
                    for i, w in enumerate(m["windows"])]}


def preview_stream(sample, stream, labels):
    values = getattr(sample, stream + "_values")
    mask = getattr(sample, stream + "_feature_mask")
    times = getattr(sample, stream + "_timestamps_s")
    names = getattr(sample.schema, stream + "_feature_names")
    preferred = [names.index(name) for name in labels if name in names and mask[:, names.index(name)].any()]
    available = list(np.flatnonzero(mask.any(axis=0)))
    columns = (preferred + [c for c in available if c not in preferred])[:3]
    traces = []
    for c in columns:
        indices = np.flatnonzero(mask[:, c])
        # shortcut: preview is capped at 240 observed points; downloads retain the complete representation.
        if len(indices) > 240:
            indices = indices[np.linspace(0, len(indices) - 1, 240, dtype=int)]
        traces.append({"label": display_text(labels.get(names[c], f"观测字段 {c + 1}")),
                       "points": [[float(times[i]), float(values[i, c])] for i in indices]})
    return {"observed_points": int(mask.any(axis=1).sum()), "coverage": float(mask.mean()) if mask.size else 0.,
            "traces": traces}
