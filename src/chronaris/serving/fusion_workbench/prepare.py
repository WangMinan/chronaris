"""Copy a bounded Dingxin validation case into an independent demo package."""
from pathlib import Path
import shutil

import torch

from chronaris.evaluation.application_tasks.dingxin_context_data import build_dingxin_lazy_context_index, dingxin_vehicle_field_labels
from chronaris.representation import DINGXIN_INCLUDE_MANEUVER_HISTORY_POLICY
from chronaris.serving.fusion_workbench.cases import ARRAYS, Case, write_json
from chronaris.simulation.aviation_dual_stream.deterministic_npz import write_deterministic_npz, sha256_file


def prepare_dingxin_case(*, checkpoint, snapshot_root, audit_root, destination):
    destination = Path(destination).resolve()
    if destination.exists():
        raise ValueError("案例目录已存在；请使用新的独立目录")
    checkpoint = Path(checkpoint).resolve()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (payload.get("training_status") != "completed" or payload.get("method_name") != "chronaris"
        or payload.get("implementation_revision") != "causal_fusion_v4"
        or payload.get("label_used_for_encoder_training")):
        raise ValueError("请选择已完成的当前 Chronaris 自监督冻结模型")
    ids = payload["fold"]["validation_sample_ids"][:3]
    if not ids:
        raise ValueError("模型没有内部验证窗口")
    audit_root = Path(audit_root)
    index = build_dingxin_lazy_context_index(snapshot_root=snapshot_root,
        field_role_manifest_path=audit_root / "field_role_manifest.csv",
        context_manifest_path=audit_root / "context_sample_manifest.jsonl",
        maneuver_history_policy=DINGXIN_INCLUDE_MANEUVER_HISTORY_POLICY)
    for stream in ("physiology", "vehicle"):
        if tuple(payload[stream + "_feature_names"]) != getattr(index.plan.schema, stream + "_feature_names"):
            raise ValueError("数据字段与模型不兼容")
    field_labels = {name: "脑电 " + name.rsplit(".", 1)[-1].upper() + "（原始标度）"
                    for name in index.plan.schema.physiology_feature_names if ".eeg." in name}
    vehicle_labels = dingxin_vehicle_field_labels(index, field_role_manifest_path=audit_root / "field_role_manifest.csv")
    for keyword in ("载机俯仰角", "载机横滚角", "载机真航向"):
        for name, label in vehicle_labels:
            if keyword in label:
                field_labels[name] = keyword + ("（毫弧度）" if "毫弧度" in label else "（原始标度）")
                break
    destination.mkdir(parents=True)
    before = sha256_file(checkpoint)
    shutil.copyfile(checkpoint, destination / "encoder.pt")
    if sha256_file(destination / "encoder.pt") != before:
        raise ValueError("复制模型时来源发生变化")
    windows = []
    for i, sample_id in enumerate(ids):
        sample = index.load_sample(sample_id)
        filename = f"window-{i + 1:02}.npz"
        digest = write_deterministic_npz(destination / filename, {k: getattr(sample, k) for k in ARRAYS})
        context = index.context_by_id[sample_id]
        windows.append({"file": filename, "sha256": digest, "sample_id": sample.sample_id,
            "group_id": sample.group_id, "source_sample_hash": sample.source_sample_hash,
            "start_s": float(context["start_offset_ms"]) / 1000, "duration_s": sample.context_duration_s})
    write_json(destination / "case.json", {"format": "chronaris.demo_case.v1", "id": "dingxin-validation",
        "title": "鼎新 · 人机双流离线案例", "source_label": "鼎新实测 · 内部验证片段",
        "description": "选取三个相邻的 30 秒观测窗口，复现冻结模型的连续对齐与融合特征输出。",
        "model_label": "Chronaris 连续融合 · 冻结自监督模型", "checkpoint": "encoder.pt",
        "checkpoint_sha256": before, "schema": index.plan.schema.to_dict(), "windows": windows,
        "field_labels": field_labels, "source": {"checkpoint": str(checkpoint), "snapshot": str(Path(snapshot_root).resolve()),
            "selection": "first_three_internal_validation_windows", "seed": payload["seed"],
            "model_config": {k: payload.get("encoder_manifest", {}).get("backbone_config", {}).get(k)
                             for k in ("variant", "fusion_kind", "semantic_event_enabled", "attention_kind")}}})
    return Case(destination / "case.json")
