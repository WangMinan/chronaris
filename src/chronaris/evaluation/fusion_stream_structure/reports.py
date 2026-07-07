"""Report and artifact writers for E3 fusion stream structure evaluation."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Mapping, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/chronaris-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from chronaris.evaluation.fusion_stream_structure.contracts import FusionStreamRecord
from chronaris.evaluation.fusion_stream_structure.metrics import metric_rows_to_frame


def write_e3_long_table(rows: Sequence[Mapping[str, object]], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = metric_rows_to_frame(rows).copy()
    if "details" in frame.columns:
        frame["details"] = [
            json.dumps(value, ensure_ascii=False, sort_keys=True, default=_json_default)
            for value in frame["details"]
        ]
    frame.to_csv(output_path, index=False)
    return output_path


def write_e3_summary(
    rows: Sequence[Mapping[str, object]],
    path: str | Path,
    *,
    extra: Mapping[str, object] | None = None,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = [row for row in rows if row.get("status") == "completed"]
    unavailable = [row for row in rows if row.get("status") != "completed"]
    summary = {
        "metric_row_count": len(rows),
        "completed_metric_row_count": len(completed),
        "unavailable_metric_row_count": len(unavailable),
        "training_invoked": False,
        "metrics_changed": False,
        "confirmed_metrics_changed": False,
    }
    if extra:
        summary.update(extra)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    return output_path


def write_evidence_manifest(manifest: Mapping[str, object], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    return output_path


def plot_state_timeline(
    record: FusionStreamRecord,
    clasp_result: Mapping[str, object],
    path: str | Path,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 2.8))
    status = str(clasp_result.get("status", "not_run"))
    times = np.asarray(record.times, dtype=float)
    if status == "completed" and clasp_result.get("state_sequence"):
        states = np.asarray(clasp_result["state_sequence"], dtype=float)
        ax.step(times[: states.size], states, where="post", label="state")
        for point in clasp_result.get("change_points", []):
            if 0 <= int(point) < len(times):
                ax.axvline(times[int(point)], color="#b23a48", linewidth=1.0, alpha=0.8)
        ax.set_ylabel("state")
        ax.legend(loc="upper right", fontsize=8)
    else:
        ax.text(0.5, 0.55, f"ClaSP/CLaP unavailable: {status}", ha="center", va="center", transform=ax.transAxes)
        ax.set_yticks([])
    ax.set_title(f"E3 state timeline: {record.method_name} / {_display_view(record)}")
    ax.set_xlabel("window time")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_transition_graph(
    record: FusionStreamRecord,
    clasp_result: Mapping[str, object],
    path: str | Path,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    graph = clasp_result.get("state_transition_graph", {})
    transitions = graph.get("transitions", []) if isinstance(graph, Mapping) else []
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    if transitions:
        labels = [f"{item['from']}->{item['to']}" for item in transitions]
        counts = [int(item["count"]) for item in transitions]
        ax.bar(labels, counts, color="#3b6ea8")
        ax.set_ylabel("transition count")
        ax.tick_params(axis="x", rotation=35)
    else:
        ax.text(0.5, 0.55, f"No transition graph: {clasp_result.get('status', 'not_run')}", ha="center", va="center", transform=ax.transAxes)
        ax.set_yticks([])
    ax.set_title(f"E3 transition graph: {record.method_name} / {_display_view(record)}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_fragment_replay(
    record: FusionStreamRecord,
    stumpy_result: Mapping[str, object],
    path: str | Path,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matrix = record.feature_matrix()
    signal = matrix[:, 0] if matrix.shape[1] else np.zeros(record.T)
    fig, ax = plt.subplots(figsize=(8, 2.8))
    ax.plot(record.times, signal, linewidth=1.0, color="#2a6f57", label="fusion feature 1")
    status = str(stumpy_result.get("status", "not_run"))
    if status == "completed":
        for label, color in (("discord_segment", "#b23a48"),):
            segment = stumpy_result.get(label, {})
            if isinstance(segment, Mapping) and "start_index" in segment:
                start = int(segment["start_index"])
                end = int(segment["end_index"])
                ax.axvspan(record.times[start], record.times[max(start, end - 1)], color=color, alpha=0.22, label=label)
        motif = stumpy_result.get("motif_pair", {})
        if isinstance(motif, Mapping):
            for side, color in (("left", "#3b6ea8"), ("right", "#f2b134")):
                segment = motif.get(side)
                if isinstance(segment, Mapping):
                    start = int(segment["start_index"])
                    end = int(segment["end_index"])
                    ax.axvspan(record.times[start], record.times[max(start, end - 1)], color=color, alpha=0.18, label=f"motif {side}")
    else:
        ax.text(0.5, 0.85, f"STUMPY unavailable: {status}", ha="center", va="center", transform=ax.transAxes)
    ax.set_title(f"E3 fragment replay: {record.method_name} / {_display_view(record)}")
    ax.set_xlabel("window time")
    ax.set_ylabel("feature")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def write_markdown_report(
    path: str | Path,
    *,
    run_id: str,
    summary: Mapping[str, object],
    manifest: Mapping[str, object],
    output_files: Sequence[str],
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    available_methods = ", ".join(manifest.get("available_methods", [])) or "无"
    unavailable = manifest.get("unavailable_methods", [])
    unavailable_text = "无"
    if unavailable:
        unavailable_text = "; ".join(
            f"{item.get('method_name')}={item.get('reason')}"
            for item in unavailable
            if isinstance(item, Mapping)
        )
    external = manifest.get("external_libraries", {})
    evaluator_counts = manifest.get("evaluator_status_counts", {})
    lines = [
        "# E3 融合表示流结构评价执行报告",
        "",
        f"- run_id: `{run_id}`",
        "- 评价定位：E3 是无监督结构诊断，用于观察融合表示流是否形成连续、稳定、可复盘的状态轨迹。",
        "- 论文边界：E3 不替代分类任务和回归任务；历史检索任务 artifact 未删除。",
        "- 本轮是否训练：`false`。",
        "- 是否修改 confirmed metrics：`false`。",
        f"- 可用方法：{available_methods}。",
        f"- 不可用方法：{unavailable_text}。",
        f"- 外部库状态：claspy={_external_text(external.get('claspy', {}))}；stumpy={_external_text(external.get('stumpy', {}))}。",
        f"- evaluator 状态：ClaSP={_status_count_text(evaluator_counts.get('clasp', {}))}；CLaP={_status_count_text(evaluator_counts.get('clap', {}))}；STUMPY={_status_count_text(evaluator_counts.get('stumpy', {}))}。",
        "",
        "## 输出文件",
        "",
    ]
    lines.extend(f"- `{item}`" for item in output_files)
    lines.extend([
        "",
        "## 结果边界",
        "",
        "本轮输出只进入 E3 专属 long 表，`evidence_quadrant = fusion_stream_structure`。Composite score 只作为固定权重汇总展示，不作为 winner 结论。若外部库缺失、短序列状态检测不足或方法无可复用融合流，对应指标保留为 unavailable，不静默删除。",
        "",
        "合成数据 run 只证明工程链路和 evaluator API 可执行；Dingxin 小规模 dry run 只证明现有可用融合流可以进入 E3 结构诊断流程，不能直接写成正式论文结论。",
        "",
        "## 当前摘要",
        "",
        f"- metric rows: {summary.get('metric_row_count')}",
        f"- completed metric rows: {summary.get('completed_metric_row_count')}",
        f"- unavailable metric rows: {summary.get('unavailable_metric_row_count')}",
    ])
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def _external_text(payload: object) -> str:
    if not isinstance(payload, Mapping):
        return "unknown"
    status = str(payload.get("status", "unknown"))
    version = payload.get("version")
    import_available = payload.get("import_available")
    if version:
        return f"{status}({version}, import_available={import_available})"
    return f"{status}(import_available={import_available})"


def _status_count_text(payload: object) -> str:
    if not isinstance(payload, Mapping) or not payload:
        return "unknown"
    return ", ".join(f"{key}:{value}" for key, value in sorted(payload.items()))


def _display_view(record: FusionStreamRecord) -> str:
    text = str(record.view_id)
    if text and text.isascii() and len(text) <= 64:
        return text
    return "Dingxin stream"
