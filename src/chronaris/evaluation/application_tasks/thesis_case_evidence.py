"""Input-rule-selected Dingxin cases for bounded mechanism attribution."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from chronaris.evaluation.application_tasks.dingxin_fold_pretraining_data import (
    load_dingxin_fold_pretraining_data,
)
from chronaris.evaluation.application_tasks.thesis_native_outer_run import (
    _missing_provider,
)
from chronaris.modeling.fusion_encoders.chronaris_physics import physics_audit_to_rows
from chronaris.modeling.fusion_encoders.single_stream import move_observation_batch
from chronaris.modeling.training import (
    TrainedFusionAdapter,
    load_common_pretraining_checkpoint,
)
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file


REPO = Path(__file__).resolve().parents[4]
SNAPSHOT = REPO / "artifacts/application_evaluation/2026-07-10_dingxin-input-snapshot"
FIXED_AUDIT = REPO / "docs/artifacts/runs/2026-07-10_fixed-data-audit"
INNER_SPLIT = REPO / "docs/artifacts/runs/2026-07-11_dingxin-inner-splits"
OUTER_COMPACT = (
    REPO / "docs/artifacts/runs/2026-09-03_thesis-dingxin-confirmation-v3p2p2"
)
FOLDS = ("leave_one_sortie_out__fold01", "leave_one_sortie_out__fold02")


def run_thesis_case_evidence(*, output_root: str | Path, device="cuda"):
    if device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("thesis case evidence requires CUDA")
    selected = []
    fold_data = {}
    for fold_id in FOLDS:
        data = load_dingxin_fold_pretraining_data(
            fold_id=fold_id,
            snapshot_root=SNAPSHOT,
            fixed_audit_root=FIXED_AUDIT,
            inner_split_root=INNER_SPLIT,
        )
        fold_data[fold_id] = data
        candidates = []
        for sample_id in data.fold.held_out_sample_ids:
            batch = data.load_batch((sample_id,))
            candidates.append(
                {
                    "fold": fold_id,
                    "sample_id": sample_id,
                    "group_id": batch.group_ids[0],
                    "source_sample_hash": batch.source_sample_hashes[0],
                    "input_activity_score": input_activity_score(batch),
                }
            )
        selected.append(
            min(
                candidates,
                key=lambda row: (-row["input_activity_score"], row["sample_id"]),
            )
        )
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    selection_path = root / "case_selection.json"
    _atomic_json(
        selection_path,
        {
            "selection_rule": (
                "per held-out sortie choose the lexicographically first window at "
                "maximum 90th-percentile within-feature normalized vehicle change"
            ),
            "model_metrics_used_for_selection": False,
            "cases": selected,
        },
    )
    state = json.loads(
        (OUTER_COMPACT / "outer_results.json").read_text(encoding="utf-8")
    )
    if state.get("completed") is not True:
        raise RuntimeError("case evidence requires completed Dingxin confirmation")
    case_rows = []
    physics_rows = []
    occlusion_rows = []
    context_states = {}
    training = next(
        row
        for row in state["training_rows"]
        if row["fold"] == FOLDS[0]
        and row["seed"] == 17
        and row["method"] == "chronaris"
    )
    checkpoint = Path(training["checkpoint_path"])
    if not checkpoint.is_absolute():
        checkpoint = REPO / checkpoint
    adapter = _adapter(checkpoint, FOLDS[0], device)
    for selection in selected:
        fold_id = selection["fold"]
        data = fold_data[fold_id]
        sample_id = selection["sample_id"]
        raw = data.load_batch((sample_id,))
        output = _encoding(adapter, raw)
        semantic = output.auxiliary["semantic_event_output"]
        names = tuple(semantic.query_names)
        for query_index, query_name in enumerate(names):
            token_index = int(
                semantic.query_to_event_attention[0, query_index].argmax().cpu()
            )
            case_rows.append(
                {
                    "fold": fold_id,
                    "sample_id": sample_id,
                    "group_id": raw.group_ids[0],
                    "query_role": query_name,
                    "top_event_offset_s": float(
                        semantic.event_token_center_offsets_s[0, token_index].cpu()
                    ),
                    "query_to_event_weight": float(
                        semantic.query_to_event_attention[
                            0, query_index, token_index
                        ].cpu()
                    ),
                    "query_attribution_score": float(
                        semantic.query_attribution_scores[0, query_index].cpu()
                    ),
                    "learned_query_residual_norm": _query_residual_norm(
                        adapter, query_index
                    ),
                }
            )
        context_states[fold_id] = {
            name: semantic.query_context_states[0, names.index(name)].detach().cpu()
            for name in names
        }
        physics_rows.extend(
            {"fold": fold_id, "sample_id": sample_id, **row}
            for row in physics_audit_to_rows(
                output.auxiliary["physical_consistency"]
            )
        )
        full_embedding = _pooled(output)
        for scenario, provider in (
            ("central_missing", _missing_provider(data.load_batch, "physiology")),
            ("aircraft_context_missing", _missing_provider(data.load_batch, "vehicle")),
        ):
            changed = _pooled(_encoding(adapter, provider((sample_id,))))
            occlusion_rows.append(
                {
                    "fold": fold_id,
                    "sample_id": sample_id,
                    "scenario": scenario,
                    "relative_embedding_change": float(
                        torch.linalg.vector_norm(changed - full_embedding)
                        / torch.linalg.vector_norm(full_embedding).clamp_min(1e-12)
                    ),
                }
            )
    del adapter
    torch.cuda.empty_cache()
    pairing = _pairing_diagnostic(context_states)
    audit = {
        "format": "chronaris.thesis_case_evidence.v1",
        "protocol_version": "v3.2.2",
        "source_commit": subprocess.check_output(
            ("git", "rev-parse", "HEAD"), cwd=REPO, text=True
        ).strip(),
        "evaluation_code_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "attribution_checkpoint_fold": FOLDS[0],
        "checkpoint_sha256": sha256_file(checkpoint),
        "case_selection_sha256": sha256_file(selection_path),
        "dingxin_outer_state_sha256": sha256_file(
            OUTER_COMPACT / "outer_results.json"
        ),
        "selection_uses_model_outcomes": False,
        "pairing_diagnostic": pairing,
        "interpretation_boundary": (
            "both cases use one frozen fold model so context similarities share a "
            "coordinate system; query weights and attention are attribution material, "
            "not causal proof or held-out performance"
        ),
    }
    _write_csv(root / "semantic_attribution.csv", case_rows)
    _write_csv(root / "physical_residuals.csv", physics_rows)
    _write_csv(root / "occlusion.csv", occlusion_rows)
    _atomic_json(root / "audit.json", audit)
    (root / "report.md").write_text(_report(audit, selected), encoding="utf-8")
    return audit


def input_activity_score(batch):
    values = batch.vehicle_values[0].detach().cpu().numpy()
    mask = (
        batch.vehicle_feature_mask[0]
        & batch.vehicle_point_mask[0].unsqueeze(-1)
    ).detach().cpu().numpy()
    scores = []
    for feature in range(values.shape[1]):
        observed = values[mask[:, feature], feature]
        observed = observed[np.isfinite(observed)]
        if len(observed) < 2:
            continue
        scale = np.quantile(observed, 0.75) - np.quantile(observed, 0.25)
        scores.append(float(np.median(np.abs(np.diff(observed))) / max(scale, 1e-6)))
    return float(np.quantile(scores, 0.90)) if scores else 0.0


def _adapter(checkpoint, fold_id, device):
    encoder, _heads, normalizer, _payload = load_common_pretraining_checkpoint(
        checkpoint,
        device=device,
    )
    return TrainedFusionAdapter(
        encoder=encoder,
        normalizer=normalizer,
        fold_id=fold_id,
        checkpoint_sha256=sha256_file(checkpoint),
    )


def _encoding(adapter, raw):
    normalized = move_observation_batch(
        adapter.normalizer.transform(raw),
        device=next(adapter.encoder.parameters()).device,
    )
    adapter.encoder.eval()
    with torch.inference_mode():
        return adapter.encoder(normalized, compute_chronaris_diagnostics=True)


def _pooled(output):
    valid = output.modality_available_mask
    count = valid.sum(dim=1, keepdim=True).clamp_min(1).to(
        output.sequence_embedding.dtype
    )
    return (
        output.sequence_embedding
        * valid.unsqueeze(-1).to(output.sequence_embedding.dtype)
    ).sum(dim=1) / count


def _query_residual_norm(adapter, query_index):
    residual = (
        adapter.encoder.backbone.semantic_event_fusion.query_bank.query_residual
    )
    return float(torch.linalg.vector_norm(residual[query_index]).detach().cpu())


def _pairing_diagnostic(states):
    first, second = FOLDS
    correct = torch.stack(
        (
            F.cosine_similarity(
                states[first]["flight_event"],
                states[first]["physiology_response"],
                dim=0,
            ),
            F.cosine_similarity(
                states[second]["flight_event"],
                states[second]["physiology_response"],
                dim=0,
            ),
        )
    ).mean()
    wrong = torch.stack(
        (
            F.cosine_similarity(
                states[first]["flight_event"],
                states[second]["physiology_response"],
                dim=0,
            ),
            F.cosine_similarity(
                states[second]["flight_event"],
                states[first]["physiology_response"],
                dim=0,
            ),
        )
    ).mean()
    return {
        "correct_similarity": float(correct),
        "cross_sortie_wrong_similarity": float(wrong),
        "similarity_gap": float(correct - wrong),
    }


def _write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _report(audit, selected):
    pairing = audit["pairing_diagnostic"]
    fold_labels = {
        FOLDS[0]: "架次一作为留出组",
        FOLDS[1]: "架次二作为留出组",
    }
    lines = [
        "# 鼎新代表性窗口机制案例",
        "",
        "本案例先按输入侧航电变化规则分别从两个架次选择一个窗口，再使用同一个冻结分组模型读取归因；选择过程不使用模型输赢或预测误差。两个窗口共享同一表示坐标系，其中一个是该模型的留出架次，案例结果不作为外层性能结论。",
        "",
        "| 留出架次折 | 窗口 | 输入活动得分 |",
        "|---|---|---:|",
    ]
    lines.extend(
        f"| {fold_labels[row['fold']]} | `{row['sample_id']}` | "
        f"{row['input_activity_score']:.4f} |"
        for row in selected
    )
    lines.extend(
        (
            "",
            f"正确事件—响应配对的平均余弦相似度为 {pairing['correct_similarity']:.4f}，跨架次错误配对为 {pairing['cross_sortie_wrong_similarity']:.4f}，差值为 {pairing['similarity_gap']:.4f}。语义查询权重、事件注意力、单模态遮挡变化和运动学残差共同作为归因材料，不作为因果证明。",
            "",
        )
    )
    return "\n".join(lines)
