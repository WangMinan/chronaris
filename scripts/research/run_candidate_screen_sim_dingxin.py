"""Three-seed four-candidate training-internal screen for protocol v3.1."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold

from chronaris.dataset.clare_native import build_clare_native_dataset
from chronaris.dataset.cogpilot_native import build_cogpilot_difficulty_dataset
from chronaris.dataset.group_splits import split_group_train_validation
from chronaris.evaluation.application_tasks.dingxin_fold_pretraining_data import (
    load_dingxin_fold_pretraining_data,
)
from chronaris.evaluation.application_tasks.thesis_candidate_screen_metrics import (
    balanced_shift_accuracy,
    dingxin_validation_metrics,
    public_validation_metrics,
    summarize_candidate_training,
)
from chronaris.modeling.training import (
    CommonPretrainingConfig,
    EncoderCandidateConfig,
    TrainedFusionAdapter,
    load_common_pretraining_checkpoint,
    train_common_pretext_method,
)
from chronaris.modeling.training.candidate_checkpoint import (
    candidate_source_code_sha256,
)
from chronaris.representation import (
    AugmentationPolicy,
    FoldLineage,
    TrainOnlyRobustNormalizer,
    collate_observation_samples,
    load_simulation_observed_context,
)


REPO = Path(__file__).resolve().parents[2]
SIM_ROOT = REPO / "artifacts/application_evaluation/2026-07-10_aviation-simulation-formal"
DINGXIN_SNAPSHOT = REPO / "artifacts/application_evaluation/2026-07-10_dingxin-input-snapshot"
FIXED_AUDIT = REPO / "docs/artifacts/runs/2026-07-10_fixed-data-audit"
INNER_SPLIT = REPO / "docs/artifacts/runs/2026-07-11_dingxin-inner-splits"
NESTED_TARGETS = REPO / "docs/artifacts/runs/2026-07-11_dingxin-nested-targets/nested_targets.csv"
COGPILOT_ROOT = Path("/home/wangminan/dataset/chronaris/physio_net/physionet.org/files/virtual-reality-piloting/1.0.0/dataPackage/task-ils")
CLARE_ROOT = Path("/home/wangminan/dataset/chronaris/clare")
PROTOCOL_VERSION = "v3.1"
RUN_ROOT = REPO / "docs/artifacts/runs/2026-09-01_candidate-screen-v3p1"
HEAVY_ROOT = REPO / "artifacts/application_evaluation/2026-09-01_candidate-screen-v3p1"
FOLD_IDS = (
    "leave_one_view_out__fold01",
    "leave_one_view_out__fold02",
    "leave_one_view_out__fold03",
    "leave_one_sortie_out__fold01",
    "leave_one_sortie_out__fold02",
)


def main() -> int:
    args = _parse_args()
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("paper-facing candidate screen requires available CUDA")
    state = _load_state(args.resume, args)
    if "simulation" in args.stages:
        _run_simulation(state, args)
    if "dingxin" in args.stages:
        _run_dingxin(state, args)
    if "cogpilot" in args.stages:
        _run_cogpilot(state, args)
    if "clare" in args.stages:
        _run_clare(state, args)
    expected = len(args.seeds) * len(_candidate_specs()) * (
        (1 if "simulation" in args.stages else 0)
        + (len(FOLD_IDS) if "dingxin" in args.stages else 0)
        + (1 if "cogpilot" in args.stages else 0)
        + (5 if "clare" in args.stages else 0)
    )
    state["completed_for_requested_stages"] = sum(
        row["stage"] in args.stages for row in state["rows"]
    ) >= expected
    _write_state(state)
    _atomic_text(RUN_ROOT / "report.md", _render_report(state))
    return 0


def _run_simulation(state, args) -> None:
    train_paths = _simulation_paths("train")[:24]
    validation_paths = _simulation_paths("validation")[:12]
    samples = [
        load_simulation_observed_context(
            path,
            context_start_s=0.0,
            context_duration_s=30.0,
            sample_id=f"screen_train_{index:03d}",
            group_id=path.parents[2].name,
        )
        for index, path in enumerate(train_paths)
    ]
    samples.extend(
        load_simulation_observed_context(
            path,
            context_start_s=0.0,
            context_duration_s=30.0,
            sample_id=f"screen_validation_{index:03d}",
            group_id=path.parents[2].name,
        )
        for index, path in enumerate(validation_paths)
    )
    batch = collate_observation_samples(samples)
    fold = FoldLineage(
        fold_id="simulation_training_internal_screen",
        train_sample_ids=tuple(sample.sample_id for sample in samples[:24]),
        validation_sample_ids=tuple(sample.sample_id for sample in samples[24:-1]),
        held_out_sample_ids=(samples[-1].sample_id,),
    )
    normalizer = TrainOnlyRobustNormalizer().fit(
        batch,
        train_sample_ids=fold.train_sample_ids,
        held_out_sample_ids=fold.validation_sample_ids + fold.held_out_sample_ids,
    )
    schema = samples[0].schema
    for seed in args.seeds:
        for candidate in _candidate_specs():
            unit = f"simulation::seed{seed}::{candidate['id']}"
            if _done(state, unit):
                continue
            result, adapter, payload = _train(
                unit=unit,
                seed=seed,
                candidate=candidate,
                fold=fold,
                batch=batch,
                schema=schema,
                normalizer=normalizer,
                args=args,
            )
            _append(
                state,
                {
                    "unit": unit,
                    "stage": "simulation",
                    "fold": fold.fold_id,
                    "seed": seed,
                    "candidate": candidate["id"],
                    "balanced_shift_accuracy": balanced_shift_accuracy(
                        adapter,
                        payload,
                        batch=batch,
                        provider=None,
                        sample_ids=fold.validation_sample_ids,
                        batch_size=args.batch_size,
                    ),
                    **summarize_candidate_training(result, payload),
                },
            )


def _run_dingxin(state, args) -> None:
    targets = pd.read_csv(NESTED_TARGETS)
    for fold_id in FOLD_IDS:
        data = load_dingxin_fold_pretraining_data(
            fold_id=fold_id,
            snapshot_root=DINGXIN_SNAPSHOT,
            fixed_audit_root=FIXED_AUDIT,
            inner_split_root=INNER_SPLIT,
        )
        provider = _guarded_provider(
            data.load_batch,
            allowed=data.fold.train_sample_ids + data.fold.validation_sample_ids,
            forbidden=data.fold.held_out_sample_ids,
        )
        normalizer = TrainOnlyRobustNormalizer().fit_from_batch_provider(
            provider,
            train_sample_ids=data.fold.train_sample_ids,
            held_out_sample_ids=data.fold.validation_sample_ids + data.fold.held_out_sample_ids,
            batch_size=2,
        )
        fold_targets = targets[
            (targets["fold_id"].astype(str) == fold_id)
            & (targets["status"].astype(str) == "completed")
        ].copy()
        for seed in args.seeds:
            for candidate in _candidate_specs():
                unit = f"dingxin::{fold_id}::seed{seed}::{candidate['id']}"
                if _done(state, unit):
                    continue
                result, adapter, payload = _train(
                    unit=unit,
                    seed=seed,
                    candidate=candidate,
                    fold=data.fold,
                    provider=provider,
                    schema=data.index.plan.schema,
                    normalizer=normalizer,
                    vehicle_labels=data.vehicle_field_labels,
                    args=args,
                )
                application = dingxin_validation_metrics(
                    adapter,
                    provider,
                    data.fold,
                    fold_targets,
                    args.batch_size,
                    seed,
                )
                _append(
                    state,
                    {
                        "unit": unit,
                        "stage": "dingxin",
                        "fold": fold_id,
                        "seed": seed,
                        "candidate": candidate["id"],
                        **application,
                        **summarize_candidate_training(result, payload),
                    },
                )


def _run_cogpilot(state, args) -> None:
    dataset = build_cogpilot_difficulty_dataset(
        COGPILOT_ROOT,
        subject_limit=args.cogpilot_subjects,
        cache_root=HEAVY_ROOT / "cogpilot/native_cache",
    )
    selected = _first_indices(dataset.records, lambda row: (row.group_id, row.label))
    sample_ids = tuple(dataset.sample_ids[index] for index in selected)
    groups = np.asarray([dataset.group_ids[index] for index in selected])
    labels = np.asarray([dataset.labels[index] for index in selected], dtype=int)
    outer_train, outer_test = next(
        GroupKFold(n_splits=5).split(sample_ids, labels, groups)
    )
    for seed in args.seeds:
        train, validation = split_group_train_validation(
            outer_train,
            groups,
            seed=seed,
        )
        fold = FoldLineage(
            fold_id=f"cogpilot_candidate_outer_train_seed{seed}",
            train_sample_ids=tuple(sample_ids[index] for index in train),
            validation_sample_ids=tuple(sample_ids[index] for index in validation),
            held_out_sample_ids=tuple(sample_ids[index] for index in outer_test),
        )
        provider = _guarded_provider(
            dataset.batch_provider,
            allowed=fold.train_sample_ids + fold.validation_sample_ids,
            forbidden=fold.held_out_sample_ids,
        )
        normalizer = TrainOnlyRobustNormalizer().fit_from_batch_provider(
            provider,
            train_sample_ids=fold.train_sample_ids,
            held_out_sample_ids=fold.validation_sample_ids + fold.held_out_sample_ids,
            batch_size=args.public_batch_size,
        )
        label_by_id = dict(zip(sample_ids, labels, strict=True))
        for candidate in _candidate_specs():
            unit = f"cogpilot::seed{seed}::{candidate['id']}"
            if _done(state, unit):
                continue
            result, adapter, payload = _train(
                unit=unit,
                seed=seed,
                candidate=candidate,
                fold=fold,
                provider=provider,
                schema=dataset.schema,
                normalizer=normalizer,
                args=args,
                batch_size=args.public_batch_size,
            )
            application = public_validation_metrics(
                adapter,
                provider,
                fold,
                label_by_id,
                args.public_batch_size,
                seed,
            )
            _append(
                state,
                {
                    "unit": unit,
                    "stage": "cogpilot",
                    "fold": fold.fold_id,
                    "seed": seed,
                    "candidate": candidate["id"],
                    "selected_sample_count": len(sample_ids),
                    **application,
                    **summarize_candidate_training(result, payload),
                },
            )


def _run_clare(state, args) -> None:
    dataset = build_clare_native_dataset(
        CLARE_ROOT,
        subject_limit=args.clare_subjects,
        window_stride=args.clare_window_stride,
        cache_root=HEAVY_ROOT / "clare/native_cache",
    )
    selected = _middle_dual_stream_indices(
        dataset,
        lambda row: (row.group_id, row.sample_id.split("__", 1)[1].split("_", 1)[0]),
    )
    sample_ids = tuple(dataset.sample_ids[index] for index in selected)
    groups = np.asarray([dataset.group_ids[index] for index in selected])
    scores = np.asarray([dataset.labels[index] for index in selected], dtype=int)
    binary = (scores >= 7).astype(int)
    label_by_id = dict(zip(sample_ids, binary, strict=True))
    score_by_id = dict(zip(sample_ids, scores, strict=True))
    splitter = GroupKFold(n_splits=5)
    for fold_index, (outer_train, outer_test) in enumerate(
        splitter.split(sample_ids, binary, groups),
        start=1,
    ):
        for seed in args.seeds:
            train, validation = split_group_train_validation(
                outer_train,
                groups,
                seed=seed + fold_index,
            )
            fold = FoldLineage(
                fold_id=f"clare_candidate_outer_train_fold{fold_index}_seed{seed}",
                train_sample_ids=tuple(sample_ids[index] for index in train),
                validation_sample_ids=tuple(
                    sample_ids[index] for index in validation
                ),
                held_out_sample_ids=tuple(sample_ids[index] for index in outer_test),
            )
            provider = _guarded_provider(
                dataset.batch_provider,
                allowed=fold.train_sample_ids + fold.validation_sample_ids,
                forbidden=fold.held_out_sample_ids,
            )
            normalizer = TrainOnlyRobustNormalizer().fit_from_batch_provider(
                provider,
                train_sample_ids=fold.train_sample_ids,
                held_out_sample_ids=(
                    fold.validation_sample_ids + fold.held_out_sample_ids
                ),
                batch_size=args.public_batch_size,
            )
            for candidate in _candidate_specs():
                unit = (
                    f"clare::fold{fold_index}::seed{seed}::{candidate['id']}"
                )
                if _done(state, unit):
                    continue
                result, adapter, payload = _train(
                    unit=unit,
                    seed=seed,
                    candidate=candidate,
                    fold=fold,
                    provider=provider,
                    schema=dataset.schema,
                    normalizer=normalizer,
                    args=args,
                    batch_size=args.public_batch_size,
                )
                application = public_validation_metrics(
                    adapter,
                    provider,
                    fold,
                    label_by_id,
                    args.public_batch_size,
                    seed,
                    score_by_id=score_by_id,
                )
                _append(
                    state,
                    {
                        "unit": unit,
                        "stage": "clare",
                        "fold": fold_index,
                        "seed": seed,
                        "candidate": candidate["id"],
                        "selected_sample_count": len(sample_ids),
                        **application,
                        **summarize_candidate_training(result, payload),
                    },
                )


def _train(
    *,
    unit,
    seed,
    candidate,
    fold,
    schema,
    normalizer,
    args,
    batch=None,
    provider=None,
    vehicle_labels=None,
    batch_size=None,
):
    print(f"[candidate-screen] start {unit}", flush=True)
    started = time.perf_counter()
    resolved_batch_size = batch_size or args.batch_size
    result = train_common_pretext_method(
        "chronaris",
        batch=batch,
        batch_provider=provider,
        fold=fold,
        physiology_feature_names=schema.physiology_feature_names,
        vehicle_feature_names=schema.vehicle_feature_names,
        vehicle_field_labels=vehicle_labels
        or tuple((name, name) for name in schema.vehicle_feature_names),
        normalizer=normalizer,
        output_root=HEAVY_ROOT / unit.replace("::", "/"),
        config=CommonPretrainingConfig(
            epochs=args.epochs,
            batch_size=resolved_batch_size,
            seed=seed,
            device=args.device,
            semantic_event_enabled=candidate["semantic"],
            learnable_semantic_queries=candidate["semantic"],
            heartbeat_interval_s=30.0,
        ),
        candidate_config=EncoderCandidateConfig(candidate_id="C", hidden_dim=32),
        augmentation_policy=AugmentationPolicy(),
        chronaris_fusion_kind="safe_lag",
        chronaris_mechanism_enabled=True,
        chronaris_explicit_shift_enabled=True,
        chronaris_explicit_shift_weight=candidate["shift_weight"],
        chronaris_event_pair_weight=candidate["pair_weight"],
        resume=args.resume,
    )
    encoder, _heads, loaded_normalizer, payload = load_common_pretraining_checkpoint(
        result.best_checkpoint_path,
        device=args.device,
    )
    print(
        f"[candidate-screen] done {unit} elapsed_s={time.perf_counter() - started:.1f}",
        flush=True,
    )
    return result, TrainedFusionAdapter(
        encoder=encoder,
        normalizer=loaded_normalizer,
        fold_id=fold.fold_id,
        checkpoint_sha256=payload["canonical_training_state_sha256"],
    ), payload


def _guarded_provider(base_provider, *, allowed, forbidden):
    allowed = set(allowed)
    forbidden = set(forbidden)

    def provider(sample_ids):
        requested = set(sample_ids)
        if requested & forbidden or not requested <= allowed:
            raise ValueError("candidate screen rejected outer-test sample request")
        return base_provider(sample_ids)

    return provider


def _first_indices(records, key):
    seen = set()
    selected = []
    for index, record in enumerate(records):
        value = key(record)
        if value not in seen:
            selected.append(index)
            seen.add(value)
    return tuple(selected)


def _middle_dual_stream_indices(dataset, key):
    groups = {}
    for index, record in enumerate(dataset.records):
        groups.setdefault(key(record), []).append(index)
    selected = []
    for indices in groups.values():
        middle = (len(indices) - 1) / 2
        for position in sorted(range(len(indices)), key=lambda value: abs(value - middle)):
            index = indices[position]
            sample = dataset.load_sample(dataset.sample_ids[index])
            if sample.physiology_feature_mask.any() and sample.vehicle_feature_mask.any():
                selected.append(index)
                break
    return tuple(selected)


def _candidate_specs():
    return (
        {"id": "base", "semantic": False, "shift_weight": 0.0, "pair_weight": 0.0},
        {"id": "explicit_shift", "semantic": False, "shift_weight": 0.1, "pair_weight": 0.0},
        {"id": "semantic_pair", "semantic": True, "shift_weight": 0.0, "pair_weight": 0.1},
        {"id": "both_objectives", "semantic": True, "shift_weight": 0.1, "pair_weight": 0.1},
    )


def _simulation_paths(split):
    return sorted(
        (SIM_ROOT / split).glob("*/*/clean_asynchronous/raw_dual_stream.npz")
    )


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=("simulation", "dingxin", "cogpilot", "clare"),
        default=("simulation", "dingxin"),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=(17, 29, 43))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--public-batch-size", type=int, default=16)
    parser.add_argument("--cogpilot-subjects", type=int, default=5)
    parser.add_argument("--clare-subjects", type=int, default=8)
    parser.add_argument("--clare-window-stride", type=int, default=1)
    parser.add_argument(
        "--protocol-version",
        choices=(PROTOCOL_VERSION,),
        default=PROTOCOL_VERSION,
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _load_state(resume, args):
    path = RUN_ROOT / "screen_results.json"
    source_sha256 = candidate_source_code_sha256()
    source_commit = subprocess.check_output(
        ("git", "rev-parse", "HEAD"),
        cwd=REPO,
        text=True,
    ).strip()
    runner_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if resume and path.exists():
        state = json.loads(path.read_text(encoding="utf-8"))
        expected = {
            "protocol_version": args.protocol_version,
            "source_commit": source_commit,
            "source_code_sha256": source_sha256,
            "runner_sha256": runner_sha256,
        }
        if any(state.get(name) != value for name, value in expected.items()):
            raise RuntimeError("candidate screen resume rejected source or protocol drift")
        return state
    return {
        "format": "chronaris.training_internal_candidate_screen.v3.1",
        "protocol_version": args.protocol_version,
        "source_commit": source_commit,
        "source_code_sha256": source_sha256,
        "runner_sha256": runner_sha256,
        "seeds": [17, 29, 43],
        "candidates": [value["id"] for value in _candidate_specs()],
        "outer_results_opened": False,
        "rows": [],
        "completed_for_requested_stages": False,
    }


def _done(state, unit):
    return any(row["unit"] == unit for row in state["rows"])


def _append(state, row):
    state["rows"].append(_json_safe(row))
    _write_state(state)


def _write_state(state):
    _atomic_text(
        RUN_ROOT / "screen_results.json",
        json.dumps(state, ensure_ascii=False, indent=2) + "\n",
    )


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _atomic_text(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _render_report(state):
    stages = sorted(set(row["stage"] for row in state["rows"]))
    return "\n".join(
        [
            "# 四候选三随机种子训练内筛选",
            "",
            f"当前完成 `{len(state['rows'])}` 个单元，覆盖：{', '.join(stages) or '无'}。",
            "",
            "本运行只使用仿真验证集、鼎新内部验证集以及公开数据各外层训练集中的受试者分组验证集；外层确认样本由数据提供器拒绝访问，未生成外层预测或指标。",
            "",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
