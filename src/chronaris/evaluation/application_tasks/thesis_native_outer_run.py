"""Frozen subject-grouped outer evaluation for CogPilot and CLARE."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, f1_score, mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from chronaris.dataset.group_splits import split_group_train_validation
from chronaris.evaluation.application_tasks.application_consumer_representations import APPLICATION_METHODS
from chronaris.evaluation.application_tasks.thesis_candidate_screen_metrics import (
    export_pooled_embeddings,
)
from chronaris.evaluation.application_tasks.thesis_native_data import (
    load_frozen_native_task,
)
from chronaris.modeling.fusion_encoders.single_stream import move_observation_batch
from chronaris.evaluation.application_tasks.thesis_outer_training import (
    train_frozen_outer_adapters,
)
from chronaris.modeling.training.candidate_checkpoint import (
    candidate_source_code_sha256,
)
from chronaris.representation import FoldLineage, TrainOnlyRobustNormalizer
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file


TASKS = ("cogpilot_difficulty", "cogpilot_event_response", "clare_cognitive_load")
REPO = Path(__file__).resolve().parents[4]
PROTOCOL = REPO / "docs/requirements/thesis-frozen-paper-evaluation-v3.2.2.md"


@dataclass(frozen=True, slots=True)
class ThesisNativeOuterConfig:
    run_id: str = "2026-09-03_thesis-native-outer-v3p2p2"
    compact_output_root: str = "docs/artifacts/runs"
    heavy_output_root: str = "artifacts/application_evaluation"
    selected_models_path: str = "docs/requirements/thesis-frozen-models-v3.2.json"
    protocol_version: str = "v3.2.2"
    runner_sha256: str = ""
    seeds: tuple[int, ...] = (17, 29, 43)
    tasks: tuple[str, ...] = TASKS
    max_epochs: int = 50
    patience: int = 8
    batch_size: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    device: str = "cuda"
    resume: bool = True

    def __post_init__(self):
        if self.device != "cuda" or not torch.cuda.is_available():
            raise ValueError("paper-facing native outer evaluation requires CUDA")
        if self.protocol_version != "v3.2.2":
            raise ValueError("native outer protocol version changed")
        if self.seeds != (17, 29, 43) or self.tasks != TASKS:
            raise ValueError("native outer seeds or tasks changed")
        if (
            min(self.max_epochs, self.patience, self.batch_size) <= 0
            or self.learning_rate <= 0
            or self.weight_decay < 0
        ):
            raise ValueError("native outer training budget is invalid")
        if not self.resume:
            raise ValueError("native outer results may only continue with resume enabled")
        if not self.runner_sha256:
            raise ValueError("native outer runner SHA-256 is required")


def run_thesis_native_outer(config: ThesisNativeOuterConfig):
    compact = Path(config.compact_output_root) / config.run_id
    heavy = Path(config.heavy_output_root) / config.run_id
    compact.mkdir(parents=True, exist_ok=True)
    heavy.mkdir(parents=True, exist_ok=True)
    selected = json.loads(
        Path(config.selected_models_path).read_text(encoding="utf-8")
    )
    state = _load_state(config, compact)
    for task_id in config.tasks:
        task = load_frozen_native_task(
            task_id,
            cache_root=heavy / "native_cache",
        )
        _bind_task_lineage(state, task)
        groups = np.asarray(task.group_ids)
        split_target = np.asarray(
            task.class_targets
            if task.class_targets is not None
            else task.regression_targets,
        )
        for fold_index, (outer_train, outer_test) in enumerate(
            GroupKFold(n_splits=5).split(task.sample_ids, split_target, groups),
            start=1,
        ):
            for seed in config.seeds:
                unit = f"{task_id}::fold{fold_index}::seed{seed}"
                if unit in state["completed_units"]:
                    continue
                train, validation = split_group_train_validation(
                    outer_train,
                    groups,
                    seed=seed + fold_index,
                )
                fold = FoldLineage(
                    fold_id=unit,
                    train_sample_ids=tuple(task.sample_ids[index] for index in train),
                    validation_sample_ids=tuple(
                        task.sample_ids[index] for index in validation
                    ),
                    held_out_sample_ids=tuple(
                        task.sample_ids[index] for index in outer_test
                    ),
                )
                _run_unit(
                    state=state,
                    compact=compact,
                    heavy=heavy,
                    task=task,
                    fold=fold,
                    outer_train_ids=tuple(
                        task.sample_ids[index] for index in outer_train
                    ),
                    seed=seed,
                    selected=selected,
                    config=config,
                )
                _write_state(compact, state)
    state["completed_for_requested_tasks"] = all(
        sum(unit.startswith(task_id + "::") for unit in state["completed_units"])
        == 15
        for task_id in config.tasks
    )
    _write_state(compact, state)
    return state

def _run_unit(
    *, state, compact, heavy, task, fold, outer_train_ids, seed, selected, config
):
    unit = fold.fold_id
    training_provider = _guarded_provider(
        task.dataset.batch_provider,
        allowed=fold.train_sample_ids + fold.validation_sample_ids,
        forbidden=fold.held_out_sample_ids,
    )
    normalizer = TrainOnlyRobustNormalizer().fit_from_batch_provider(
        training_provider,
        train_sample_ids=fold.train_sample_ids,
        held_out_sample_ids=fold.validation_sample_ids + fold.held_out_sample_ids,
        batch_size=config.batch_size,
    )
    adapters, checkpoint_hashes, training_rows = train_frozen_outer_adapters(
        provider=training_provider,
        fold=fold,
        schema=task.dataset.schema,
        normalizer=normalizer,
        selected_models=selected,
        output_root=heavy / "training" / Path(*fold.fold_id.split("::")),
        seed=seed,
        max_epochs=config.max_epochs,
        patience=config.patience,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        device=config.device,
        resume=config.resume,
    )
    for row in training_rows:
        key = f"{fold.fold_id}::{row['method']}"
        if not any(value["key"] == key for value in state["training_rows"]):
            state["training_rows"].append(
                {
                    "key": key,
                    "task": task.task_id,
                    "fold": fold.fold_id,
                    "seed": seed,
                    **row,
                }
            )
    _write_state(compact, state)
    protocol = {
        "unit": unit,
        "fold": fold.to_dict(),
        "outer_train_sample_ids": list(outer_train_ids),
        "checkpoint_sha256": checkpoint_hashes,
        "task_lineage_sha256": task.lineage_sha256,
        "consumer": {
            "classification": "StandardScaler+balanced LogisticRegression(C=1)",
            "regression": "StandardScaler+Ridge(alpha=10)",
        },
    }
    protocol_sha256 = hashlib.sha256(
        json.dumps(protocol, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()
    unit_path = heavy / "outer_units" / Path(*unit.split("::")) / "result.json"
    if config.resume and unit_path.is_file():
        payload = json.loads(unit_path.read_text(encoding="utf-8"))
        if payload.get("protocol_sha256") != protocol_sha256:
            raise RuntimeError(f"outer unit protocol changed: {unit}")
    else:
        metrics, predictions = _evaluate_unit(
            task=task,
            fold=fold,
            outer_train_ids=outer_train_ids,
            adapters=adapters,
            seed=seed,
            batch_size=config.batch_size,
        )
        payload = {
            "format": "chronaris.thesis_native_outer_unit.v1",
            "protocol_sha256": protocol_sha256,
            "protocol": protocol,
            "metric_rows": metrics,
            "prediction_rows": predictions,
        }
        _atomic_json(unit_path, payload)
    state["metric_rows"].extend(payload["metric_rows"])
    state["completed_units"].append(unit)
    state["outer_results_opened"] = True
    del adapters
    torch.cuda.empty_cache()

def _evaluate_unit(*, task, fold, outer_train_ids, adapters, seed, batch_size):
    class_by_id = (
        dict(zip(task.sample_ids, task.class_targets, strict=True))
        if task.class_targets is not None
        else None
    )
    regression_by_id = (
        dict(zip(task.sample_ids, task.regression_targets, strict=True))
        if task.regression_targets is not None
        else None
    )
    scenarios = [("full", task.dataset.batch_provider)]
    if task.task_id == "clare_cognitive_load":
        scenarios.extend(
            (
                (
                    "central_missing",
                    _missing_provider(task.dataset.batch_provider, "physiology"),
                ),
                (
                    "peripheral_missing",
                    _missing_provider(task.dataset.batch_provider, "vehicle"),
                ),
            )
        )
    metrics = []
    predictions = []
    for method in APPLICATION_METHODS:
        adapter = adapters[method]
        train_embedding = export_pooled_embeddings(
            adapter,
            task.dataset.batch_provider,
            outer_train_ids,
            batch_size,
        )
        classifier = None
        regressor = None
        if class_by_id is not None:
            train_class = np.asarray(
                [class_by_id[value] for value in outer_train_ids], dtype=int
            )
            if set(train_class) != set(task.class_values):
                raise RuntimeError("outer-train classification classes are incomplete")
            classifier = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=seed,
                ),
            ).fit(train_embedding, train_class)
        if regression_by_id is not None:
            regressor = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(
                train_embedding,
                np.asarray(
                    [regression_by_id[value] for value in outer_train_ids],
                    dtype=float,
                ),
            )
        for scenario, provider in scenarios:
            test_embedding = export_pooled_embeddings(
                adapter,
                provider,
                fold.held_out_sample_ids,
                batch_size,
            )
            if classifier is not None:
                truth = np.asarray(
                    [class_by_id[value] for value in fold.held_out_sample_ids],
                    dtype=int,
                )
                prediction = classifier.predict(test_embedding)
                metrics.extend(
                    _classification_rows(
                        task,
                        fold,
                        seed,
                        method,
                        scenario,
                        truth,
                        prediction,
                    )
                )
                predictions.extend(
                    {
                        "sample_id": sample_id,
                        "group_id": group_id,
                        "task": task.task_id,
                        "fold": fold.fold_id,
                        "seed": seed,
                        "method": method,
                        "scenario": scenario,
                        "target_kind": "classification",
                        "truth": int(truth[index]),
                        "prediction": int(prediction[index]),
                    }
                    for index, (sample_id, group_id) in enumerate(
                        zip(
                            fold.held_out_sample_ids,
                            _groups(task, fold.held_out_sample_ids),
                            strict=True,
                        )
                    )
                )
            if regressor is not None:
                truth = np.asarray(
                    [
                        regression_by_id[value]
                        for value in fold.held_out_sample_ids
                    ],
                    dtype=float,
                )
                prediction = regressor.predict(test_embedding)
                metrics.extend(
                    _regression_rows(
                        task,
                        fold,
                        seed,
                        method,
                        scenario,
                        truth,
                        prediction,
                    )
                )
                predictions.extend(
                    {
                        "sample_id": sample_id,
                        "group_id": group_id,
                        "task": task.task_id,
                        "fold": fold.fold_id,
                        "seed": seed,
                        "method": method,
                        "scenario": scenario,
                        "target_kind": "regression",
                        "truth": float(truth[index]),
                        "prediction": float(prediction[index]),
                    }
                    for index, (sample_id, group_id) in enumerate(
                        zip(
                            fold.held_out_sample_ids,
                            _groups(task, fold.held_out_sample_ids),
                            strict=True,
                        )
                    )
                )
        if method == "chronaris":
            pair = event_pair_retrieval_metrics(
                adapter,
                task.dataset.batch_provider,
                fold.held_out_sample_ids,
                batch_size=batch_size,
            )
            for name in ("similarity_gap", "recall_at_1", "mrr"):
                value = pair[name]
                if value is not None:
                    metrics.append(
                        _metric_row(
                            task,
                            fold,
                            seed,
                            method,
                            "full",
                            f"event_pair_{name}",
                            value,
                            "higher",
                        )
                    )
    return metrics, predictions


def _classification_rows(task, fold, seed, method, scenario, truth, prediction):
    return [
        _metric_row(
            task,
            fold,
            seed,
            method,
            scenario,
            "macro_f1",
            f1_score(
                truth,
                prediction,
                labels=task.class_values,
                average="macro",
                zero_division=0,
            ),
            "higher",
        ),
        _metric_row(
            task,
            fold,
            seed,
            method,
            scenario,
            "balanced_accuracy",
            balanced_accuracy_score(truth, prediction),
            "higher",
        ),
    ]


def _regression_rows(task, fold, seed, method, scenario, truth, prediction):
    rho = spearmanr(truth, prediction).correlation
    return [
        _metric_row(
            task,
            fold,
            seed,
            method,
            scenario,
            "rmse",
            math.sqrt(mean_squared_error(truth, prediction)),
            "lower",
        ),
        _metric_row(
            task,
            fold,
            seed,
            method,
            scenario,
            "spearman",
            float(rho) if math.isfinite(rho) else None,
            "higher",
        ),
    ]


def _metric_row(task, fold, seed, method, scenario, metric, value, direction):
    return {
        "task": task.task_id,
        "fold": fold.fold_id,
        "seed": seed,
        "method": method,
        "scenario": scenario,
        "metric": metric,
        "value": None if value is None else float(value),
        "direction": direction,
        "sample_count": len(fold.held_out_sample_ids),
        "group_count": len(set(_groups(task, fold.held_out_sample_ids))),
    }


def event_pair_retrieval_metrics(adapter, provider, sample_ids, *, batch_size):
    event_rows = []
    response_rows = []
    groups = []
    device = next(adapter.encoder.parameters()).device
    for offset in range(0, len(sample_ids), batch_size):
        raw = provider(sample_ids[offset : offset + batch_size])
        normalized = move_observation_batch(
            adapter.normalizer.transform(raw), device=device
        )
        adapter.encoder.eval()
        with torch.inference_mode():
            semantic = adapter.encoder(normalized).auxiliary["semantic_event_output"]
        names = tuple(semantic.query_names)
        event_rows.append(
            semantic.query_context_states[:, names.index("flight_event")].cpu()
        )
        response_rows.append(
            semantic.query_context_states[:, names.index("physiology_response")].cpu()
        )
        groups.extend(raw.group_ids)
    event = torch.nn.functional.normalize(torch.cat(event_rows), dim=-1)
    response = torch.nn.functional.normalize(torch.cat(response_rows), dim=-1)
    similarity = event @ response.T
    positives = []
    negatives = []
    ranks = []
    for anchor, group in enumerate(groups):
        candidates = [
            index
            for index, candidate_group in enumerate(groups)
            if index == anchor or candidate_group != group
        ]
        mismatch = next(
            (index for index in candidates if index != anchor),
            None,
        )
        if mismatch is None:
            continue
        positives.append(similarity[anchor, anchor])
        negatives.append(similarity[anchor, mismatch])
        scores = similarity[anchor, candidates]
        ranks.append(1 + int((scores > similarity[anchor, anchor]).sum()))
    if not negatives:
        return {
            "status": "unavailable",
            "similarity_gap": None,
            "recall_at_1": None,
            "mrr": None,
            "count": 0,
        }
    negative = torch.stack(negatives)
    kept_positive = torch.stack(positives)
    return {
        "status": "available",
        "similarity_gap": float((kept_positive - negative).mean()),
        "recall_at_1": float(np.mean(np.asarray(ranks) == 1)),
        "mrr": float(np.mean(1.0 / np.asarray(ranks))),
        "count": len(ranks),
    }


def _missing_provider(base_provider, stream):
    def provider(sample_ids):
        batch = base_provider(sample_ids)
        if stream == "physiology":
            return replace(
                batch,
                physiology_values=torch.zeros_like(batch.physiology_values),
                physiology_point_mask=torch.zeros_like(batch.physiology_point_mask),
                physiology_feature_mask=torch.zeros_like(
                    batch.physiology_feature_mask
                ),
                physiology_observation_age_s=torch.full_like(
                    batch.physiology_observation_age_s,
                    torch.inf,
                ),
            )
        return replace(
            batch,
            vehicle_values=torch.zeros_like(batch.vehicle_values),
            vehicle_point_mask=torch.zeros_like(batch.vehicle_point_mask),
            vehicle_feature_mask=torch.zeros_like(batch.vehicle_feature_mask),
            vehicle_observation_age_s=torch.full_like(
                batch.vehicle_observation_age_s,
                torch.inf,
            ),
        )

    return provider


def _groups(task, sample_ids):
    mapping = dict(zip(task.sample_ids, task.group_ids, strict=True))
    return tuple(mapping[value] for value in sample_ids)


def _guarded_provider(base_provider, *, allowed, forbidden):
    allowed = set(allowed)
    forbidden = set(forbidden)

    def provider(sample_ids):
        requested = set(sample_ids)
        if requested & forbidden or not requested <= allowed:
            raise ValueError("native training provider rejected outer-test access")
        return base_provider(sample_ids)

    return provider


def _load_state(config, compact):
    commit = subprocess.check_output(
        ("git", "rev-parse", "HEAD"), cwd=REPO, text=True
    ).strip()
    identity = {
        "protocol_version": config.protocol_version,
        "source_commit": commit,
        "source_code_sha256": candidate_source_code_sha256(),
        "runner_sha256": config.runner_sha256,
        "evaluation_protocol_sha256": sha256_file(PROTOCOL),
        "evaluation_code_sha256": _evaluation_code_sha256(
            Path(__file__),
            Path(__file__).with_name("thesis_native_data.py"),
            Path(__file__).with_name("thesis_outer_training.py"),
        ),
        "selected_models_sha256": sha256_file(config.selected_models_path),
        "config": asdict(config),
    }
    path = compact / "outer_results.json"
    if config.resume and path.is_file():
        state = json.loads(path.read_text(encoding="utf-8"))
        if any(state.get(name) != value for name, value in identity.items()):
            raise RuntimeError("native outer resume rejected frozen identity drift")
        return state
    return {
        "format": "chronaris.thesis_native_outer.v1",
        **identity,
        "outer_results_opened": False,
        "task_manifests": {},
        "training_rows": [],
        "metric_rows": [],
        "completed_units": [],
        "completed_for_requested_tasks": False,
    }


def _bind_task_lineage(state, task):
    manifest = {
        "lineage_sha256": task.lineage_sha256,
        "sample_count": len(task.sample_ids),
        "group_count": len(set(task.group_ids)),
        "class_values": list(task.class_values),
    }
    stored = state["task_manifests"].get(task.task_id)
    if stored is not None and stored != manifest:
        raise RuntimeError(f"native task lineage changed: {task.task_id}")
    state["task_manifests"][task.task_id] = manifest


def _write_state(compact, state):
    _atomic_json(compact / "outer_results.json", state)
    _write_csv(compact / "training_summary.csv", state["training_rows"])
    _write_csv(compact / "metric_long.csv", state["metric_rows"])
    completed = len(state["completed_units"])
    expected = len(state["config"]["tasks"]) * 15
    compact.joinpath("report.md").write_text(
        "\n".join(
            (
                "# 公开原生时间数据论文级分组评价",
                "",
                f"当前完成 {completed}/{expected} 个外层折与随机种子单元。",
                "编码器只使用外层训练集内部的训练与验证角色；消费者在完整外层训练集拟合，留出集只用于评价。",
                "",
            )
        ),
        encoding="utf-8",
    )
    compact.joinpath("resume_command.txt").write_text(
        "/home/wangminan/env/anaconda3/envs/chronaris/bin/python "
        "scripts/research/run_thesis_native_outer.py --protocol-version v3.2.2 "
        "--device cuda --resume\n",
        encoding="utf-8",
    )


def _write_csv(path, rows):
    if not rows:
        return
    import csv

    fields = tuple(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _evaluation_code_sha256(*paths):
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()
