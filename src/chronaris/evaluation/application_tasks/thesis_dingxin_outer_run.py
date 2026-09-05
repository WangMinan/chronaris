"""Frozen Dingxin grouped confirmation for the thesis main line."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, f1_score, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from chronaris.evaluation.application_tasks.dingxin_fold_pretraining_data import (
    load_dingxin_fold_pretraining_data,
)
from chronaris.evaluation.application_tasks.thesis_candidate_screen_metrics import (
    export_pooled_embeddings,
)
from chronaris.evaluation.application_tasks.thesis_native_outer_run import (
    event_pair_retrieval_metrics,
)
from chronaris.evaluation.application_tasks.thesis_outer_training import (
    train_frozen_outer_adapters,
)
from chronaris.modeling.training.candidate_checkpoint import (
    candidate_source_code_sha256,
)
from chronaris.representation import TrainOnlyRobustNormalizer
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file


REPO = Path(__file__).resolve().parents[4]
PROTOCOL = REPO / "docs/requirements/thesis-frozen-paper-evaluation-v3.2.2.md"
SNAPSHOT = REPO / "artifacts/application_evaluation/2026-07-10_dingxin-input-snapshot"
FIXED_AUDIT = REPO / "docs/artifacts/runs/2026-07-10_fixed-data-audit"
INNER_SPLIT = REPO / "docs/artifacts/runs/2026-07-11_dingxin-inner-splits"
TARGETS = REPO / "docs/artifacts/runs/2026-07-11_dingxin-nested-targets/nested_targets.csv"
FOLDS = (
    "leave_one_view_out__fold01",
    "leave_one_view_out__fold02",
    "leave_one_view_out__fold03",
    "leave_one_sortie_out__fold01",
    "leave_one_sortie_out__fold02",
)
METHODS = (
    "physiology_only",
    "vehicle_only",
    "naive_time_sync",
    "mult",
    "contiformer",
    "chronaris",
)


@dataclass(frozen=True, slots=True)
class ThesisDingxinOuterConfig:
    run_id: str = "2026-09-03_thesis-dingxin-confirmation-v3p2p2"
    compact_output_root: str = "docs/artifacts/runs"
    heavy_output_root: str = "artifacts/application_evaluation"
    selected_models_path: str = "docs/requirements/thesis-frozen-models-v3.2.json"
    protocol_version: str = "v3.2.2"
    runner_sha256: str = ""
    seeds: tuple[int, ...] = (17, 29, 43)
    max_epochs: int = 50
    patience: int = 8
    batch_size: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    device: str = "cuda"
    resume: bool = True

    def __post_init__(self):
        if self.device != "cuda" or not torch.cuda.is_available():
            raise ValueError("paper-facing Dingxin confirmation requires CUDA")
        if self.protocol_version != "v3.2.2":
            raise ValueError("Dingxin confirmation protocol version changed")
        if self.seeds != (17, 29, 43):
            raise ValueError("Dingxin confirmation seeds changed")
        if (
            min(self.max_epochs, self.patience, self.batch_size) <= 0
            or self.learning_rate <= 0
            or self.weight_decay < 0
        ):
            raise ValueError("Dingxin confirmation training budget is invalid")
        if not self.resume:
            raise ValueError("Dingxin confirmation may only continue with resume enabled")
        if not self.runner_sha256:
            raise ValueError("Dingxin confirmation runner SHA-256 is required")


def run_thesis_dingxin_outer(config: ThesisDingxinOuterConfig):
    compact = Path(config.compact_output_root) / config.run_id
    heavy = Path(config.heavy_output_root) / config.run_id
    compact.mkdir(parents=True, exist_ok=True)
    heavy.mkdir(parents=True, exist_ok=True)
    selected = json.loads(
        Path(config.selected_models_path).read_text(encoding="utf-8")
    )
    targets = pd.read_csv(TARGETS)
    state = _load_state(config, compact)
    for fold_id in FOLDS:
        data = load_dingxin_fold_pretraining_data(
            fold_id=fold_id,
            snapshot_root=SNAPSHOT,
            fixed_audit_root=FIXED_AUDIT,
            inner_split_root=INNER_SPLIT,
        )
        _bind_fold_lineage(state, data)
        for seed in config.seeds:
            unit = f"{fold_id}::seed{seed}"
            if unit in state["completed_units"]:
                continue
            _run_unit(
                state=state,
                compact=compact,
                heavy=heavy,
                data=data,
                targets=targets,
                seed=seed,
                selected=selected,
                config=config,
            )
            _write_state(compact, state)
    state["completed"] = len(state["completed_units"]) == 15
    _write_state(compact, state)
    return state


def _run_unit(*, state, compact, heavy, data, targets, seed, selected, config):
    fold = data.fold
    unit = f"{fold.fold_id}::seed{seed}"
    training_provider = _guarded_provider(
        data.load_batch,
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
        schema=data.index.plan.schema,
        normalizer=normalizer,
        selected_models=selected,
        output_root=heavy / "training" / fold.fold_id / f"seed{seed}",
        seed=seed,
        max_epochs=config.max_epochs,
        patience=config.patience,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        device=config.device,
        resume=config.resume,
        vehicle_field_labels=data.vehicle_field_labels,
    )
    for row in training_rows:
        key = f"{unit}::{row['method']}"
        if not any(value["key"] == key for value in state["training_rows"]):
            state["training_rows"].append(
                {
                    "key": key,
                    "fold": fold.fold_id,
                    "seed": seed,
                    **row,
                }
            )
    _write_state(compact, state)
    protocol = {
        "unit": unit,
        "fold": fold.to_dict(),
        "checkpoint_sha256": checkpoint_hashes,
        "source_hashes": data.source_hashes,
        "target_sha256": sha256_file(TARGETS),
        "consumer": {
            "classification": "StandardScaler+balanced LogisticRegression(C=1)",
            "regression": "StandardScaler+Ridge(alpha=1)",
        },
    }
    protocol_sha256 = hashlib.sha256(
        json.dumps(protocol, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()
    result_path = heavy / "outer_units" / fold.fold_id / f"seed{seed}.json"
    if config.resume and result_path.is_file():
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        if payload.get("protocol_sha256") != protocol_sha256:
            raise RuntimeError(f"Dingxin outer unit protocol changed: {unit}")
    else:
        metric_rows, prediction_rows = _evaluate(
            data=data,
            targets=targets,
            adapters=adapters,
            seed=seed,
            batch_size=config.batch_size,
        )
        payload = {
            "format": "chronaris.thesis_dingxin_outer_unit.v1",
            "protocol_sha256": protocol_sha256,
            "protocol": protocol,
            "metric_rows": metric_rows,
            "prediction_rows": prediction_rows,
        }
        _atomic_json(result_path, payload)
    state["metric_rows"].extend(payload["metric_rows"])
    state["completed_units"].append(unit)
    state["outer_results_opened"] = True
    del adapters
    torch.cuda.empty_cache()


def _evaluate(*, data, targets, adapters, seed, batch_size):
    fold = data.fold
    fold_targets = targets[
        (targets["fold_id"].astype(str) == fold.fold_id)
        & (targets["status"].astype(str) == "completed")
    ]
    outer_train_ids = fold.train_sample_ids + fold.validation_sample_ids
    metrics = []
    predictions = []
    for method in METHODS:
        adapter = adapters[method]
        train_values = export_pooled_embeddings(
            adapter, data.load_batch, outer_train_ids, batch_size
        )
        test_values = export_pooled_embeddings(
            adapter, data.load_batch, fold.held_out_sample_ids, batch_size
        )
        train_index = dict(zip(outer_train_ids, train_values, strict=True))
        test_index = dict(zip(fold.held_out_sample_ids, test_values, strict=True))
        for task_slug in (
            "maneuver_intensity_classification",
            "physiology_response_prediction",
        ):
            frame = fold_targets[fold_targets["task_slug"] == task_slug]
            train = frame[frame["role"].isin(("train", "validation"))]
            test = frame[frame["role"] == "held_out"]
            if task_slug == "maneuver_intensity_classification":
                model = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(
                        max_iter=3000,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ).fit(
                    np.stack([train_index[value] for value in train["context_id"]]),
                    train["class_target"].to_numpy(dtype=int),
                )
                truth = test["class_target"].to_numpy(dtype=int)
                prediction = model.predict(
                    np.stack([test_index[value] for value in test["context_id"]])
                )
                values = (
                    (
                        "macro_f1",
                        f1_score(
                            truth,
                            prediction,
                            labels=(0, 1, 2),
                            average="macro",
                            zero_division=0,
                        ),
                        "higher",
                    ),
                    (
                        "balanced_accuracy",
                        balanced_accuracy_score(truth, prediction),
                        "higher",
                    ),
                )
            else:
                model = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(
                    np.stack([train_index[value] for value in train["context_id"]]),
                    train["continuous_target"].to_numpy(dtype=float),
                )
                truth = test["continuous_target"].to_numpy(dtype=float)
                prediction = model.predict(
                    np.stack([test_index[value] for value in test["context_id"]])
                )
                rho = spearmanr(truth, prediction).correlation
                values = (
                    (
                        "rmse",
                        math.sqrt(mean_squared_error(truth, prediction)),
                        "lower",
                    ),
                    (
                        "spearman",
                        float(rho) if math.isfinite(rho) else None,
                        "higher",
                    ),
                )
            metrics.extend(
                {
                    "fold": fold.fold_id,
                    "fold_kind": (
                        "view_adaptation"
                        if "view" in fold.fold_id
                        else "sortie_confirmation"
                    ),
                    "seed": seed,
                    "method": method,
                    "task": task_slug,
                    "metric": name,
                    "value": None if value is None else float(value),
                    "direction": direction,
                    "sample_count": len(test),
                }
                for name, value, direction in values
            )
            predictions.extend(
                {
                    "fold": fold.fold_id,
                    "seed": seed,
                    "method": method,
                    "task": task_slug,
                    "context_id": context_id,
                    "truth": float(truth[index]),
                    "prediction": float(prediction[index]),
                }
                for index, context_id in enumerate(test["context_id"])
            )
        if method == "chronaris":
            pair = event_pair_retrieval_metrics(
                adapter,
                data.load_batch,
                fold.held_out_sample_ids,
                batch_size=batch_size,
            )
            for name in ("similarity_gap", "recall_at_1", "mrr"):
                if pair[name] is not None:
                    metrics.append(
                        {
                            "fold": fold.fold_id,
                            "fold_kind": (
                                "view_adaptation"
                                if "view" in fold.fold_id
                                else "sortie_confirmation"
                            ),
                            "seed": seed,
                            "method": method,
                            "task": "event_response_pairing",
                            "metric": name,
                            "value": float(pair[name]),
                            "direction": "higher",
                            "sample_count": pair["count"],
                        }
                    )
    return metrics, predictions


def _guarded_provider(base_provider, *, allowed, forbidden):
    allowed = set(allowed)
    forbidden = set(forbidden)

    def provider(sample_ids):
        requested = set(sample_ids)
        if requested & forbidden or not requested <= allowed:
            raise ValueError("Dingxin training provider rejected outer-test access")
        return base_provider(sample_ids)

    return provider


def _load_state(config, compact):
    identity = {
        "protocol_version": config.protocol_version,
        "source_commit": subprocess.check_output(
            ("git", "rev-parse", "HEAD"), cwd=REPO, text=True
        ).strip(),
        "source_code_sha256": candidate_source_code_sha256(),
        "runner_sha256": config.runner_sha256,
        "evaluation_protocol_sha256": sha256_file(PROTOCOL),
        "evaluation_code_sha256": _evaluation_code_sha256(
            Path(__file__),
            Path(__file__).with_name("thesis_native_outer_run.py"),
            Path(__file__).with_name("thesis_outer_training.py"),
        ),
        "selected_models_sha256": sha256_file(config.selected_models_path),
        "config": asdict(config),
        "target_sha256": sha256_file(TARGETS),
    }
    path = compact / "outer_results.json"
    if config.resume and path.is_file():
        state = json.loads(path.read_text(encoding="utf-8"))
        if any(state.get(key) != value for key, value in identity.items()):
            raise RuntimeError("Dingxin outer resume rejected frozen identity drift")
        return state
    return {
        "format": "chronaris.thesis_dingxin_confirmation.v1",
        **identity,
        "outer_results_opened": False,
        "fold_manifests": {},
        "training_rows": [],
        "metric_rows": [],
        "completed_units": [],
        "completed": False,
    }


def _bind_fold_lineage(state, data):
    manifest = {
        "fold": data.fold.to_dict(),
        "source_hashes": data.source_hashes,
    }
    stored = state["fold_manifests"].get(data.fold.fold_id)
    if stored is not None and stored != manifest:
        raise RuntimeError(f"Dingxin fold lineage changed: {data.fold.fold_id}")
    state["fold_manifests"][data.fold.fold_id] = manifest


def _write_state(compact, state):
    _atomic_json(compact / "outer_results.json", state)
    _write_csv(compact / "training_summary.csv", state["training_rows"])
    _write_csv(compact / "metric_long.csv", state["metric_rows"])
    compact.joinpath("report.md").write_text(
        "\n".join(
            (
                "# 鼎新论文级分组确认",
                "",
                f"当前完成 {len(state['completed_units'])}/15 个分组与随机种子单元。",
                "三个留一视图折用于视图适配诊断，两个留一架次折用于分组确认；不报告显著性结论。",
                "",
            )
        ),
        encoding="utf-8",
    )
    compact.joinpath("resume_command.txt").write_text(
        "/home/wangminan/env/anaconda3/envs/chronaris/bin/python "
        "scripts/research/run_thesis_dingxin_outer.py "
        "--protocol-version v3.2.2 --device cuda --resume\n",
        encoding="utf-8",
    )


def _write_csv(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
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
