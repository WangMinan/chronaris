"""Stage I public-benchmark dataset and baseline tests."""

from __future__ import annotations

import json
import math
import sys
import tempfile
import warnings
from pathlib import Path
import unittest

import pandas as pd

warnings.filterwarnings("ignore", message=".*is_sparse.*", category=DeprecationWarning)

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chronaris.dataset import build_nasa_csm_task_entries, build_uab_task_entries, load_stage_i_task_entries
from chronaris.features import build_nasa_csm_feature_table, build_uab_feature_table
from chronaris.pipelines.stage_i_baseline import run_stage_i_baselines
from chronaris.pipelines.stage_i_phase3 import (
    StageIPhase3Config,
    compose_stage_i_phase3_closure,
    run_stage_i_phase3,
)

REAL_DATASET_ROOT = Path("/home/wangminan/dataset/chronaris")


class StageIContractCompatibilityTest(unittest.TestCase):
    def test_load_legacy_session_entries_backfills_window_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "legacy.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "sample_id": "legacy_uab_session",
                        "dataset_id": "uab_workload_dataset",
                        "subset_id": "n_back",
                        "subject_id": "subject_01",
                        "session_id": "legacy_uab_session",
                        "split_group": "subject_01",
                        "training_role": "primary",
                        "window_start_utc": "2020-01-01T00:00:00Z",
                        "window_end_utc": "2020-01-01T00:10:00Z",
                        "source_refs": {"eeg_parquet": "uab_workload_dataset/data_n_back_test/eeg/eeg.parquet"},
                        "objective_label_name": "workload_level",
                        "objective_label_value": 1,
                        "subjective_target_name": "tlx_mean",
                        "subjective_target_value": 42.0,
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            entries = load_stage_i_task_entries(path)
            self.assertEqual(len(entries), 1)
            entry = entries[0]
            self.assertEqual(entry.sample_granularity, "session")
            self.assertEqual(entry.recording_id, "legacy_uab_session")
            self.assertEqual(entry.window_index, 0)
            self.assertEqual(entry.task_family, "workload")
            self.assertEqual(entry.label_namespace, "workload_level")


class StageIUABLiveDatasetTest(unittest.TestCase):
    @unittest.skipUnless((REAL_DATASET_ROOT / "uab_workload_dataset").exists(), "local UAB dataset not present")
    def test_live_session_manifest_counts_and_known_labels(self) -> None:
        prepared = build_uab_task_entries(REAL_DATASET_ROOT, profile="session_v1")

        self.assertEqual(prepared.subset_counts["n_back"], 48)
        self.assertEqual(prepared.subset_counts["heat_the_chair"], 34)
        self.assertEqual(prepared.subset_counts["flight_simulator"], 5)
        self.assertEqual(len(prepared.entries), 87)

        by_sample_id = {entry.sample_id: entry for entry in prepared.entries}
        n_back_sample = by_sample_id["uab_n_back__subject_01__test_1"]
        self.assertEqual(n_back_sample.objective_label_value, 0)
        self.assertAlmostEqual(n_back_sample.subjective_target_value or 0.0, 5.0 / 6.0, places=6)

        heat_sample = by_sample_id["uab_heat__subject_01__test_2"]
        self.assertEqual(heat_sample.objective_label_value, 1)
        self.assertEqual(heat_sample.context_payload["game_mode"], "with")


class StageINASALiveDatasetTest(unittest.TestCase):
    @unittest.skipUnless((REAL_DATASET_ROOT / "nasa_csm").exists(), "local NASA CSM dataset not present")
    def test_live_nasa_manifest_covers_benchmark_and_loft(self) -> None:
        prepared = build_nasa_csm_task_entries(REAL_DATASET_ROOT)
        subsets = {entry.subset_id for entry in prepared.entries}
        roles = {entry.training_role for entry in prepared.entries}
        labels = {int(entry.objective_label_value) for entry in prepared.entries}

        self.assertEqual(subsets, {"benchmark", "loft"})
        self.assertIn("primary", roles)
        self.assertIn("inventory_only", roles)
        self.assertEqual(labels, {0, 1, 2, 5})


class StageISyntheticPipelineTest(unittest.TestCase):
    def test_uab_window_pipeline_and_ablations(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_root = Path(temp_dir)
            _write_mini_uab_dataset(dataset_root)

            prepared = build_uab_task_entries(dataset_root, profile="window_v2")
            feature_result = build_uab_feature_table(dataset_root, prepared.entries)
            feature_table = feature_result.feature_table

            self.assertTrue(all(entry.sample_granularity == "window" for entry in prepared.entries))
            self.assertTrue(all(entry.window_duration_s == 5.0 for entry in prepared.entries))
            self.assertEqual(feature_result.missing_ecg_session_counts["n_back"], 3)

            subject_windows = [
                entry for entry in prepared.entries
                if entry.session_id == "uab_n_back__subject_03__test_3"
            ]
            self.assertEqual(len(subject_windows), 3)
            missing_rows = feature_table.loc[
                feature_table["session_id"] == "uab_n_back__subject_03__test_3"
            ]
            ecg_columns = [column for column in feature_result.ecg_feature_columns]
            self.assertTrue(missing_rows[ecg_columns].isna().all(axis=1).all())

            artifacts = run_stage_i_baselines(
                feature_table,
                dataset_id="uab_workload_dataset",
                profile="window_v2",
                task_family="workload",
                artifact_root=dataset_root / "artifacts",
            )

            self.assertEqual(
                set(artifacts.objective_metrics["primary_results"]),
                {"n_back", "heat_the_chair"},
            )
            self.assertEqual(
                set(artifacts.subjective_metrics["primary_results"]),
                {"n_back", "heat_the_chair"},
            )
            for group_name in ("n_back", "heat_the_chair"):
                self.assertEqual(
                    set(artifacts.objective_metrics["ablation_results"][group_name]),
                    {"eeg_ecg", "eeg_only", "ecg_only"},
                )
                objective_models = set(
                    artifacts.objective_metrics["ablation_results"][group_name]["eeg_ecg"]["models"]
                )
                self.assertEqual(objective_models, {"logistic_regression", "linear_svc"})
                self.assertEqual(
                    set(artifacts.subjective_metrics["ablation_results"][group_name]),
                    {"eeg_ecg", "eeg_only", "ecg_only"},
                )
                subjective_models = set(
                    artifacts.subjective_metrics["ablation_results"][group_name]["eeg_ecg"]["models"]
                )
                self.assertEqual(subjective_models, {"ridge_regression", "linear_svr"})
            self.assertEqual(
                set(artifacts.fold_predictions["split_group"].unique()),
                {"subject_01", "subject_02", "subject_03"},
            )

    def test_nasa_attention_pipeline_filters_background_and_keeps_subject_loso(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_root = Path(temp_dir)
            _write_mini_nasa_csm_dataset(dataset_root)

            prepared = build_nasa_csm_task_entries(dataset_root)
            feature_result = build_nasa_csm_feature_table(dataset_root, prepared.entries)
            feature_table = feature_result.feature_table

            self.assertIn("benchmark", prepared.subset_counts)
            self.assertIn("loft", prepared.subset_counts)
            background_entries = [entry for entry in prepared.entries if entry.objective_label_value == 0]
            self.assertTrue(background_entries)
            self.assertTrue(all(entry.training_role == "inventory_only" for entry in background_entries))

            artifacts = run_stage_i_baselines(
                feature_table,
                dataset_id="nasa_csm",
                profile="window_v2",
                task_family="attention_state",
                artifact_root=dataset_root / "artifacts",
            )

            self.assertEqual(
                set(artifacts.objective_metrics["primary_results"]),
                {"benchmark_only", "loft_only", "combined"},
            )
            for group_name in ("benchmark_only", "loft_only", "combined"):
                self.assertEqual(
                    set(artifacts.objective_metrics["ablation_results"][group_name]),
                    {"all_sensors", "eeg_only", "peripheral_only"},
                )
                self.assertEqual(
                    set(artifacts.objective_metrics["ablation_results"][group_name]["all_sensors"]["models"]),
                    {"logistic_regression", "linear_svc"},
                )
            combined = artifacts.fold_predictions.loc[
                artifacts.fold_predictions["evaluation_group"] == "combined"
            ]
            self.assertTrue((combined["split_group"] == combined["subject_id"]).all())

    def test_phase3_orchestration_writes_summary_and_reports(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_root = Path(temp_dir)
            _write_mini_uab_dataset(dataset_root)
            _write_mini_nasa_csm_dataset(dataset_root)
            prior_root = Path(temp_dir) / "prior_session"
            prior_root.mkdir(parents=True, exist_ok=True)
            _write_fake_session_comparison_assets(prior_root)

            result = run_stage_i_phase3(
                StageIPhase3Config(
                    dataset_root=str(dataset_root),
                    output_root=str(Path(temp_dir) / "artifacts"),
                    run_id="phase3-test",
                    prior_uab_session_artifact_root=str(prior_root),
                )
            )

            summary_path = Path(result.closure_summary_path)
            report_path = Path(result.closure_report_path)
            self.assertTrue(summary_path.exists())
            self.assertTrue(report_path.exists())
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertIn("uab_window", summary)
            self.assertIn("nasa_attention", summary)
            self.assertIn("uab_session_comparison", summary)
            self.assertTrue((Path(result.uab_artifact_root) / "baseline_report.md").exists())
            self.assertTrue((Path(result.nasa_artifact_root) / "baseline_report.md").exists())

    def test_compose_phase3_closure_rebuilds_local_baseline_reports(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_root = Path(temp_dir)
            _write_mini_uab_dataset(dataset_root)
            _write_mini_nasa_csm_dataset(dataset_root)
            prior_root = Path(temp_dir) / "prior_session"
            prior_root.mkdir(parents=True, exist_ok=True)
            _write_fake_session_comparison_assets(prior_root)

            result = run_stage_i_phase3(
                StageIPhase3Config(
                    dataset_root=str(dataset_root),
                    output_root=str(Path(temp_dir) / "artifacts"),
                    run_id="phase3-reuse-test",
                    prior_uab_session_artifact_root=str(prior_root),
                )
            )
            Path(result.uab_artifact_root, "baseline_report.md").unlink()
            Path(result.nasa_artifact_root, "baseline_report.md").unlink()

            recomposed = compose_stage_i_phase3_closure(
                artifact_root=result.artifact_root,
                prior_uab_session_artifact_root=str(prior_root),
            )

            self.assertTrue(Path(recomposed.closure_summary_path).exists())
            self.assertTrue(Path(recomposed.uab_artifact_root, "baseline_report.md").exists())
            self.assertTrue(Path(recomposed.nasa_artifact_root, "baseline_report.md").exists())


def _write_mini_uab_dataset(dataset_root: Path) -> None:
    root = dataset_root / "uab_workload_dataset"
    _write_n_back(root)
    _write_heat(root)
    _write_flight(root)


def _write_n_back(root: Path) -> None:
    subject_ids = ("subject_01", "subject_02", "subject_03")
    eeg_rows = []
    hr_rows = []
    ibi_rows = []
    br_rows = []
    tlx_rows = []
    base_time = pd.Timestamp("2020-01-01T00:00:00")
    for subject_index, subject_id in enumerate(subject_ids, start=1):
        for test_value in (1, 2, 3):
            session_start = base_time + pd.Timedelta(minutes=subject_index * 10 + test_value)
            for second in range(16):
                current = session_start + pd.Timedelta(seconds=second)
                eeg_rows.append(
                    {
                        "subject": subject_id,
                        "test": test_value,
                        "phase": 1,
                        "datetime": current,
                        "POW.AF3.Theta": float(test_value * 10 + second),
                        "PM.Focus.Scaled": float(subject_index + test_value + second / 10.0),
                    }
                )
                if not (subject_id == "subject_03" and test_value == 3):
                    hr_rows.append(
                        {
                            "subject": subject_id,
                            "test": test_value,
                            "phase": 1,
                            "datetime": current,
                            "hr": float(70 + test_value * 2 + second / 10.0),
                        }
                    )
                    ibi_rows.append(
                        {
                            "subject": subject_id,
                            "test": test_value,
                            "phase": 1,
                            "datetime": current,
                            "rr_int": float(700 + test_value * 10 + second),
                        }
                    )
                    br_rows.append(
                        {
                            "subject": subject_id,
                            "test": test_value,
                            "phase": 1,
                            "datetime": current,
                            "br": float(10 + test_value + second / 10.0),
                        }
                    )
            tlx_rows.append(
                {
                    "subject": subject_id,
                    "test": test_value,
                    "mental_demand": test_value * 2,
                    "physical_demand": test_value,
                    "temporal_demand": test_value * 3,
                    "performance": test_value * 2 + 1,
                    "effort": test_value * 4,
                    "frustration": test_value * 2,
                }
            )
    _write_parquet(root / "data_n_back_test" / "eeg" / "eeg.parquet", eeg_rows)
    _write_parquet(root / "data_n_back_test" / "ecg" / "ecg_hr.parquet", hr_rows)
    _write_parquet(root / "data_n_back_test" / "ecg" / "ecg_ibi.parquet", ibi_rows)
    _write_parquet(root / "data_n_back_test" / "ecg" / "ecg_br.parquet", br_rows)
    _write_parquet(root / "data_n_back_test" / "subjective_performance" / "tlx_answers.parquet", tlx_rows)
    _write_parquet(
        root / "data_n_back_test" / "game_performance" / "game_scores.parquet",
        [{"subject": row["subject"], "test": row["test"], "score": 100 - row["test"]} for row in tlx_rows],
    )


def _write_heat(root: Path) -> None:
    subject_ids = ("subject_01", "subject_02", "subject_03")
    eeg_rows = []
    ecg_rows = []
    tlx_rows = []
    base_time = pd.Timestamp("2021-06-01T00:00:00")
    for subject_index, subject_id in enumerate(subject_ids, start=1):
        for test_value, game in ((2, "with"), (1, "without")):
            session_start = base_time + pd.Timedelta(minutes=subject_index * 10 + (0 if test_value == 2 else 2))
            for second in range(11):
                current = session_start + pd.Timedelta(seconds=second)
                eeg_rows.append(
                    {
                        "subject": subject_id,
                        "test": test_value,
                        "phase": 1,
                        "datetime": current,
                        "POW.AF3.Theta": float(test_value * 20 + second),
                        "PM.Focus.Scaled": float(subject_index + test_value + second / 10.0),
                    }
                )
                ecg_rows.append(
                    {
                        "subject": subject_id,
                        "test": test_value,
                        "datetime": current,
                        "hr": float(75 + test_value * 3 + second / 10.0),
                        "rr_int": float(650 + test_value * 10 + second),
                    }
                )
            tlx_rows.append(
                {
                    "subject": subject_id,
                    "game": game,
                    "mental_demand": test_value * 3,
                    "physical_demand": test_value,
                    "temporal_demand": test_value * 2,
                    "performance": test_value * 2,
                    "effort": test_value * 4,
                    "frustration": test_value,
                }
            )
            timestamps = [int((session_start + pd.Timedelta(seconds=offset)).timestamp()) for offset in (1, 5, 9)]
            performance_path = root / "data_heat_the_chair" / "game_performance" / f"{subject_id}_{game}.csv"
            performance_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "game": [1, 0, 0],
                    "piece": [0, 1, 1],
                    "objective": [0, 0, 0],
                    "interruption_in": [0, 0, 0],
                    "interruption_out": [0, 0, 0],
                    "interruption_type": ["none", "none", "none"],
                    "interruption_begin_time": [0, 0, 0],
                    "interruption_due_time": [0, 0, 0],
                }
            ).to_csv(performance_path, index=False)
    _write_parquet(root / "data_heat_the_chair" / "eeg" / "eeg.parquet", eeg_rows)
    _write_parquet(root / "data_heat_the_chair" / "ecg" / "ecg.parquet", ecg_rows)
    _write_parquet(root / "data_heat_the_chair" / "subjective_performance" / "tlx_answers.parquet", tlx_rows)


def _write_flight(root: Path) -> None:
    eeg_rows = []
    hr_rows = []
    ibi_rows = []
    base_time = pd.Timestamp("2020-12-22T00:00:00")
    sessions = ((1, 1, 2.0, 2.0), (2, 2, 1.0, 1.0))
    for subject_id, flight_id, perceived, theoretical in sessions:
        session_start = base_time + pd.Timedelta(minutes=subject_id * 5 + flight_id)
        for second in range(11):
            current = session_start + pd.Timedelta(seconds=second)
            eeg_rows.append(
                {
                    "subject": subject_id,
                    "flight": flight_id,
                    "datetime": current,
                    "role": "flying",
                    "perceived_difficulty": perceived,
                    "theoretical_difficulty": theoretical,
                    "POW.AF3.Theta": float(flight_id * 30 + second),
                    "PM.Focus.Scaled": float(subject_id + flight_id + second / 10.0),
                }
            )
            hr_rows.append(
                {
                    "subject": subject_id,
                    "flight": flight_id,
                    "datetime": current,
                    "hr": float(72 + flight_id + second / 10.0),
                }
            )
            ibi_rows.append(
                {
                    "subject": subject_id,
                    "flight": flight_id,
                    "datetime": current,
                    "rr_int": float(710 + flight_id * 10 + second),
                }
            )
    _write_parquet(root / "data_flight_simulator" / "eeg" / "eeg.parquet", eeg_rows)
    _write_parquet(root / "data_flight_simulator" / "ecg" / "ecg_hr.parquet", hr_rows)
    _write_parquet(root / "data_flight_simulator" / "ecg" / "ecg_ibi.parquet", ibi_rows)
    difficulty_dir = root / "data_flight_simulator" / "perceived_difficulty"
    difficulty_dir.mkdir(parents=True, exist_ok=True)
    (difficulty_dir / "flight_1.json").write_text("{}", encoding="utf-8")
    (difficulty_dir / "flight_2_4.json").write_text("{}", encoding="utf-8")


def _write_mini_nasa_csm_dataset(dataset_root: Path) -> None:
    root = dataset_root / "nasa_csm" / "extracted"
    for subject in ("10", "11"):
        subject_root = root / subject
        subject_root.mkdir(parents=True, exist_ok=True)
        _write_nasa_csv(subject_root / f"{subject}_CA.csv", _benchmark_events(2))
        _write_nasa_csv(subject_root / f"{subject}_DA.csv", _benchmark_events(5))
        _write_nasa_csv(subject_root / f"{subject}_SS.csv", _benchmark_events(1))
        _write_nasa_csv(subject_root / f"{subject}_LOFT.csv", _loft_events())


def _benchmark_events(event_code: int) -> list[tuple[float, int]]:
    rows: list[tuple[float, int]] = []
    for second in range(18):
        if second < 6:
            code = 0
        elif second < 16:
            code = event_code
        else:
            code = 0
        rows.append((float(second), code))
    return rows


def _loft_events() -> list[tuple[float, int]]:
    rows: list[tuple[float, int]] = []
    for second in range(48):
        if second < 6:
            code = 0
        elif second < 16:
            code = 2
        elif second < 22:
            code = 0
        elif second < 32:
            code = 1
        elif second < 38:
            code = 0
        elif second < 48:
            code = 5
        rows.append((float(second), code))
    return rows


def _write_nasa_csv(path: Path, event_rows: list[tuple[float, int]]) -> None:
    rows = []
    for index, (time_secs, event_code) in enumerate(event_rows):
        rows.append(
            {
                "Unnamed: 0.1": index,
                "Unnamed: 0": index,
                "TimeSecs": time_secs,
                "EEG_FP1": 10.0 + index,
                "EEG_F7": 5.0 + index / 10.0,
                "EEG_F8": 3.0 + index / 20.0,
                "EEG_C3": 1.0 + index / 30.0,
                "ECG": 700.0 + index,
                "R": 12.0 + index / 10.0,
                "GSR": 100.0 + event_code,
                "Event": event_code,
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_fake_session_comparison_assets(root: Path) -> None:
    (root / "objective_metrics.json").write_text(
        json.dumps(
            {
                "primary_results": {
                    "n_back": {
                        "best_model_name": "rf",
                        "models": {"rf": {"macro_f1": 0.45}},
                    },
                    "heat_the_chair": {
                        "best_model_name": "rf",
                        "models": {"rf": {"macro_f1": 0.55}},
                    },
                }
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "subjective_metrics.json").write_text(
        json.dumps(
            {
                "primary_results": {
                    "n_back": {
                        "best_model_name": "rf",
                        "models": {"rf": {"rmse": 5.0}},
                    },
                    "heat_the_chair": {
                        "best_model_name": "rf",
                        "models": {"rf": {"rmse": 2.0}},
                    },
                }
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_parquet(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)
