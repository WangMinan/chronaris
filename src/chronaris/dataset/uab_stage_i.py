"""UAB public-dataset adapter for Stage I task manifests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pandas as pd
import pyarrow.parquet as pq

from chronaris.dataset.stage_i_contracts import StageITaskEntry, isoformat_utc

DATASET_ID = "uab_workload_dataset"
PRIMARY_ROLE = "primary"
AUXILIARY_ROLE = "auxiliary"
SESSION_V1 = "session_v1"
WINDOW_V2 = "window_v2"
WINDOW_DURATION_SECONDS = 5.0

N_BACK_OBJECTIVE_LABELS = {
    1: ("low", 0),
    2: ("medium", 1),
    3: ("high", 2),
}

HEAT_OBJECTIVE_LABELS = {
    "without": ("low", 0),
    "with": ("medium", 1),
}


@dataclass(frozen=True, slots=True)
class UABPreparedTaskSet:
    """All UAB Stage I entries plus helper metadata."""

    entries: tuple[StageITaskEntry, ...]
    subset_counts: Mapping[str, int]
    heat_test_mapping: Mapping[str, Mapping[int, str]]


@dataclass(frozen=True, slots=True)
class _UABSessionSpec:
    subset_id: str
    subject_id: str
    session_id: str
    recording_id: str
    split_group: str
    training_role: str
    start: datetime
    end: datetime
    source_refs: Mapping[str, str]
    objective_label_name: str | None
    objective_label_value: float | int | None
    subjective_target_name: str | None
    subjective_target_value: float | None
    context_payload: Mapping[str, object]


def build_uab_task_entries(
    dataset_root: str | Path,
    *,
    profile: str = SESSION_V1,
) -> UABPreparedTaskSet:
    """Build the Stage I task manifest for the local UAB dataset."""

    if profile not in {SESSION_V1, WINDOW_V2}:
        raise ValueError(f"unsupported UAB Stage I profile: {profile}")

    root = Path(dataset_root) / DATASET_ID
    use_overlap_bounds = profile == WINDOW_V2
    n_back_specs = _build_n_back_specs(root, use_overlap_bounds=use_overlap_bounds)
    heat_specs, heat_mapping = _build_heat_specs(root, use_overlap_bounds=use_overlap_bounds)
    flight_specs = _build_flight_specs(root, use_overlap_bounds=use_overlap_bounds)
    session_specs = tuple((*n_back_specs, *heat_specs, *flight_specs))
    entries = _materialize_entries(session_specs, profile=profile)
    subset_counts = {
        "n_back": sum(1 for entry in entries if entry.subset_id == "n_back"),
        "heat_the_chair": sum(1 for entry in entries if entry.subset_id == "heat_the_chair"),
        "flight_simulator": sum(1 for entry in entries if entry.subset_id == "flight_simulator"),
    }
    return UABPreparedTaskSet(
        entries=entries,
        subset_counts=subset_counts,
        heat_test_mapping=heat_mapping,
    )


def _build_n_back_specs(root: Path, *, use_overlap_bounds: bool) -> tuple[_UABSessionSpec, ...]:
    tlx = pd.read_parquet(root / "data_n_back_test" / "subjective_performance" / "tlx_answers.parquet")
    eeg_bounds = _stream_group_bounds(
        root / "data_n_back_test" / "eeg" / "eeg.parquet",
        key_columns=("subject", "test"),
        time_column="datetime",
    )
    hr_bounds = _stream_group_bounds(
        root / "data_n_back_test" / "ecg" / "ecg_hr.parquet",
        key_columns=("subject", "test"),
        time_column="datetime",
    ) if use_overlap_bounds else {}
    ibi_bounds = _stream_group_bounds(
        root / "data_n_back_test" / "ecg" / "ecg_ibi.parquet",
        key_columns=("subject", "test"),
        time_column="datetime",
    ) if use_overlap_bounds else {}
    br_bounds = _stream_group_bounds(
        root / "data_n_back_test" / "ecg" / "ecg_br.parquet",
        key_columns=("subject", "test"),
        time_column="datetime",
    ) if use_overlap_bounds else {}
    specs: list[_UABSessionSpec] = []
    for row in tlx.sort_values(["subject", "test"]).itertuples(index=False):
        objective_name, objective_value = N_BACK_OBJECTIVE_LABELS[int(row.test)]
        subject_id = str(row.subject)
        session_suffix = f"test_{int(row.test)}"
        session_id = f"uab_n_back__{subject_id}__{session_suffix}"
        key = (subject_id, int(row.test))
        session_bounds = _choose_overlap_bounds(
            eeg_bounds[key],
            hr_bounds.get(key),
            ibi_bounds.get(key),
            br_bounds.get(key),
        )
        specs.append(
            _UABSessionSpec(
                subset_id="n_back",
                subject_id=subject_id,
                session_id=session_id,
                recording_id=session_id,
                split_group=subject_id,
                training_role=PRIMARY_ROLE,
                start=session_bounds["start"],
                end=session_bounds["end"],
                source_refs={
                    "eeg_parquet": "uab_workload_dataset/data_n_back_test/eeg/eeg.parquet",
                    "ecg_hr_parquet": "uab_workload_dataset/data_n_back_test/ecg/ecg_hr.parquet",
                    "ecg_ibi_parquet": "uab_workload_dataset/data_n_back_test/ecg/ecg_ibi.parquet",
                    "ecg_br_parquet": "uab_workload_dataset/data_n_back_test/ecg/ecg_br.parquet",
                    "tlx_answers_parquet": "uab_workload_dataset/data_n_back_test/subjective_performance/tlx_answers.parquet",
                    "game_scores_parquet": "uab_workload_dataset/data_n_back_test/game_performance/game_scores.parquet",
                },
                objective_label_name="workload_level",
                objective_label_value=objective_value,
                subjective_target_name="tlx_mean",
                subjective_target_value=_tlx_mean(row),
                context_payload={
                    "objective_label_text": objective_name,
                    "task_name": "n_back",
                    "task_variant": _n_back_variant(int(row.test)),
                    "source_partition": "benchmark",
                    "label_sources": {
                        "objective": "task_definition",
                        "subjective": "tlx_answers",
                    },
                },
            )
        )
    return tuple(specs)


def _build_heat_specs(root: Path, *, use_overlap_bounds: bool) -> tuple[tuple[_UABSessionSpec, ...], dict[str, dict[int, str]]]:
    tlx = pd.read_parquet(root / "data_heat_the_chair" / "subjective_performance" / "tlx_answers.parquet")
    eeg_bounds = _stream_group_bounds(
        root / "data_heat_the_chair" / "eeg" / "eeg.parquet",
        key_columns=("subject", "test"),
        time_column="datetime",
    )
    ecg_bounds = _stream_group_bounds(
        root / "data_heat_the_chair" / "ecg" / "ecg.parquet",
        key_columns=("subject", "test"),
        time_column="datetime",
    ) if use_overlap_bounds else {}
    test_mapping = _infer_heat_test_mapping(root, eeg_bounds)
    specs: list[_UABSessionSpec] = []
    for row in tlx.sort_values(["subject", "game"]).itertuples(index=False):
        subject_id = str(row.subject)
        game = str(row.game)
        reverse_map = {mapped_game: test_value for test_value, mapped_game in test_mapping[subject_id].items()}
        test_value = reverse_map[game]
        session_id = f"uab_heat__{subject_id}__test_{test_value}"
        key = (subject_id, test_value)
        session_bounds = _choose_overlap_bounds(eeg_bounds[key], ecg_bounds.get(key))
        objective_name, objective_value = HEAT_OBJECTIVE_LABELS[game]
        specs.append(
            _UABSessionSpec(
                subset_id="heat_the_chair",
                subject_id=subject_id,
                session_id=session_id,
                recording_id=session_id,
                split_group=subject_id,
                training_role=PRIMARY_ROLE,
                start=session_bounds["start"],
                end=session_bounds["end"],
                source_refs={
                    "eeg_parquet": "uab_workload_dataset/data_heat_the_chair/eeg/eeg.parquet",
                    "ecg_parquet": "uab_workload_dataset/data_heat_the_chair/ecg/ecg.parquet",
                    "tlx_answers_parquet": "uab_workload_dataset/data_heat_the_chair/subjective_performance/tlx_answers.parquet",
                    "game_performance_dir": "uab_workload_dataset/data_heat_the_chair/game_performance",
                },
                objective_label_name="workload_level",
                objective_label_value=objective_value,
                subjective_target_name="tlx_mean",
                subjective_target_value=_tlx_mean(row),
                context_payload={
                    "objective_label_text": objective_name,
                    "task_name": "heat_the_chair",
                    "game_mode": game,
                    "source_partition": "benchmark",
                    "label_sources": {
                        "objective": "task_definition",
                        "subjective": "tlx_answers",
                    },
                },
            )
        )
    return tuple(specs), test_mapping


def _build_flight_specs(root: Path, *, use_overlap_bounds: bool) -> tuple[_UABSessionSpec, ...]:
    eeg = pd.read_parquet(
        root / "data_flight_simulator" / "eeg" / "eeg.parquet",
        columns=[
            "subject",
            "flight",
            "datetime",
            "role",
            "perceived_difficulty",
            "theoretical_difficulty",
        ],
    )
    eeg = eeg.sort_values(["subject", "flight", "datetime"])
    hr_bounds = _stream_group_bounds(
        root / "data_flight_simulator" / "ecg" / "ecg_hr.parquet",
        key_columns=("subject", "flight"),
        time_column="datetime",
    ) if use_overlap_bounds else {}
    ibi_bounds = _stream_group_bounds(
        root / "data_flight_simulator" / "ecg" / "ecg_ibi.parquet",
        key_columns=("subject", "flight"),
        time_column="datetime",
    ) if use_overlap_bounds else {}
    specs: list[_UABSessionSpec] = []
    for (subject, flight), frame in eeg.groupby(["subject", "flight"], sort=False):
        subject_id = str(subject)
        nonbaseline = frame.loc[frame["perceived_difficulty"] >= 0]
        perceived = float(nonbaseline["perceived_difficulty"].mean()) if not nonbaseline.empty else None
        theoretical_series = frame.loc[frame["theoretical_difficulty"] >= 0, "theoretical_difficulty"]
        theoretical = float(theoretical_series.max()) if not theoretical_series.empty else None
        role_values = tuple(sorted(set(frame["role"].dropna().astype(str))))
        session_id = f"uab_flight__subject_{subject_id}__flight_{int(flight)}"
        primary_bounds = {
            "start": frame["datetime"].iloc[0].to_pydatetime(),
            "end": frame["datetime"].iloc[-1].to_pydatetime(),
        }
        key = (subject, int(flight))
        session_bounds = _choose_overlap_bounds(primary_bounds, hr_bounds.get(key), ibi_bounds.get(key))
        specs.append(
            _UABSessionSpec(
                subset_id="flight_simulator",
                subject_id=subject_id,
                session_id=session_id,
                recording_id=session_id,
                split_group=subject_id,
                training_role=AUXILIARY_ROLE,
                start=session_bounds["start"],
                end=session_bounds["end"],
                source_refs={
                    "eeg_parquet": "uab_workload_dataset/data_flight_simulator/eeg/eeg.parquet",
                    "ecg_hr_parquet": "uab_workload_dataset/data_flight_simulator/ecg/ecg_hr.parquet",
                    "ecg_ibi_parquet": "uab_workload_dataset/data_flight_simulator/ecg/ecg_ibi.parquet",
                    "difficulty_dir": "uab_workload_dataset/data_flight_simulator/perceived_difficulty",
                },
                objective_label_name="theoretical_difficulty",
                objective_label_value=theoretical,
                subjective_target_name="perceived_difficulty",
                subjective_target_value=perceived,
                context_payload={
                    "task_name": "flight_simulator",
                    "flight_id": int(flight),
                    "role_values": role_values,
                    "source_partition": "benchmark",
                    "label_sources": {
                        "objective": "eeg.parquet theoretical_difficulty",
                        "subjective": "eeg.parquet perceived_difficulty",
                    },
                },
            )
        )
    return tuple(specs)


def _materialize_entries(
    session_specs: Sequence[_UABSessionSpec],
    *,
    profile: str,
) -> tuple[StageITaskEntry, ...]:
    entries: list[StageITaskEntry] = []
    if profile == SESSION_V1:
        for spec in session_specs:
            entries.append(
                StageITaskEntry(
                    sample_id=spec.session_id,
                    dataset_id=DATASET_ID,
                    subset_id=spec.subset_id,
                    subject_id=spec.subject_id,
                    session_id=spec.session_id,
                    split_group=spec.split_group,
                    training_role=spec.training_role,
                    sample_granularity="session",
                    recording_id=spec.recording_id,
                    window_index=0,
                    window_duration_s=max((spec.end - spec.start).total_seconds(), 0.0),
                    task_family="workload",
                    label_namespace=spec.objective_label_name,
                    window_start_utc=isoformat_utc(spec.start),
                    window_end_utc=isoformat_utc(spec.end),
                    source_refs=spec.source_refs,
                    objective_label_name=spec.objective_label_name,
                    objective_label_value=spec.objective_label_value,
                    subjective_target_name=spec.subjective_target_name,
                    subjective_target_value=spec.subjective_target_value,
                    context_payload=dict(spec.context_payload),
                )
            )
        return tuple(entries)

    for spec in session_specs:
        entries.extend(_build_window_entries_for_spec(spec))
    return tuple(entries)


def _build_window_entries_for_spec(spec: _UABSessionSpec) -> list[StageITaskEntry]:
    entries: list[StageITaskEntry] = []
    window_delta = timedelta(seconds=WINDOW_DURATION_SECONDS)
    window_index = 0
    cursor = spec.start
    while cursor + window_delta <= spec.end:
        window_start = cursor
        window_end = cursor + window_delta
        context_payload = dict(spec.context_payload)
        context_payload["window_strategy"] = "fixed_5s_no_tail"
        entries.append(
            StageITaskEntry(
                sample_id=f"{spec.recording_id}__window_{window_index:05d}",
                dataset_id=DATASET_ID,
                subset_id=spec.subset_id,
                subject_id=spec.subject_id,
                session_id=spec.session_id,
                split_group=spec.split_group,
                training_role=spec.training_role,
                sample_granularity="window",
                recording_id=spec.recording_id,
                window_index=window_index,
                window_duration_s=WINDOW_DURATION_SECONDS,
                task_family="workload",
                label_namespace=spec.objective_label_name,
                window_start_utc=isoformat_utc(window_start),
                window_end_utc=isoformat_utc(window_end),
                source_refs=spec.source_refs,
                objective_label_name=spec.objective_label_name,
                objective_label_value=spec.objective_label_value,
                subjective_target_name=spec.subjective_target_name,
                subjective_target_value=spec.subjective_target_value,
                context_payload=context_payload,
            )
        )
        window_index += 1
        cursor = window_end
    return entries


def _tlx_mean(row: object) -> float:
    fields = (
        float(getattr(row, "mental_demand")),
        float(getattr(row, "physical_demand")),
        float(getattr(row, "temporal_demand")),
        float(getattr(row, "performance")),
        float(getattr(row, "effort")),
        float(getattr(row, "frustration")),
    )
    return sum(fields) / len(fields)


def _n_back_variant(test_value: int) -> str:
    if test_value == 1:
        return "position_1_back"
    if test_value == 2:
        return "arithmetic_1_back"
    if test_value == 3:
        return "dual_arithmetic_2_back"
    raise ValueError(f"unexpected n-back test value: {test_value}")


def _stream_group_bounds(
    parquet_path: Path,
    *,
    key_columns: tuple[str, ...],
    time_column: str,
) -> dict[tuple[object, ...], dict[str, datetime]]:
    reader = pq.ParquetFile(parquet_path)
    bounds: dict[tuple[object, ...], dict[str, datetime]] = {}
    for batch in reader.iter_batches(batch_size=65_536, columns=[*key_columns, time_column]):
        frame = batch.to_pandas()
        if frame.empty:
            continue
        frame = frame.sort_values([*key_columns, time_column])
        grouped = frame.groupby(list(key_columns), sort=False)[time_column].agg(["min", "max"])
        for key, row in grouped.iterrows():
            normalized = _normalize_key(key if isinstance(key, tuple) else (key,))
            candidate_start = row["min"].to_pydatetime()
            candidate_end = row["max"].to_pydatetime()
            current = bounds.get(normalized)
            if current is None:
                bounds[normalized] = {
                    "start": candidate_start,
                    "end": candidate_end,
                }
                continue
            if candidate_start < current["start"]:
                current["start"] = candidate_start
            if candidate_end > current["end"]:
                current["end"] = candidate_end
    return bounds


def _infer_heat_test_mapping(
    root: Path,
    bounds: Mapping[tuple[object, ...], Mapping[str, object]],
) -> dict[str, dict[int, str]]:
    performance_dir = root / "data_heat_the_chair" / "game_performance"
    subject_mapping: dict[str, dict[int, str]] = {}
    for path in sorted(performance_dir.glob("subject_*_*.csv")):
        subject_id, game = _parse_heat_performance_filename(path)
        frame = pd.read_csv(path)
        timestamps = frame["timestamp"].dropna().astype(float)
        if timestamps.empty:
            raise ValueError(f"cannot infer heat test mapping from empty file: {path}")
        first_timestamp = datetime.fromtimestamp(float(timestamps.iloc[0]), tz=timezone.utc).replace(tzinfo=None)
        subject_bounds = {
            int(test_value): window
            for (bound_subject, test_value), window in bounds.items()
            if str(bound_subject) == subject_id
        }
        best_test = min(
            subject_bounds,
            key=lambda test_value: abs((subject_bounds[test_value]["start"] - first_timestamp).total_seconds()),
        )
        subject_mapping.setdefault(subject_id, {})[best_test] = game
    return subject_mapping


def _parse_heat_performance_filename(path: Path) -> tuple[str, str]:
    stem = path.stem
    if stem.endswith("_with"):
        return stem.removesuffix("_with"), "with"
    if stem.endswith("_without"):
        return stem.removesuffix("_without"), "without"
    raise ValueError(f"unexpected heat performance filename: {path.name}")


def _choose_overlap_bounds(
    primary: Mapping[str, datetime],
    *optional_bounds: Mapping[str, datetime] | None,
) -> dict[str, datetime]:
    available = [primary, *(bounds for bounds in optional_bounds if bounds is not None)]
    if len(available) == 1:
        return {"start": primary["start"], "end": primary["end"]}
    start = max(bounds["start"] for bounds in available)
    end = min(bounds["end"] for bounds in available)
    if start >= end:
        return {"start": primary["start"], "end": primary["end"]}
    return {"start": start, "end": end}


def _normalize_key(values: Sequence[object]) -> tuple[object, ...]:
    normalized: list[object] = []
    for value in values:
        if isinstance(value, str):
            normalized.append(value)
        elif isinstance(value, float) and value.is_integer():
            normalized.append(int(value))
        else:
            normalized.append(value)
    return tuple(normalized)
