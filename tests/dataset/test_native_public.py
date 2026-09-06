from __future__ import annotations

import hashlib
import numpy as np
import pandas as pd

from chronaris.dataset.clare_native import build_clare_native_dataset
from chronaris.dataset.cogpilot_native import (
    PHYS_NAMES,
    _event_times_and_responses,
    _resistance_to_conductance,
    build_cogpilot_difficulty_dataset,
)
from chronaris.dataset.lazy_observed import (
    LazyObservedDataset,
    NativeSampleRecord,
    merge_native_feature_series,
)
from chronaris.representation import ObservationSchema, ObservedDualStreamSample


def test_native_series_union_preserves_sparse_feature_masks() -> None:
    timestamps, values, mask = merge_native_feature_series(
        (
            (np.array([0.0, 1.0]), np.array([[1.0], [2.0]]), (0,)),
            (np.array([0.5, 1.0]), np.array([[3.0], [np.nan]]), (1,)),
        ),
        feature_count=2,
    )

    np.testing.assert_array_equal(timestamps, [0.0, 0.5, 1.0])
    np.testing.assert_array_equal(mask, [[True, False], [False, True], [True, False]])
    np.testing.assert_array_equal(values[mask], [1.0, 3.0, 2.0])


def test_lazy_dataset_reuses_lineage_checked_disk_cache(tmp_path) -> None:
    schema = ObservationSchema(
        schema_id="lazy_test.v1",
        source_kind="unit_test",
        physiology_feature_names=("p",),
        vehicle_feature_names=("v",),
        physiology_feature_roles=("observed",),
        vehicle_feature_roles=("observed",),
    )
    record = NativeSampleRecord(
        sample_id="sample",
        group_id="group",
        label=1,
        context_duration_s=2.0,
        source_sample_hash=hashlib.sha256(b"sample").hexdigest(),
    )
    calls = []

    def load(value):
        calls.append(value.sample_id)
        return ObservedDualStreamSample(
            sample_id=value.sample_id,
            group_id=value.group_id,
            schema=schema,
            physiology_values=np.array([[1.0]], dtype=np.float32),
            physiology_timestamps_s=np.array([0.25]),
            physiology_feature_mask=np.array([[True]]),
            vehicle_values=np.array([[2.0]], dtype=np.float32),
            vehicle_timestamps_s=np.array([0.5]),
            vehicle_feature_mask=np.array([[True]]),
            source_sample_hash=value.source_sample_hash,
            context_duration_s=value.context_duration_s,
        )

    first = LazyObservedDataset([record], schema=schema, loader=load, cache_root=tmp_path)
    first.load_sample("sample")
    second = LazyObservedDataset(
        [record],
        schema=schema,
        loader=lambda _record: (_ for _ in ()).throw(AssertionError("cache miss")),
        cache_root=tmp_path,
    )
    restored = second.load_sample("sample")

    assert calls == ["sample"]
    assert restored.physiology_values.item() == 1.0

    from chronaris.dataset.native_table_cache import native_file_sha256
    cache_path = second._cache_path(record)
    verified = LazyObservedDataset([record], schema=schema, loader=load, cache_root=tmp_path,
        cache_file_sha256={record.sample_id: native_file_sha256(cache_path)})
    verified.load_sample(record.sample_id)
    cache_path.write_bytes(cache_path.read_bytes() + b"changed")
    import pytest
    with pytest.raises(ValueError, match="cache bytes changed"):
        verified.load_sample(record.sample_id)


def test_cogpilot_builder_keeps_native_stream_densities(tmp_path) -> None:
    run = tmp_path / "sub-cp001" / "ses-1" / "level-02B_run-001"
    run.mkdir(parents=True)
    origin = 738000.0

    def write(token, frame):
        frame.to_csv(run / f"sample_stream-{token}_dat.csv", index=False)

    seconds = np.arange(0.0, 100.0, 1.0)
    days = origin + seconds / 86400.0
    write(
        "lslshimmereda",
        pd.DataFrame({"time_dn": days, "ppg_finger_mV": seconds, "eda_hand_l_kOhms": seconds + 1}),
    )
    resp_seconds = np.arange(0.0, 100.0, 2.0)
    write(
        "lslshimmerresp",
        pd.DataFrame({"time_dn": origin + resp_seconds / 86400.0, "respiration_trace_mV": resp_seconds}),
    )
    write(
        "lslshimmerecg",
        pd.DataFrame({"time_dn": days, "ecg_projection_ll_ra_mV": np.sin(seconds * np.pi)}),
    )
    vehicle_seconds = np.arange(0.0, 100.0, 0.5)
    vehicle = {"time_dn": origin + vehicle_seconds / 86400.0}
    from chronaris.dataset.cogpilot_native import VEH_COLS

    vehicle.update({name: vehicle_seconds + index for index, name in enumerate(VEH_COLS)})
    write("lslxp11xpcac", pd.DataFrame(vehicle))

    dataset = build_cogpilot_difficulty_dataset(
        tmp_path,
        subject_limit=1,
        window_start_s=60.0,
        context_duration_s=30.0,
        cache_root=tmp_path / "cache",
    )
    sample = dataset.load_sample(dataset.sample_ids[0])

    assert sample.schema.schema_id == "cogpilot_native.v4"
    assert PHYS_NAMES[-1] == "physiology.ecg"
    np.testing.assert_allclose(
        sample.physiology_values[sample.physiology_feature_mask[:, 1], 1],
        1000.0 / np.arange(61.0, 91.0),
        rtol=1e-6,
    )
    assert len(sample.vehicle_timestamps_s) > len(sample.physiology_timestamps_s)
    assert (~sample.physiology_feature_mask[:, 2]).any()
    assert not np.array_equal(sample.physiology_timestamps_s, sample.vehicle_timestamps_s)

    ecg_path = next(run.glob("*stream-lslshimmerecg*_dat.csv"))
    ecg = pd.read_csv(ecg_path)
    ecg.loc[seconds >= 90.0, "ecg_projection_ll_ra_mV"] = 1e9
    ecg.to_csv(ecg_path, index=False)
    replay = build_cogpilot_difficulty_dataset(
        tmp_path,
        subject_limit=1,
        window_start_s=60.0,
        context_duration_s=30.0,
        cache_root=tmp_path / "cache_replay",
    ).load_sample(dataset.sample_ids[0])
    np.testing.assert_array_equal(
        sample.physiology_timestamps_s,
        replay.physiology_timestamps_s,
    )
    np.testing.assert_array_equal(sample.physiology_values, replay.physiology_values)


def test_cogpilot_builder_skips_runs_with_missing_streams(tmp_path) -> None:
    incomplete = tmp_path / "sub-cp001" / "ses-1" / "level-01B_run-001"
    complete = tmp_path / "sub-cp001" / "ses-1" / "level-02B_run-002"
    incomplete.mkdir(parents=True)
    complete.mkdir(parents=True)
    (incomplete / "placeholder.txt").write_text("missing streams")

    origin = 738000.0
    seconds = np.arange(0.0, 100.0)
    paths = {
        "lslshimmereda": pd.DataFrame(
            {
                "time_dn": origin + seconds / 86400.0,
                "ppg_finger_mV": seconds,
                "eda_hand_l_kOhms": seconds,
            }
        ),
        "lslshimmerresp": pd.DataFrame(
            {
                "time_dn": origin + seconds / 86400.0,
                "respiration_trace_mV": seconds,
            }
        ),
        "lslshimmerecg": pd.DataFrame(
            {
                "time_dn": origin + seconds / 86400.0,
                "ecg_projection_ll_ra_mV": seconds,
            }
        ),
    }
    from chronaris.dataset.cogpilot_native import VEH_COLS

    vehicle = {"time_dn": origin + seconds / 86400.0}
    vehicle.update({name: seconds for name in VEH_COLS})
    paths["lslxp11xpcac"] = pd.DataFrame(vehicle)
    for token, frame in paths.items():
        frame.to_csv(complete / f"sample_stream-{token}_dat.csv", index=False)

    dataset = build_cogpilot_difficulty_dataset(tmp_path, subject_limit=1)

    assert dataset.sample_ids == ("sub-cp001::level-02B_run-002",)


def test_clare_windows_start_at_recording_timestamp_without_interpolation(tmp_path) -> None:
    subject = "1001"
    for folder in ("EEG", "EDA", "ECG", "Labels"):
        (tmp_path / folder / subject if folder != "Labels" else tmp_path / folder).mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"level_0": [5.0]}).to_csv(tmp_path / "Labels" / f"{subject}.csv", index=False)
    eeg_t = np.arange(120.0, 130.0, 0.5)
    eeg = {"Timestamp": eeg_t}
    eeg.update({name: eeg_t + index for index, name in enumerate(("TP9", "AF7", "AF8", "TP10"))})
    pd.DataFrame(eeg).to_csv(tmp_path / "EEG" / subject / "eeg_data_exp_0.csv", index=False)
    periph_t = np.arange(120.25, 130.0, 0.25)
    pd.DataFrame({"Timestamp": periph_t, "GSR Conductance CAL": periph_t}).to_csv(
        tmp_path / "EDA" / subject / "eda_data_experiment_0.csv", index=False
    )
    ecg_path = tmp_path / "ECG" / subject / "ecg_data_experiment_0.csv"
    ecg_t = np.concatenate((periph_t, (130.25,)))
    pd.DataFrame(
        {"Timestamp": ecg_t, "ECG LL-RA CAL": np.sin(ecg_t * 8)}
    ).to_csv(ecg_path, index=False)

    dataset = build_clare_native_dataset(tmp_path, subject_limit=1, cache_root=tmp_path / "cache")
    sample = dataset.load_sample(dataset.sample_ids[0])

    assert sample.schema.schema_id == "clare_native.v4"
    assert sample.schema.vehicle_feature_names[-1] == "peripheral.ecg"
    assert sample.physiology_timestamps_s[0] == 0.0
    assert sample.vehicle_timestamps_s[0] == 0.25
    assert len(sample.physiology_timestamps_s) != len(sample.vehicle_timestamps_s)

    ecg = pd.read_csv(ecg_path)
    ecg.loc[ecg["Timestamp"] >= 130.0, "ECG LL-RA CAL"] = 1e9
    ecg.to_csv(ecg_path, index=False)
    replay = build_clare_native_dataset(
        tmp_path,
        subject_limit=1,
        cache_root=tmp_path / "cache_replay",
    ).load_sample(dataset.sample_ids[0])
    np.testing.assert_array_equal(sample.vehicle_timestamps_s, replay.vehicle_timestamps_s)
    np.testing.assert_array_equal(sample.vehicle_values, replay.vehicle_values)


def test_cogpilot_resistance_conversion_masks_nonphysical_values() -> None:
    converted = _resistance_to_conductance(
        np.asarray((1000.0, 500.0, 0.0, -1.0, np.nan))
    )

    np.testing.assert_allclose(converted[:2], (1.0, 2.0))
    assert np.isnan(converted[2:]).all()


def test_cogpilot_event_response_uses_conductance_median_delta(tmp_path) -> None:
    seconds = np.arange(0.0, 100.0, 0.1)
    origin = 738000.0
    vehicle_path = tmp_path / "vehicle.csv"
    eda_path = tmp_path / "eda.csv"
    pd.DataFrame(
        {
            "time_dn": origin + seconds / 86400.0,
            "aircraft_roll_deg": np.sin(seconds * 0.5),
        }
    ).to_csv(vehicle_path, index=False)
    conductance = 1.0 + seconds * 0.01
    resistance = 1000.0 / conductance
    resistance[500] = -1_000_000.0
    pd.DataFrame(
        {
            "time_dn": origin + seconds / 86400.0,
            "eda_hand_l_kOhms": resistance,
        }
    ).to_csv(eda_path, index=False)

    events = _event_times_and_responses(
        vehicle_path,
        eda_path,
        context_duration_s=12.0,
        response_pre_s=2.0,
        response_post_s=8.0,
        minimum_event_gap_s=5.0,
    )

    assert events
    np.testing.assert_allclose([value for _time, value in events], 0.05, atol=0.003)
