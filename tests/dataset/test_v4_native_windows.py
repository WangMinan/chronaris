import numpy as np
import pandas as pd
import pytest

from chronaris.dataset.clare_native import build_clare_native_dataset, CENTRAL_COLUMNS
from chronaris.dataset.cogpilot_native import (
    build_cogpilot_difficulty_dataset, build_cogpilot_event_response_dataset, VEH_COLS,
)


def test_all_cogpilot_windows_events_reuse_record_tables_and_preserve_native_points(tmp_path, monkeypatch):
    run = tmp_path / "sub-cp001/ses-1/level-02B_run-001"
    run.mkdir(parents=True)
    origin = 738000.
    for token, step, columns in (
        ("lslshimmereda", .5, ("ppg_finger_mV", "eda_hand_l_kOhms")),
        ("lslshimmerresp", 1., ("respiration_trace_mV",)),
        ("lslshimmerecg", .25, ("ecg_projection_ll_ra_mV",)),
        ("lslxp11xpcac", .5, VEH_COLS),
    ):
        seconds = np.arange(0., 185., step)
        values = {"time_dn": origin + seconds / 86400., **{name: seconds + i for i, name in enumerate(columns)}}
        if "aircraft_roll_deg" in values:
            values["aircraft_roll_deg"] = np.sin(seconds * .5)
        if "eda_hand_l_kOhms" in values:
            values["eda_hand_l_kOhms"] = 1000. / (1. + seconds * .01)
        pd.DataFrame(values).to_csv(run / f"sample_stream-{token}_dat.csv", index=False)
    dataset = build_cogpilot_difficulty_dataset(tmp_path, subject_ids=("sub-cp001",),
        all_legal_windows=True, max_memory_cache_bytes=0)
    assert len(dataset.records) == 4
    np.testing.assert_allclose([(row.window_start_native - origin) * 86400. for row in dataset.records],
                               [60., 90., 120., 150.], atol=1e-5)
    calls = []
    original = pd.read_csv
    def track(*args, **kwargs):
        calls.append(args[0])
        return original(*args, **kwargs)
    monkeypatch.setattr(pd, "read_csv", track)
    for sample_id in dataset.sample_ids:
        sample = dataset.load_sample(sample_id)
        assert sample.physiology_feature_mask[:, -1].sum() >= 119
        assert len(sample.physiology_timestamps_s) > 96
    assert calls == []  # each recording was parsed once for bounds, then reused for all windows
    events = build_cogpilot_event_response_dataset(tmp_path, subject_ids=("sub-cp001",),
        max_events_per_run=None, minimum_event_gap_s=5.)
    assert len(events.records) > 3
    assert calls == []
    assert all(row.context_duration_s == 12. for row in events.records)
    with pytest.raises(ValueError, match="fixed native subject list is missing"):
        build_cogpilot_difficulty_dataset(tmp_path, subject_ids=("sub-cp999",), all_legal_windows=True)


def test_clare_preserves_all_ten_second_labels_and_fractional_scores(tmp_path):
    subject = "1390"
    for folder in ("EEG", "EDA", "ECG"):
        (tmp_path / folder / subject).mkdir(parents=True)
    (tmp_path / "Labels").mkdir()
    pd.DataFrame({"level_0": [5.5, 7.5, 8.5, 9.5]}).to_csv(tmp_path / "Labels/1390.csv", index=False)
    times = np.arange(120., 155., .5)
    pd.DataFrame({"Timestamp": times, **{name: times for name in CENTRAL_COLUMNS}}).to_csv(
        tmp_path / "EEG/1390/eeg_data_exp_0.csv", index=False)
    for folder, column, filename in (("EDA", "GSR Conductance CAL", "eda_data_experiment_0.csv"),
                                      ("ECG", "ECG LL-RA CAL", "ecg_data_experiment_0.csv")):
        pd.DataFrame({"Timestamp": times, column: times}).to_csv(tmp_path / folder / subject / filename, index=False)
    dataset = build_clare_native_dataset(tmp_path, subject_ids=(subject,), window_stride=1, all_legal_windows=True)
    assert dataset.labels == (5.5, 7.5, 8.5)
    assert [row.window_start_s for row in dataset.records] == [120., 130., 140.]
    for sample_id in dataset.sample_ids:
        sample = dataset.load_sample(sample_id)
        np.testing.assert_array_equal(sample.physiology_timestamps_s, np.arange(0., 10., .5))
