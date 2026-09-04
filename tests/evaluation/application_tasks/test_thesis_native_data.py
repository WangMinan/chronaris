from dataclasses import dataclass

import numpy as np

from chronaris.evaluation.application_tasks.thesis_native_data import (
    first_record_indices,
    middle_dual_stream_indices,
    middle_record_indices,
)


@dataclass
class Record:
    sample_id: str
    group_id: str
    label: int


def test_frozen_native_selection_is_label_independent_where_required():
    records = [
        Record(f"g{group}__run_window{window}", f"g{group}", window % 2)
        for group in range(2)
        for window in range(5)
    ]

    assert first_record_indices(records, lambda row: (row.group_id, row.label)) == (
        0,
        1,
        5,
        6,
    )
    assert middle_record_indices(records, lambda row: row.group_id) == (2, 7)


def test_middle_dual_stream_selection_skips_invalid_central_window():
    records = [Record(f"g__run_window{index}", "g", 0) for index in range(5)]

    class Dataset:
        sample_ids = tuple(row.sample_id for row in records)

        def __init__(self):
            self.records = records

        def load_sample(self, sample_id):
            valid = not sample_id.endswith("window2")
            return type(
                "Sample",
                (),
                {
                    "physiology_feature_mask": np.asarray([[valid]]),
                    "vehicle_feature_mask": np.asarray([[valid]]),
                },
            )()

    assert middle_dual_stream_indices(Dataset(), lambda row: row.group_id) == (1,)
