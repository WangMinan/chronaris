from __future__ import annotations

import numpy as np

from chronaris.evaluation.application_tasks.application_metrics import (
    classification_metrics,
)


def test_macro_f1_keeps_absent_validation_class_in_fixed_label_set():
    metrics = classification_metrics(
        np.asarray([1, 1, 2, 2]),
        np.asarray([1, 1, 2, 2]),
        np.asarray(
            [
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
            ]
        ),
        np.asarray([0, 1, 2]),
    )

    assert metrics["macro_f1"][0] == 2.0 / 3.0
