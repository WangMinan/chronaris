import hashlib

import numpy as np

from chronaris.evaluation.application_tasks.thesis_case_evidence import (
    input_activity_score,
)
from chronaris.representation import (
    ObservationSchema,
    ObservedDualStreamSample,
    collate_observation_samples,
)


def test_input_activity_score_uses_only_observed_vehicle_changes():
    schema = ObservationSchema(
        schema_id="case-test",
        source_kind="unit_test",
        physiology_feature_names=("p",),
        vehicle_feature_names=("steady", "changing"),
        physiology_feature_roles=("observed",),
        vehicle_feature_roles=("observed", "observed"),
    )
    sample = ObservedDualStreamSample(
        sample_id="sample",
        group_id="sortie",
        schema=schema,
        physiology_values=np.ones((2, 1), np.float32),
        physiology_timestamps_s=np.asarray((0.0, 1.0)),
        physiology_feature_mask=np.ones((2, 1), bool),
        vehicle_values=np.asarray(((1.0, 0.0), (1.0, 1.0), (1.0, 3.0))),
        vehicle_timestamps_s=np.asarray((0.0, 0.5, 1.0)),
        vehicle_feature_mask=np.ones((3, 2), bool),
        source_sample_hash=hashlib.sha256(b"sample").hexdigest(),
    )

    assert input_activity_score(collate_observation_samples((sample,))) > 0
