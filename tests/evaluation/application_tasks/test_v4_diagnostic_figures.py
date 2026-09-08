from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from chronaris.evaluation.application_tasks.v4_diagnostic_figures import _select_scores, _matrix


def test_initial_stage_scope_and_missing_statistics_are_explicit():
    root = Path(__file__).resolve().parents[3]
    frame = pd.read_csv(root / 'docs/artifacts/runs/2026-09-07_v4-initial-diagnostics/metric_long.csv')
    selected = _select_scores(frame)
    assert len(selected) == 180
    for changed in (selected.iloc[1:], pd.concat([selected, selected.iloc[:1]]),
                    selected.assign(seed=29), selected.assign(fold='expanded_training512')):
        with pytest.raises(ValueError):
            _select_scores(changed)
    fig, axis = plt.subplots()
    values = np.ones((2,8));values[1,3] = np.nan
    plot = _matrix(axis, values, ['自监督', '任务引导'], '状态统计')
    assert plot.get_array().mask[1,3]
    assert [text.get_text() for text in axis.texts].count('不可用') == 1
    plt.close(fig)
