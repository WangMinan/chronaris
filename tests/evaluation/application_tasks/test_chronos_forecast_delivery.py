import pandas as pd
import pytest
from chronaris.evaluation.application_tasks.chronos_forecast_delivery import delivery_text

def test_delivery_requires_complete_verified_unit_and_discloses_selection():
    units=[dict(fold=f,view=v,epochs=2,best_epoch=0,updates=3) for f in ('fold01','fold02') for v in ('aircraft','joint')]
    run=dict(status='completed',completed_models=4,updates=12,windows_seen=48,units=units)
    result=dict(status='completed',paired_routes=[])
    check=dict(status='passed',models=4,updates=12,protected_files=10,predictions=20,subject_metrics=8,maximum_metric_delta=0.,replay={'one':0.})
    methods=['native_frozen_aircraft','native_frozen_joint','native_adapted_aircraft','native_adapted_joint','reference_aircraft']
    summary=pd.DataFrame([dict(fold=f,method=m,rmse=.3) for f in ('fold01','fold02') for m in methods])
    text=delivery_text(run,result,check,summary)
    assert '尚未同时通过' in text and '选择第0轮' in text and '重新冻结' in text
    with pytest.raises(AssertionError):delivery_text({**run,'status':'budget_saved'},result,check,summary)
    with pytest.raises(AssertionError):delivery_text(run,result,{**check,'status':'failed'},summary)
