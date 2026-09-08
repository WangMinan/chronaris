from pathlib import Path
import runpy


def test_confirmation_compatibility_entry_reuses_the_single_pipeline(monkeypatch):
    from chronaris.evaluation.application_tasks import v4_pipeline
    calls=[]
    monkeypatch.setattr(v4_pipeline,'main',lambda **kwargs:calls.append(kwargs))
    script=Path(__file__).parents[3]/'scripts/evaluation/application_tasks/run_v4_confirmation_chain.py'
    runpy.run_path(str(script),run_name='__main__')
    assert calls==[{'default_until':'confirmation'}]
