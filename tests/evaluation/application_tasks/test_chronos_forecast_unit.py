from pathlib import Path
import json
import datetime
import runpy
import subprocess
import sys
import pytest

@pytest.mark.parametrize('failure,expected',[(None,4),(2,2)])
@pytest.mark.parametrize('extension',[False,True])
def test_unit_stops_at_first_failure_and_preserves_receipt(tmp_path,monkeypatch,failure,expected,extension):
    script=Path(__file__).resolve().parents[3]/'scripts/evaluation/application_tasks/run_chronos_forecast_unit.py'
    main=runpy.run_path(str(script))['main'];report=tmp_path/'report';report.mkdir();assets=tmp_path/'assets'
    deadline=(datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(hours=4)).isoformat()
    argv=['unit','--report',str(report),'--assets',str(assets)]
    monkeypatch.setattr(sys,'argv',argv+(['--deadline',deadline] if extension else []))
    calls=[]
    def run(command,stdout,stderr):
        calls.append(command);stdout.write('stage retained\n')
        return subprocess.CompletedProcess(command,9 if len(calls)==failure else 0)
    monkeypatch.setattr(subprocess,'run',run)
    assert main()==(9 if failure else 0)
    receipt=json.loads((report/'pipeline_receipt.json').read_text())
    assert len(calls)==len(receipt['stages'])==expected
    assert receipt['status']==('stopped' if failure else 'completed')
    for command in calls:
        assert ('--deadline' in command)==(extension and not command[1].endswith('deliver_chronos_forecast.py'))
        if '--deadline' in command:assert command[-1]==deadline
    assert all(Path(s['log']).read_text()=='stage retained\n' for s in receipt['stages'])
    with pytest.raises(FileExistsError):main()
