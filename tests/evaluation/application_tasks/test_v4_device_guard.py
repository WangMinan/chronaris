import pytest

from chronaris.evaluation.application_tasks.v4_diagnostic_run import _require_diagnostic_device


@pytest.mark.parametrize('name', ['NVIDIA GeForce RTX 4090', 'NVIDIA GeForce RTX 4070 SUPER'])
@pytest.mark.parametrize('seed', [17, 29, 43])
def test_known_training_devices_accept_declared_seeds(monkeypatch, name, seed):
    monkeypatch.setattr('torch.cuda.is_available', lambda: True)
    monkeypatch.setattr('torch.cuda.get_device_name', lambda: name)
    _require_diagnostic_device(seed)


@pytest.mark.parametrize('available,name,seed', [
    (False, 'NVIDIA GeForce RTX 4070 SUPER', 17),
    (True, 'NVIDIA GeForce RTX 4070 SUPER', 99),
    (True, 'unqualified GPU', 17),
])
def test_device_migration_keeps_invalid_runtime_rejection(monkeypatch, available, name, seed):
    monkeypatch.setattr('torch.cuda.is_available', lambda: available)
    monkeypatch.setattr('torch.cuda.get_device_name', lambda: name)
    with pytest.raises(ValueError, match='approved seed'):
        _require_diagnostic_device(seed)
