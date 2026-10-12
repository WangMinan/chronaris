import pytest

from chronaris.evaluation.application_tasks.device_resume import compatible_guard, verify_copy, GUARD


def test_device_guard_change_preserves_seed_and_cuda_rejections():
    source = 'if seed not in (17, 29, 43) or not torch.cuda.is_available() or "4090" not in torch.cuda.get_device_name():\n    raise ValueError("v4 experiments require RTX 4090 and an approved seed")\n'
    revised = compatible_guard(source)
    assert 'seed not in (17, 29, 43) or not torch.cuda.is_available()' in revised
    assert 'RTX 4070 SUPER' in revised
    with pytest.raises(ValueError, match='unexpected'):
        compatible_guard(revised)


def test_compatibility_rejects_unreviewed_source_change(tmp_path):
    old, new = tmp_path/'old', tmp_path/'new'
    source = 'if "4090" not in torch.cuda.get_device_name(): pass\n'
    for path, text in ((old/GUARD, source), (new/GUARD, compatible_guard(source))):
        path.parent.mkdir(parents=True)
        path.write_text(text)
    receipt = dict(files={}, legacy_workspace=str(old), workspace=str(new))
    verify_copy(receipt)
    (new/GUARD).write_text(compatible_guard(source)+'pass\n')
    with pytest.raises(ValueError, match='beyond the device guard'):
        verify_copy(receipt)
