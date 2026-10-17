"""Register native observed validation contexts for the already-trained full model."""
from pathlib import Path
import shutil

from chronaris.representation import load_simulation_observed_context
from chronaris.serving.fusion_workbench.catalog import inspect_model, compatible
from chronaris.serving.fusion_workbench.cases import ARRAYS, Case, read_json, write_json
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file, write_deterministic_npz


SIMULATION_CHECKPOINT = Path('/mnt/e/chronaris-v4-results/2026-10-11-chapter5-reload/core_training/simulation/chronaris/reference/fold01/seed17/self_supervised/chronaris/C/best.pt')
SIMULATION_ROOT = Path('/home/wangminan/projects/chronaris/artifacts/application_evaluation/2026-09-07_thesis-v4-simulation-expanded')


def simulation_samples(payload, ids):
    registry_path = Path('docs/requirements/thesis-v4-simulation-manifest.json')
    registry = read_json(registry_path)
    audit = read_json(SIMULATION_ROOT/'v4_generation_audit.json')
    assert audit['registry_sha256'] == sha256_file(registry_path)
    observed = {r['trajectory_id']: r for r in read_json(SIMULATION_ROOT/'simulation_manifest.json')['scenario_rows']}
    index = {sample: (r, start) for r in registry['trajectories'] if r['role'] in ('train', 'validation')
             for sample, start in zip(r['context_sample_ids'], registry['context_starts_s'], strict=True)}
    allowed = set(payload['fold']['train_sample_ids'] + payload['fold']['validation_sample_ids'])
    if not set(ids) <= allowed:
        raise ValueError('application sampling cannot open held-out observations')
    samples, provenance = [], []
    for sample_id in ids:
        planned, start = index[sample_id]
        saved = observed[planned['trajectory_id']]
        path = Path(saved['scenario_manifest_path']).with_name('raw_dual_stream.npz')
        assert sha256_file(path) == saved['raw_sha256']
        sample = load_simulation_observed_context(path, context_start_s=start)
        assert sample.sample_id == sample_id and sample.group_id == planned['profile_id']
        samples.append(sample)
        provenance.append(dict(sample_id=sample_id, role=planned['role'], start_s=start,
            trajectory_id=planned['trajectory_id'], raw_path=str(path), raw_sha256=saved['raw_sha256']))
    return samples, provenance


def prepare_full_cases(root):
    root = Path(root)
    if root.exists():
        raise ValueError('preserve the prior application root; choose a new directory')
    root.mkdir(parents=True)
    source = Path('artifacts/chapter6-final/service/cases/dingxin-validation')
    shutil.copytree(source, root/'cases/dingxin-validation')
    payload = inspect_model(SIMULATION_CHECKPOINT)
    config = payload['encoder_manifest']['backbone_config']
    assert config['semantic_event_enabled'] and config['learnable_semantic_queries']
    assert len(config['physics_calibration']['relations']) == 5
    labels = {'vehicle.speed_mps':'飞机速度（米/秒）', 'vehicle.altitude_m':'飞机高度（米）',
        'vehicle.roll_rad':'滚转角（弧度）', 'physiology.heart_rate_bpm':'心率（次/分）',
        'physiology.eeg_low_relative':'脑电低频相对分量', 'physiology.eeg_high_relative':'脑电高频相对分量',
        'physiology.eeg_complexity':'脑电复杂度', 'physiology.spo2_percent':'血氧饱和度（%）',
        'physiology.physiology_variability':'生理变异量', 'physiology.individual_baseline':'个体基线' }
    for case_id, count, title in (('simulation-validation', 3, '仿真数据 · 完整方法案例'),
                                  ('simulation-windows', 16, '仿真数据 · 独立轨迹窗口集合')):
        destination = root/'cases'/case_id; destination.mkdir()
        ids = payload['fold']['validation_sample_ids'][:count]
        samples, sources = simulation_samples(payload, ids)
        windows = []
        for i, (sample, source) in enumerate(zip(samples, sources)):
            assert compatible(sample.schema, payload)
            filename = f'window-{i+1:02}.npz'
            digest = write_deterministic_npz(destination/filename, {k:getattr(sample,k) for k in ARRAYS})
            windows.append(dict(file=filename, sha256=digest, sample_id=sample.sample_id,
                source_sample_hash=sample.source_sample_hash, group_id=sample.group_id,
                start_s=source['start_s'], duration_s=sample.context_duration_s))
        shutil.copyfile(SIMULATION_CHECKPOINT, destination/'encoder.pt')
        write_json(destination/'case.json', dict(format='chronaris.observations.v1', id=case_id, title=title,
            description='原生仿真观测与受物理约束训练的完整语义模型。', source_label='受控仿真数据',
            model_label='连续对齐与物理约束语义融合', checkpoint='encoder.pt',
            checkpoint_sha256=sha256_file(SIMULATION_CHECKPOINT), schema=samples[0].schema.to_dict(),
            field_labels=labels, windows=windows, source=dict(checkpoint=str(SIMULATION_CHECKPOINT),
            selection='first_internal_validation_contexts_without_outcome_selection', observations=sources)))
        Case(destination/'case.json')
    return root
