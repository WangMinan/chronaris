"""Final-model timing observations from the registered training/development trajectories."""
from dataclasses import replace
from pathlib import Path
import json

from chronaris.evaluation.application_tasks.v4_candidates import EXPANDED_SIMULATION_ROOT
from chronaris.evaluation.application_tasks.v4_pipeline_steps import write_result
from chronaris.evaluation.application_tasks.v4_simulation_confirmation_data import read_simulation_model_freeze
from chronaris.evaluation.application_tasks.v4_simulation_confirmation import SIMULATION_REGISTRY
from chronaris.modeling.training.candidate_screen import _periodic_training_heartbeat
from chronaris.representation import collate_observation_samples, load_simulation_observed_context
from chronaris.simulation.aviation_dual_stream.benchmark import SimulationBenchmarkConfig, SimulationSplitSpec, generate_benchmark
from chronaris.simulation.aviation_dual_stream.config import ObservationScenarioConfig, locked_stress_observation_scenarios
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file


def timing_scenarios():
    return (ObservationScenarioConfig('clean_asynchronous'), *[
        scenario for scenario in locked_stress_observation_scenarios()
        if scenario.scenario_id.startswith(('clock_offset_', 'physiology_lag_'))])


def generate_mechanism_data(*, formal_root, output_root):
    formal, root = Path(formal_root), Path(output_root).resolve()
    freeze, models = formal/'frozen_configuration.json', formal/'simulation_frozen_models.json'
    read_simulation_model_freeze(models, sha256_file(models), freeze_path=freeze, freeze_sha256=sha256_file(freeze))
    clean_root = Path(EXPANDED_SIMULATION_ROOT)
    if root==clean_root.resolve() or ((root/'simulation_manifest.json').exists() and not (root/'timing_protocol.json').exists()):
        raise ValueError('timing observations require a separate, dedicated output directory')
    clean = json.loads((clean_root/'simulation_manifest.json').read_text())
    registry = json.loads(Path(SIMULATION_REGISTRY).read_text())
    splits = tuple(SimulationSplitSpec(**spec) for spec in clean['split_specs'])
    if {(s.split_id, s.generator_family, s.profile_count, s.trajectories_per_profile) for s in splits} != {
        ('v4_train', 'g1_state_space', 64, 8), ('v4_development', 'g1_state_space', 8, 8)}:
        raise ValueError('timing probes require exactly the expanded training and fixed development trajectories')
    source = dict(model_freeze_sha256=sha256_file(models), clean_manifest_sha256=sha256_file(clean_root/'simulation_manifest.json'),
        registry_sha256=sha256_file(SIMULATION_REGISTRY), scenarios=[s.to_dict() for s in timing_scenarios()])
    path = root/'timing_protocol.json'
    if path.exists() and json.loads(path.read_text()) != source:
        raise ValueError('timing data source changed')
    write_result(path, source)
    for manifest in root.glob('v4_*/*/*/*/scenario_manifest.json'):
        stored = json.loads(manifest.read_text())
        for kind in ('raw_dual_stream', 'ground_truth'):
            if sha256_file(stored[kind+'_path']) != stored[kind+'_sha256']:
                raise ValueError('existing timing observations changed')
    with _periodic_training_heartbeat('final_timing_data', 30, root=root) as progress:
        result = generate_benchmark(SimulationBenchmarkConfig(run_id=root.name, output_root=str(root.parent),
            duration_s=registry['duration_s'], truth_rate_hz=registry['truth_rate_hz'], split_specs=splits,
            observation_scenarios=timing_scenarios(), paired_observation_seed=True, resume=True),
            progress_callback=lambda event, values: progress.update(phase=event, **values))
    generated = json.loads(Path(result.simulation_manifest_path).read_text())['scenario_rows']
    expected = {row['trajectory_id']: row for row in clean['scenario_rows']}
    actual = {(row['trajectory_id'], row['scenario_id']): row for row in generated}
    if len(actual) != len(generated) or set(actual) != {(t, s.scenario_id) for t in expected for s in timing_scenarios()}:
        raise ValueError('timing data escaped the complete training/development identity inventory')
    for trajectory, original in expected.items():
        if actual[(trajectory, 'clean_asynchronous')]['raw_sha256'] != original['raw_sha256']:
            raise ValueError('timing generation did not reproduce the original clean trajectory')
    if not all(row['latent_hash_shared'] and row['trajectory_id_shared'] and row['scenario_ids_unique'] for row in result.paired_rows):
        raise ValueError('timing factors changed the latent trajectory')
    if not all(row['vehicle_values_finite'] and row['physiology_values_finite']
        and row['vehicle_clock_mapping_max_error_s']<=1e-9 and row['physiology_clock_mapping_max_error_s']<=1e-9
        for row in result.validation_rows):
        raise ValueError('timing generation failed finite values or clock validation')
    return write_result(root/'timing_audit.json', dict(status='completed', source=source,
        simulation_manifest_sha256=sha256_file(root/'simulation_manifest.json'), trajectory_count=576,
        condition_count=len(timing_scenarios()), model_scores_produced=False, encoder_training=False))


def timing_batch(root, condition, manifest_rows):
    """Load native observations only, retaining the original fold's canonical IDs."""
    root = Path(root)
    audit = json.loads((root/'timing_audit.json').read_text())
    if audit['simulation_manifest_sha256'] != sha256_file(root/'simulation_manifest.json'):
        raise ValueError('timing manifest changed')
    scenarios = {row['trajectory_id']: row for row in json.loads((root/'simulation_manifest.json').read_text())['scenario_rows']
                 if row['scenario_id'] == condition}
    samples, rows = [], []
    checked = set()
    for original in manifest_rows:
        if original['role'] not in ('train', 'validation'):
            continue
        item = scenarios[original['trajectory_id']]
        path = Path(item['scenario_manifest_path']).with_name('raw_dual_stream.npz')
        if path not in checked:
            if sha256_file(path) != item['raw_sha256'] or sha256_file(path.with_name('ground_truth.npz')) != item['ground_truth_sha256']:
                raise ValueError('timing raw observations or targets changed')
            checked.add(path)
        sample = load_simulation_observed_context(path, context_start_s=original['context_start_s'])
        samples.append(replace(sample, sample_id=original['sample_id']))
        rows.append(dict(original, observed_path=str(path), scenario_id=condition))
    return collate_observation_samples(samples), rows
