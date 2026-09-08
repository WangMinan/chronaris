"""Generate the fixed G2 confirmation inventory only after all selected models freeze."""
from dataclasses import fields, replace
import json
from pathlib import Path

from chronaris.evaluation.application_tasks.application_consumer_smoke_data import ApplicationConsumerSmokeData
from chronaris.evaluation.application_tasks.v4_confirmation_training import read_frozen_configuration
from chronaris.evaluation.application_tasks.v4_development_data import v4_workflow_source_sha256
from chronaris.evaluation.application_tasks.v4_simulation_confirmation import SIMULATION_REGISTRY, simulation_confirmation_units
from chronaris.modeling.training.candidate_screen import _periodic_training_heartbeat
from chronaris.representation import collate_observation_samples, load_simulation_observed_context
from chronaris.simulation.aviation_dual_stream.benchmark import SimulationBenchmarkConfig, SimulationSplitSpec, generate_benchmark
from chronaris.simulation.aviation_dual_stream.config import ObservationScenarioConfig, locked_stress_observation_scenarios
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file
from chronaris.simulation.aviation_dual_stream.profiles import sample_pilot_profile


def read_simulation_model_freeze(path, expected_sha256, *, freeze_path, freeze_sha256):
    frozen = read_frozen_configuration(freeze_path, freeze_sha256)
    if sha256_file(path) != expected_sha256:
        raise ValueError('frozen simulation model inventory changed')
    models = json.loads(Path(path).read_text())
    if (models['format'] != 'chronaris.v4_simulation_model_freeze.v1' or models['status'] != 'frozen'
        or models['freeze_sha256'] != freeze_sha256 or models['source_code_sha256'] != v4_workflow_source_sha256()
        or models['simulation_registry_sha256'] != sha256_file(SIMULATION_REGISTRY)):
        raise ValueError('simulation model inventory is not frozen under the selected configuration')
    expected = simulation_confirmation_units(frozen)
    actual = [{key: row[key] for key in ('method','candidate_name','seed','routes')} for row in models['records']]
    if actual != expected or models['evaluation_units'] != 36 or not models['files']:
        raise ValueError('simulation model inventory lacks the complete six-method three-seed scope')
    for filename, digest in models['files'].items():
        if sha256_file(filename) != digest:
            raise ValueError('frozen simulation model evidence changed')
    return models


def generate_simulation_confirmation(*, freeze_path, freeze_sha256, model_freeze_path, model_freeze_sha256, output_root):
    read_simulation_model_freeze(model_freeze_path, model_freeze_sha256, freeze_path=freeze_path, freeze_sha256=freeze_sha256)
    registry = json.loads(Path(SIMULATION_REGISTRY).read_text())
    if registry['seed_collision_audit']['collisions'] or not registry['confirmation_generation_requires_frozen_models']:
        raise ValueError('simulation confirmation seed isolation or model-freeze contract changed')
    for filename, digest in registry['generator_source_sha256'].items():
        if sha256_file(filename) != digest:
            raise ValueError('simulation generator changed since the parameter inventory was frozen')
    for item in registry['historical_generation_manifests']:
        if sha256_file(item['path']) != item['sha256']:
            raise ValueError('historical seed-collision audit source changed')
    spec = next(row for row in registry['split_specs'] if row['role'] == 'confirmation')
    split = SimulationSplitSpec(**{field.name: spec[field.name] for field in fields(SimulationSplitSpec)})
    if (split.generator_family, split.profile_count, split.trajectories_per_profile) != ('g2_event_spline',16,8):
        raise ValueError('confirmation requires the frozen 16 profiles and 128 G2 trajectories')
    profiles = {row['profile_id']: row for row in registry['profiles'] if row['role'] == 'confirmation'}
    for index in range(split.profile_count):
        profile = sample_pilot_profile(split_id=split.split_id, profile_index=index, seed=split.profile_seed_base+index)
        if any(profiles[profile.profile_id][key] != value for key, value in profile.to_dict().items()):
            raise ValueError('confirmation parameters differ from the preselected profile inventory')
    stress = locked_stress_observation_scenarios()
    if len(stress) != 35:
        raise ValueError('formal simulation requires the fixed 35 scenarios')
    scenarios = (ObservationScenarioConfig('clean_asynchronous'), *stress)
    root = Path(output_root).resolve(); root.mkdir(parents=True, exist_ok=True)
    contract = dict(format='chronaris.v4_simulation_confirmation_generation.v1', freeze_sha256=freeze_sha256,
        model_freeze_sha256=model_freeze_sha256, registry_sha256=sha256_file(SIMULATION_REGISTRY),
        source_code_sha256=v4_workflow_source_sha256(), scenarios=[item.to_dict() for item in scenarios], split_spec=split.to_dict())
    path = root/'confirmation_generation_contract.json'
    if path.exists() and json.loads(path.read_text()) != contract:
        raise ValueError('confirmation generation source, models or data configuration changed')
    if not path.exists():
        path.write_text(json.dumps(contract, indent=2)+'\n')
    # The shared historical generator can repair files; reject changed evidence before calling it.
    for path in root.glob('v4_confirmation/*/*/*/scenario_manifest.json'):
        stored = json.loads(path.read_text())
        for kind, key in (('raw_dual_stream','raw_dual_stream_sha256'),('ground_truth','ground_truth_sha256')):
            if sha256_file(stored[kind+'_path']) != stored[key]:
                raise ValueError('existing confirmation data changed; do not overwrite evidence')
    with _periodic_training_heartbeat('v4_confirmation_generation',30,root=root) as progress:
        result = generate_benchmark(SimulationBenchmarkConfig(run_id=root.name,output_root=str(root.parent),
            duration_s=registry['duration_s'],truth_rate_hz=registry['truth_rate_hz'],split_specs=(split,),
            observation_scenarios=scenarios,paired_observation_seed=True,resume=True),
            progress_callback=lambda event,values: progress.update(phase=event,**values))
    generated = json.loads(Path(result.simulation_manifest_path).read_text())['scenario_rows']
    planned = {row['trajectory_id']: row for row in registry['trajectories'] if row['role']=='confirmation'}
    if len(generated) != 128*36 or {(row['trajectory_id'],row['scenario_id']) for row in generated} != {
        (trajectory,scenario.scenario_id) for trajectory in planned for scenario in scenarios}:
        raise ValueError('confirmation generated an incomplete trajectory/scenario inventory')
    if any(row['observation_seed'] != planned[row['trajectory_id']]['observation_seed'] for row in generated):
        raise ValueError('confirmation observations used another seed namespace')
    if not all(row['all_states_present'] and row['event_count']>=2 and row['vehicle_values_finite']
        and row['physiology_values_finite'] and row['vehicle_clock_mapping_max_error_s']<=1e-9
        and row['physiology_clock_mapping_max_error_s']<=1e-9 for row in result.validation_rows):
        raise ValueError('confirmation generator failed its data-only acceptance')
    if not all(row['latent_hash_shared'] and row['trajectory_id_shared'] and row['scenario_ids_unique'] for row in result.paired_rows):
        raise ValueError('confirmation pressure scenarios do not share their clean latent truth')
    audit = dict(status='completed', model_freeze_sha256=model_freeze_sha256,
        contract_sha256=sha256_file(root/'confirmation_generation_contract.json'),
        simulation_manifest_sha256=sha256_file(root/'simulation_manifest.json'),
        registry_sha256=sha256_file(SIMULATION_REGISTRY), trajectory_count=128, profile_count=16,
        scenario_count=36, stress_scenario_count=35, contexts_per_scenario=512, model_scores_generated=False)
    (root/'confirmation_generation_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    return audit


def load_simulation_confirmation(root, *, model_freeze_sha256, condition='clean_asynchronous'):
    root = Path(root)
    audit = json.loads((root/'confirmation_generation_audit.json').read_text())
    if (audit['status']!='completed' or audit['model_freeze_sha256']!=model_freeze_sha256
        or audit['simulation_manifest_sha256']!=sha256_file(root/'simulation_manifest.json')
        or audit['registry_sha256']!=sha256_file(SIMULATION_REGISTRY)
        or audit['contract_sha256']!=sha256_file(root/'confirmation_generation_contract.json')):
        raise ValueError('confirmation observations differ from the frozen generation audit')
    registry = json.loads(Path(SIMULATION_REGISTRY).read_text())
    original_condition = condition
    if condition in ('physiology_missing','vehicle_missing'):
        condition = 'clean_asynchronous'
    generated = {row['trajectory_id']: row for row in json.loads((root/'simulation_manifest.json').read_text())['scenario_rows']
                 if row['scenario_id']==condition}
    planned = [row for row in registry['trajectories'] if row['role']=='confirmation']
    if set(generated) != {row['trajectory_id'] for row in planned}:
        raise ValueError('confirmation condition lacks the complete preselected trajectories')
    samples = []; rows = []
    for trajectory in planned:
        item = generated[trajectory['trajectory_id']]
        path = Path(item['scenario_manifest_path']).with_name('raw_dual_stream.npz')
        if sha256_file(path)!=item['raw_sha256'] or sha256_file(path.with_name('ground_truth.npz'))!=item['ground_truth_sha256']:
            raise ValueError('confirmation observations or targets changed')
        for start, sample_id in zip(registry['context_starts_s'],trajectory['context_sample_ids'],strict=True):
            sample = load_simulation_observed_context(path, context_start_s=start)
            if sample.group_id!=trajectory['profile_id']:
                raise ValueError('confirmation parameter group changed')
            if original_condition in ('physiology_missing','vehicle_missing'):
                from chronaris.evaluation.application_tasks.v4_development_conditions import apply_development_missingness
                sample = apply_development_missingness(sample, original_condition)
            sample = replace(sample, sample_id=sample_id)
            samples.append(sample)
            rows.append(dict(sample_id=sample_id,group_id=sample.group_id,profile_id=sample.group_id,
                trajectory_id=trajectory['trajectory_id'],role='held_out',context_start_s=start,context_end_s=start+30.,
                observed_path=str(path),observed_sha256=item['raw_sha256'],source_sample_hash=sample.source_sample_hash,
                condition=original_condition,oracle_opened_for_representation=False))
    roles = dict(train=(),validation=(),held_out=tuple(sample.sample_id for sample in samples))
    return ApplicationConsumerSmokeData(collate_observation_samples(samples),samples[0].schema,roles,tuple(rows))
