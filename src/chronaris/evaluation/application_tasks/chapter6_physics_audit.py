"""Verify trained five-relation provenance and gradients without optimizer updates."""
from pathlib import Path
import torch

from chronaris.evaluation.application_tasks.chapter6_full_assets import SIMULATION_CHECKPOINT, simulation_samples
from chronaris.modeling.training import load_common_pretraining_checkpoint
from chronaris.modeling.training.chronaris_auxiliary import build_chronaris_auxiliary_losses
from chronaris.modeling.training.pretext import chronaris_auxiliary_weight_schedule
from chronaris.modeling.fusion_encoders.alignment_bridge import build_alignment_batch_from_observations
from chronaris.models.alignment.calibrated_physics import (SIMULATION_RELATIONS, fit_physics_calibration,
    calibrated_kinematic_losses, relation_samples, KinematicRelation)
from chronaris.representation import collate_observation_samples, select_observation_batch, TrainOnlyRobustNormalizer
from chronaris.serving.fusion_workbench.cases import write_json
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file


def audit_full_physics(output):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    before = sha256_file(SIMULATION_CHECKPOINT)
    encoder, _, saved_normalizer, payload = load_common_pretraining_checkpoint(SIMULATION_CHECKPOINT, device='cpu')
    fold = payload['fold']; config = payload['encoder_manifest']['backbone_config']
    calibration = config['physics_calibration']
    assert len(calibration['relations']) == 5 and payload['chronaris_mechanism_enabled']
    assert config['semantic_event_enabled'] and config['learnable_semantic_queries']
    samples, sources = simulation_samples(payload, fold['train_sample_ids'])
    training = collate_observation_samples(samples)
    provider = lambda ids: select_observation_batch(training, ids)
    normalizer = TrainOnlyRobustNormalizer().fit_from_batch_provider(provider,
        train_sample_ids=fold['train_sample_ids'],
        held_out_sample_ids=fold['validation_sample_ids']+fold['held_out_sample_ids'], batch_size=4)
    assert normalizer.to_manifest() == saved_normalizer.to_manifest()
    fitted = fit_physics_calibration(normalizer, provider, train_sample_ids=fold['train_sample_ids'],
        vehicle_feature_names=tuple(payload['vehicle_feature_names']), relations=SIMULATION_RELATIONS)
    assert fitted == calibration, 'training-only physical calibration changed'
    training_pairs = []
    for relation in SIMULATION_RELATIONS:
        residual, rhs = relation_samples(training.vehicle_values.double(), training.vehicle_timestamps_s.double(),
            training.vehicle_feature_mask & training.vehicle_point_mask.unsqueeze(-1),
            tuple(payload['vehicle_feature_names']), relation)
        training_pairs.append(dict(source=relation.source, target=relation.target, count=residual.numel()))
    chosen, validation_sources = simulation_samples(payload, fold['validation_sample_ids'][:3])
    raw = collate_observation_samples(chosen)
    normalized = normalizer.transform(raw)
    encoder.eval()
    encoded = encoder(normalized, compute_chronaris_diagnostics=True)
    aligned = encoded.auxiliary['alignment_output']
    batch = build_alignment_batch_from_observations(normalized,
        physiology_feature_names=tuple(payload['physiology_feature_names']),
        vehicle_feature_names=tuple(payload['vehicle_feature_names']))
    rows = calibrated_kinematic_losses(aligned.vehicle.reconstructions, batch.vehicle, calibration)
    physical = (aligned.vehicle.reconstructions.float()*torch.tensor(calibration['scale'])
                +torch.tensor(calibration['center']))
    parameters = tuple(encoder.backbone.continuous_backbone.vehicle_stream.parameters())
    relation_rows = []
    for spec, (_, loss, count) in zip(calibration['relations'], rows, strict=True):
        gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
        norm = sum(float(g.detach().square().sum()) for g in gradients if g is not None)**.5
        assert count > 0 and norm > 0 and torch.isfinite(loss)
        residual, _ = relation_samples(physical, batch.vehicle.offsets_s,
            batch.vehicle.feature_valid_mask & batch.vehicle.mask.unsqueeze(-1),
            tuple(payload['vehicle_feature_names']), KinematicRelation(**spec))
        relation_rows.append(dict(**spec, valid_pairs=count, raw_loss=float(loss.detach()),
            physical_residual_mean=float(residual.detach().mean()),
            physical_residual_abs_mean=float(residual.detach().abs().mean()),
            physical_residual_rmse=float(residual.detach().square().mean().sqrt()),
            mean_contribution=float(loss.detach())/5, weighted_contribution=float(loss.detach())/5*config['physics_weight'],
            vehicle_parameter_gradient_norm=norm))
    audit = encoded.auxiliary['physical_consistency']
    weights = chronaris_auxiliary_weight_schedule(1, optimizer_updates=payload['best_update'])
    losses = build_chronaris_auxiliary_losses(encoded, encoded, weights=weights)
    torch.testing.assert_close(losses.physical_consistency, torch.stack([r[1] for r in rows]).mean(), atol=0, rtol=0)
    assert abs(losses.physical_effective_weight-config['physics_weight']) < 1e-12
    total_gradient = torch.autograd.grad(losses.physical_consistency*losses.physical_effective_weight,
        parameters, allow_unused=True)
    total_norm = sum(float(g.detach().square().sum()) for g in total_gradient if g is not None)**.5
    assert total_norm > 0
    active_training = [r for r in payload['training_rows'] if r['term_name']=='chronaris_physical_consistency'
        and r.get('weight',0)>0 and r.get('count',0)>0]
    assert active_training, 'no actual active physics training record'
    assert sha256_file(SIMULATION_CHECKPOINT)==before
    write_json(output/'physics-training.json', dict(status='passed', checkpoint_sha256=before,
        checkpoint_path=str(SIMULATION_CHECKPOINT), training_status=payload['training_status'],
        source_code_sha256=payload['source_code_sha256'], config=payload['config'],
        best_update=payload['best_update'], total_optimizer_updates=payload['total_optimizer_updates'],
        training_roles={k:len(v) for k,v in fold.items() if isinstance(v,list)},
        label_used_for_encoder_training=payload['label_used_for_encoder_training'],
        normalization_refit_equal=True, calibration_refit_equal=True, calibration=calibration,
        training_pairs=training_pairs, validation_relations=relation_rows,
        weighted_loss=float(audit.total_weighted_value.detach()), weighted_gradient_norm=total_norm,
        active_training_rows=len(active_training), training_row_examples=active_training[:2]+active_training[-2:],
        validation_history=payload['epoch_rows'], observations=validation_sources,
        optimizer_updates_in_this_audit=0, checkpoint_unchanged=True))
    return relation_rows
