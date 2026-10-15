"""Real-weight participation and historical-information checks for application acceptance."""
from dataclasses import replace

import torch

from chronaris.representation import collate_observation_samples
from chronaris.serving.fusion_workbench.inference import load_adapter
from chronaris.modeling.fusion_encoders.chronaris_physics import physics_audit_to_rows
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file


def verify_application_mechanisms(case, *, device="cpu"):
    adapter, payload = load_adapter(case, device=device)
    encoder = adapter.encoder.eval()
    raw = collate_observation_samples([case.sample(i) for i in range(3)])
    from chronaris.modeling.fusion_encoders.single_stream import move_observation_batch
    normalized = move_observation_batch(adapter.normalizer.transform(raw), device=device)
    with torch.inference_mode():
        output = encoder(normalized, compute_chronaris_diagnostics=True)
        semantic = output.auxiliary['semantic_event_output']
        assert semantic is not None and semantic.event_token_mask.any()
        module = encoder.backbone.semantic_event_fusion
        try:
            encoder.backbone.semantic_event_fusion = None
            without = encoder(normalized).sequence_embedding
        finally:
            encoder.backbone.semantic_event_fusion = module
        difference = float((output.sequence_embedding - without).abs().max())
        assert difference > 1e-6, 'event semantics did not affect the exported sequence'
        # Perturb only observations after a fixed query cutoff, leaving times and masks unchanged.
        changes = {}
        for stream in ('physiology', 'vehicle'):
            values = getattr(normalized, stream + '_values').clone()
            future = (getattr(normalized, stream + '_timestamps_s') > 15.)
            mask = future.unsqueeze(-1) & getattr(normalized, stream + '_feature_mask')
            values[mask] += 10.
            changes[stream + '_values'] = values
        changed = encoder(replace(normalized, **changes)).sequence_embedding
        prefix = (raw.query_timestamps_s <= 15.).to(output.sequence_embedding.device)
        torch.testing.assert_close(output.sequence_embedding[prefix], changed[prefix], atol=1e-5, rtol=1e-5)
        query = module.query_bank.query_residual
        assert query.abs().max() > 0, 'learnable semantic queries retained zero initialization'
        state = payload['optimizer_state_dict']['state']
        trained_steps = sorted({int(v['step']) for v in state.values() if 'step' in v})
    config = payload['encoder_manifest']['backbone_config']
    return dict(status='passed', device=str(next(encoder.parameters()).device), checkpoint_sha256=sha256_file(case.checkpoint),
        strict_load=True, training_status=payload['training_status'], optimizer_updates=payload['total_optimizer_updates'],
        optimizer_state_steps=trained_steps, selected_update=payload.get('best_update'),
        label_used_for_encoder_training=payload['label_used_for_encoder_training'],
        config=config, normalizer_sha256=payload['normalizer']['transform_sha256'],
        source_code_sha256=payload['source_code_sha256'], transfer_initialization=payload['transfer_initialization'],
        semantic_query_max_abs=float(query.detach().abs().max()), semantic_output_difference=difference,
        active_terminal_events=int(semantic.event_token_mask.sum()),
        event_times_s=semantic.event_token_center_offsets_s.tolist(),
        event_valid=semantic.event_token_mask.tolist(),
        query_to_event_attention=semantic.query_to_event_attention.tolist(),
        prefix_max_abs_difference=float((output.sequence_embedding[prefix]-changed[prefix]).abs().max()),
        tolerance=dict(atol=1e-5, rtol=1e-5),
        physics_audit=list(physics_audit_to_rows(output.auxiliary['physical_consistency'])),
        scope='application participation check, not downstream effectiveness comparison')
