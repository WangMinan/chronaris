"""Explicit execution identity and synchronized inference using the standard feature contract."""
import time
import torch

from chronaris.modeling.fusion_encoders.single_stream import move_observation_batch
from chronaris.modeling.training import TrainedFusionAdapter
from chronaris.representation import FusionStreamBatch
from chronaris.representation.contracts import pool_exported_sequence


def execution_identity(device='cpu'):
    if device not in ('cpu', 'cuda', 'cuda:0'):
        raise ValueError('运行设备须为 cpu 或 cuda')
    resolved = 'cuda:0' if device.startswith('cuda') else 'cpu'
    if resolved != 'cpu' and not torch.cuda.is_available():
        raise RuntimeError('请求的GPU不可用；未回退到CPU')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return dict(device=resolved, precision='float32', matmul_tf32=False,
        device_name=torch.cuda.get_device_name(0) if resolved != 'cpu' else 'CPU',
        torch_version=torch.__version__, cuda_runtime=torch.version.cuda if resolved != 'cpu' else None)


class DeviceFusionAdapter(TrainedFusionAdapter):
    """Keep normalization, transfer, forward and CPU export separately measurable."""
    def __call__(self, batch):
        device = next(self.encoder.parameters()).device
        def sync():
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
        sync(); start = time.perf_counter()
        normalized = self.normalizer.transform(batch)
        normalized_s = time.perf_counter() - start
        start = time.perf_counter()
        normalized = move_observation_batch(normalized, device=device)
        sync(); transfer_s = time.perf_counter() - start
        tensor_devices = {str(v.device) for v in vars_from_batch(normalized)}
        if tensor_devices != {str(device)}:
            raise RuntimeError('模型与输入张量的运行设备不一致')
        self.encoder.eval()
        start = time.perf_counter()
        with torch.inference_mode():
            encoded = self.encoder(normalized)
            sequence = encoded.sequence_embedding.masked_fill(~encoded.modality_available_mask.unsqueeze(-1), 0)
        sync(); forward_s = time.perf_counter() - start
        start = time.perf_counter()
        output_device = str(sequence.device)
        sequence = sequence.detach().cpu()
        valid = encoded.modality_available_mask.detach().cpu()
        pooled = pool_exported_sequence(sequence, valid)
        sync(); export_s = time.perf_counter() - start
        self.last_execution = dict(parameter_device=str(device), input_tensor_devices=sorted(tensor_devices),
            forward_output_device=output_device, normalized_s=normalized_s, input_transfer_s=transfer_s,
            forward_s=forward_s, output_transfer_pool_s=export_s)
        return FusionStreamBatch(sample_ids=batch.sample_ids, timestamps_s=batch.query_timestamps_s.cpu(),
            sequence_embedding=sequence, valid_mask=valid, pooled_embedding=pooled,
            method_name=self.method_name, fold_id=self.fold_id, checkpoint_sha256=self.checkpoint_sha256,
            source_sample_hashes=batch.source_sample_hashes)


def vars_from_batch(batch):
    from dataclasses import fields
    return [v for field in fields(batch) if torch.is_tensor(v := getattr(batch, field.name))]
