"""Execute the same standard readback as the service, with independent value checks."""
import csv
from pathlib import Path
import sys

import torch
from chronaris.representation import load_fusion_stream_batch

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / 'feature-example'
features = load_fusion_stream_batch(root)
sequence, valid, pooled = features.sequence_embedding, features.valid_mask, features.pooled_embedding
assert sequence.shape[1:] == (96, 64) and valid.dtype == torch.bool
assert torch.isfinite(sequence).all() and torch.equal(sequence[~valid], torch.zeros_like(sequence[~valid]))
expected = sequence.masked_fill(~valid.unsqueeze(-1), 0).sum(1) / valid.sum(1, keepdim=True).clamp_min(1)
torch.testing.assert_close(pooled, expected, atol=0, rtol=0)
with (root / 'features.csv').open() as handle:
    rows = list(csv.DictReader(handle))
assert len(rows) == len(sequence) * 96
for i, row in enumerate(rows):
    assert row['sample_id'] == features.sample_ids[i // 96]
    assert float(row['query_time_s']) == float(features.timestamps_s[i // 96, i % 96])
    assert (row['valid'] == 'True') == bool(valid[i // 96, i % 96])
    torch.testing.assert_close(torch.tensor([float(row[f'feature_{d:02}']) for d in range(64)]),
                               sequence[i // 96, i % 96], atol=0, rtol=0)
print({'sequence_shape': list(sequence.shape), 'valid_points': int(valid.sum()),
       'pooled_shape': list(pooled.shape), 'checkpoint_sha256': features.checkpoint_sha256, 'status': 'passed'})
