"""Window-end features without inventing a time sequence for scalar consumers."""
from dataclasses import dataclass

import torch

from chronaris.representation.contracts import CHECKPOINT_HASH_PATTERN, RepresentationContractError


@dataclass(frozen=True)
class WindowFeatureBatch:
    sample_ids: tuple[str, ...]
    timestamps_s: torch.Tensor
    pooled_embedding: torch.Tensor
    valid_mask: torch.Tensor
    method_name: str
    fold_id: str
    checkpoint_sha256: str
    source_sample_hashes: tuple[str, ...]

    @property
    def sequence_embedding(self):
        return None

    def __post_init__(self):
        count = len(self.sample_ids)
        if not count or len(set(self.sample_ids)) != count or len(self.source_sample_hashes) != count:
            raise RepresentationContractError('window feature identities are invalid')
        if not self.method_name or not self.fold_id or not CHECKPOINT_HASH_PATTERN.fullmatch(self.checkpoint_sha256):
            raise RepresentationContractError('window feature provenance is invalid')
        if self.pooled_embedding.ndim != 2 or self.pooled_embedding.shape[0] != count or self.pooled_embedding.shape[1] < 1:
            raise RepresentationContractError('window features require [N,D], D > 0')
        if self.timestamps_s.shape != (count,) or self.valid_mask.shape != (count,) or self.valid_mask.dtype != torch.bool:
            raise RepresentationContractError('window cutoffs and observation mask require [N]')
        if not torch.isfinite(self.timestamps_s).all() or not torch.isfinite(self.pooled_embedding).all():
            raise RepresentationContractError('window features and cutoffs must be finite')
        if (self.pooled_embedding[~self.valid_mask] != 0).any():
            raise RepresentationContractError('unobserved windows must have zero features')
