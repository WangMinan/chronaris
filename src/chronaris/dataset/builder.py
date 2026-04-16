"""High-level builders that turn raw sortie bundles into datasets."""

from __future__ import annotations

from dataclasses import dataclass, field

from chronaris.dataset.timebase import TimebasePolicy, align_sortie_bundle
from chronaris.dataset.windows import build_sample_windows
from chronaris.schema.models import DatasetBuildResult, SortieBundle, WindowConfig


@dataclass(slots=True)
class SortieDatasetBuilder:
    """Builds one sortie into aligned window samples."""

    timebase_policy: TimebasePolicy = field(default_factory=TimebasePolicy)
    window_config: WindowConfig = field(default_factory=lambda: WindowConfig(duration_ms=5_000, stride_ms=5_000))

    def build(self, bundle: SortieBundle) -> DatasetBuildResult:
        aligned_bundle = align_sortie_bundle(bundle, self.timebase_policy)
        windows = build_sample_windows(aligned_bundle, self.window_config)
        return DatasetBuildResult(aligned_bundle=aligned_bundle, windows=windows)
