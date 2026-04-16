"""First minimal pipeline for sortie-level dataset construction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from chronaris.access.loader import SortieLoader
from chronaris.dataset.builder import SortieDatasetBuilder
from chronaris.schema.models import DatasetBuildResult, SortieLocator


@dataclass(slots=True)
class DatasetPipelineV1:
    """Loads one sortie and turns it into aligned windows."""

    loader: SortieLoader
    builder: SortieDatasetBuilder = field(default_factory=SortieDatasetBuilder)

    def run(self, locator: SortieLocator) -> DatasetBuildResult:
        bundle = self.loader.load(locator)
        return self.builder.build(bundle)

    def run_many(self, locators: Sequence[SortieLocator]) -> tuple[DatasetBuildResult, ...]:
        return tuple(self.run(locator) for locator in locators)
