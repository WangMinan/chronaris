"""Pipeline for building E0 minimal experiment inputs from one sortie."""

from __future__ import annotations

from dataclasses import dataclass, field

from chronaris.dataset.builder import SortieDatasetBuilder
from chronaris.features.experiment_input import E0ExperimentSample, E0InputConfig, build_e0_experiment_samples
from chronaris.access.loader import SortieLoader
from chronaris.schema.models import SortieLocator


@dataclass(slots=True)
class E0PreviewPipeline:
    """Builds E0 experiment samples directly from one sortie locator."""

    loader: SortieLoader
    dataset_builder: SortieDatasetBuilder = field(default_factory=SortieDatasetBuilder)
    input_config: E0InputConfig = field(default_factory=E0InputConfig)

    def run(self, locator: SortieLocator) -> tuple[E0ExperimentSample, ...]:
        bundle = self.loader.load(locator)
        dataset_result = self.dataset_builder.build(bundle)
        return build_e0_experiment_samples(dataset_result, config=self.input_config)
