"""In-memory access implementations for tests and local prototyping."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from chronaris.schema.models import RawPoint, SortieLocator, SortieMetadata, StreamKind


@dataclass(slots=True)
class InMemoryPointReader:
    """A minimal point reader backed by a dictionary keyed by sortie id."""

    points_by_sortie: Mapping[str, Sequence[RawPoint]]
    expected_kind: StreamKind

    def fetch_points(self, locator: SortieLocator) -> Sequence[RawPoint]:
        points = tuple(self.points_by_sortie.get(locator.sortie_id, ()))
        for point in points:
            if point.stream_kind != self.expected_kind:
                raise ValueError(
                    f"Point '{point.measurement}' has kind {point.stream_kind}, "
                    f"expected {self.expected_kind}."
                )
        return points


@dataclass(slots=True)
class InMemoryMetadataReader:
    """A minimal metadata reader backed by a dictionary keyed by sortie id."""

    metadata_by_sortie: Mapping[str, SortieMetadata] = field(default_factory=dict)

    def fetch_metadata(self, locator: SortieLocator) -> SortieMetadata:
        return self.metadata_by_sortie.get(locator.sortie_id, SortieMetadata(sortie_id=locator.sortie_id))
