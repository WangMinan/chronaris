"""Protocols for loading Chronaris upstream data."""

from __future__ import annotations

from typing import Protocol, Sequence

from chronaris.schema.models import RawPoint, SortieLocator, SortieMetadata


class PhysiologyPointReader(Protocol):
    """Reads physiology time-series points for one sortie."""

    def fetch_points(self, locator: SortieLocator) -> Sequence[RawPoint]:
        """Return physiology points for the given sortie locator."""


class VehiclePointReader(Protocol):
    """Reads vehicle-side time-series points for one sortie."""

    def fetch_points(self, locator: SortieLocator) -> Sequence[RawPoint]:
        """Return vehicle points for the given sortie locator."""


class MetadataReader(Protocol):
    """Reads sortie metadata from the business metadata store."""

    def fetch_metadata(self, locator: SortieLocator) -> SortieMetadata:
        """Return metadata for the given sortie locator."""
