"""Coordinated loaders that aggregate upstream sources."""

from __future__ import annotations

from dataclasses import dataclass

from chronaris.access.contracts import MetadataReader, PhysiologyPointReader, VehiclePointReader
from chronaris.schema.models import RawPoint, SortieBundle, SortieLocator, StreamKind


@dataclass(slots=True)
class SortieLoader:
    """Aggregates physiology, vehicle, and metadata reads into one bundle."""

    physiology_reader: PhysiologyPointReader
    vehicle_reader: VehiclePointReader
    metadata_reader: MetadataReader

    def load(self, locator: SortieLocator) -> SortieBundle:
        physiology_points = tuple(sorted(self.physiology_reader.fetch_points(locator), key=self._sort_key))
        vehicle_points = tuple(sorted(self.vehicle_reader.fetch_points(locator), key=self._sort_key))

        self._validate_stream_kinds(physiology_points, StreamKind.PHYSIOLOGY)
        self._validate_stream_kinds(vehicle_points, StreamKind.VEHICLE)

        return SortieBundle(
            locator=locator,
            metadata=self.metadata_reader.fetch_metadata(locator),
            physiology_points=physiology_points,
            vehicle_points=vehicle_points,
        )

    @staticmethod
    def _sort_key(point: RawPoint) -> tuple:
        return point.timestamp, point.measurement

    @staticmethod
    def _validate_stream_kinds(points: tuple[RawPoint, ...], expected: StreamKind) -> None:
        for point in points:
            if point.stream_kind != expected:
                raise ValueError(
                    f"Unexpected stream kind for measurement '{point.measurement}': "
                    f"expected {expected}, got {point.stream_kind}."
                )
