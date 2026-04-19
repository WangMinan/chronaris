"""Direct Influx readers for known overlap-focused preview runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Mapping

from chronaris.access.influx_cli import InfluxCliRunner, InfluxMeasurementPointReader, InfluxQueryRunner, InfluxQuerySpec
from chronaris.access.loader import SortieLoader
from chronaris.access.memory import InMemoryMetadataReader
from chronaris.access.settings import InfluxSettings
from chronaris.schema.models import SortieLocator, SortieMetadata, StreamKind


@dataclass(frozen=True, slots=True)
class DirectInfluxScopeConfig:
    """A fixed Influx scope for one stream during overlap-focused preview runs."""

    bucket: str
    measurements: tuple[str, ...]
    start_time_utc: datetime
    stop_time_utc: datetime
    tag_filters: Mapping[str, str] = field(default_factory=dict)
    point_limit_per_measurement: int | None = None

    def __post_init__(self) -> None:
        if not self.measurements:
            raise ValueError("measurements must contain at least one measurement name.")
        if self.stop_time_utc < self.start_time_utc:
            raise ValueError("stop_time_utc must be greater than or equal to start_time_utc.")
        if self.point_limit_per_measurement is not None and self.point_limit_per_measurement <= 0:
            raise ValueError("point_limit_per_measurement must be positive when provided.")


@dataclass(frozen=True, slots=True)
class OverlapPreviewSortieLoaderConfig:
    """Configuration for a direct overlap-focused sortie loader."""

    sortie_id: str
    physiology_scope: DirectInfluxScopeConfig
    vehicle_scope: DirectInfluxScopeConfig
    metadata: SortieMetadata | None = None


@dataclass(frozen=True, slots=True)
class ScopedInfluxPointReader:
    """Reads one or more fixed measurements from Influx for one stream."""

    runner: InfluxQueryRunner
    stream_kind: StreamKind
    scope: DirectInfluxScopeConfig

    def fetch_points(self, locator: SortieLocator):
        points = []
        for measurement_name in self.scope.measurements:
            scoped_reader = InfluxMeasurementPointReader(
                runner=self.runner,
                stream_kind=self.stream_kind,
                query_builder=lambda _, measurement_name=measurement_name: InfluxQuerySpec(
                    bucket=self.scope.bucket,
                    measurement=measurement_name,
                    start=self.scope.start_time_utc,
                    stop=self.scope.stop_time_utc,
                    tag_filters=self.scope.tag_filters,
                    limit=self.scope.point_limit_per_measurement,
                ),
            )
            points.extend(scoped_reader.fetch_points(locator))

        points.sort(key=lambda point: (point.timestamp, point.measurement))
        return tuple(points)


def build_overlap_preview_sortie_loader(
    config: OverlapPreviewSortieLoaderConfig,
    *,
    influx_settings: InfluxSettings | None = None,
    runner: InfluxQueryRunner | None = None,
) -> SortieLoader:
    """Build a direct Influx-backed sortie loader for known overlap-focused runs."""

    if (influx_settings is None) == (runner is None):
        raise ValueError("Provide exactly one of influx_settings or runner.")

    resolved_runner = runner or InfluxCliRunner(influx_settings)
    metadata = config.metadata or SortieMetadata(
        sortie_id=config.sortie_id,
        sortie_number=config.sortie_id,
    )
    return SortieLoader(
        physiology_reader=ScopedInfluxPointReader(
            runner=resolved_runner,
            stream_kind=StreamKind.PHYSIOLOGY,
            scope=config.physiology_scope,
        ),
        vehicle_reader=ScopedInfluxPointReader(
            runner=resolved_runner,
            stream_kind=StreamKind.VEHICLE,
            scope=config.vehicle_scope,
        ),
        metadata_reader=InMemoryMetadataReader({config.sortie_id: metadata}),
    )
