"""Factories for Stage B live readers and loaders."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from chronaris.access.influx_cli import InfluxCliRunner
from chronaris.access.live_readers import PhysiologyInfluxPointReader, RealBusInfluxPointReader
from chronaris.access.loader import SortieLoader
from chronaris.access.mysql_cli import MySQLCliRunner
from chronaris.access.mysql_metadata import (
    MySQLCollectTaskReader,
    MySQLFlightTaskReader,
    MySQLRealBusContextReader,
    MySQLSortieMetadataReader,
)
from chronaris.access.settings import InfluxSettings, MySQLSettings


@dataclass(frozen=True, slots=True)
class StageBLiveLoaderConfig:
    """Configuration for building a live Stage B sortie loader."""

    java_properties_path: str | Path
    access_rule_id: int
    analysis_id: int
    mysql_database: str = "rjgx_backend"
    physiology_measurements: tuple[str, ...] | None = None
    physiology_point_limit_per_measurement: int | None = None
    bus_point_limit: int | None = None
    start_time_override_utc: datetime | None = None
    stop_time_override_utc: datetime | None = None


def build_stage_b_live_sortie_loader(config: StageBLiveLoaderConfig) -> SortieLoader:
    """Build a live SortieLoader backed by real MySQL and Influx access."""

    mysql_settings = MySQLSettings.from_java_properties(
        config.java_properties_path,
        database=config.mysql_database,
    )
    influx_settings = InfluxSettings.from_java_properties(config.java_properties_path)

    mysql_runner = MySQLCliRunner(mysql_settings)
    influx_runner = InfluxCliRunner(influx_settings)

    flight_task_reader = MySQLFlightTaskReader(mysql_runner)
    collect_task_reader = MySQLCollectTaskReader(mysql_runner)
    metadata_reader = MySQLSortieMetadataReader(flight_task_reader=flight_task_reader)

    return SortieLoader(
        physiology_reader=PhysiologyInfluxPointReader(
            flight_task_reader=flight_task_reader,
            collect_task_reader=collect_task_reader,
            runner=influx_runner,
            measurement_names=config.physiology_measurements,
            point_limit_per_measurement=config.physiology_point_limit_per_measurement,
            start_time_override_utc=config.start_time_override_utc,
            stop_time_override_utc=config.stop_time_override_utc,
        ),
        vehicle_reader=RealBusInfluxPointReader(
            context_reader=MySQLRealBusContextReader(
                runner=mysql_runner,
                flight_task_reader=flight_task_reader,
            ),
            runner=influx_runner,
            access_rule_id=config.access_rule_id,
            analysis_id=config.analysis_id,
            point_limit=config.bus_point_limit,
            start_time_override_utc=config.start_time_override_utc,
            stop_time_override_utc=config.stop_time_override_utc,
        ),
        metadata_reader=metadata_reader,
    )
