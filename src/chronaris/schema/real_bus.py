"""Business metadata contracts borrowed from the historical real-bus pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from datetime import datetime


@dataclass(frozen=True, slots=True)
class FlightTaskMetadata:
    """Relevant fields from rjgx_backend.flight_task."""

    flight_task_id: int
    sortie_number: str
    flight_batch_id: int | None = None
    source_sortie_id: str | None = None
    batch_number: str | None = None
    flight_date: date | None = None
    mission_code: str | None = None
    aircraft_model: str | None = None
    aircraft_number: str | None = None
    pilot_code: str | None = None
    fly_num: int | None = None
    car_start_time: datetime | None = None
    car_end_time: datetime | None = None
    up_pilot_id: int | None = None
    down_pilot_id: int | None = None


@dataclass(frozen=True, slots=True)
class CollectTaskMetadata:
    """Relevant fields from rjgx_backend.collect_task."""

    collect_task_id: int
    coding: str | None = None
    collect_date: date | None = None
    subject: str | None = None
    collect_start_time: datetime | None = None
    collect_end_time: datetime | None = None


@dataclass(frozen=True, slots=True)
class AccessRuleDetail:
    """Subset of access_rule_detail used by the real-bus import chain."""

    access_rule_id: int
    storage_data_analysis_id: int
    col_field: str
    col_name: str | None = None
    col_type: str | None = None
    measurement: str | None = None
    bucket: str | None = None


@dataclass(frozen=True, slots=True)
class StorageAnalysis:
    """Subset of storage_data_analysis used for bus metadata derivation."""

    analysis_id: int
    category: str
    bucket: str | None = None
    measurement: str | None = None
    sortie_number: str | None = None
    md5_val: str | None = None


@dataclass(frozen=True, slots=True)
class StorageAnalysisDetail:
    """Subset of storage_data_analysis_detail used for column lookup."""

    analysis_id: int
    order_num: int
    col_name: str
    col_field: str


@dataclass(frozen=True, slots=True)
class StorageStructure:
    """Subset of storage_data_structure used for parent-child measurement splits."""

    analysis_id: int
    col_field: str
    col_name: str
    parent_id: int | None = None
    is_leaf: bool | None = None


@dataclass(frozen=True, slots=True)
class RealBusContext:
    """Aggregated business metadata required by the real-bus chain."""

    flight_task: FlightTaskMetadata
    analysis: StorageAnalysis
    access_rule_details: tuple[AccessRuleDetail, ...]
    detail_list: tuple[StorageAnalysisDetail, ...]
    structure_list: tuple[StorageStructure, ...]
