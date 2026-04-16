"""Helpers for deriving context from real-bus business metadata."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from chronaris.schema.real_bus import AccessRuleDetail, FlightTaskMetadata, StorageAnalysisDetail

BUS_TIME_COLUMN_NAME = "消息发生时间"
ACMI_TIME_COLUMN_NAME = "消息时间"


@dataclass(frozen=True, slots=True)
class RealBusDerivedContext:
    """Derived lookup values needed by the bus parsing and normalization flow."""

    flight_date: date
    selected_fields: frozenset[str]
    time_column_index: int | None


def resolve_selected_fields(access_rule_details: tuple[AccessRuleDetail, ...]) -> frozenset[str]:
    """Collect the selected col_field values from access_rule_detail."""

    return frozenset(detail.col_field for detail in access_rule_details if detail.col_field)


def resolve_time_column_index(category: str, detail_list: tuple[StorageAnalysisDetail, ...]) -> int | None:
    """
    Mirror the time-column lookup logic from RealBusFileReceiver.

    For BUS, prefer the exact column named '消息发生时间'.
    For ACMI-like categories, prefer '消息时间'; otherwise fall back to the first
    column whose display name ends with '时间'.
    """

    if category.upper() == "BUS":
        for detail in detail_list:
            if detail.col_name == BUS_TIME_COLUMN_NAME:
                return detail.order_num - 1
        return None

    fallback_index: int | None = None
    for detail in detail_list:
        if detail.col_name == ACMI_TIME_COLUMN_NAME:
            return detail.order_num - 1
        if fallback_index is None and detail.col_name.endswith("时间"):
            fallback_index = detail.order_num - 1
    return fallback_index


def derive_flight_date(flight_task: FlightTaskMetadata) -> date:
    """Return the takeoff date from flight_task metadata."""

    if flight_task.flight_date is None:
        raise ValueError(
            f"flight_task.flight_date is required to attach full datetimes for sortie "
            f"{flight_task.sortie_number}."
        )
    return flight_task.flight_date


def derive_real_bus_context(
    *,
    flight_task: FlightTaskMetadata,
    category: str,
    access_rule_details: tuple[AccessRuleDetail, ...],
    detail_list: tuple[StorageAnalysisDetail, ...],
) -> RealBusDerivedContext:
    """Build the minimal derived context needed for real-bus normalization."""

    return RealBusDerivedContext(
        flight_date=derive_flight_date(flight_task),
        selected_fields=resolve_selected_fields(access_rule_details),
        time_column_index=resolve_time_column_index(category, detail_list),
    )
