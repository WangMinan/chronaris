"""Metadata readers backed by MySQL CLI queries."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

from chronaris.access.contracts import MetadataReader
from chronaris.access.mysql_cli import SQLQueryRunner
from chronaris.schema.models import SortieLocator, SortieMetadata
from chronaris.schema.real_bus import (
    AccessRuleDetail,
    CollectTaskMetadata,
    FlightTaskMetadata,
    RealBusContext,
    StorageAnalysis,
    StorageAnalysisDetail,
    StorageStructure,
)


@dataclass(frozen=True, slots=True)
class MySQLFlightTaskReader:
    """Reads joined flight_task and flight_batch metadata."""

    runner: SQLQueryRunner

    def fetch_by_locator(self, locator: SortieLocator) -> FlightTaskMetadata:
        row = self.runner.query_one(_build_flight_task_query(locator))
        if row is None:
            raise LookupError(f"No flight_task found for sortie locator: {locator.sortie_id}")
        return _map_flight_task_metadata(row)

    def fetch_by_id(self, flight_task_id: int) -> FlightTaskMetadata:
        row = self.runner.query_one(_build_flight_task_by_id_query(flight_task_id))
        if row is None:
            raise LookupError(f"No flight_task found for id: {flight_task_id}")
        return _map_flight_task_metadata(row)


@dataclass(frozen=True, slots=True)
class MySQLSortieMetadataReader(MetadataReader):
    """Adapts flight_task metadata to the generic SortieMetadata contract."""

    flight_task_reader: MySQLFlightTaskReader

    def fetch_metadata(self, locator: SortieLocator) -> SortieMetadata:
        flight_task = self.flight_task_reader.fetch_by_locator(locator)
        return SortieMetadata(
            sortie_id=flight_task.sortie_number,
            flight_task_id=flight_task.flight_task_id,
            flight_batch_id=flight_task.flight_batch_id,
            flight_date=flight_task.flight_date,
            mission_code=flight_task.mission_code,
            aircraft_model=flight_task.aircraft_model,
            aircraft_tail=flight_task.aircraft_number,
            pilot_code=flight_task.pilot_code,
            batch_id=str(flight_task.flight_batch_id) if flight_task.flight_batch_id is not None else None,
            sortie_number=flight_task.sortie_number,
            batch_number=flight_task.batch_number,
            extra={
                "source_sortie_id": flight_task.source_sortie_id,
                "fly_num": flight_task.fly_num,
                "car_start_time": flight_task.car_start_time.isoformat() if flight_task.car_start_time else None,
                "car_end_time": flight_task.car_end_time.isoformat() if flight_task.car_end_time else None,
            },
        )


@dataclass(frozen=True, slots=True)
class MySQLRealBusContextReader:
    """Reads business metadata required by the real-bus normalization chain."""

    runner: SQLQueryRunner
    flight_task_reader: MySQLFlightTaskReader

    def fetch_context(
        self,
        *,
        locator: SortieLocator | None = None,
        flight_task_id: int | None = None,
        access_rule_id: int,
        analysis_id: int,
    ) -> RealBusContext:
        flight_task = self._resolve_flight_task(locator=locator, flight_task_id=flight_task_id)

        analysis_row = self.runner.query_one(_build_storage_analysis_query(analysis_id))
        if analysis_row is None:
            raise LookupError(f"No storage_data_analysis found for id: {analysis_id}")

        return RealBusContext(
            flight_task=flight_task,
            analysis=_map_storage_analysis(analysis_row),
            access_rule_details=tuple(
                _map_access_rule_detail(row)
                for row in self.runner.query(_build_access_rule_detail_query(access_rule_id, analysis_id))
            ),
            detail_list=tuple(
                _map_storage_analysis_detail(row)
                for row in self.runner.query(_build_storage_analysis_detail_query(analysis_id))
            ),
            structure_list=tuple(
                _map_storage_structure(row)
                for row in self.runner.query(_build_storage_structure_query(analysis_id))
            ),
        )

    def _resolve_flight_task(
        self,
        *,
        locator: SortieLocator | None,
        flight_task_id: int | None,
    ) -> FlightTaskMetadata:
        if flight_task_id is not None:
            return self.flight_task_reader.fetch_by_id(flight_task_id)
        if locator is None:
            raise ValueError("Either locator or flight_task_id must be provided.")
        return self.flight_task_reader.fetch_by_locator(locator)


@dataclass(frozen=True, slots=True)
class MySQLStorageAnalysisReader:
    """Lists storage_data_analysis rows for one sortie."""

    runner: SQLQueryRunner

    def list_for_sortie(
        self,
        locator: SortieLocator,
        *,
        category: str | None = None,
    ) -> tuple[StorageAnalysis, ...]:
        return tuple(
            _map_storage_analysis(row)
            for row in self.runner.query(
                _build_storage_analysis_for_sortie_query(
                    sortie_number=locator.sortie_id,
                    category=category,
                )
            )
        )


@dataclass(frozen=True, slots=True)
class MySQLCollectTaskReader:
    """Reads collect_task metadata for physiology-side lookup."""

    runner: SQLQueryRunner

    def fetch_by_id(self, collect_task_id: int) -> CollectTaskMetadata:
        row = self.runner.query_one(_build_collect_task_by_id_query(collect_task_id))
        if row is None:
            raise LookupError(f"No collect_task found for id: {collect_task_id}")
        return _map_collect_task_metadata(row)

    def fetch_for_flight_task(self, flight_task: FlightTaskMetadata) -> CollectTaskMetadata:
        row = self.runner.query_one(_build_collect_task_for_flight_task_query(flight_task))
        if row is None:
            raise LookupError(
                f"No collect_task could be resolved for sortie {flight_task.sortie_number} "
                f"with subject {flight_task.mission_code}."
            )
        return _map_collect_task_metadata(row)


def _build_flight_task_query(locator: SortieLocator) -> str:
    key = _escape_sql_literal(locator.sortie_id)
    return f"""
SELECT
    ft.id AS flight_task_id,
    ft.flight_batch_id,
    ft.sortie_number,
    ft.batch_number,
    ft.subject,
    ft.airplane_model,
    ft.airplane_number,
    ft.fly_num,
    ft.up_pilot_id,
    ft.down_pilot_id,
    ft.source_sortie_id,
    ft.car_star_time,
    ft.car_end_time,
    fb.fly_date
FROM flight_task ft
LEFT JOIN flight_batch fb ON fb.id = ft.flight_batch_id
WHERE ft.sortie_number = '{key}'
   OR ft.source_sortie_id = '{key}'
ORDER BY ft.id DESC
LIMIT 1
""".strip()


def _build_flight_task_by_id_query(flight_task_id: int) -> str:
    return f"""
SELECT
    ft.id AS flight_task_id,
    ft.flight_batch_id,
    ft.sortie_number,
    ft.batch_number,
    ft.subject,
    ft.airplane_model,
    ft.airplane_number,
    ft.fly_num,
    ft.up_pilot_id,
    ft.down_pilot_id,
    ft.source_sortie_id,
    ft.car_star_time,
    ft.car_end_time,
    fb.fly_date
FROM flight_task ft
LEFT JOIN flight_batch fb ON fb.id = ft.flight_batch_id
WHERE ft.id = {flight_task_id}
LIMIT 1
""".strip()


def _build_access_rule_detail_query(access_rule_id: int, analysis_id: int) -> str:
    return f"""
SELECT
    access_rule_id,
    storage_data_analysis_id,
    col_field,
    col_name,
    col_type,
    measurement,
    bucket
FROM access_rule_detail
WHERE access_rule_id = {access_rule_id}
  AND storage_data_analysis_id = {analysis_id}
""".strip()


def _build_collect_task_by_id_query(collect_task_id: int) -> str:
    return f"""
SELECT
    id,
    coding,
    collect_date,
    subject,
    collect_start_date,
    collect_end_date
FROM collect_task
WHERE id = {collect_task_id}
LIMIT 1
""".strip()


def _build_collect_task_for_flight_task_query(flight_task: FlightTaskMetadata) -> str:
    conditions = []
    if flight_task.flight_date is not None:
        conditions.append(f"collect_date = '{flight_task.flight_date.isoformat()}'")
    if flight_task.mission_code is not None:
        conditions.append(f"subject = '{_escape_sql_literal(flight_task.mission_code)}'")
    if flight_task.car_start_time is not None:
        conditions.append(f"collect_start_date <= '{flight_task.car_start_time.strftime('%Y-%m-%d %H:%M:%S')}'")
    if flight_task.car_end_time is not None:
        conditions.append(f"collect_end_date >= '{flight_task.car_end_time.strftime('%Y-%m-%d %H:%M:%S')}'")

    if not conditions:
        raise ValueError(
            f"Not enough metadata to resolve collect_task for sortie {flight_task.sortie_number}."
        )

    return f"""
SELECT
    id,
    coding,
    collect_date,
    subject,
    collect_start_date,
    collect_end_date
FROM collect_task
WHERE {' AND '.join(conditions)}
ORDER BY id DESC
LIMIT 1
""".strip()


def _build_storage_analysis_query(analysis_id: int) -> str:
    return f"""
SELECT
    id,
    category,
    bucket,
    measurement,
    sortie_number,
    md5_val
FROM storage_data_analysis
WHERE id = {analysis_id}
LIMIT 1
""".strip()


def _build_storage_analysis_for_sortie_query(*, sortie_number: str, category: str | None = None) -> str:
    conditions = [f"sortie_number = '{_escape_sql_literal(sortie_number)}'"]
    if category is not None:
        conditions.append(f"category = '{_escape_sql_literal(category)}'")
    return f"""
SELECT
    id,
    category,
    bucket,
    measurement,
    sortie_number,
    md5_val
FROM storage_data_analysis
WHERE {' AND '.join(conditions)}
ORDER BY measurement, id
""".strip()


def _build_storage_analysis_detail_query(analysis_id: int) -> str:
    return f"""
SELECT
    storage_data_analysis_id,
    order_num,
    col_name,
    col_field
FROM storage_data_analysis_detail
WHERE storage_data_analysis_id = {analysis_id}
ORDER BY order_num
""".strip()


def _build_storage_structure_query(analysis_id: int) -> str:
    return f"""
SELECT
    storage_data_analysis_id,
    col_field,
    col_name,
    parent_id,
    is_leaf
FROM storage_data_structure
WHERE storage_data_analysis_id = {analysis_id}
""".strip()


def _map_flight_task_metadata(row: dict[str, str | None]) -> FlightTaskMetadata:
    sortie_number = _require_value(row, "sortie_number")
    return FlightTaskMetadata(
        flight_task_id=int(_require_value(row, "flight_task_id")),
        flight_batch_id=_optional_int(row, "flight_batch_id"),
        sortie_number=sortie_number,
        batch_number=row.get("batch_number"),
        flight_date=_optional_date(row, "fly_date"),
        mission_code=row.get("subject"),
        aircraft_model=row.get("airplane_model"),
        aircraft_number=row.get("airplane_number"),
        pilot_code=_derive_pilot_code(sortie_number),
        source_sortie_id=row.get("source_sortie_id"),
        fly_num=_optional_int(row, "fly_num"),
        car_start_time=_optional_datetime(row, "car_star_time"),
        car_end_time=_optional_datetime(row, "car_end_time"),
        up_pilot_id=_optional_int(row, "up_pilot_id"),
        down_pilot_id=_optional_int(row, "down_pilot_id"),
    )


def _map_access_rule_detail(row: dict[str, str | None]) -> AccessRuleDetail:
    return AccessRuleDetail(
        access_rule_id=int(_require_value(row, "access_rule_id")),
        storage_data_analysis_id=int(_require_value(row, "storage_data_analysis_id")),
        col_field=_require_value(row, "col_field"),
        col_name=row.get("col_name"),
        col_type=row.get("col_type"),
        measurement=row.get("measurement"),
        bucket=row.get("bucket"),
    )


def _map_storage_analysis(row: dict[str, str | None]) -> StorageAnalysis:
    return StorageAnalysis(
        analysis_id=int(_require_value(row, "id")),
        category=_require_value(row, "category"),
        bucket=row.get("bucket"),
        measurement=row.get("measurement"),
        sortie_number=row.get("sortie_number"),
        md5_val=row.get("md5_val"),
    )


def _map_storage_analysis_detail(row: dict[str, str | None]) -> StorageAnalysisDetail:
    return StorageAnalysisDetail(
        analysis_id=int(_require_value(row, "storage_data_analysis_id")),
        order_num=int(_require_value(row, "order_num")),
        col_name=_require_value(row, "col_name"),
        col_field=_require_value(row, "col_field"),
    )


def _map_storage_structure(row: dict[str, str | None]) -> StorageStructure:
    return StorageStructure(
        analysis_id=int(_require_value(row, "storage_data_analysis_id")),
        col_field=_require_value(row, "col_field"),
        col_name=_require_value(row, "col_name"),
        parent_id=_optional_int(row, "parent_id"),
        is_leaf=_optional_bool(row, "is_leaf"),
    )


def _map_collect_task_metadata(row: dict[str, str | None]) -> CollectTaskMetadata:
    return CollectTaskMetadata(
        collect_task_id=int(_require_value(row, "id")),
        coding=row.get("coding"),
        collect_date=_optional_date(row, "collect_date"),
        subject=row.get("subject"),
        collect_start_time=_optional_datetime(row, "collect_start_date"),
        collect_end_time=_optional_datetime(row, "collect_end_date"),
    )


def _derive_pilot_code(sortie_number: str) -> str | None:
    parts = sortie_number.split("_")
    if len(parts) >= 6:
        return parts[3]
    return None


def _optional_date(row: dict[str, str | None], key: str) -> date | None:
    value = row.get(key)
    if value is None:
        return None
    return date.fromisoformat(value)


def _optional_datetime(row: dict[str, str | None], key: str) -> datetime | None:
    value = row.get(key)
    if value is None:
        return None
    return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")


def _optional_int(row: dict[str, str | None], key: str) -> int | None:
    value = row.get(key)
    if value is None:
        return None
    return int(value)


def _optional_bool(row: dict[str, str | None], key: str) -> bool | None:
    value = row.get(key)
    if value is None:
        return None
    return bool(int(value))


def _require_value(row: dict[str, str | None], key: str) -> str:
    value = row.get(key)
    if value is None:
        raise ValueError(f"Missing required column '{key}' in MySQL result row.")
    return value


def _escape_sql_literal(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")
