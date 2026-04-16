"""Reference-time selection and relative-time projection."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from chronaris.schema.models import AlignedPoint, AlignedSortieBundle, RawPoint, SortieBundle


class ReferenceStrategy(StrEnum):
    """Supported reference-time selection strategies."""

    EARLIEST_OBSERVATION = "earliest_observation"
    LOCATOR_START = "locator_start"
    FIRST_PHYSIOLOGY_POINT = "first_physiology_point"
    FIRST_VEHICLE_POINT = "first_vehicle_point"


@dataclass(frozen=True, slots=True)
class TimebasePolicy:
    """Controls how relative time is computed for one sortie."""

    reference_strategy: ReferenceStrategy = ReferenceStrategy.EARLIEST_OBSERVATION


class TimebaseError(ValueError):
    """Raised when Chronaris cannot establish a valid shared timebase."""


def align_sortie_bundle(bundle: SortieBundle, policy: TimebasePolicy | None = None) -> AlignedSortieBundle:
    """Project one sortie bundle into a single reference-time coordinate system."""

    active_policy = policy or TimebasePolicy()
    reference_time = choose_reference_time(bundle, active_policy)
    _validate_timestamp_family(bundle, reference_time)

    return AlignedSortieBundle(
        locator=bundle.locator,
        metadata=bundle.metadata,
        reference_time=reference_time,
        physiology_points=_align_points(bundle.physiology_points, reference_time),
        vehicle_points=_align_points(bundle.vehicle_points, reference_time),
    )


def choose_reference_time(bundle: SortieBundle, policy: TimebasePolicy) -> datetime:
    """Return the absolute reference time for one sortie bundle."""

    strategy = policy.reference_strategy

    if strategy == ReferenceStrategy.LOCATOR_START:
        if bundle.locator.start_time is None:
            raise TimebaseError("Locator start_time is required for locator_start strategy.")
        return bundle.locator.start_time

    if strategy == ReferenceStrategy.FIRST_PHYSIOLOGY_POINT:
        if not bundle.physiology_points:
            raise TimebaseError("No physiology points available for first_physiology_point strategy.")
        return min(point.timestamp for point in bundle.physiology_points)

    if strategy == ReferenceStrategy.FIRST_VEHICLE_POINT:
        if not bundle.vehicle_points:
            raise TimebaseError("No vehicle points available for first_vehicle_point strategy.")
        return min(point.timestamp for point in bundle.vehicle_points)

    candidates = [point.timestamp for point in bundle.physiology_points]
    candidates.extend(point.timestamp for point in bundle.vehicle_points)

    if bundle.locator.start_time is not None:
        candidates.append(bundle.locator.start_time)

    if not candidates:
        raise TimebaseError("Cannot choose reference time from an empty sortie bundle.")

    _validate_datetime_awareness(candidates)
    return min(candidates)


def _align_points(points: tuple[RawPoint, ...], reference_time: datetime) -> tuple[AlignedPoint, ...]:
    return tuple(
        AlignedPoint(
            point=point,
            offset_ms=int((point.timestamp - reference_time).total_seconds() * 1000),
        )
        for point in points
    )


def _validate_timestamp_family(bundle: SortieBundle, reference_time: datetime) -> None:
    timestamps = [point.timestamp for point in bundle.physiology_points]
    timestamps.extend(point.timestamp for point in bundle.vehicle_points)
    timestamps.append(reference_time)

    _validate_datetime_awareness(timestamps)


def _is_aware(timestamp: datetime) -> bool:
    return timestamp.tzinfo is not None and timestamp.utcoffset() is not None


def _validate_datetime_awareness(timestamps: list[datetime]) -> None:
    awareness = {_is_aware(timestamp) for timestamp in timestamps}
    if len(awareness) > 1:
        raise TimebaseError("Aware and naive datetimes cannot be mixed in one sortie build.")
