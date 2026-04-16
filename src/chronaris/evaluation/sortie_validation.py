"""Single-sortie validation summaries for Stage C."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime

from chronaris.dataset.timebase import TimebasePolicy, align_sortie_bundle
from chronaris.dataset.windows import build_sample_windows
from chronaris.schema.models import RawPoint, SortieBundle, StreamKind, WindowConfig


@dataclass(frozen=True, slots=True)
class MeasurementCoverageSummary:
    """Coverage summary for one measurement within one stream."""

    measurement: str
    point_count: int
    first_timestamp: datetime
    last_timestamp: datetime
    span_ms: int


@dataclass(frozen=True, slots=True)
class StreamCoverageSummary:
    """Coverage summary for one stream across all measurements."""

    stream_kind: StreamKind
    point_count: int
    measurement_count: int
    first_timestamp: datetime | None
    last_timestamp: datetime | None
    span_ms: int
    measurements: tuple[MeasurementCoverageSummary, ...]


@dataclass(frozen=True, slots=True)
class CrossStreamTimingSummary:
    """Describes timing overlap or separation between physiology and vehicle streams."""

    relation: str
    overlap_start: datetime | None
    overlap_end: datetime | None
    overlap_duration_ms: int
    gap_duration_ms: int
    leading_stream: StreamKind | None


@dataclass(frozen=True, slots=True)
class WindowTrialSummary:
    """Window-build result for one WindowConfig trial."""

    duration_ms: int
    stride_ms: int
    min_physiology_points: int
    min_vehicle_points: int
    window_count: int
    first_window_start_offset_ms: int | None
    first_window_end_offset_ms: int | None


@dataclass(frozen=True, slots=True)
class SortieValidationSummary:
    """Validation summary for one sortie bundle."""

    sortie_id: str
    physiology: StreamCoverageSummary
    vehicle: StreamCoverageSummary
    cross_stream_timing: CrossStreamTimingSummary
    window_trials: tuple[WindowTrialSummary, ...]


def validate_sortie_bundle(
    bundle: SortieBundle,
    *,
    window_configs: tuple[WindowConfig, ...],
    timebase_policy: TimebasePolicy | None = None,
) -> SortieValidationSummary:
    """Build a validation summary for one sortie bundle."""

    aligned_bundle = align_sortie_bundle(bundle, timebase_policy)
    physiology_summary = summarize_stream(StreamKind.PHYSIOLOGY, bundle.physiology_points)
    vehicle_summary = summarize_stream(StreamKind.VEHICLE, bundle.vehicle_points)
    timing_summary = summarize_cross_stream_timing(physiology_summary, vehicle_summary)
    window_trials = tuple(
        summarize_window_trial(aligned_bundle, config)
        for config in window_configs
    )

    return SortieValidationSummary(
        sortie_id=bundle.locator.sortie_id,
        physiology=physiology_summary,
        vehicle=vehicle_summary,
        cross_stream_timing=timing_summary,
        window_trials=window_trials,
    )


def summarize_stream(stream_kind: StreamKind, points: tuple[RawPoint, ...]) -> StreamCoverageSummary:
    """Summarize point counts and coverage for one stream."""

    if not points:
        return StreamCoverageSummary(
            stream_kind=stream_kind,
            point_count=0,
            measurement_count=0,
            first_timestamp=None,
            last_timestamp=None,
            span_ms=0,
            measurements=(),
        )

    by_measurement: dict[str, list[RawPoint]] = defaultdict(list)
    for point in sorted(points, key=lambda item: (item.timestamp, item.measurement)):
        by_measurement[point.measurement].append(point)

    measurement_summaries = tuple(
        _summarize_measurement(measurement, grouped_points)
        for measurement, grouped_points in sorted(by_measurement.items())
    )
    first_timestamp = min(point.timestamp for point in points)
    last_timestamp = max(point.timestamp for point in points)

    return StreamCoverageSummary(
        stream_kind=stream_kind,
        point_count=len(points),
        measurement_count=len(by_measurement),
        first_timestamp=first_timestamp,
        last_timestamp=last_timestamp,
        span_ms=_duration_ms(first_timestamp, last_timestamp),
        measurements=measurement_summaries,
    )


def summarize_cross_stream_timing(
    physiology: StreamCoverageSummary,
    vehicle: StreamCoverageSummary,
) -> CrossStreamTimingSummary:
    """Describe whether two streams overlap in time and by how much."""

    if physiology.first_timestamp is None or vehicle.first_timestamp is None:
        return CrossStreamTimingSummary(
            relation="missing_stream",
            overlap_start=None,
            overlap_end=None,
            overlap_duration_ms=0,
            gap_duration_ms=0,
            leading_stream=None,
        )

    overlap_start = max(physiology.first_timestamp, vehicle.first_timestamp)
    overlap_end = min(physiology.last_timestamp, vehicle.last_timestamp)

    if overlap_end >= overlap_start:
        return CrossStreamTimingSummary(
            relation="overlap",
            overlap_start=overlap_start,
            overlap_end=overlap_end,
            overlap_duration_ms=_duration_ms(overlap_start, overlap_end),
            gap_duration_ms=0,
            leading_stream=None,
        )

    if physiology.last_timestamp < vehicle.first_timestamp:
        return CrossStreamTimingSummary(
            relation="physiology_before_vehicle",
            overlap_start=None,
            overlap_end=None,
            overlap_duration_ms=0,
            gap_duration_ms=_duration_ms(physiology.last_timestamp, vehicle.first_timestamp),
            leading_stream=StreamKind.PHYSIOLOGY,
        )

    return CrossStreamTimingSummary(
        relation="vehicle_before_physiology",
        overlap_start=None,
        overlap_end=None,
        overlap_duration_ms=0,
        gap_duration_ms=_duration_ms(vehicle.last_timestamp, physiology.first_timestamp),
        leading_stream=StreamKind.VEHICLE,
    )


def summarize_window_trial(aligned_bundle, config: WindowConfig) -> WindowTrialSummary:
    """Run one window configuration and summarize the result."""

    windows = build_sample_windows(aligned_bundle, config)
    first_window = windows[0] if windows else None
    return WindowTrialSummary(
        duration_ms=config.duration_ms,
        stride_ms=config.stride_ms,
        min_physiology_points=config.min_physiology_points,
        min_vehicle_points=config.min_vehicle_points,
        window_count=len(windows),
        first_window_start_offset_ms=first_window.start_offset_ms if first_window else None,
        first_window_end_offset_ms=first_window.end_offset_ms if first_window else None,
    )


def render_validation_markdown(
    summary: SortieValidationSummary,
    *,
    title: str | None = None,
    notes: tuple[str, ...] = (),
) -> str:
    """Render a validation summary into a compact Markdown report."""

    lines = [f"# {title or 'Sortie Validation Summary'}", ""]
    lines.append(f"- sortie: `{summary.sortie_id}`")
    lines.append(
        f"- physiology points: `{summary.physiology.point_count}` across `{summary.physiology.measurement_count}` measurements"
    )
    lines.append(
        f"- vehicle points: `{summary.vehicle.point_count}` across `{summary.vehicle.measurement_count}` measurements"
    )
    lines.append("")

    lines.append("## Stream Coverage")
    lines.append("")
    lines.extend(_render_stream_block(summary.physiology))
    lines.append("")
    lines.extend(_render_stream_block(summary.vehicle))
    lines.append("")

    lines.append("## Cross-Stream Timing")
    lines.append("")
    lines.append(f"- relation: `{summary.cross_stream_timing.relation}`")
    if summary.cross_stream_timing.relation == "overlap":
        lines.append(
            f"- overlap: `{summary.cross_stream_timing.overlap_start.isoformat()}` -> "
            f"`{summary.cross_stream_timing.overlap_end.isoformat()}`"
        )
        lines.append(f"- overlap duration ms: `{summary.cross_stream_timing.overlap_duration_ms}`")
    else:
        lines.append(f"- gap duration ms: `{summary.cross_stream_timing.gap_duration_ms}`")
        if summary.cross_stream_timing.leading_stream is not None:
            lines.append(f"- leading stream: `{summary.cross_stream_timing.leading_stream}`")
    lines.append("")

    lines.append("## Window Trials")
    lines.append("")
    for trial in summary.window_trials:
        lines.append(
            f"- duration `{trial.duration_ms}` ms / stride `{trial.stride_ms}` ms -> "
            f"`{trial.window_count}` windows"
        )
        if trial.first_window_start_offset_ms is not None:
            lines.append(
                f"  first window offsets: `{trial.first_window_start_offset_ms}` -> "
                f"`{trial.first_window_end_offset_ms}`"
            )
    if not summary.window_trials:
        lines.append("- no window trials")

    if notes:
        lines.append("")
        lines.append("## Notes")
        lines.append("")
        for note in notes:
            lines.append(f"- {note}")

    lines.append("")
    return "\n".join(lines)


def _summarize_measurement(measurement: str, points: list[RawPoint]) -> MeasurementCoverageSummary:
    first_timestamp = points[0].timestamp
    last_timestamp = points[-1].timestamp
    return MeasurementCoverageSummary(
        measurement=measurement,
        point_count=len(points),
        first_timestamp=first_timestamp,
        last_timestamp=last_timestamp,
        span_ms=_duration_ms(first_timestamp, last_timestamp),
    )


def _duration_ms(start: datetime, stop: datetime) -> int:
    return int((stop - start).total_seconds() * 1000)


def _render_stream_block(summary: StreamCoverageSummary) -> list[str]:
    lines = [f"### `{summary.stream_kind}`"]
    lines.append(f"- points: `{summary.point_count}`")
    lines.append(f"- measurements: `{summary.measurement_count}`")
    if summary.first_timestamp is not None:
        lines.append(f"- first timestamp: `{summary.first_timestamp.isoformat()}`")
        lines.append(f"- last timestamp: `{summary.last_timestamp.isoformat()}`")
        lines.append(f"- span ms: `{summary.span_ms}`")
    for measurement in summary.measurements:
        lines.append(
            f"- measurement `{measurement.measurement}`: `{measurement.point_count}` points, "
            f"`{measurement.first_timestamp.isoformat()}` -> `{measurement.last_timestamp.isoformat()}`"
        )
    return lines
