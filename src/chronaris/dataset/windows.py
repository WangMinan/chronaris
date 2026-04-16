"""Window generation for aligned sortie bundles."""

from __future__ import annotations

from chronaris.schema.models import AlignedPoint, AlignedSortieBundle, SampleWindow, WindowConfig


def build_sample_windows(
    aligned_bundle: AlignedSortieBundle,
    config: WindowConfig,
) -> tuple[SampleWindow, ...]:
    """Slice one aligned sortie bundle into windows."""

    all_points = aligned_bundle.physiology_points + aligned_bundle.vehicle_points
    if not all_points:
        return ()

    min_offset = min(point.offset_ms for point in all_points)
    terminal_end = max(point.offset_ms for point in all_points) + 1

    windows: list[SampleWindow] = []
    index = 0
    start = min_offset

    while start < terminal_end:
        requested_end = start + config.duration_ms
        if requested_end > terminal_end:
            if not config.allow_partial_last_window:
                break
            actual_end = terminal_end
        else:
            actual_end = requested_end

        physiology_points = _slice_points(aligned_bundle.physiology_points, start, actual_end)
        vehicle_points = _slice_points(aligned_bundle.vehicle_points, start, actual_end)

        if (
            len(physiology_points) >= config.min_physiology_points
            and len(vehicle_points) >= config.min_vehicle_points
        ):
            windows.append(
                SampleWindow(
                    sample_id=f"{aligned_bundle.locator.sortie_id}:{index:04d}",
                    sortie_id=aligned_bundle.locator.sortie_id,
                    window_index=index,
                    start_offset_ms=start,
                    end_offset_ms=actual_end,
                    physiology_points=physiology_points,
                    vehicle_points=vehicle_points,
                )
            )
            index += 1

        start += config.stride_ms

    return tuple(windows)


def _slice_points(
    points: tuple[AlignedPoint, ...],
    start_offset_ms: int,
    end_offset_ms: int,
) -> tuple[AlignedPoint, ...]:
    return tuple(point for point in points if start_offset_ms <= point.offset_ms < end_offset_ms)
