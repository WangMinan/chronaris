"""Incremental window emission for replay or near-real-time inference."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from chronaris.schema.models import AlignedPoint, AlignedSortieBundle, SampleWindow, StreamKind, WindowConfig


@dataclass(frozen=True, slots=True)
class StreamingPointEvent:
    """One aligned point tagged with its source stream."""

    stream_kind: StreamKind
    point: AlignedPoint


@dataclass(frozen=True, slots=True)
class StreamingWindowBufferDiagnostics:
    """Operational counters for one streaming window buffer."""

    latest_offset_ms: int
    out_of_order_event_count: int
    max_cached_points_per_stream: int | None
    physiology_cached_points: int
    vehicle_cached_points: int


def iter_aligned_sortie_events(bundle: AlignedSortieBundle) -> tuple[StreamingPointEvent, ...]:
    """Return all aligned points sorted into one replay-ready event stream."""

    events = [
        StreamingPointEvent(stream_kind=StreamKind.PHYSIOLOGY, point=point)
        for point in bundle.physiology_points
    ]
    events.extend(
        StreamingPointEvent(stream_kind=StreamKind.VEHICLE, point=point)
        for point in bundle.vehicle_points
    )
    return tuple(sorted(events, key=lambda item: (item.point.offset_ms, item.stream_kind.value)))


class StreamingWindowBuffer:
    """Incrementally materialize `SampleWindow`s from an aligned point stream."""

    def __init__(
        self,
        *,
        sortie_id: str,
        window_config: WindowConfig,
        sample_id_prefix: str | None = None,
        max_cached_points_per_stream: int | None = None,
        allow_out_of_order: bool = True,
    ) -> None:
        self.sortie_id = sortie_id
        self.window_config = window_config
        self.sample_id_prefix = sample_id_prefix or sortie_id
        self.max_cached_points_per_stream = max_cached_points_per_stream
        self.allow_out_of_order = allow_out_of_order
        self._physiology_points: deque[AlignedPoint] = deque()
        self._vehicle_points: deque[AlignedPoint] = deque()
        self._next_window_start_ms = 0
        self._window_index = 0
        self._latest_offset_ms = -1
        self._out_of_order_event_count = 0

    def push(self, event: StreamingPointEvent) -> tuple[SampleWindow, ...]:
        """Push one aligned point and emit any newly completed windows."""

        if event.point.offset_ms < self._latest_offset_ms:
            self._out_of_order_event_count += 1
            if not self.allow_out_of_order:
                raise ValueError("out-of-order streaming point is not allowed.")
        self._latest_offset_ms = max(self._latest_offset_ms, int(event.point.offset_ms))
        if event.stream_kind == StreamKind.PHYSIOLOGY:
            self._append_point(self._physiology_points, event.point)
        elif event.stream_kind == StreamKind.VEHICLE:
            self._append_point(self._vehicle_points, event.point)
        else:  # pragma: no cover - defensive guard for future stream kinds.
            raise ValueError(f"unsupported stream kind: {event.stream_kind}")
        return self._emit_ready_windows(allow_partial=False)

    def flush(self) -> tuple[SampleWindow, ...]:
        """Emit the trailing partial window when configured to allow it."""

        return self._emit_ready_windows(allow_partial=self.window_config.allow_partial_last_window, force_flush=True)

    def _emit_ready_windows(
        self,
        *,
        allow_partial: bool,
        force_flush: bool = False,
    ) -> tuple[SampleWindow, ...]:
        emitted: list[SampleWindow] = []
        duration_ms = self.window_config.duration_ms
        stride_ms = self.window_config.stride_ms
        while True:
            window_end_ms = self._next_window_start_ms + duration_ms
            is_complete = self._latest_offset_ms >= window_end_ms
            can_emit_partial = force_flush and allow_partial and self._latest_offset_ms >= self._next_window_start_ms
            if not is_complete and not can_emit_partial:
                break
            emitted_window = self._build_window(
                start_offset_ms=self._next_window_start_ms,
                end_offset_ms=window_end_ms,
            )
            if emitted_window is not None:
                emitted.append(emitted_window)
            self._next_window_start_ms += stride_ms
            self._window_index += 1
            self._prune_points(before_offset_ms=self._next_window_start_ms)
            if can_emit_partial and not is_complete:
                break
        return tuple(emitted)

    def _build_window(
        self,
        *,
        start_offset_ms: int,
        end_offset_ms: int,
    ) -> SampleWindow | None:
        physiology_points = tuple(
            point for point in self._physiology_points if start_offset_ms <= point.offset_ms < end_offset_ms
        )
        vehicle_points = tuple(
            point for point in self._vehicle_points if start_offset_ms <= point.offset_ms < end_offset_ms
        )
        if len(physiology_points) < self.window_config.min_physiology_points:
            return None
        if len(vehicle_points) < self.window_config.min_vehicle_points:
            return None
        return SampleWindow(
            sample_id=f"{self.sample_id_prefix}::window-{self._window_index:04d}",
            sortie_id=self.sortie_id,
            window_index=self._window_index,
            start_offset_ms=start_offset_ms,
            end_offset_ms=end_offset_ms,
            physiology_points=physiology_points,
            vehicle_points=vehicle_points,
        )

    def _prune_points(self, *, before_offset_ms: int) -> None:
        while self._physiology_points and self._physiology_points[0].offset_ms < before_offset_ms:
            self._physiology_points.popleft()
        while self._vehicle_points and self._vehicle_points[0].offset_ms < before_offset_ms:
            self._vehicle_points.popleft()

    @property
    def diagnostics(self) -> StreamingWindowBufferDiagnostics:
        return StreamingWindowBufferDiagnostics(
            latest_offset_ms=self._latest_offset_ms,
            out_of_order_event_count=self._out_of_order_event_count,
            max_cached_points_per_stream=self.max_cached_points_per_stream,
            physiology_cached_points=len(self._physiology_points),
            vehicle_cached_points=len(self._vehicle_points),
        )

    def _append_point(self, target: deque[AlignedPoint], point: AlignedPoint) -> None:
        if not target or point.offset_ms >= target[-1].offset_ms:
            target.append(point)
        else:
            rows = list(target)
            insert_index = 0
            while insert_index < len(rows) and rows[insert_index].offset_ms <= point.offset_ms:
                insert_index += 1
            rows.insert(insert_index, point)
            target.clear()
            target.extend(rows)
        if self.max_cached_points_per_stream is not None:
            while len(target) > self.max_cached_points_per_stream:
                target.popleft()
