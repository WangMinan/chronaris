"""Small matplotlib helpers for readable task evaluation bar-chart labels."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np


def short_value_label(value: object, *, max_chars: int = 8) -> str:
    """Format a numeric label compactly enough to sit on a bar."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(number):
        return ""
    if abs(number) >= 1.0 and abs(number - round(number)) < 1e-9:
        text = str(int(round(number)))
    else:
        magnitude = abs(number)
        if magnitude == 0.0:
            text = "0"
        elif magnitude >= 10000.0 or magnitude < 0.001:
            text = f"{number:.2e}"
        elif magnitude >= 100.0:
            text = f"{number:.1f}"
        elif magnitude >= 10.0:
            text = f"{number:.2f}"
        elif magnitude >= 1.0:
            text = f"{number:.3f}"
        else:
            text = f"{number:.4f}"
        if "e" not in text:
            text = text.rstrip("0").rstrip(".")
    if len(text) > max_chars:
        text = f"{number:.2g}"
    return text[:max_chars]


def label_vertical_bars(axis, bars: Iterable[object], values: Iterable[object] | None = None, *, fontsize: int = 7) -> None:
    bar_list = list(bars)
    vals = list(values) if values is not None else [bar.get_height() for bar in bar_list]
    for bar, value in zip(bar_list, vals, strict=False):
        label = short_value_label(value)
        if not label:
            continue
        height = float(bar.get_height())
        y = float(bar.get_y()) + height
        offset = 3 if height >= 0 else -3
        axis.annotate(
            label,
            xy=(float(bar.get_x()) + float(bar.get_width()) / 2.0, y),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va="bottom" if height >= 0 else "top",
            fontsize=fontsize,
            clip_on=False,
        )
    axis.margins(y=0.14)


def label_horizontal_bars(axis, bars: Iterable[object], values: Iterable[object] | None = None, *, fontsize: int = 7) -> None:
    bar_list = list(bars)
    vals = list(values) if values is not None else [bar.get_width() for bar in bar_list]
    widths = [float(value) for value in vals if _is_finite_number(value)]
    span = max(max(widths) - min(widths), max((abs(value) for value in widths), default=1.0), 1.0) if widths else 1.0
    for bar, value in zip(bar_list, vals, strict=False):
        if not _is_finite_number(value):
            continue
        label = short_value_label(value)
        if not label:
            continue
        width = float(bar.get_width())
        x = width + (0.01 * span if width >= 0 else -0.01 * span)
        axis.text(
            x,
            float(bar.get_y()) + float(bar.get_height()) / 2.0,
            label,
            va="center",
            ha="left" if width >= 0 else "right",
            fontsize=fontsize,
            clip_on=False,
        )
    axis.margins(x=0.18)


def label_stack_totals(axis, x_positions: Iterable[object], totals: Iterable[object], *, fontsize: int = 7) -> None:
    finite_totals = [float(value) for value in totals if _is_finite_number(value)]
    if not finite_totals:
        return
    for x, total in zip(x_positions, totals, strict=False):
        if not _is_finite_number(total):
            continue
        label = short_value_label(total)
        if not label:
            continue
        axis.annotate(
            label,
            xy=(x, float(total)),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=fontsize,
            clip_on=False,
        )
    axis.margins(y=0.14)


def _is_finite_number(value: object) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False
