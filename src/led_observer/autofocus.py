"""Software autofocus: maximize image sharpness by driving the camera's
lens position, instead of trusting the camera's own hardware autofocus.

Continuous hardware autofocus tends to "breathe" (hunt/refocus) when aimed
at a small, high-contrast target like an LED against a dark PCB -- exactly
the kind of scene this library points a camera at -- which is disruptive to
a fixed calibrated ROI. The fix is the same one embedded-vision projects
converge on: disable the camera's autofocus and drive focus to a fixed
position chosen once by a software sweep (`autofocus_sweep`) that maximizes
a sharpness metric (variance of the Laplacian -- a standard, cheap blur
detector: higher is sharper) via a coarse linear scan followed by a
localized hill-climb.

The sweep is decoupled from any specific camera API via three callables
(``get_frame``/``get_focus``/``set_focus``) so it's unit-testable against a
synthetic focus-vs-sharpness curve, with no real camera or ``time.sleep``
needed (see :class:`OpenCVCameraSource.autofocus` for the real-camera glue).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import cv2
import numpy as np

from led_observer.calibration import ROI, Frame


@dataclass(frozen=True, slots=True)
class AutofocusResult:
    """Outcome of one :func:`autofocus_sweep` run."""

    focus: int
    """The focus position left active on the camera."""
    sharpness: float
    initial_focus: int
    initial_sharpness: float
    improved: bool
    """False means the sweep didn't beat ``min_improvement_pct`` and the
    camera was left at ``initial_focus`` rather than a worse or
    not-meaningfully-better position."""
    samples: tuple[tuple[int, float], ...]
    """``(focus_position, sharpness)`` for every position sampled, in the
    order tested -- useful for debugging/plotting a focus curve."""


def measure_sharpness(frame: Frame, roi: ROI | None = None) -> float:
    """Variance of the Laplacian over (a crop of) ``frame`` -- a standard,
    cheap blur-detection metric: higher means sharper. Restricting to
    ``roi`` avoids out-of-focus background clutter dominating the score
    when the calibrated LED ROI is already known."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    gray = gray.astype(np.uint8, copy=False)
    if roi is not None:
        gray = roi.crop(gray)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _detect_focus_range(
    get_focus: Callable[[], int],
    set_focus: Callable[[int], None],
    probe_margin: int = 1000,
    settle_s: float = 0.1,
) -> tuple[int, int]:
    """Probes the camera's real focus range by driving it well past a
    plausible max/min and reading back whatever the driver clamped it to,
    rather than assuming a fixed range -- it varies by camera model."""
    initial = get_focus()

    set_focus(initial + probe_margin)
    if settle_s > 0:
        time.sleep(settle_s)
    upper = get_focus()

    set_focus(0)
    if settle_s > 0:
        time.sleep(settle_s)
    lower = get_focus()

    set_focus(initial)
    if settle_s > 0:
        time.sleep(settle_s)

    return min(lower, initial), max(upper, initial)


def autofocus_sweep(
    get_frame: Callable[[], Frame | None],
    get_focus: Callable[[], int],
    set_focus: Callable[[int], None],
    *,
    roi: ROI | None = None,
    focus_range: tuple[int, int] | None = None,
    settle_s: float = 0.1,
    min_improvement_pct: float = 5.0,
) -> AutofocusResult:
    """Coarse-to-fine hill-climbing software autofocus.

    A coarse linear scan across the focus range finds the right
    neighborhood; a localized search around the best coarse position then
    refines it (compare a midpoint against its immediate neighbors, step
    toward whichever is sharper, shrinking the interval each time -- a
    binary search that assumes the sharpness-vs-focus curve is
    unimodal near the coarse-scan optimum, which holds for a normal lens).

    Falls back to ``initial_focus`` if the sweep doesn't improve sharpness
    by at least ``min_improvement_pct`` -- a lens that was already
    well-focused, or a target with too little texture for this metric to
    read reliably, shouldn't be made worse by running the sweep.

    ``settle_s`` is real settle time for the physical focus motor between
    setting a position and reading a frame back; pass ``0.0`` against a
    synthetic frame source in tests, where there's no motor to wait on.
    """

    def sample(focus_pos: int) -> float:
        set_focus(focus_pos)
        if settle_s > 0:
            time.sleep(settle_s)
        frame = get_frame()
        return 0.0 if frame is None else measure_sharpness(frame, roi)

    initial_focus = get_focus()
    initial_sharpness = sample(initial_focus)
    samples: list[tuple[int, float]] = [(initial_focus, initial_sharpness)]

    focus_min, focus_max = (
        focus_range
        if focus_range is not None
        else _detect_focus_range(get_focus, set_focus, settle_s=settle_s)
    )

    best_focus, best_sharpness = initial_focus, initial_sharpness

    if focus_max > focus_min:
        coarse_step = max(5, (focus_max - focus_min) // 20)
        for focus_pos in range(focus_min, focus_max + 1, coarse_step):
            sharpness = sample(focus_pos)
            samples.append((focus_pos, sharpness))
            if sharpness > best_sharpness:
                best_focus, best_sharpness = focus_pos, sharpness

        left = max(focus_min, best_focus - coarse_step)
        right = min(focus_max, best_focus + coarse_step)
        while left <= right:
            mid = (left + right) // 2
            mid_sharpness = sample(mid)
            samples.append((mid, mid_sharpness))

            left_sharpness = None
            if mid - 1 >= focus_min:
                left_sharpness = sample(mid - 1)
                samples.append((mid - 1, left_sharpness))
            right_sharpness = None
            if mid + 1 <= focus_max:
                right_sharpness = sample(mid + 1)
                samples.append((mid + 1, right_sharpness))

            if left_sharpness is not None and left_sharpness > mid_sharpness:
                right = mid - 1
            elif right_sharpness is not None and right_sharpness > mid_sharpness:
                left = mid + 1
            else:
                if mid_sharpness > best_sharpness:
                    best_focus, best_sharpness = mid, mid_sharpness
                break

    improved = initial_sharpness > 0 and best_sharpness > initial_sharpness * (
        1 + min_improvement_pct / 100
    )
    final_focus = best_focus if improved else initial_focus
    final_sharpness = best_sharpness if improved else initial_sharpness

    set_focus(final_focus)
    if settle_s > 0:
        time.sleep(settle_s)

    return AutofocusResult(
        focus=final_focus,
        sharpness=final_sharpness,
        initial_focus=initial_focus,
        initial_sharpness=initial_sharpness,
        improved=improved,
        samples=tuple(samples),
    )
