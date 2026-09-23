from __future__ import annotations

import cv2
import numpy as np

from led_observer.autofocus import AutofocusResult, autofocus_sweep, measure_sharpness
from led_observer.calibration import ROI, Frame
from tests.mocks import make_frame


def test_measure_sharpness_is_higher_for_sharp_than_blurred() -> None:
    sharp = np.zeros((64, 64), dtype=np.uint8)
    cv2.rectangle(sharp, (20, 20), (44, 44), 255, -1)
    blurred = cv2.GaussianBlur(sharp, (15, 15), 0).astype(np.uint8, copy=False)

    assert measure_sharpness(sharp) > measure_sharpness(blurred)


def test_measure_sharpness_accepts_color_frames() -> None:
    frame = make_frame(True, size=(64, 64))
    assert measure_sharpness(frame) >= 0.0


def test_measure_sharpness_restricts_to_roi() -> None:
    # Identical inside a 40x40 ROI in both frames; only a region well
    # outside it differs (sharp vs. heavily blurred) -- the ROI crop must
    # ignore that difference entirely.
    base = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(base, (10, 10), (30, 30), 255, -1)

    with_blurred_clutter = base.copy()
    cv2.rectangle(with_blurred_clutter, (60, 60), (90, 90), 255, -1)
    clutter = with_blurred_clutter[55:95, 55:95]
    clutter[:] = cv2.GaussianBlur(clutter, (25, 25), 0)

    roi = ROI(x=0, y=0, width=40, height=40)
    assert measure_sharpness(with_blurred_clutter, roi=roi) == measure_sharpness(base, roi=roi)


class _SyntheticLens:
    """Simulates a focus motor: sharpness peaks at `peak_focus` and falls
    off with distance from it, via a Gaussian-blur kernel scaled by that
    distance -- the same synthetic-focus-curve idea used to test this in
    istorrs/led_detection's test_autofocus.py, adapted to this project's
    decoupled callable API (no camera/driver mock needed)."""

    def __init__(self, peak_focus: int = 150, size: tuple[int, int] = (240, 320)) -> None:
        self.peak_focus = peak_focus
        self.current_focus = 0
        self._size = size

    def get_focus(self) -> int:
        return self.current_focus

    def set_focus(self, position: int) -> None:
        self.current_focus = position

    def get_frame(self) -> Frame:
        height, width = self._size
        frame = np.zeros((height, width), dtype=np.uint8)
        cv2.circle(frame, (width // 2, height // 2), min(height, width) // 4, 255, -1)
        distance = abs(self.current_focus - self.peak_focus)
        kernel = int(distance / 3) * 2 + 1
        if kernel > 1:
            frame = cv2.GaussianBlur(frame, (kernel, kernel), 0).astype(np.uint8, copy=False)
        return frame


def test_autofocus_sweep_converges_near_the_sharpness_peak() -> None:
    lens = _SyntheticLens(peak_focus=150)
    lens.set_focus(0)

    result = autofocus_sweep(
        get_frame=lens.get_frame,
        get_focus=lens.get_focus,
        set_focus=lens.set_focus,
        focus_range=(0, 300),
        settle_s=0.0,
    )

    assert 140 <= result.focus <= 160
    assert result.improved is True
    assert lens.current_focus == result.focus


def test_autofocus_sweep_detects_focus_range_when_not_given() -> None:
    lens = _SyntheticLens(peak_focus=150)
    lens.set_focus(50)

    # No focus_range passed -- must probe for it via get_focus/set_focus.
    result = autofocus_sweep(
        get_frame=lens.get_frame,
        get_focus=lens.get_focus,
        set_focus=lens.set_focus,
        settle_s=0.0,
    )

    assert 140 <= result.focus <= 160


def test_autofocus_sweep_keeps_initial_focus_when_already_sharp() -> None:
    lens = _SyntheticLens(peak_focus=150)
    lens.set_focus(150)  # already at the peak

    result = autofocus_sweep(
        get_frame=lens.get_frame,
        get_focus=lens.get_focus,
        set_focus=lens.set_focus,
        focus_range=(0, 300),
        settle_s=0.0,
    )

    assert result.improved is False
    assert result.focus == 150
    assert lens.current_focus == 150


def test_autofocus_sweep_handles_a_flat_zero_focus_range() -> None:
    lens = _SyntheticLens(peak_focus=150)
    lens.set_focus(75)

    result = autofocus_sweep(
        get_frame=lens.get_frame,
        get_focus=lens.get_focus,
        set_focus=lens.set_focus,
        focus_range=(75, 75),  # camera reports no adjustable range
        settle_s=0.0,
    )

    assert result.focus == 75
    assert result.samples == ((75, result.initial_sharpness),)


def test_autofocus_sweep_returns_none_frame_as_zero_sharpness() -> None:
    def get_frame() -> None:
        return None

    result = autofocus_sweep(
        get_frame=get_frame,
        get_focus=lambda: 0,
        set_focus=lambda _position: None,
        focus_range=(0, 20),
        settle_s=0.0,
    )
    assert result.sharpness == 0.0
    assert result.improved is False


def test_autofocus_result_records_every_sample() -> None:
    lens = _SyntheticLens(peak_focus=150)
    result = autofocus_sweep(
        get_frame=lens.get_frame,
        get_focus=lens.get_focus,
        set_focus=lens.set_focus,
        focus_range=(0, 300),
        settle_s=0.0,
    )
    assert isinstance(result, AutofocusResult)
    assert len(result.samples) > 1
    focuses = [focus for focus, _ in result.samples]
    assert all(0 <= focus <= 300 for focus in focuses)
