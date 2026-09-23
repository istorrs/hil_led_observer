#!/usr/bin/env python3
"""Example: a HIL test script driving led_observer against a DUT.

Demonstrates the intended integration shape:

1. Calibrate against the DUT's status LED (ON/OFF reference frames).
2. Trigger the DUT into the behavior under test.
3. Open an observation window with the expected cadence schema and a
   timeout, so a hung/incorrect DUT fails fast instead of blocking forever.
4. Stop the window and inspect the pass/fail verification, exiting non-zero
   on failure (as a pytest test or CI step would).

Runs standalone (``python examples/hil_integration_demo.py``) against an
in-process synthetic LED, so it's runnable without camera hardware. Against
real hardware, swap ``InProcessSyntheticLED`` for
``led_observer.capture.OpenCVCameraSource`` and load a
:class:`~led_observer.calibration.CalibrationProfile` captured on the bench
with :func:`~led_observer.calibration.build_calibration_profile`.
"""

from __future__ import annotations

import sys
import time

import numpy as np

from led_observer.calibration import ROI, Frame, build_calibration_profile
from led_observer.capture import FrameSource
from led_observer.models import LEDState, SequenceStep
from led_observer.service import LEDObserverService

ROI_SIZE = (32, 32)
ON_BGR = (60, 220, 60)
OFF_BGR = (12, 12, 12)
BLINK_WINDOW_S = 3.0

# A live classifier needs to see two full pulses of a *new* cadence before
# it can confirm periodicity (one pulse alone is indistinguishable from a
# SINGLE_FLASH or a steady ON) -- so the first cycle of the slow blink is
# briefly, correctly, seen as SINGLE_FLASH-then-ON before settling into
# SLOW_BLINK. This is a real property of any live/causal detector, not
# noise to filter out, so the expected schema names it explicitly.
EXPECTED_SEQUENCE = [
    SequenceStep(LEDState.OFF, duration_s=None),
    SequenceStep(LEDState.SINGLE_FLASH, duration_s=None),
    SequenceStep(LEDState.OFF, duration_s=None),
    SequenceStep(LEDState.SINGLE_FLASH, duration_s=None),
    SequenceStep(LEDState.ON, duration_s=None),
    SequenceStep(LEDState.SLOW_BLINK, duration_s=None),
    SequenceStep(LEDState.FAST_BLINK, duration_s=None),
]

# Matches EXPECTED_SEQUENCE, so this demo shows a PASS end-to-end.
_DUT_SCHEDULE: list[tuple[float, bool]] = [
    (6.0, False),
    (0.3, True),
    (8.0, False),
]
_SLOW_BLINK_START_S = sum(d for d, _ in _DUT_SCHEDULE)
_SLOW_BLINK_DURATION_S = 8.0
_FAST_BLINK_START_S = _SLOW_BLINK_START_S + _SLOW_BLINK_DURATION_S
_FAST_BLINK_DURATION_S = 6.0


def _make_frame(is_on: bool) -> Frame:
    color = np.array(ON_BGR if is_on else OFF_BGR, dtype=np.uint8)
    return np.tile(color, (ROI_SIZE[0], ROI_SIZE[1], 1))


class InProcessSyntheticLED(FrameSource):
    """Stands in for a real camera + DUT: plays back ``_DUT_SCHEDULE`` plus a
    faked slow-then-fast blink, so this demo passes without hardware."""

    def __init__(self) -> None:
        self._start_time: float | None = None

    def read(self) -> tuple[bool, Frame | None]:
        if self._start_time is None:
            self._start_time = time.monotonic()
        elapsed = time.monotonic() - self._start_time
        return True, _make_frame(self._is_on_at(elapsed))

    def _is_on_at(self, elapsed: float) -> bool:
        if elapsed < _SLOW_BLINK_START_S:
            cursor = 0.0
            for duration, is_on in _DUT_SCHEDULE:
                if elapsed < cursor + duration:
                    return is_on
                cursor += duration
            return False
        if elapsed < _FAST_BLINK_START_S:
            return (elapsed - _SLOW_BLINK_START_S) % 1.0 < 0.5  # ~1 Hz
        return (elapsed - _FAST_BLINK_START_S) % 0.2 < 0.1  # ~5 Hz

    def release(self) -> None:
        pass


def stimulate_dut() -> None:
    """Placeholder for whatever triggers the DUT in a real rig (a GPIO
    pulse, a relay, a message over its control channel, ...)."""
    print("[HIL] Stimulating DUT: requesting connect sequence")


def main() -> int:
    roi = ROI(x=0, y=0, width=ROI_SIZE[1], height=ROI_SIZE[0])
    calibration = build_calibration_profile(
        roi,
        on_frames=[_make_frame(True) for _ in range(5)],
        off_frames=[_make_frame(False) for _ in range(5)],
    )

    with LEDObserverService(
        calibration, InProcessSyntheticLED(), blink_window_s=BLINK_WINDOW_S
    ) as service:
        stimulate_dut()
        service.start_observation(expected_sequence=EXPECTED_SEQUENCE, max_timeout_s=30.0)
        print("[HIL] Observing LED cadence...")
        time.sleep(_FAST_BLINK_START_S + _FAST_BLINK_DURATION_S)
        result = service.stop_observation()

    print(f"[HIL] Observed {len(result.segments)} segment(s) over {result.duration:.2f}s:")
    for segment in result.segments:
        print(
            f"  {segment.state.value:<12} {segment.start_time:6.2f}s -> "
            f"{segment.end_time:6.2f}s ({segment.duration:.2f}s)"
        )

    report = result.verification
    assert report is not None
    if report.matched:
        print("[HIL] PASS: observed cadence matches expected sequence")
        return 0

    assert report.divergence_point is not None
    print(f"[HIL] FAIL at step {report.divergence_point.index}: {report.divergence_point.reason}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
