"""Frame source abstraction: a real OpenCV camera, or (in tests) a synthetic
generator implementing the same :class:`FrameSource` protocol.
"""

from __future__ import annotations

from typing import Protocol

import cv2
import numpy as np

from led_observer.autofocus import AutofocusResult, autofocus_sweep
from led_observer.calibration import ROI, Frame


class CameraUnavailableError(RuntimeError):
    """Raised when the configured video device cannot be opened."""


class FrameSource(Protocol):
    """Anything that can hand back timestamped frames, real or synthetic."""

    def read(self) -> tuple[bool, Frame | None]:
        """Returns ``(success, frame)``. ``success=False`` means a dropped
        frame or transient read failure, not necessarily end-of-stream."""
        ...

    def release(self) -> None: ...


class OpenCVCameraSource:
    """Wraps ``cv2.VideoCapture`` for a real USB camera device."""

    def __init__(self, device_index: int = 0) -> None:
        self._capture = cv2.VideoCapture(device_index)
        if not self._capture.isOpened():
            self._capture.release()
            raise CameraUnavailableError(f"could not open video device {device_index}")

    def read(self) -> tuple[bool, Frame | None]:
        try:
            success, frame = self._capture.read()
        except cv2.error:
            return False, None
        if not success or frame is None:
            return False, None
        return True, frame.astype(np.uint8, copy=False)

    def release(self) -> None:
        self._capture.release()

    def disable_hardware_autofocus(self) -> None:
        """Turns off the camera's own continuous autofocus.

        Many UVC webcams keep "breathing" (continuously hunting/refocusing)
        on a small, high-contrast target like an LED against a dark PCB --
        exactly this library's use case -- which drifts a calibrated ROI
        out of focus over time. Setting ``CAP_PROP_AUTOFOCUS`` to 0 isn't
        always enough by itself on some drivers; writing back the current
        focus value immediately after is a documented workaround some
        cameras need to actually leave continuous-AF mode.
        """
        self._capture.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        current_focus = self._capture.get(cv2.CAP_PROP_FOCUS)
        self._capture.set(cv2.CAP_PROP_FOCUS, current_focus)

    def set_focus(self, position: int) -> None:
        """Directly sets a fixed focus position -- e.g. one already chosen
        by an earlier :meth:`autofocus` run -- without re-running the
        sweep. Does not itself touch ``CAP_PROP_AUTOFOCUS``; call
        :meth:`disable_hardware_autofocus` first if it hasn't been already.
        """
        self._capture.set(cv2.CAP_PROP_FOCUS, position)

    def get_focus(self) -> int:
        """Current focus position, as reported by the driver."""
        return int(self._capture.get(cv2.CAP_PROP_FOCUS))

    def autofocus(
        self,
        roi: ROI | None = None,
        focus_range: tuple[int, int] | None = None,
        settle_s: float = 0.1,
        min_improvement_pct: float = 5.0,
    ) -> AutofocusResult:
        """Disables hardware autofocus and runs :func:`~led_observer.autofocus.autofocus_sweep`
        against this camera to pick a fixed, sharp focus position.

        Pass ``roi`` (the calibrated LED ROI, once known) to score sharpness
        there instead of the full frame -- out-of-focus background clutter
        can otherwise dominate the metric. Re-asserts hardware-autofocus-off
        plus the chosen focus a few times afterward, since some cameras
        drift back into continuous AF otherwise.
        """
        self.disable_hardware_autofocus()

        result = autofocus_sweep(
            get_frame=lambda: self.read()[1],
            get_focus=self.get_focus,
            set_focus=self.set_focus,
            roi=roi,
            focus_range=focus_range,
            settle_s=settle_s,
            min_improvement_pct=min_improvement_pct,
        )

        for _ in range(3):
            self._capture.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            self.set_focus(result.focus)

        return result
