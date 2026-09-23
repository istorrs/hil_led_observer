"""Synthetic camera stream generation for headless, hardware-free tests.

``make_frame`` builds a single solid-color BGR frame standing in for a
camera capture of the LED's ROI. ``SyntheticFrameSource`` implements the
same :class:`~led_observer.capture.FrameSource` protocol as a real camera,
playing back a schedule of ON/OFF segments paced against wall-clock time
(so it exercises :class:`~led_observer.service.LEDObserverService`'s real
threaded polling loop) with optional frame-drop and pixel-noise simulation.
"""

from __future__ import annotations

import time

import numpy as np

from led_observer.calibration import Frame

ON_BGR: tuple[int, int, int] = (60, 220, 60)
OFF_BGR: tuple[int, int, int] = (12, 12, 12)


def make_frame(
    is_on: bool,
    size: tuple[int, int] = (32, 32),
    on_bgr: tuple[int, int, int] = ON_BGR,
    off_bgr: tuple[int, int, int] = OFF_BGR,
    noise_std: float = 0.0,
    rng: np.random.Generator | None = None,
) -> Frame:
    """A single solid-color frame representing the LED's ROI in one state."""
    height, width = size
    color = np.array(on_bgr if is_on else off_bgr, dtype=np.float64)
    frame = np.tile(color, (height, width, 1))
    if noise_std:
        generator = rng if rng is not None else np.random.default_rng()
        frame = frame + generator.normal(0.0, noise_std, frame.shape)
    return np.clip(frame, 0, 255).astype(np.uint8)


class SyntheticFrameSource:
    """A scripted ON/OFF cadence, played back against wall-clock time.

    ``schedule`` is a list of ``(duration_s, is_on)`` segments played in
    order; once exhausted, the last segment's state holds indefinitely.
    """

    def __init__(
        self,
        schedule: list[tuple[float, bool]],
        *,
        frame_size: tuple[int, int] = (32, 32),
        noise_std: float = 0.0,
        drop_rate: float = 0.0,
        rng_seed: int = 0,
    ) -> None:
        if not schedule:
            raise ValueError("schedule must contain at least one segment")
        self._schedule = schedule
        self._frame_size = frame_size
        self._noise_std = noise_std
        self._drop_rate = drop_rate
        self._rng = np.random.default_rng(rng_seed)
        self._start_time: float | None = None
        self.read_count = 0
        self.released = False

    @property
    def total_scheduled_duration_s(self) -> float:
        return sum(duration for duration, _ in self._schedule)

    def _is_on_at(self, elapsed_s: float) -> bool:
        cursor = 0.0
        for duration, is_on in self._schedule:
            if elapsed_s < cursor + duration:
                return is_on
            cursor += duration
        return self._schedule[-1][1]

    def read(self) -> tuple[bool, Frame | None]:
        if self._start_time is None:
            self._start_time = time.monotonic()
        self.read_count += 1

        if self._drop_rate and self._rng.random() < self._drop_rate:
            return False, None

        elapsed = time.monotonic() - self._start_time
        is_on = self._is_on_at(elapsed)
        frame = make_frame(is_on, size=self._frame_size, noise_std=self._noise_std, rng=self._rng)
        return True, frame

    def release(self) -> None:
        self.released = True


class FailingFrameSource:
    """Always reports a read failure -- exercises the dropped-frame path."""

    def __init__(self) -> None:
        self.released = False

    def read(self) -> tuple[bool, Frame | None]:
        return False, None

    def release(self) -> None:
        self.released = True


class CrashingFrameSource:
    """Returns a frame that blows up during classification (wrong shape),
    exercising the worker thread's per-frame exception handling."""

    def __init__(self) -> None:
        self.released = False

    def read(self) -> tuple[bool, Frame | None]:
        return True, np.zeros((1, 1, 3), dtype=np.uint8)

    def release(self) -> None:
        self.released = True
