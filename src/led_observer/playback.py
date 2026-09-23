"""Record a real frame sequence once, then replay it offline as many times
as needed while tuning calibration/detection parameters -- without
repeating a slow, manual, real-hardware trigger sequence for every attempt.

A recording is a directory written by ``scripts/record_frames.py``: lossless
per-frame PNGs (deliberately not a compressed video -- see that script's
docstring for why) plus a ``manifest.json`` of each frame's original
capture-relative timestamp. :func:`load_recording` reads one back into
memory; :func:`replay_recording` feeds it through the real classification
pipeline using each frame's original timestamp directly, with no real-time
waiting -- the same instant, no-threads pattern
``tests/test_real_dut_sequences.py`` uses for synthetic data, applied here
to real recorded frames instead. :class:`RecordedFrameSource` is the
complementary `FrameSource`-protocol implementation, for running the real
threaded :class:`~led_observer.service.LEDObserverService` against a
recording at (approximately) its original real-time pacing instead.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from led_observer.calibration import CalibrationProfile, Frame
from led_observer.detector import BlinkClassifier, InstantClassifier
from led_observer.models import LEDState, ObservedSegment, Pulse
from led_observer.sequence import PulseRecorder, TimelineRecorder


@dataclass(frozen=True, slots=True)
class RecordedFrame:
    """One frame from a recording, with its original capture-relative
    timestamp (seconds since the recording started)."""

    timestamp: float
    frame: Frame


def load_recording(directory: str | Path) -> list[RecordedFrame]:
    """Loads a recording written by ``scripts/record_frames.py``: reads
    ``manifest.json`` and every listed PNG, in order."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))

    recording: list[RecordedFrame] = []
    for entry in manifest["frames"]:
        path = directory / entry["path"]
        frame = cv2.imread(str(path))
        if frame is None:
            raise ValueError(f"could not read recorded frame: {path}")
        recording.append(
            RecordedFrame(
                timestamp=entry["timestamp"],
                frame=frame.astype(np.uint8, copy=False),
            )
        )

    if not recording:
        raise ValueError(f"recording at {directory} has no frames")
    return recording


def replay_recording(
    recording: list[RecordedFrame],
    calibration: CalibrationProfile,
    blink_window_s: float = 3.0,
) -> tuple[list[ObservedSegment], list[Pulse]]:
    """Feeds a loaded recording through the real
    :class:`~led_observer.detector.InstantClassifier` /
    :class:`~led_observer.detector.BlinkClassifier` /
    :class:`~led_observer.sequence.TimelineRecorder` /
    :class:`~led_observer.sequence.PulseRecorder` pipeline, using each
    frame's *original recorded timestamp* directly rather than wall-clock
    time -- so trying a different ``calibration`` or ``blink_window_s``
    against the same captured footage is near-instant, with no real-time
    waiting and no camera needed.

    A per-frame classification failure (e.g. a corrupt frame) is skipped,
    mirroring :class:`~led_observer.service.LEDObserverService`'s handling
    of the same case against a live camera.
    """
    instant = InstantClassifier(calibration)
    blink = BlinkClassifier(window_s=blink_window_s)
    recorder = TimelineRecorder()
    pulse_recorder = PulseRecorder()

    for entry in recording:
        try:
            is_on = instant.classify_frame(entry.frame)
        except Exception:  # noqa: BLE001 - a bad frame must not abort the whole replay
            continue

        blink.add_sample(entry.timestamp, is_on)
        pulse_recorder.add(entry.timestamp, is_on)
        state = blink.classify()
        if state is not LEDState.UNKNOWN:
            recorder.add(state, entry.timestamp)

    segments = recorder.finalize(recording[-1].timestamp)
    return segments, pulse_recorder.pulses


class RecordedFrameSource:
    """Replays a loaded recording at (approximately) its original real-time
    pacing, implementing the same :class:`~led_observer.capture.FrameSource`
    protocol as a live camera -- so
    :class:`~led_observer.service.LEDObserverService`'s real background
    thread and ``time.monotonic()``-based classification run unmodified
    against previously captured footage.

    Prefer :func:`replay_recording` instead for fast, repeated offline
    parameter tuning -- this class exists for exercising the real
    threaded/live code path against recorded data instead of live hardware,
    e.g. to reproduce a service-level issue deterministically.
    """

    def __init__(
        self,
        recording: list[RecordedFrame],
        speed: float = 1.0,
        loop: bool = False,
    ) -> None:
        if not recording:
            raise ValueError("recording must contain at least one frame")
        if speed <= 0:
            raise ValueError("speed must be positive")
        self._recording = recording
        self._speed = speed
        self._loop = loop
        self._index = 0
        self._playback_start: float | None = None
        self.released = False

    def read(self) -> tuple[bool, Frame | None]:
        if self._index >= len(self._recording):
            if not self._loop:
                return False, None
            self.reset()

        if self._playback_start is None:
            self._playback_start = time.monotonic()

        entry = self._recording[self._index]
        target_time = self._playback_start + entry.timestamp / self._speed
        remaining = target_time - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)

        self._index += 1
        return True, entry.frame

    def reset(self) -> None:
        """Restarts playback from the first frame."""
        self._index = 0
        self._playback_start = None

    def release(self) -> None:
        self.released = True
