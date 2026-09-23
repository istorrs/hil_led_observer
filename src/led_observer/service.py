"""High-level, thread-safe observation API meant to be driven directly by a
HIL test script.

A background daemon thread continuously pulls frames from a
:class:`~led_observer.capture.FrameSource`, classifies them, and keeps
``get_current_state()`` fresh at all times. ``start_observation`` /
``stop_observation`` bracket a window during which classified states are
also appended to a timeline for later :func:`~led_observer.sequence.verify_sequence`
comparison -- this lets a test harness watch the LED continuously while only
recording the segment relevant to one test step.
"""

from __future__ import annotations

import threading
import time
from types import TracebackType

from led_observer.calibration import CalibrationProfile
from led_observer.capture import FrameSource
from led_observer.detector import BlinkClassifier, InstantClassifier
from led_observer.models import (
    LEDState,
    ObservationResult,
    ObservedSegment,
    Pulse,
    SequenceStep,
    VerificationReport,
)
from led_observer.sequence import PulseRecorder, TimelineRecorder
from led_observer.sequence import verify_sequence as _verify_sequence


class ObservationInProgressError(RuntimeError):
    """Raised by :meth:`LEDObserverService.start_observation` on re-entry."""


class LEDObserverService:
    """Thread-safe façade over calibration + detection + timeline recording."""

    def __init__(
        self,
        calibration: CalibrationProfile,
        frame_source: FrameSource,
        blink_window_s: float = 3.0,
        poll_interval_s: float = 0.0,
        default_tolerance_pct: float = 0.20,
    ) -> None:
        self._instant = InstantClassifier(calibration)
        self._blink = BlinkClassifier(window_s=blink_window_s)
        self._frame_source = frame_source
        self._poll_interval_s = poll_interval_s
        self._default_tolerance_pct = default_tolerance_pct

        self._lock = threading.Lock()
        self._current_state = LEDState.UNKNOWN
        self._recorder = TimelineRecorder()
        self._pulse_recorder = PulseRecorder()
        self._observing = False
        self._timed_out = False
        self._deadline: float | None = None
        self._obs_start = 0.0
        self._obs_end = 0.0
        self._dropped_frames = 0
        self._expected_sequence: list[SequenceStep] | None = None

        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, name="led-observer-worker", daemon=True)
        self._thread.start()

    def start_observation(
        self,
        expected_sequence: list[SequenceStep] | None = None,
        max_timeout_s: float | None = None,
    ) -> None:
        """Opens a new recording window. Raises :class:`ObservationInProgressError`
        if a window is already open; call :meth:`stop_observation` first."""
        with self._lock:
            if self._observing:
                raise ObservationInProgressError(
                    "an observation window is already open; call stop_observation() first"
                )
            self._recorder.reset()
            self._pulse_recorder.reset()
            self._timed_out = False
            self._dropped_frames = 0
            self._expected_sequence = expected_sequence
            self._obs_start = time.monotonic()
            self._obs_end = self._obs_start
            self._deadline = self._obs_start + max_timeout_s if max_timeout_s is not None else None
            self._observing = True

    def stop_observation(self) -> ObservationResult:
        """Closes the current recording window (a no-op on the timing fields
        if it already auto-closed due to ``max_timeout_s``) and returns
        everything recorded during it."""
        with self._lock:
            if self._observing:
                self._obs_end = time.monotonic()
                self._observing = False
            segments = self._recorder.finalize(self._obs_end)
            pulses = self._pulse_recorder.pulses
            expected_sequence = self._expected_sequence
            result = ObservationResult(
                segments=segments,
                transitions=self._recorder.transitions,
                pulses=pulses,
                start_time=self._obs_start,
                end_time=self._obs_end,
                timed_out=self._timed_out,
                dropped_frames=self._dropped_frames,
            )
        if expected_sequence is not None:
            result.verification = _verify_sequence(
                expected_sequence, segments, self._default_tolerance_pct, pulses=pulses
            )
        return result

    def get_current_state(self) -> LEDState:
        """Latest cadence classification, independent of any observation
        window -- safe to poll at any time."""
        with self._lock:
            return self._current_state

    @staticmethod
    def verify_sequence(
        expected_sequence: list[SequenceStep],
        actual_sequence: list[ObservedSegment],
        tolerance_pct: float = 0.20,
        pulses: list[Pulse] | None = None,
    ) -> VerificationReport:
        """Convenience wrapper; see :func:`led_observer.sequence.verify_sequence`."""
        return _verify_sequence(expected_sequence, actual_sequence, tolerance_pct, pulses=pulses)

    def close(self) -> None:
        """Stops the worker thread and releases the underlying frame source."""
        self._stop_event.set()
        self._thread.join(timeout=2.0)
        self._frame_source.release()

    def __enter__(self) -> LEDObserverService:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            success, frame = self._frame_source.read()
            timestamp = time.monotonic()

            if not success or frame is None:
                with self._lock:
                    self._dropped_frames += 1
                self._sleep()
                continue

            try:
                is_on = self._instant.classify_frame(frame)
            except Exception:  # a malformed/corrupt frame must never kill the worker thread
                with self._lock:
                    self._dropped_frames += 1
                self._sleep()
                continue

            self._blink.add_sample(timestamp, is_on)
            state = self._blink.classify()

            with self._lock:
                self._current_state = state
                if self._observing:
                    self._pulse_recorder.add(timestamp, is_on)
                    if state is not LEDState.UNKNOWN:
                        # A transient UNKNOWN (e.g. right after start, before the
                        # cadence window has enough history) is not a real LED
                        # state and must not appear as a spurious leading/gap
                        # segment in the recorded timeline.
                        self._recorder.add(state, timestamp)
                    if self._deadline is not None and timestamp >= self._deadline:
                        self._obs_end = timestamp
                        self._observing = False
                        self._timed_out = True

            self._sleep()

    def _sleep(self) -> None:
        if self._poll_interval_s > 0:
            time.sleep(self._poll_interval_s)
