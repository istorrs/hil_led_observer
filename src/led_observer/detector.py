"""Frame-by-frame ON/OFF classification and windowed cadence detection.

Two layers, deliberately decoupled so each is independently testable:

* :class:`InstantClassifier` turns one OpenCV frame into a boolean on/off
  reading using the calibrated luminosity + color thresholds, with
  hysteresis to avoid chatter at the boundary.
* :class:`BlinkClassifier` turns a *stream* of ``(timestamp, is_on)``
  readings into a :class:`~led_observer.models.LEDState` cadence
  classification over a sliding time-domain window (edge/peak detection).
  :func:`estimate_frequency_fft` provides the frequency-domain alternative
  named in the spec, for cross-checking the time-domain estimate.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from led_observer.calibration import CalibrationProfile, Frame, FrameMetrics, compute_frame_metrics
from led_observer.models import LEDState

#: [f_low, f_high) bounds, Hz.
SLOW_BLINK_RANGE: tuple[float, float] = (0.5, 2.0)
#: [f_low, f_high] bounds, Hz.
FAST_BLINK_RANGE: tuple[float, float] = (2.0, 10.0)
#: Max ON duration, seconds, for an isolated pulse to count as a single flash.
SINGLE_FLASH_MAX_DURATION_S: float = 1.0


class InstantClassifier:
    """Classifies one frame's ROI as ON/OFF using hysteresis thresholds."""

    def __init__(self, profile: CalibrationProfile) -> None:
        self._profile = profile
        self._is_on = False

    def classify_frame(self, frame: Frame) -> bool:
        return self.classify_metrics(compute_frame_metrics(frame, self._profile))

    def classify_metrics(self, metrics: FrameMetrics) -> bool:
        """Pure-data entry point for the same hysteresis decision, useful for
        unit-testing threshold behavior without constructing image frames."""
        if (
            metrics.luminosity >= self._profile.luminosity_on_threshold
            and metrics.color_ratio >= self._profile.color_ratio_min
        ):
            self._is_on = True
        elif metrics.luminosity <= self._profile.luminosity_off_threshold:
            self._is_on = False
        # else: inside the hysteresis band -> retain previous reading.
        return self._is_on

    def reset(self) -> None:
        self._is_on = False


@dataclass(frozen=True, slots=True)
class _Pulse:
    rise_time: float
    fall_time: float

    @property
    def duration(self) -> float:
        return self.fall_time - self.rise_time


class BlinkClassifier:
    """Classifies LED cadence over a sliding window of on/off samples."""

    def __init__(
        self,
        window_s: float = 3.0,
        *,
        slow_blink_range: tuple[float, float] = SLOW_BLINK_RANGE,
        fast_blink_range: tuple[float, float] = FAST_BLINK_RANGE,
        single_flash_max_duration_s: float = SINGLE_FLASH_MAX_DURATION_S,
    ) -> None:
        if window_s <= 0:
            raise ValueError("window_s must be positive")
        self._window_s = window_s
        self._slow_range = slow_blink_range
        self._fast_range = fast_blink_range
        self._single_flash_max_duration_s = single_flash_max_duration_s
        self._history: deque[tuple[float, bool]] = deque()

    def add_sample(self, timestamp: float, is_on: bool) -> None:
        self._history.append((timestamp, is_on))
        cutoff = timestamp - self._window_s
        while len(self._history) > 1 and self._history[0][0] < cutoff:
            self._history.popleft()

    def reset(self) -> None:
        self._history.clear()

    def _pulses(self) -> tuple[list[_Pulse], float | None]:
        """Completed on/off pulses in history, plus the rise time of a
        still-open pulse (the LED is currently on and hasn't fallen yet
        within the retained window), if any."""
        pulses: list[_Pulse] = []
        rise_time: float | None = None
        for timestamp, is_on in self._history:
            if is_on and rise_time is None:
                rise_time = timestamp
            elif not is_on and rise_time is not None:
                pulses.append(_Pulse(rise_time=rise_time, fall_time=timestamp))
                rise_time = None
        return pulses, rise_time

    def classify(self) -> LEDState:
        if len(self._history) < 2:
            return LEDState.UNKNOWN

        span = self._history[-1][0] - self._history[0][0]
        if span < self._window_s / 2:
            return LEDState.UNKNOWN

        currently_on = self._history[-1][1]
        now = self._history[-1][0]
        pulses, open_rise_time = self._pulses()

        if not pulses:
            if open_rise_time is None:
                return LEDState.OFF
            # A live classifier can't yet know whether this still-open pulse
            # will fall back to OFF quickly enough to count as a
            # SINGLE_FLASH -- hold at UNKNOWN rather than guessing ON, up to
            # single_flash_max_duration_s, after which it's unambiguously a
            # steady ON.
            if now - open_rise_time < self._single_flash_max_duration_s:
                return LEDState.UNKNOWN
            return LEDState.ON

        if len(pulses) == 1 and open_rise_time is None:
            pulse = pulses[0]
            if pulse.duration < self._single_flash_max_duration_s:
                return LEDState.SINGLE_FLASH
            return LEDState.OFF

        if len(pulses) < 2:
            # One completed pulse plus a second one now in progress: not
            # enough periods yet for a frequency estimate.
            return LEDState.ON if currently_on else LEDState.OFF

        rise_times = [p.rise_time for p in pulses]
        periods = [rise_times[i + 1] - rise_times[i] for i in range(len(rise_times) - 1)]
        avg_period = sum(periods) / len(periods)
        if avg_period <= 0:
            return LEDState.UNKNOWN
        frequency = 1.0 / avg_period

        if self._slow_range[0] <= frequency < self._slow_range[1]:
            return LEDState.SLOW_BLINK
        if frequency >= self._fast_range[0]:
            return LEDState.FAST_BLINK
        return LEDState.UNKNOWN


def estimate_frequency_fft(
    samples: list[tuple[float, bool]],
    resample_hz: float = 60.0,
) -> float:
    """Frequency-domain alternative to :class:`BlinkClassifier`'s edge counting.

    Resamples the boolean on/off series onto a uniform grid (step-hold of
    the last known reading) and returns the frequency of the largest
    non-DC magnitude bin of its real FFT. Returns ``0.0`` when there is not
    enough history to resolve a frequency.
    """
    if len(samples) < 2:
        return 0.0

    ordered = sorted(samples, key=lambda s: s[0])
    t0, t1 = ordered[0][0], ordered[-1][0]
    duration = t1 - t0
    if duration <= 0:
        return 0.0

    n_samples = max(int(duration * resample_hz), 2)
    grid = np.linspace(t0, t1, n_samples)
    sample_times = np.array([s[0] for s in ordered])
    sample_values = np.array([1.0 if s[1] else 0.0 for s in ordered])
    indices = np.searchsorted(sample_times, grid, side="right") - 1
    indices = np.clip(indices, 0, len(sample_values) - 1)
    signal = sample_values[indices]

    if np.all(signal == signal[0]):
        return 0.0

    signal = signal - float(np.mean(signal))
    spectrum = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / resample_hz)
    magnitudes = np.abs(spectrum)
    magnitudes[0] = 0.0  # drop DC
    peak_index = int(np.argmax(magnitudes))
    return float(freqs[peak_index])
