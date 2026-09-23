from __future__ import annotations

import pytest

from led_observer.calibration import CalibrationProfile, FrameMetrics
from led_observer.detector import BlinkClassifier, InstantClassifier, estimate_frequency_fft
from led_observer.models import LEDState
from tests.mocks import make_frame


def test_instant_classifier_reads_on_and_off(calibration_profile: CalibrationProfile) -> None:
    classifier = InstantClassifier(calibration_profile)
    assert classifier.classify_frame(make_frame(True)) is True
    assert classifier.classify_frame(make_frame(False)) is False


def test_instant_classifier_holds_last_value_inside_hysteresis_band(
    calibration_profile: CalibrationProfile,
) -> None:
    classifier = InstantClassifier(calibration_profile)
    assert classifier.classify_frame(make_frame(True)) is True

    midpoint = (
        calibration_profile.luminosity_on_threshold + calibration_profile.luminosity_off_threshold
    ) / 2.0
    ambiguous_metrics = FrameMetrics(luminosity=midpoint, color_ratio=1.0)
    # Inside the hysteresis band: neither threshold is crossed, so the
    # previous reading (ON) must be retained rather than flipping.
    assert classifier.classify_metrics(ambiguous_metrics) is True

    classifier.reset()
    assert classifier.classify_metrics(ambiguous_metrics) is False


def test_blink_classifier_requires_minimum_history() -> None:
    classifier = BlinkClassifier(window_s=2.0)
    assert classifier.classify() is LEDState.UNKNOWN
    classifier.add_sample(0.0, True)
    assert classifier.classify() is LEDState.UNKNOWN


def test_blink_classifier_reports_steady_off() -> None:
    off_classifier = BlinkClassifier(window_s=1.0)
    for t in (0.0, 0.3, 0.6, 0.9, 1.2):
        off_classifier.add_sample(t, False)
    assert off_classifier.classify() is LEDState.OFF


def test_blink_classifier_reports_steady_on_once_past_single_flash_window() -> None:
    # A pulse still shorter than single_flash_max_duration_s is ambiguous
    # (it might yet resolve into a SINGLE_FLASH), so window_s here is chosen
    # comfortably larger than the 1.0s default threshold.
    on_classifier = BlinkClassifier(window_s=2.5)
    for t in (0.0, 0.5, 1.0, 1.5, 2.0, 2.2):
        on_classifier.add_sample(t, True)
    assert on_classifier.classify() is LEDState.ON


def test_blink_classifier_holds_unknown_while_pulse_could_still_be_a_flash() -> None:
    classifier = BlinkClassifier(window_s=2.5)
    for t in (0.0, 0.3, 0.6):
        classifier.add_sample(t, True)
    assert classifier.classify() is LEDState.UNKNOWN


def _feed_blink(
    classifier: BlinkClassifier, frequency_hz: float, total_s: float, fps: float = 60.0
) -> None:
    period = 1.0 / frequency_hz
    dt = 1.0 / fps
    t = 0.0
    while t <= total_s:
        phase = (t % period) / period
        classifier.add_sample(t, phase < 0.5)
        t += dt


def test_blink_classifier_detects_slow_blink() -> None:
    classifier = BlinkClassifier(window_s=3.0)
    _feed_blink(classifier, frequency_hz=1.0, total_s=4.0)
    assert classifier.classify() is LEDState.SLOW_BLINK


def test_blink_classifier_detects_fast_blink() -> None:
    classifier = BlinkClassifier(window_s=3.0)
    _feed_blink(classifier, frequency_hz=5.0, total_s=4.0)
    assert classifier.classify() is LEDState.FAST_BLINK


def test_blink_classifier_detects_single_flash() -> None:
    classifier = BlinkClassifier(window_s=3.0)
    for t, is_on in [
        (0.0, False),
        (0.5, False),
        (1.0, True),
        (1.3, True),
        (1.6, False),
        (2.0, False),
    ]:
        classifier.add_sample(t, is_on)
    assert classifier.classify() is LEDState.SINGLE_FLASH


def test_blink_classifier_rejects_long_pulse_as_single_flash() -> None:
    classifier = BlinkClassifier(window_s=3.0)
    for t, is_on in [
        (0.0, False),
        (0.5, True),
        (1.0, True),
        (1.5, True),
        (2.0, False),
        (2.2, False),
    ]:
        classifier.add_sample(t, is_on)
    assert classifier.classify() is LEDState.OFF


def test_blink_classifier_reset_clears_history() -> None:
    classifier = BlinkClassifier(window_s=1.0)
    _feed_blink(classifier, frequency_hz=5.0, total_s=2.0)
    assert classifier.classify() is not LEDState.UNKNOWN
    classifier.reset()
    assert classifier.classify() is LEDState.UNKNOWN


def test_blink_classifier_rejects_non_positive_window() -> None:
    with pytest.raises(ValueError, match="window_s"):
        BlinkClassifier(window_s=0)


def test_blink_classifier_transitional_state_before_second_period_completes() -> None:
    # One completed pulse, and a second pulse now in progress: not enough
    # periods yet for a frequency estimate, so it reports the instant state.
    classifier = BlinkClassifier(window_s=3.0)
    for t, is_on in [
        (0.0, False),
        (1.0, True),
        (2.0, False),
        (3.0, True),
    ]:
        classifier.add_sample(t, is_on)
    assert classifier.classify() is LEDState.ON


def test_blink_classifier_returns_unknown_below_slow_blink_range() -> None:
    # Two full periods at 0.2 Hz (period 5s) -- slower than SLOW_BLINK's
    # 0.5 Hz floor, so the cadence isn't a recognized one.
    classifier = BlinkClassifier(window_s=12.0)
    for t, is_on in [
        (0.0, True),
        (1.0, False),
        (5.0, True),
        (6.0, False),
        (10.0, True),
        (11.0, False),
    ]:
        classifier.add_sample(t, is_on)
    assert classifier.classify() is LEDState.UNKNOWN


def test_blink_classifier_returns_unknown_for_non_advancing_rise_times() -> None:
    # Malformed/out-of-order timestamps (e.g. a clock hiccup) must not raise;
    # a non-positive average period is reported as UNKNOWN rather than a
    # bogus frequency.
    classifier = BlinkClassifier(window_s=3.0)
    for t, is_on in [
        (2.0, True),
        (3.0, False),
        (2.0, True),
        (4.0, False),
    ]:
        classifier.add_sample(t, is_on)
    assert classifier.classify() is LEDState.UNKNOWN


def test_estimate_frequency_fft_returns_zero_for_zero_duration_span() -> None:
    assert estimate_frequency_fft([(1.0, True), (1.0, False)]) == 0.0


def test_estimate_frequency_fft_matches_time_domain_slow_blink() -> None:
    samples: list[tuple[float, bool]] = []
    period = 1.0 / 1.0
    dt = 1.0 / 60.0
    t = 0.0
    while t <= 4.0:
        phase = (t % period) / period
        samples.append((t, phase < 0.5))
        t += dt
    frequency = estimate_frequency_fft(samples)
    assert 0.7 <= frequency <= 1.3


def test_estimate_frequency_fft_returns_zero_for_flat_signal() -> None:
    samples = [(0.0, False), (1.0, False), (2.0, False)]
    assert estimate_frequency_fft(samples) == 0.0


def test_estimate_frequency_fft_returns_zero_for_insufficient_samples() -> None:
    assert estimate_frequency_fft([(0.0, True)]) == 0.0
