from __future__ import annotations

import pytest

from led_observer.models import LEDState, ObservedSegment, Pulse, PulsePattern, SequenceStep
from led_observer.sequence import PulseRecorder, TimelineRecorder, verify_sequence


def test_timeline_recorder_collapses_repeated_states() -> None:
    recorder = TimelineRecorder()
    recorder.add(LEDState.OFF, 0.0)
    recorder.add(LEDState.OFF, 0.5)
    recorder.add(LEDState.ON, 1.0)
    recorder.add(LEDState.ON, 1.4)
    segments = recorder.finalize(2.0)

    assert segments == [
        ObservedSegment(LEDState.OFF, 0.0, 1.0),
        ObservedSegment(LEDState.ON, 1.0, 2.0),
    ]


def test_timeline_recorder_transitions_only_recorded_on_change() -> None:
    recorder = TimelineRecorder()
    recorder.add(LEDState.OFF, 0.0)
    recorder.add(LEDState.OFF, 0.5)
    recorder.add(LEDState.SLOW_BLINK, 1.0)
    assert [t.state for t in recorder.transitions] == [LEDState.OFF, LEDState.SLOW_BLINK]


def test_timeline_recorder_finalize_ignores_zero_length_open_segment() -> None:
    recorder = TimelineRecorder()
    recorder.add(LEDState.OFF, 5.0)
    segments = recorder.finalize(5.0)
    assert segments == []


def test_timeline_recorder_reset() -> None:
    recorder = TimelineRecorder()
    recorder.add(LEDState.ON, 0.0)
    recorder.finalize(1.0)
    recorder.reset()
    assert recorder.segments == []
    assert recorder.transitions == []


_EXPECTED_SEQUENCE = [
    SequenceStep(LEDState.OFF, duration_s=None),
    SequenceStep(LEDState.SINGLE_FLASH, duration_s=None),
    SequenceStep(LEDState.OFF, duration_s=5.0),
    SequenceStep(LEDState.SLOW_BLINK, duration_s=5.0),
    SequenceStep(LEDState.FAST_BLINK, duration_s=3.0),
    SequenceStep(LEDState.OFF, duration_s=15.0),
    SequenceStep(LEDState.FAST_BLINK, duration_s=60.0),
]


def _matching_actual_sequence() -> list[ObservedSegment]:
    t = 0.0
    segments = []
    for state, duration in [
        (LEDState.OFF, 2.0),
        (LEDState.SINGLE_FLASH, 0.4),
        (LEDState.OFF, 5.1),
        (LEDState.SLOW_BLINK, 4.8),
        (LEDState.FAST_BLINK, 3.2),
        (LEDState.OFF, 14.5),
        (LEDState.FAST_BLINK, 60.0),
    ]:
        segments.append(ObservedSegment(state, t, t + duration))
        t += duration
    return segments


def test_verify_sequence_matches_within_tolerance() -> None:
    report = verify_sequence(_EXPECTED_SEQUENCE, _matching_actual_sequence(), tolerance_pct=0.20)
    assert report.matched is True
    assert report.divergence_point is None


def test_verify_sequence_reports_state_mismatch() -> None:
    actual = _matching_actual_sequence()
    actual[3] = ObservedSegment(LEDState.OFF, actual[3].start_time, actual[3].end_time)

    report = verify_sequence(_EXPECTED_SEQUENCE, actual)

    assert report.matched is False
    assert report.divergence_point is not None
    assert report.divergence_point.index == 3
    assert "SLOW_BLINK" in report.divergence_point.reason
    assert "OFF" in report.divergence_point.reason


def test_verify_sequence_reports_duration_out_of_tolerance() -> None:
    actual = _matching_actual_sequence()
    start = actual[4].start_time
    actual[4] = ObservedSegment(LEDState.FAST_BLINK, start, start + 1.0)  # expected ~3.0s ±20%

    report = verify_sequence(_EXPECTED_SEQUENCE, actual)

    assert report.matched is False
    assert report.divergence_point is not None
    assert report.divergence_point.index == 4
    assert "duration" in report.divergence_point.reason


def test_verify_sequence_reports_premature_end_of_observation() -> None:
    actual = _matching_actual_sequence()[:3]

    report = verify_sequence(_EXPECTED_SEQUENCE, actual)

    assert report.matched is False
    assert report.divergence_point is not None
    assert report.divergence_point.index == 3
    assert "ended" in report.divergence_point.reason


def test_verify_sequence_ignores_trailing_extra_segments() -> None:
    actual = _matching_actual_sequence()
    actual.append(ObservedSegment(LEDState.OFF, 200.0, 210.0))
    report = verify_sequence(_EXPECTED_SEQUENCE, actual)
    assert report.matched is True


def test_verify_sequence_any_duration_step_ignores_actual_duration() -> None:
    expected = [SequenceStep(LEDState.OFF, duration_s=None)]
    actual = [ObservedSegment(LEDState.OFF, 0.0, 999.0)]
    assert verify_sequence(expected, actual).matched is True


def test_verify_sequence_per_step_tolerance_override() -> None:
    expected = [SequenceStep(LEDState.ON, duration_s=10.0, tolerance_pct=0.5)]
    actual = [ObservedSegment(LEDState.ON, 0.0, 14.0)]  # 40% over, allowed by 50% override
    assert verify_sequence(expected, actual, tolerance_pct=0.20).matched is True


def test_verify_sequence_empty_expected_always_matches() -> None:
    report = verify_sequence([], _matching_actual_sequence())
    assert report.matched is True


def test_sequence_step_requires_exactly_one_of_state_or_pulse_pattern() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        SequenceStep()
    with pytest.raises(ValueError, match="exactly one"):
        SequenceStep(state=LEDState.OFF, pulse_pattern=PulsePattern(pulses=[(0.1, 0.2)]))


def test_pulse_pattern_rejects_empty_pulses() -> None:
    with pytest.raises(ValueError, match="at least one pulse"):
        PulsePattern(pulses=[])


class TestPulseRecorder:
    def test_records_completed_pulses_only(self) -> None:
        recorder = PulseRecorder()
        for t, is_on in [(0.0, False), (1.0, True), (1.1, False), (2.0, True)]:
            recorder.add(t, is_on)
        # the final rise at t=2.0 never falls -- not a completed pulse.
        assert recorder.pulses == [Pulse(rise_time=1.0, fall_time=1.1)]

    def test_ignores_repeated_same_state_samples(self) -> None:
        recorder = PulseRecorder()
        for t, is_on in [(0.0, False), (0.5, False), (1.0, True), (1.2, True), (1.5, False)]:
            recorder.add(t, is_on)
        assert recorder.pulses == [Pulse(rise_time=1.0, fall_time=1.5)]

    def test_reset_clears_history(self) -> None:
        recorder = PulseRecorder()
        recorder.add(0.0, True)
        recorder.add(0.1, False)
        assert recorder.pulses
        recorder.reset()
        assert recorder.pulses == []


_MULTI_BOND_PATTERN = PulsePattern(pulses=[(0.1, 0.2), (0.1, 0.2)], tolerance_pct=0.3)


def _multi_bond_pulses(start: float = 10.0) -> list[Pulse]:
    return [
        Pulse(rise_time=start, fall_time=start + 0.1),
        Pulse(rise_time=start + 0.3, fall_time=start + 0.4),
    ]


def test_verify_sequence_matches_pulse_pattern_step() -> None:
    expected = [SequenceStep(pulse_pattern=_MULTI_BOND_PATTERN)]
    report = verify_sequence(expected, actual_sequence=[], pulses=_multi_bond_pulses())
    assert report.matched is True


def test_verify_sequence_pulse_pattern_on_duration_out_of_tolerance() -> None:
    expected = [SequenceStep(pulse_pattern=_MULTI_BOND_PATTERN)]
    pulses = [
        Pulse(rise_time=10.0, fall_time=10.5),  # 0.5s on, expected 0.1s ±30%
        Pulse(rise_time=10.7, fall_time=10.8),
    ]
    report = verify_sequence(expected, actual_sequence=[], pulses=pulses)
    assert report.matched is False


def test_verify_sequence_pulse_pattern_out_of_tolerance() -> None:
    expected = [SequenceStep(pulse_pattern=_MULTI_BOND_PATTERN)]
    pulses = [
        Pulse(rise_time=10.0, fall_time=10.1),
        Pulse(rise_time=10.9, fall_time=11.0),  # gap ~0.8s, way outside 0.2s ±30%
    ]
    report = verify_sequence(expected, actual_sequence=[], pulses=pulses)
    assert report.matched is False
    assert report.divergence_point is not None
    assert report.divergence_point.index == 0


def test_verify_sequence_pulse_pattern_missing_raises_without_pulses_arg() -> None:
    expected = [SequenceStep(pulse_pattern=_MULTI_BOND_PATTERN)]
    with pytest.raises(ValueError, match="pulses"):
        verify_sequence(expected, actual_sequence=[])


def test_verify_sequence_pulse_pattern_skips_transient_actual_segments() -> None:
    """A burst too short for BlinkClassifier to average a stable frequency
    from typically shows up as a few noisy, transient segments (this
    mirrors real classifier output, not just a hypothetical), the last of
    which lingers past the burst's physical end for roughly blink_window_s.
    The pulse_pattern step must consume all of them -- via settle_time_s for
    the lingering one -- and let the *next* expected state step match
    whatever segment comes after, without any of them being named
    individually in the schema.
    """
    pattern_with_settle = PulsePattern(
        pulses=_MULTI_BOND_PATTERN.pulses,
        tolerance_pct=_MULTI_BOND_PATTERN.tolerance_pct,
        settle_time_s=1.0,
    )
    expected = [
        SequenceStep(LEDState.OFF, duration_s=None),
        SequenceStep(pulse_pattern=pattern_with_settle),
        SequenceStep(LEDState.OFF, duration_s=None),
    ]
    pulses = _multi_bond_pulses(start=5.0)
    actual = [
        ObservedSegment(LEDState.OFF, 0.0, 5.0),
        ObservedSegment(LEDState.SINGLE_FLASH, 5.0, 5.1),  # transient, from pulse 1 falling
        ObservedSegment(LEDState.ON, 5.3, 5.4),  # transient, from pulse 2 rising
        ObservedSegment(LEDState.FAST_BLINK, 5.4, 6.5),  # transient, lingers in the window
        ObservedSegment(LEDState.OFF, 6.5, 30.0),  # the real post-burst OFF period
    ]
    report = verify_sequence(expected, actual, pulses=pulses)
    assert report.matched is True


def test_verify_sequence_mixed_schema_still_detects_a_real_mismatch_after_burst() -> None:
    expected = [
        SequenceStep(pulse_pattern=_MULTI_BOND_PATTERN),
        SequenceStep(LEDState.SLOW_BLINK, duration_s=None),
    ]
    pulses = _multi_bond_pulses(start=5.0)
    actual = [
        ObservedSegment(LEDState.SINGLE_FLASH, 5.0, 5.1),
        ObservedSegment(LEDState.FAST_BLINK, 5.4, 6.5),
        ObservedSegment(LEDState.OFF, 6.5, 30.0),  # not SLOW_BLINK as expected
    ]
    report = verify_sequence(expected, actual, pulses=pulses)
    assert report.matched is False
    assert report.divergence_point is not None
    assert report.divergence_point.index == 1
    assert "SLOW_BLINK" in report.divergence_point.reason
