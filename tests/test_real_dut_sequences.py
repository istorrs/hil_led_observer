"""End-to-end verification of the model against two real, documented DUT LED
sequences (traced from actual firmware source in `bapi_app.c`), replayed as
synthetic (timestamp, is_on) edge streams through the real classification +
recording + verify_sequence pipeline -- no threads, no camera, no real-time
sleep, just the actual production logic fed mocked timing data.

Both scenarios share the same "factory reset" hold up through the multi-bond
entry confirmation blink (a fixed 2-pulse burst): a burst too short for
BlinkClassifier to average a stable period from, which is exactly the
motivating case for PulsePattern/PulseRecorder (see models.py and
sequence.py). This is deliberately tested against the *actual* transient
misclassifications the classifier produces for that burst (traced via
scratch runs against the real code, not hand-guessed), not an idealized
version of them -- if BlinkClassifier's internals change and the transient
labels change with it, these tests should fail and need re-tracing, which is
the point.
"""

from __future__ import annotations

from led_observer.detector import BlinkClassifier
from led_observer.models import LEDState, ObservedSegment, Pulse, PulsePattern, SequenceStep
from led_observer.sequence import PulseRecorder, TimelineRecorder, verify_sequence

# Firmware fires the multi-bond confirmation as ON 100ms / OFF 200ms, twice,
# ending OFF (see BAPI_LED_FAST_BLINK_ON_TIME / _OFF_TIME in bapi_app.h).
MULTI_BOND_CONFIRM = PulsePattern(pulses=[(0.1, 0.2), (0.1, 0.2)], tolerance_pct=0.3)


def _replay(
    phases: list[tuple[float, bool]], window_s: float, sample_hz: float = 200.0
) -> tuple[list[ObservedSegment], list[Pulse], float]:
    """Feeds a scripted (duration_s, is_on) phase list through the real
    BlinkClassifier + TimelineRecorder + PulseRecorder, at a fixed sample
    rate far above any real camera's -- this tests the classification and
    verification *logic*, not camera-side sampling limits (covered
    separately, empirically, against real hardware)."""
    blink = BlinkClassifier(window_s=window_s)
    recorder = TimelineRecorder()
    pulse_recorder = PulseRecorder()

    dt = 1.0 / sample_hz
    t = 0.0
    for duration, is_on in phases:
        end = t + duration
        while t < end - 1e-9:
            blink.add_sample(t, is_on)
            pulse_recorder.add(t, is_on)
            state = blink.classify()
            if state is not LEDState.UNKNOWN:
                recorder.add(state, t)
            t += dt

    segments = recorder.finalize(t)
    return segments, pulse_recorder.pulses, t


def test_factory_reset_sequence_verifies_end_to_end() -> None:
    """Traced sequence (continuous button hold through to factory reset):

    0 -> 0.125s:    wake flash (ON then OFF, no repeat)
    0.125 -> 5s:    OFF
    5 -> 10s:       slow blink, 125ms ON / 1s OFF, repeating (~4 pulses)
    ~10 -> 10.6s:   multi-bond confirmation: ON100/OFF200 x2, ending OFF
    10.6 -> 30s:    OFF (multi-bond advertising window)
    30s onward:     fast blink, 31.25ms ON / 31.25ms OFF, continuous
                    (about-to-reset warning, then flash erase -- visually
                    one continuous fast blink of indeterminate length)
    """
    phases = [
        (0.125, True),
        (4.875, False),
        *([(0.125, True), (1.0, False)] * 4),  # ~4 slow-blink pulses over 5s
        (0.5, False),  # pad phase 3 to exactly 5.0s
        (0.1, True),
        (0.2, False),
        (0.1, True),
        (0.2, False),  # multi-bond confirmation burst
        (19.4, False),  # multi-bond advertising window
    ]
    fast_blink_cycles = int(12.0 / 0.0625)  # mock stand-in for "5s then tens of seconds"
    phases += [(0.03125, True), (0.03125, False)] * fast_blink_cycles

    segments, pulses, _ = _replay(phases, window_s=3.0)

    # A live classifier's segment boundaries lag the real transitions by up
    # to blink_window_s, and can't confirm a new cadence until it has seen
    # two full pulses of it (see CLAUDE.md's "Windowed cadence
    # classification is causal, not retroactive") -- so the wake flash and
    # each blink phase's first cycle show up as their own transient
    # SINGLE_FLASH/ON steps before settling. This is real, traced
    # classifier output, not a hypothetical.
    expected = [
        SequenceStep(LEDState.SINGLE_FLASH, duration_s=None),  # wake flash (lingers in the window)
        SequenceStep(LEDState.OFF, duration_s=None),
        SequenceStep(LEDState.SINGLE_FLASH, duration_s=None),  # slow-blink ramp-up
        SequenceStep(LEDState.ON, duration_s=None),  # slow-blink ramp-up
        SequenceStep(LEDState.SLOW_BLINK, duration_s=None),
        SequenceStep(
            pulse_pattern=PulsePattern(
                pulses=MULTI_BOND_CONFIRM.pulses,
                tolerance_pct=MULTI_BOND_CONFIRM.tolerance_pct,
                settle_time_s=2.9,  # a bit under blink_window_s=3.0 -- see PulsePattern docs
            )
        ),
        SequenceStep(LEDState.OFF, duration_s=None),  # multi-bond advertising window
        SequenceStep(LEDState.SINGLE_FLASH, duration_s=None),  # fast-blink ramp-up
        SequenceStep(LEDState.ON, duration_s=None),  # fast-blink ramp-up
        SequenceStep(LEDState.FAST_BLINK, duration_s=None),  # about-to-reset + flash erase
    ]

    report = verify_sequence(expected, segments, pulses=pulses)
    assert report.matched is True, report.divergence_point


def test_factory_reset_sequence_catches_a_missing_confirmation_burst() -> None:
    """Sanity-check the negative case: if the multi-bond confirmation burst
    never happened (e.g. a firmware regression skipped it), the pulse
    pattern step must fail to match rather than silently passing."""
    phases = [
        (0.125, True),
        (4.875, False),
        *([(0.125, True), (1.0, False)] * 4),
        (0.5, False),
        (20.0, False),  # no confirmation burst here
    ]
    segments, pulses, _ = _replay(phases, window_s=3.0)

    expected = [
        SequenceStep(LEDState.SINGLE_FLASH, duration_s=None),
        SequenceStep(LEDState.OFF, duration_s=None),
        SequenceStep(LEDState.SINGLE_FLASH, duration_s=None),
        SequenceStep(LEDState.ON, duration_s=None),
        SequenceStep(LEDState.SLOW_BLINK, duration_s=None),
        SequenceStep(pulse_pattern=MULTI_BOND_CONFIRM),
    ]
    report = verify_sequence(expected, segments, pulses=pulses)
    assert report.matched is False
    assert report.divergence_point is not None
    assert report.divergence_point.index == 5


def test_double_press_commissioning_sequence_verifies_end_to_end() -> None:
    """Traced sequence (hold through multi-bond entry, release, idle ~8s,
    then double-press into commissioning mode):

    0 -> ~8s:      idle OFF
    ~8s:           press #1: single ~125ms flash
    +~1s:          race window (firmware's own docs call this "a genuine
                   race", not just unknown to us) -- LED stays off either way
    from there:    steady commissioning blink, 200ms ON / 1800ms OFF (0.5Hz),
                   indefinitely

    0.5Hz sits exactly on SLOW_BLINK's inclusive lower boundary -- this is
    also a regression check that blink_window_s must be large relative to
    the period for a *stable* (non-oscillating) reading here: window_s=3.0
    (only 1.5x the 2.0s period) empirically oscillates between SLOW_BLINK
    and ON/SINGLE_FLASH for this exact signal (traced against the real
    code); window_s=5.0 (2.5x the period) is stable. See CLAUDE.md.
    """
    phases = [
        (8.0, False),
        (0.125, True),
        (0.975, False),  # race window, padded to ~1s total idle-after-flash
    ]
    phases += [(0.2, True), (1.8, False)] * 6  # steady commissioning blink

    segments, _, _ = _replay(phases, window_s=5.0)

    expected = [
        SequenceStep(LEDState.OFF, duration_s=None),
        SequenceStep(LEDState.SINGLE_FLASH, duration_s=None),  # press #1
        SequenceStep(LEDState.ON, duration_s=None),  # commissioning-blink ramp-up
        SequenceStep(LEDState.SLOW_BLINK, duration_s=None),  # steady 0.5Hz commissioning blink
    ]
    report = verify_sequence(expected, segments)
    assert report.matched is True, report.divergence_point


def test_double_press_commissioning_blink_is_unstable_below_2_5x_period_window() -> None:
    """Regression check for the window-sizing guidance above: a window only
    1.5x the signal's period is not large enough for a stable reading."""
    phases = [(8.0, False), (0.125, True), (0.975, False)]
    phases += [(0.2, True), (1.8, False)] * 6

    segments, _, _ = _replay(phases, window_s=3.0)
    slow_blink_segments = [s for s in segments if s.state is LEDState.SLOW_BLINK]
    assert len(slow_blink_segments) > 1, (
        "expected the classic oscillation symptom (multiple separate SLOW_BLINK "
        "segments instead of one stable one) at window_s=3.0 for a 2.0s-period "
        "signal -- if this now produces one stable segment, the window-sizing "
        "guidance in CLAUDE.md may be out of date"
    )
