"""Regression test against a real recorded factory-reset sequence.

Fixture: ``tests/fixtures/factory_reset_recording/`` -- captured from actual
DUT hardware (see CLAUDE.md's "Recording real hardware for offline tuning"),
then:

* **trimmed**: the original ~125s capture included an 18.5s pre-trigger lead-in
  and a genuine ~44s silent gap (the DUT asleep after the reset completed,
  before a second button press confirmed a clean wake) that add nothing to
  what this test exercises. Both were cut, splicing the silent gap down to
  ~5s (enough to still exercise a real, non-trivial OFF period) rather than
  removing it entirely -- down to 1986 frames / 72.25s from 3692 / 125.01s.
* **cropped + recompressed**: down to a 110x130 crop around the calibrated
  ROI, saved as JPEG q=95 -- verified beforehand against the uncompressed
  capture to produce **zero** ON/OFF classification differences across all
  1986 frames (checked at q=95/85/70/50/30; q=95 was the first with zero
  mismatches). This took the fixture from what would have been 600MB+ of
  full-frame lossless PNGs down to ~16MB.

Both edits used the splice/crop machinery directly against the real capture,
not synthetic data -- this is why the pulse count and expected-sequence
comments below cite specific measured values rather than nominal firmware
constants (contrast with ``test_real_dut_sequences.py``, which mocks the
documented timing exactly).
"""

from __future__ import annotations

from led_observer.calibration import CalibrationProfile
from led_observer.models import LEDState, ObservedSegment, Pulse, PulsePattern, SequenceStep
from led_observer.playback import load_recording, replay_recording
from led_observer.sequence import verify_sequence

FIXTURE_DIR = "tests/fixtures/factory_reset_recording"

# Firmware fires the multi-bond confirmation as ON 100ms / OFF 200ms, twice
# (see the factory-reset sequence doc). Real captured durations were
# ~132ms/100ms with a ~332ms gap -- tolerance widened slightly versus the
# synthetic-data tests (test_real_dut_sequences.py) to account for real
# camera sampling/rolling-shutter timing, not just idealized durations.
MULTI_BOND_CONFIRM = PulsePattern(
    pulses=[(0.1, 0.2), (0.1, 0.2)],
    tolerance_pct=0.35,
    settle_time_s=2.9,  # a bit under blink_window_s=3.0 -- see PulsePattern docs
)

# Traced from the real capture (button press at original t=18.506s):
# wake flash, ~4.86s OFF, 5 slow-blink pulses (~1.13s period), the 2-pulse
# confirmation burst at t=10.0s, ~19.6s OFF, then fast blink from t=30.06s
# for 14.0s (about-to-reset warning + flash erase), then sleep -- confirmed
# by a second button press (clean wake check) after the (trimmed) gap.
EXPECTED_SEQUENCE = [
    SequenceStep(LEDState.OFF, duration_s=None),
    SequenceStep(LEDState.SINGLE_FLASH, duration_s=None),  # wake flash (lingers in the window)
    SequenceStep(LEDState.OFF, duration_s=None),
    SequenceStep(LEDState.SINGLE_FLASH, duration_s=None),  # slow-blink ramp-up
    SequenceStep(LEDState.ON, duration_s=None),  # slow-blink ramp-up
    SequenceStep(LEDState.SLOW_BLINK, duration_s=None),
    SequenceStep(pulse_pattern=MULTI_BOND_CONFIRM),
    SequenceStep(LEDState.OFF, duration_s=None),  # multi-bond advertising window
    SequenceStep(LEDState.SINGLE_FLASH, duration_s=None),  # fast-blink ramp-up
    SequenceStep(LEDState.ON, duration_s=None),  # fast-blink ramp-up
    SequenceStep(LEDState.FAST_BLINK, duration_s=None),  # about-to-reset + flash erase
    SequenceStep(LEDState.SINGLE_FLASH, duration_s=None),  # fast-blink tail ramp-down
    SequenceStep(LEDState.OFF, duration_s=None),  # sleep (trimmed from ~44s to ~5s)
    SequenceStep(LEDState.SINGLE_FLASH, duration_s=None),  # 2nd button press: clean-wake check
    SequenceStep(LEDState.OFF, duration_s=None),
]


def _replay_fixture() -> tuple[list[ObservedSegment], list[Pulse]]:
    recording = load_recording(FIXTURE_DIR)
    calibration = CalibrationProfile.load(f"{FIXTURE_DIR}/calibration.json")
    return replay_recording(recording, calibration, blink_window_s=3.0)


def test_recorded_factory_reset_sequence_verifies_end_to_end() -> None:
    segments, pulses = _replay_fixture()
    report = verify_sequence(EXPECTED_SEQUENCE, segments, pulses=pulses)
    assert report.matched is True, report.divergence_point


def test_recorded_factory_reset_sequence_has_expected_pulse_count() -> None:
    # 1 wake flash + 5 slow-blink pulses + 2 confirmation pulses +
    # 126 fast-blink pulses + 1 second wake flash (clean-wake check) = 135.
    _, pulses = _replay_fixture()
    assert len(pulses) == 135
