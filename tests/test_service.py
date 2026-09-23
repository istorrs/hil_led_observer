from __future__ import annotations

import time

import pytest

from led_observer.calibration import CalibrationProfile
from led_observer.models import LEDState, SequenceStep
from led_observer.service import LEDObserverService, ObservationInProgressError
from tests.mocks import CrashingFrameSource, FailingFrameSource, SyntheticFrameSource

POLL_INTERVAL_S = 1.0 / 120.0
BLINK_WINDOW_S = 0.5


def _make_service(
    calibration_profile: CalibrationProfile, schedule: list[tuple[float, bool]]
) -> LEDObserverService:
    source = SyntheticFrameSource(schedule)
    return LEDObserverService(
        calibration_profile,
        source,
        blink_window_s=BLINK_WINDOW_S,
        poll_interval_s=POLL_INTERVAL_S,
    )


def test_get_current_state_tracks_steady_off(calibration_profile: CalibrationProfile) -> None:
    service = _make_service(calibration_profile, [(2.0, False)])
    try:
        time.sleep(0.6)
        assert service.get_current_state() is LEDState.OFF
    finally:
        service.close()


def test_get_current_state_tracks_steady_on(calibration_profile: CalibrationProfile) -> None:
    # A live classifier can't confirm steady ON until a pulse has outlasted
    # single_flash_max_duration_s (default 1.0s) -- otherwise it might still
    # resolve into a SINGLE_FLASH -- so this needs a window comfortably
    # larger than that threshold, and enough real time to observe it.
    source = SyntheticFrameSource([(3.0, True)])
    service = LEDObserverService(
        calibration_profile, source, blink_window_s=1.5, poll_interval_s=POLL_INTERVAL_S
    )
    try:
        time.sleep(1.8)
        assert service.get_current_state() is LEDState.ON
    finally:
        service.close()


def test_start_observation_twice_raises(calibration_profile: CalibrationProfile) -> None:
    service = _make_service(calibration_profile, [(2.0, False)])
    try:
        service.start_observation()
        with pytest.raises(ObservationInProgressError):
            service.start_observation()
    finally:
        service.close()


def test_stop_without_start_returns_empty_result(
    calibration_profile: CalibrationProfile,
) -> None:
    service = _make_service(calibration_profile, [(2.0, False)])
    try:
        result = service.stop_observation()
        assert result.segments == []
        assert result.duration == 0.0
        assert result.timed_out is False
    finally:
        service.close()


def test_start_stop_records_single_flash_segment(
    calibration_profile: CalibrationProfile,
) -> None:
    service = _make_service(calibration_profile, [(0.3, False), (0.15, True), (0.3, False)])
    try:
        service.start_observation()
        time.sleep(0.9)
        result = service.stop_observation()
    finally:
        service.close()

    assert any(segment.state is LEDState.SINGLE_FLASH for segment in result.segments)
    assert result.dropped_frames == 0


def test_max_timeout_auto_stops_observation(calibration_profile: CalibrationProfile) -> None:
    service = _make_service(calibration_profile, [(5.0, True)])
    try:
        service.start_observation(max_timeout_s=0.3)
        time.sleep(0.6)
        result = service.stop_observation()
    finally:
        service.close()

    assert result.timed_out is True
    assert result.duration == pytest.approx(0.3, abs=0.2)


def test_expected_sequence_auto_verifies_on_stop(
    calibration_profile: CalibrationProfile,
) -> None:
    service = _make_service(calibration_profile, [(0.3, False), (0.15, True), (0.3, False)])
    expected = [
        SequenceStep(LEDState.OFF, duration_s=None),
        SequenceStep(LEDState.SINGLE_FLASH, duration_s=None),
        SequenceStep(LEDState.OFF, duration_s=None),
    ]
    try:
        service.start_observation(expected_sequence=expected)
        time.sleep(1.1)
        result = service.stop_observation()
    finally:
        service.close()

    assert result.verification is not None
    assert result.verification.matched is True


def test_dropped_frames_counted_and_state_stays_unknown(
    calibration_profile: CalibrationProfile,
) -> None:
    source = FailingFrameSource()
    service = LEDObserverService(
        calibration_profile, source, blink_window_s=BLINK_WINDOW_S, poll_interval_s=0.01
    )
    try:
        service.start_observation()
        time.sleep(0.2)
        result = service.stop_observation()
        assert result.dropped_frames > 0
        assert service.get_current_state() is LEDState.UNKNOWN
    finally:
        service.close()
    assert source.released is True


def test_crashing_frames_do_not_kill_worker_thread(
    calibration_profile: CalibrationProfile,
) -> None:
    source = CrashingFrameSource()
    service = LEDObserverService(
        calibration_profile, source, blink_window_s=BLINK_WINDOW_S, poll_interval_s=0.01
    )
    try:
        time.sleep(0.2)
        assert service._thread.is_alive()
        service.start_observation()
        time.sleep(0.1)
        result = service.stop_observation()
        assert result.dropped_frames > 0
    finally:
        service.close()
    assert service._thread.is_alive() is False


def test_verify_sequence_static_method_matches_module_function(
    calibration_profile: CalibrationProfile,
) -> None:
    from led_observer.models import ObservedSegment

    expected = [SequenceStep(LEDState.OFF, duration_s=None)]
    actual = [ObservedSegment(LEDState.OFF, 0.0, 1.0)]
    report = LEDObserverService.verify_sequence(expected, actual)
    assert report.matched is True


def test_context_manager_closes_service(calibration_profile: CalibrationProfile) -> None:
    source = SyntheticFrameSource([(1.0, False)])
    with LEDObserverService(
        calibration_profile, source, blink_window_s=BLINK_WINDOW_S, poll_interval_s=0.01
    ) as service:
        time.sleep(0.1)
        assert service.get_current_state() in (LEDState.OFF, LEDState.UNKNOWN)
    assert source.released is True
