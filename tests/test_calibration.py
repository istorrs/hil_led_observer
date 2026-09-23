from __future__ import annotations

from pathlib import Path

import pytest

from led_observer.calibration import (
    ROI,
    CalibrationError,
    CalibrationProfile,
    build_calibration_profile,
    compute_frame_metrics,
)
from tests.mocks import make_frame


def test_roi_rejects_non_positive_dimensions() -> None:
    with pytest.raises(CalibrationError):
        ROI(x=0, y=0, width=0, height=10)
    with pytest.raises(CalibrationError):
        ROI(x=0, y=0, width=10, height=-1)


def test_roi_rejects_negative_origin() -> None:
    with pytest.raises(CalibrationError):
        ROI(x=-1, y=0, width=10, height=10)


def test_roi_crop_out_of_bounds_raises(roi: ROI) -> None:
    tiny_frame = make_frame(True, size=(4, 4))
    with pytest.raises(CalibrationError):
        roi.crop(tiny_frame)


def test_compute_frame_metrics_on_brighter_than_off(roi: ROI) -> None:
    on_metrics = compute_frame_metrics(make_frame(True), roi)
    off_metrics = compute_frame_metrics(make_frame(False), roi)
    assert on_metrics.luminosity > off_metrics.luminosity


def test_build_calibration_profile_orders_thresholds(
    calibration_profile: CalibrationProfile,
) -> None:
    assert (
        calibration_profile.luminosity_off_threshold < calibration_profile.luminosity_on_threshold
    )
    assert calibration_profile.color_ratio_min > 0.0


def test_build_calibration_profile_rejects_inverted_samples(roi: ROI) -> None:
    dim_on = [make_frame(False, size=(32, 32)) for _ in range(3)]
    dim_off = [make_frame(True, size=(32, 32)) for _ in range(3)]
    with pytest.raises(CalibrationError):
        build_calibration_profile(roi, on_frames=dim_on, off_frames=dim_off)


def test_build_calibration_profile_rejects_empty_samples(roi: ROI) -> None:
    with pytest.raises(CalibrationError):
        build_calibration_profile(roi, on_frames=[], off_frames=[make_frame(False)])
    with pytest.raises(CalibrationError):
        build_calibration_profile(roi, on_frames=[make_frame(True)], off_frames=[])


def test_calibration_profile_round_trips_through_json(
    calibration_profile: CalibrationProfile, tmp_profile_path: Path
) -> None:
    calibration_profile.save(tmp_profile_path)
    loaded = CalibrationProfile.load(tmp_profile_path)
    assert loaded == calibration_profile


def test_classification_tolerates_pixel_noise(roi: ROI) -> None:
    profile = build_calibration_profile(
        roi,
        on_frames=[make_frame(True) for _ in range(5)],
        off_frames=[make_frame(False) for _ in range(5)],
    )
    noisy_on = make_frame(True, noise_std=3.0)
    noisy_off = make_frame(False, noise_std=3.0)
    on_metrics = compute_frame_metrics(noisy_on, profile)
    off_metrics = compute_frame_metrics(noisy_off, profile)
    assert on_metrics.luminosity >= profile.luminosity_on_threshold
    assert off_metrics.luminosity <= profile.luminosity_off_threshold
