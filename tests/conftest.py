from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from led_observer.calibration import ROI, CalibrationProfile, Frame, build_calibration_profile
from tests.mocks import make_frame

FRAME_SIZE = (32, 32)


@pytest.fixture
def roi() -> ROI:
    return ROI(x=0, y=0, width=FRAME_SIZE[1], height=FRAME_SIZE[0])


@pytest.fixture
def on_frames() -> list[Frame]:
    return [make_frame(True, size=FRAME_SIZE) for _ in range(5)]


@pytest.fixture
def off_frames() -> list[Frame]:
    return [make_frame(False, size=FRAME_SIZE) for _ in range(5)]


@pytest.fixture
def calibration_profile(
    roi: ROI, on_frames: list[Frame], off_frames: list[Frame]
) -> CalibrationProfile:
    return build_calibration_profile(roi, on_frames, off_frames)


@pytest.fixture
def tmp_profile_path(tmp_path: Path) -> Iterator[Path]:
    yield tmp_path / "calibration.json"
