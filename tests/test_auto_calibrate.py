"""Hardware-free checks for the packaged calibration command."""

from pathlib import Path

import numpy as np
import pytest

from led_observer import auto_calibrate
from led_observer.calibration import CalibrationProfile, Frame


@pytest.mark.parametrize("output_dir", [".", "nested/output"])
def test_calibration_command_writes_outputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, output_dir: str
) -> None:
    class FakeCamera:
        released = False

        def release(self) -> None:
            self.released = True

    camera = FakeCamera()
    off = np.full((32, 32, 3), 12, dtype=np.uint8)
    on = off.copy()
    on[12:20, 12:20] = (60, 220, 60)
    frames: list[Frame] = [off, on] * 3

    monkeypatch.setattr(auto_calibrate, "OpenCVCameraSource", lambda _: camera)
    monkeypatch.setattr(auto_calibrate, "capture_burst", lambda _source, _duration: frames)
    monkeypatch.chdir(tmp_path)

    args = ["--duration", "0.01"]
    if output_dir != ".":
        args.extend(["--out", f"{output_dir}/calibration.json"])
        args.extend(["--debug-image", f"{output_dir}/roi_debug.png"])

    assert auto_calibrate.main(args) == 0

    profile_path = tmp_path / output_dir / "calibration.json"
    debug_path = tmp_path / output_dir / "roi_debug.png"
    profile = CalibrationProfile.load(profile_path)
    assert profile.roi.width > 0
    assert profile.roi.height > 0
    assert debug_path.is_file()
    assert camera.released
