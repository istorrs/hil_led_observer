from __future__ import annotations

import json
from pathlib import Path

import cv2
import pytest

from led_observer.calibration import CalibrationProfile
from led_observer.models import LEDState
from led_observer.playback import (
    RecordedFrame,
    RecordedFrameSource,
    load_recording,
    replay_recording,
)
from tests.mocks import make_frame


def _write_recording(directory: Path, schedule: list[tuple[float, bool]]) -> None:
    """Writes a recording directory (PNGs + manifest.json) matching what
    scripts/record_frames.py produces, from a (timestamp, is_on) schedule."""
    frames_dir = directory / "frames"
    frames_dir.mkdir(parents=True)

    manifest_frames = []
    for index, (timestamp, is_on) in enumerate(schedule):
        rel_path = f"frames/{index:06d}.png"
        cv2.imwrite(str(directory / rel_path), make_frame(is_on))
        manifest_frames.append({"index": index, "timestamp": timestamp, "path": rel_path})

    manifest = {
        "frame_count": len(manifest_frames),
        "duration_s": schedule[-1][0] if schedule else 0.0,
        "dropped_frames": 0,
        "frames": manifest_frames,
    }
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_load_recording_reads_manifest_and_frames(tmp_path: Path) -> None:
    schedule = [(0.0, False), (0.5, True), (1.0, False)]
    _write_recording(tmp_path, schedule)

    recording = load_recording(tmp_path)

    assert [entry.timestamp for entry in recording] == [0.0, 0.5, 1.0]
    assert all(entry.frame.shape == (32, 32, 3) for entry in recording)


def test_load_recording_missing_manifest_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_recording(tmp_path)


def test_load_recording_rejects_empty_recording(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text(
        json.dumps({"frame_count": 0, "duration_s": 0.0, "dropped_frames": 0, "frames": []}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="no frames"):
        load_recording(tmp_path)


def test_load_recording_missing_frame_file_raises(tmp_path: Path) -> None:
    manifest = {
        "frame_count": 1,
        "duration_s": 0.0,
        "dropped_frames": 0,
        "frames": [{"index": 0, "timestamp": 0.0, "path": "frames/000000.png"}],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="could not read"):
        load_recording(tmp_path)


def test_replay_recording_detects_a_single_flash(
    tmp_path: Path, calibration_profile: CalibrationProfile
) -> None:
    # 0.3s off, 0.15s flash, 0.6s off -- well inside single_flash_max_duration_s.
    dt = 0.01
    expanded: list[tuple[float, bool]] = []
    t = 0.0
    for duration, is_on in [(0.3, False), (0.15, True), (0.6, False)]:
        end = t + duration
        while t < end - 1e-9:
            expanded.append((t, is_on))
            t += dt
    _write_recording(tmp_path, expanded)

    recording = load_recording(tmp_path)
    segments, pulses = replay_recording(recording, calibration_profile, blink_window_s=1.0)

    assert any(segment.state is LEDState.SINGLE_FLASH for segment in segments)
    assert len(pulses) == 1
    assert pulses[0].duration == pytest.approx(0.15, abs=0.02)


def test_replay_recording_skips_a_frame_that_fails_classification(
    calibration_profile: CalibrationProfile,
) -> None:
    # A frame too small for the calibrated ROI raises inside
    # InstantClassifier -- replay must skip it, not abort the whole replay.
    recording = [
        RecordedFrame(timestamp=0.0, frame=make_frame(False)),
        RecordedFrame(timestamp=0.01, frame=make_frame(False, size=(2, 2))),
        RecordedFrame(timestamp=0.02, frame=make_frame(False)),
    ]
    segments, pulses = replay_recording(recording, calibration_profile, blink_window_s=1.0)
    assert pulses == []
    assert segments == [] or all(s.state is not LEDState.UNKNOWN for s in segments)


def test_recorded_frame_source_replays_frames_in_order() -> None:
    recording = [
        RecordedFrame(timestamp=0.0, frame=make_frame(False)),
        RecordedFrame(timestamp=0.01, frame=make_frame(True)),
        RecordedFrame(timestamp=0.02, frame=make_frame(False)),
    ]
    source = RecordedFrameSource(recording, speed=1000.0)

    results = [source.read() for _ in range(3)]
    assert [ok for ok, _ in results] == [True, True, True]

    end_ok, end_frame = source.read()
    assert end_ok is False
    assert end_frame is None


def test_recorded_frame_source_loops() -> None:
    recording = [
        RecordedFrame(timestamp=0.0, frame=make_frame(False)),
        RecordedFrame(timestamp=0.01, frame=make_frame(True)),
    ]
    source = RecordedFrameSource(recording, speed=1000.0, loop=True)

    # Read past the natural end multiple times -- must keep succeeding.
    results = [source.read()[0] for _ in range(5)]
    assert all(results)


def test_recorded_frame_source_reset_restarts_playback() -> None:
    recording = [RecordedFrame(timestamp=0.0, frame=make_frame(False))]
    source = RecordedFrameSource(recording, speed=1000.0)

    assert source.read()[0] is True
    assert source.read()[0] is False  # exhausted, not looping

    source.reset()
    assert source.read()[0] is True


def test_recorded_frame_source_release_sets_flag() -> None:
    source = RecordedFrameSource([RecordedFrame(timestamp=0.0, frame=make_frame(False))])
    assert source.released is False
    source.release()
    assert source.released is True


def test_recorded_frame_source_rejects_empty_recording() -> None:
    with pytest.raises(ValueError, match="at least one frame"):
        RecordedFrameSource([])


def test_recorded_frame_source_rejects_non_positive_speed() -> None:
    recording = [RecordedFrame(timestamp=0.0, frame=make_frame(False))]
    with pytest.raises(ValueError, match="speed"):
        RecordedFrameSource(recording, speed=0.0)
