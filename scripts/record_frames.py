#!/usr/bin/env python3
"""Records a raw frame sequence from a live camera for later offline
analysis via ``led_observer.playback`` -- so tuning calibration/detection
parameters doesn't mean repeating a slow, manual, real-hardware DUT trigger
sequence for every attempt.

Deliberately writes lossless per-frame PNGs plus a JSON timestamp manifest,
not a compressed video file: video codecs use lossy temporal prediction and
chroma-subsample color, both actively harmful to a small, fast, low-contrast
LED signal that only occupies a few pixels for one frame at a time -- a
video's nominal frame rate also isn't the same as true per-frame capture
timestamps, which matters for a fast (tens-of-ms) blink cadence. See
`led_observer.playback` for how a recording is read back.

Usage:
    python scripts/record_frames.py --duration 60 --out recordings/run1 --autofocus
    python scripts/record_frames.py --duration 60 --out recordings/run2 --focus 663
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2

from led_observer.capture import OpenCVCameraSource


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--duration", type=float, required=True, help="recording length, seconds")
    parser.add_argument("--out", required=True, help="output directory (created if missing)")
    parser.add_argument(
        "--autofocus", action="store_true", help="run software autofocus before recording"
    )
    parser.add_argument(
        "--focus", type=int, default=None, help="set this fixed focus position instead of sweeping"
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    source = OpenCVCameraSource(args.camera)
    manifest_frames: list[dict[str, object]] = []
    dropped = 0
    try:
        if args.autofocus:
            result = source.autofocus()
            print(
                f"[record] autofocus: {result.initial_focus}->{result.focus} "
                f"(sharpness {result.initial_sharpness:.1f}->{result.sharpness:.1f})"
            )
        elif args.focus is not None:
            source.disable_hardware_autofocus()
            source.set_focus(args.focus)
            print(f"[record] focus fixed at {args.focus}")
        else:
            source.disable_hardware_autofocus()

        print(f"[record] recording for {args.duration:.1f}s to {out_dir}...")
        index = 0
        start = time.monotonic()
        while time.monotonic() - start < args.duration:
            success, frame = source.read()
            timestamp = time.monotonic() - start
            if not success or frame is None:
                dropped += 1
                continue

            rel_path = f"frames/{index:06d}.png"
            cv2.imwrite(str(out_dir / rel_path), frame)
            manifest_frames.append({"index": index, "timestamp": timestamp, "path": rel_path})
            index += 1
            if index % 200 == 0:
                print(f"[record]   {index} frames captured ({timestamp:.1f}s)...")
    finally:
        source.release()

    if not manifest_frames:
        print("[record] no frames captured -- check the camera")
        return 1

    manifest = {
        "frame_count": len(manifest_frames),
        "duration_s": manifest_frames[-1]["timestamp"],
        "dropped_frames": dropped,
        "frames": manifest_frames,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(
        f"[record] done: {len(manifest_frames)} frames, {dropped} dropped, "
        f"{manifest['duration_s']:.1f}s -> {out_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
