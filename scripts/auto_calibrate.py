#!/usr/bin/env python3
"""Auto-calibrate against a real, already-blinking LED: no manual ROI entry,
no GUI. Captures a burst of frames, finds the pixel region with the most
brightness variance (the flickering LED), and derives a CalibrationProfile
from the brightest/dimmest frames observed at that ROI.

Usage:
    python scripts/auto_calibrate.py [--camera 0] [--duration 6.0] [--out config/calibration.json]

Writes the calibration profile plus a debug PNG (roi_debug.png) with the
detected ROI drawn on a representative frame, so it can be sanity-checked
visually before trusting it.
"""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from led_observer.calibration import ROI, build_calibration_profile
from led_observer.capture import Frame, OpenCVCameraSource


def capture_burst(source: OpenCVCameraSource, duration_s: float) -> list[Frame]:
    frames: list[Frame] = []
    deadline = time.monotonic() + duration_s
    while time.monotonic() < deadline:
        success, frame = source.read()
        if success and frame is not None:
            frames.append(frame)
    return frames


def find_flicker_roi(frames: list[Frame], margin: int = 8) -> ROI:
    """Bounding box of the region with the largest ON/OFF luminosity swing
    across the captured frames -- i.e. wherever something was blinking."""
    grays = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames])
    pixel_range = grays.max(axis=0).astype(np.int16) - grays.min(axis=0).astype(np.int16)

    threshold = max(int(pixel_range.max() * 0.5), 20)
    mask = (pixel_range >= threshold).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        raise RuntimeError(
            "no consistent bright/dark region found -- is the LED in frame and blinking?"
        )

    # stats[0] is the background component; pick the largest foreground one.
    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h = (int(v) for v in stats[largest_label, :4])

    height, width = pixel_range.shape
    x0 = max(0, x - margin)
    y0 = max(0, y - margin)
    x1 = min(width, x + w + margin)
    y1 = min(height, y + h + margin)
    return ROI(x=x0, y=y0, width=x1 - x0, height=y1 - y0)


def pick_on_off_frames(
    frames: list[Frame], roi: ROI, low_pct: float = 20.0, high_pct: float = 80.0
) -> tuple[list[Frame], list[Frame]]:
    luminosities = [
        float(cv2.cvtColor(roi.crop(f), cv2.COLOR_BGR2LAB)[:, :, 0].mean()) for f in frames
    ]
    low_cut = float(np.percentile(luminosities, low_pct))
    high_cut = float(np.percentile(luminosities, high_pct))
    off_frames = [f for f, lum in zip(frames, luminosities, strict=True) if lum <= low_cut]
    on_frames = [f for f, lum in zip(frames, luminosities, strict=True) if lum >= high_cut]
    return on_frames, off_frames


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--out", default="config/calibration.json")
    parser.add_argument("--debug-image", default="roi_debug.png")
    args = parser.parse_args()

    source = OpenCVCameraSource(args.camera)
    try:
        print(f"[calibrate] capturing {args.duration:.1f}s of frames from camera {args.camera}...")
        frames = capture_burst(source, args.duration)
    finally:
        source.release()

    print(f"[calibrate] captured {len(frames)} frames; locating the blinking region...")
    roi = find_flicker_roi(frames)
    print(f"[calibrate] detected ROI: {roi}")

    on_frames, off_frames = pick_on_off_frames(frames, roi)
    print(f"[calibrate] using {len(on_frames)} ON-like frames, {len(off_frames)} OFF-like frames")

    profile = build_calibration_profile(roi, on_frames=on_frames, off_frames=off_frames)
    profile.save(args.out)
    print(f"[calibrate] saved calibration profile to {args.out}")
    print(
        f"[calibrate] luminosity thresholds: off<={profile.luminosity_off_threshold:.1f} "
        f"on>={profile.luminosity_on_threshold:.1f}, color_ratio_min={profile.color_ratio_min:.2f}"
    )

    debug_frame = frames[len(frames) // 2].copy()
    cv2.rectangle(
        debug_frame,
        (roi.x, roi.y),
        (roi.x + roi.width, roi.y + roi.height),
        (0, 0, 255),
        2,
    )
    cv2.imwrite(args.debug_image, debug_frame)
    print(f"[calibrate] wrote {args.debug_image} for visual sanity-check")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
