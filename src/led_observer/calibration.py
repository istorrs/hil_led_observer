"""ROI definition and ON/OFF reference-state calibration.

Calibration derives two independent, complementary signals from a set of
sample frames captured while the DUT LED is known to be ON and known to be
OFF:

* **Luminosity** (mean L channel in CIELAB over the ROI) drives the primary
  ON/OFF decision with a hysteresis band (``T_off`` < ``T_on``) so sensor
  noise near the boundary doesn't cause chatter.
* **Color ratio** (fraction of ROI pixels inside an HSV range derived from
  the ON samples) is a secondary gate that rejects "ON-bright but wrong
  color" false positives, e.g. a specular reflection.

A profile is a small, structured JSON document so it can be captured once on
a bench and version-controlled or reused across HIL rigs.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

Frame = np.ndarray[Any, np.dtype[np.uint8]]


class CalibrationError(ValueError):
    """Raised when calibration samples are insufficient or inconsistent."""


@dataclass(frozen=True, slots=True)
class ROI:
    """A pixel-space Region of Interest within a captured frame."""

    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise CalibrationError("ROI width and height must be positive")
        if self.x < 0 or self.y < 0:
            raise CalibrationError("ROI x and y must be non-negative")

    def crop(self, frame: Frame) -> Frame:
        if frame.shape[0] < self.y + self.height or frame.shape[1] < self.x + self.width:
            raise CalibrationError(
                f"ROI {self} does not fit within frame of shape {frame.shape[:2]}"
            )
        return frame[self.y : self.y + self.height, self.x : self.x + self.width]


@dataclass(frozen=True, slots=True)
class FrameMetrics:
    """Per-frame measurements extracted from the calibrated ROI."""

    luminosity: float
    color_ratio: float


@dataclass(frozen=True, slots=True)
class CalibrationProfile:
    """Persisted thresholds and bounds needed to classify a frame's ROI."""

    roi: ROI
    luminosity_on_threshold: float
    luminosity_off_threshold: float
    hsv_lower: tuple[int, int, int]
    hsv_upper: tuple[int, int, int]
    color_ratio_min: float

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CalibrationProfile:
        roi_data = data["roi"]
        return cls(
            roi=ROI(
                x=roi_data["x"], y=roi_data["y"], width=roi_data["width"], height=roi_data["height"]
            ),
            luminosity_on_threshold=data["luminosity_on_threshold"],
            luminosity_off_threshold=data["luminosity_off_threshold"],
            hsv_lower=tuple(data["hsv_lower"]),
            hsv_upper=tuple(data["hsv_upper"]),
            color_ratio_min=data["color_ratio_min"],
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> CalibrationProfile:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def compute_frame_metrics(frame: Frame, profile_or_roi: ROI | CalibrationProfile) -> FrameMetrics:
    """Measure luminosity and color-match ratio for a frame's ROI.

    Accepts either a bare :class:`ROI` (color ratio uses a permissive
    full-range mask, used during calibration sampling) or a full
    :class:`CalibrationProfile` (color ratio uses its calibrated HSV bounds).
    """
    if isinstance(profile_or_roi, CalibrationProfile):
        roi = profile_or_roi.roi
        lower = np.array(profile_or_roi.hsv_lower, dtype=np.uint8)
        upper = np.array(profile_or_roi.hsv_upper, dtype=np.uint8)
    else:
        roi = profile_or_roi
        lower = np.array((0, 0, 0), dtype=np.uint8)
        upper = np.array((179, 255, 255), dtype=np.uint8)

    patch = roi.crop(frame)
    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)
    luminosity = float(lab[:, :, 0].mean())

    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    color_ratio = float(np.count_nonzero(mask)) / float(mask.size)

    return FrameMetrics(luminosity=luminosity, color_ratio=color_ratio)


def build_calibration_profile(
    roi: ROI,
    on_frames: list[Frame],
    off_frames: list[Frame],
    *,
    hue_margin: int = 10,
    sat_margin: int = 60,
    hysteresis_pct: float = 0.2,
) -> CalibrationProfile:
    """Derive a :class:`CalibrationProfile` from ON/OFF reference samples.

    ``hysteresis_pct`` controls the gap between ``T_off`` and ``T_on`` as a
    fraction of the ON/OFF luminosity separation, so classification doesn't
    chatter for samples that land between the two reference states.
    """
    if not on_frames or not off_frames:
        raise CalibrationError("at least one ON frame and one OFF frame are required")

    on_metrics = [compute_frame_metrics(f, roi) for f in on_frames]
    off_metrics = [compute_frame_metrics(f, roi) for f in off_frames]

    on_luminosity = sum(m.luminosity for m in on_metrics) / len(on_metrics)
    off_luminosity = sum(m.luminosity for m in off_metrics) / len(off_metrics)

    if on_luminosity <= off_luminosity:
        raise CalibrationError(
            "ON samples must be brighter than OFF samples "
            f"(got ON={on_luminosity:.1f}, OFF={off_luminosity:.1f}); "
            "check the ROI location and sample capture order"
        )

    band = on_luminosity - off_luminosity
    half_gap = (hysteresis_pct / 2.0) * band
    midpoint = (on_luminosity + off_luminosity) / 2.0
    luminosity_on_threshold = midpoint + half_gap
    luminosity_off_threshold = midpoint - half_gap

    on_patches_hsv = [
        cv2.cvtColor(roi.crop(f), cv2.COLOR_BGR2HSV).reshape(-1, 3) for f in on_frames
    ]
    stacked = np.concatenate(on_patches_hsv, axis=0)
    hue_mean = float(stacked[:, 0].mean())
    sat_mean = float(stacked[:, 1].mean())

    hsv_lower = (max(0, int(hue_mean - hue_margin)), max(0, int(sat_mean - sat_margin)), 0)
    hsv_upper = (min(179, int(hue_mean + hue_margin)), 255, 255)

    provisional = CalibrationProfile(
        roi=roi,
        luminosity_on_threshold=luminosity_on_threshold,
        luminosity_off_threshold=luminosity_off_threshold,
        hsv_lower=hsv_lower,
        hsv_upper=hsv_upper,
        color_ratio_min=0.0,
    )
    on_color_ratios = [compute_frame_metrics(f, provisional).color_ratio for f in on_frames]
    color_ratio_min = 0.5 * (sum(on_color_ratios) / len(on_color_ratios))

    return CalibrationProfile(
        roi=roi,
        luminosity_on_threshold=luminosity_on_threshold,
        luminosity_off_threshold=luminosity_off_threshold,
        hsv_lower=hsv_lower,
        hsv_upper=hsv_upper,
        color_ratio_min=color_ratio_min,
    )
