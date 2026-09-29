"""LED cadence observer: optical state/blink-pattern verification for HIL rigs."""

from importlib.metadata import version

from led_observer.autofocus import AutofocusResult, autofocus_sweep, measure_sharpness
from led_observer.calibration import (
    ROI,
    CalibrationError,
    CalibrationProfile,
    build_calibration_profile,
    compute_frame_metrics,
)
from led_observer.capture import CameraUnavailableError, FrameSource, OpenCVCameraSource
from led_observer.detector import BlinkClassifier, InstantClassifier, estimate_frequency_fft
from led_observer.models import (
    DivergencePoint,
    LEDState,
    ObservationResult,
    ObservedSegment,
    Pulse,
    PulsePattern,
    SequenceStep,
    StateTransition,
    VerificationReport,
)
from led_observer.playback import (
    RecordedFrame,
    RecordedFrameSource,
    load_recording,
    replay_recording,
)
from led_observer.sequence import PulseRecorder, TimelineRecorder, verify_sequence
from led_observer.service import LEDObserverService, ObservationInProgressError

__all__ = [
    "ROI",
    "AutofocusResult",
    "BlinkClassifier",
    "CalibrationError",
    "CalibrationProfile",
    "CameraUnavailableError",
    "DivergencePoint",
    "FrameSource",
    "InstantClassifier",
    "LEDObserverService",
    "LEDState",
    "ObservationInProgressError",
    "ObservationResult",
    "ObservedSegment",
    "OpenCVCameraSource",
    "Pulse",
    "PulsePattern",
    "PulseRecorder",
    "RecordedFrame",
    "RecordedFrameSource",
    "SequenceStep",
    "StateTransition",
    "TimelineRecorder",
    "VerificationReport",
    "autofocus_sweep",
    "build_calibration_profile",
    "compute_frame_metrics",
    "estimate_frequency_fft",
    "load_recording",
    "measure_sharpness",
    "replay_recording",
    "verify_sequence",
]

__version__ = version("led-observer")
