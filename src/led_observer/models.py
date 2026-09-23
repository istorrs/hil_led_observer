"""Shared data types for the LED observer: states, timeline segments, and
sequence-verification structures.

Kept dependency-free (no cv2/numpy) so it can be imported by any layer,
including test code, without pulling in the vision stack.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class LEDState(str, Enum):
    """Discrete classification of the observed LED's optical behavior."""

    OFF = "OFF"
    ON = "ON"
    SLOW_BLINK = "SLOW_BLINK"
    FAST_BLINK = "FAST_BLINK"
    SINGLE_FLASH = "SINGLE_FLASH"
    UNKNOWN = "UNKNOWN"  # insufficient history to classify yet


@dataclass(frozen=True, slots=True)
class StateTransition:
    """A single instantaneous state change, timestamped monotonically."""

    state: LEDState
    timestamp: float


@dataclass(frozen=True, slots=True)
class ObservedSegment:
    """A contiguous run of a single classified state within a timeline."""

    state: LEDState
    start_time: float
    end_time: float

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass(frozen=True, slots=True)
class Pulse:
    """A single completed ON pulse: an edge rise followed by a fall."""

    rise_time: float
    fall_time: float

    @property
    def duration(self) -> float:
        return self.fall_time - self.rise_time


@dataclass(frozen=True, slots=True)
class PulsePattern:
    """A fixed, ordered burst of pulses -- e.g. a firmware's "N-blink"
    confirmation code -- matched against raw ON/OFF edge timings directly,
    rather than against :class:`~led_observer.detector.BlinkClassifier`'s
    frequency-bucketed :class:`LEDState`.

    Use this in place of ``state`` in a :class:`SequenceStep` for a short,
    fixed-count burst that a windowed frequency classifier can't reliably
    tell apart from noise, a SINGLE_FLASH, or the start of a longer blink --
    there are too few pulses to average a stable period from, so the
    classifier's output for a burst like this is inherently transient and
    unreliable (see the detector module docs).

    ``pulses`` is a list of ``(on_duration_s, off_duration_s)`` pairs, one
    per pulse, in order. The **last pulse's ``off_duration_s`` is not
    independently checked** by the matcher (there is no further pulse in
    the burst to measure that gap against) -- rely on the next
    ``SequenceStep`` in the schema (typically a long ``OFF``) to confirm
    the burst didn't continue.

    ``settle_time_s`` is how much longer, after the burst's last pulse
    falls, :func:`~led_observer.sequence.verify_sequence` keeps skipping
    observed segments before resuming ordinary state matching for the
    *next* step. A windowed frequency classifier's misleading read on a
    too-short burst (see the class docstring) doesn't vanish the instant
    the burst physically ends -- it lingers in the classifier's window for
    up to roughly its ``blink_window_s``, producing an actual segment that
    starts *after* the last pulse but is still noise from the burst, not a
    new real state. Set this to roughly the service's ``blink_window_s`` if
    the step immediately following this one is getting skipped over by
    mistake; the default of ``0.0`` assumes no such lingering segment.
    """

    pulses: list[tuple[float, float]]
    tolerance_pct: float = 0.3
    settle_time_s: float = 0.0

    def __post_init__(self) -> None:
        if not self.pulses:
            raise ValueError("PulsePattern must have at least one pulse")


@dataclass(frozen=True, slots=True)
class SequenceStep:
    """One step of an expected cadence sequence: exactly one of ``state``
    (matched against classified :class:`ObservedSegment`\\ s) or
    ``pulse_pattern`` (matched against raw :class:`Pulse` edge timings)
    must be set.

    ``duration_s=None`` means "any duration" (the step's length is not
    checked, only that the state occurs next in order) -- this is how the
    prompt's ``OFF (any)`` steps are expressed. ``duration_s`` and
    ``tolerance_pct`` are ignored for a ``pulse_pattern`` step -- the
    pattern's own per-pulse ``tolerance_pct`` governs that match instead.
    """

    state: LEDState | None = None
    pulse_pattern: PulsePattern | None = None
    duration_s: float | None = None
    # Per-step override; None defers to the tolerance passed to verify_sequence().
    tolerance_pct: float | None = None

    def __post_init__(self) -> None:
        if (self.state is None) == (self.pulse_pattern is None):
            raise ValueError("SequenceStep needs exactly one of state or pulse_pattern")


@dataclass(frozen=True, slots=True)
class DivergencePoint:
    """Where and why an observed sequence stopped matching the expected one."""

    index: int
    reason: str


@dataclass(frozen=True, slots=True)
class VerificationReport:
    """Result of comparing an observed sequence against an expected one."""

    matched: bool
    expected_sequence: list[SequenceStep]
    actual_sequence: list[ObservedSegment]
    divergence_point: DivergencePoint | None = None


@dataclass(slots=True)
class ObservationResult:
    """Everything recorded during one start/stop observation window.

    ``verification`` is populated automatically when ``expected_sequence``
    was passed to ``start_observation`` for this window; otherwise call
    :func:`~led_observer.sequence.verify_sequence` against ``segments``
    directly.
    """

    segments: list[ObservedSegment] = field(default_factory=list)
    transitions: list[StateTransition] = field(default_factory=list)
    pulses: list[Pulse] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0
    timed_out: bool = False
    dropped_frames: int = 0
    verification: VerificationReport | None = None

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time
