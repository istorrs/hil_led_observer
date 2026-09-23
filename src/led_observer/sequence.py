"""Timeline recording and expected-vs-actual cadence sequence verification."""

from __future__ import annotations

from led_observer.models import (
    DivergencePoint,
    LEDState,
    ObservedSegment,
    Pulse,
    PulsePattern,
    SequenceStep,
    StateTransition,
    VerificationReport,
)


class TimelineRecorder:
    """Collapses a stream of per-sample state classifications into
    contiguous :class:`~led_observer.models.ObservedSegment` runs.
    """

    def __init__(self) -> None:
        self._segments: list[ObservedSegment] = []
        self._transitions: list[StateTransition] = []
        self._open_state: LEDState | None = None
        self._open_start: float = 0.0

    def add(self, state: LEDState, timestamp: float) -> None:
        if self._open_state is None:
            self._open_state = state
            self._open_start = timestamp
            self._transitions.append(StateTransition(state, timestamp))
            return
        if state == self._open_state:
            return
        self._segments.append(ObservedSegment(self._open_state, self._open_start, timestamp))
        self._transitions.append(StateTransition(state, timestamp))
        self._open_state = state
        self._open_start = timestamp

    def finalize(self, timestamp: float) -> list[ObservedSegment]:
        if self._open_state is not None and timestamp > self._open_start:
            self._segments.append(ObservedSegment(self._open_state, self._open_start, timestamp))
            self._open_state = None
        return list(self._segments)

    @property
    def segments(self) -> list[ObservedSegment]:
        return list(self._segments)

    @property
    def transitions(self) -> list[StateTransition]:
        return list(self._transitions)

    def reset(self) -> None:
        self._segments.clear()
        self._transitions.clear()
        self._open_state = None
        self._open_start = 0.0


class PulseRecorder:
    """Records completed ON pulses (rise/fall edge pairs) from a raw
    ``(timestamp, is_on)`` stream.

    Unlike :class:`TimelineRecorder`, this keeps the *full*, un-windowed
    history of edges for the whole observation -- it exists specifically so
    a :class:`~led_observer.models.PulsePattern` can be matched against
    exact pulse timings, independent of whatever (possibly transient/noisy)
    :class:`~led_observer.models.LEDState` a windowed frequency classifier
    happened to report for the same span.
    """

    def __init__(self) -> None:
        self._pulses: list[Pulse] = []
        self._rise_time: float | None = None
        self._last_is_on: bool | None = None

    def add(self, timestamp: float, is_on: bool) -> None:
        if self._last_is_on is None:
            self._last_is_on = is_on
            if is_on:
                self._rise_time = timestamp
            return
        if is_on and not self._last_is_on:
            self._rise_time = timestamp
        elif not is_on and self._last_is_on and self._rise_time is not None:
            self._pulses.append(Pulse(rise_time=self._rise_time, fall_time=timestamp))
            self._rise_time = None
        self._last_is_on = is_on

    @property
    def pulses(self) -> list[Pulse]:
        return list(self._pulses)

    def reset(self) -> None:
        self._pulses.clear()
        self._rise_time = None
        self._last_is_on = None


def _pulses_match(pattern: PulsePattern, candidate: list[Pulse]) -> bool:
    for j, (on_duration, off_duration) in enumerate(pattern.pulses):
        pulse = candidate[j]
        if abs(pulse.duration - on_duration) > pattern.tolerance_pct * on_duration:
            return False
        if j < len(candidate) - 1:
            gap = candidate[j + 1].rise_time - pulse.fall_time
            if abs(gap - off_duration) > pattern.tolerance_pct * off_duration:
                return False
    return True


def _find_pulse_pattern_match(
    pattern: PulsePattern, pulses: list[Pulse], start_index: int
) -> tuple[int, int] | None:
    """Searches for the first run in ``pulses`` matching ``pattern``, at or
    after ``start_index``. Returns the matching run's
    ``(start_index, end_index_exclusive)``, or ``None``.

    Deliberately indexed into ``pulses``, not gated by a wall-clock time
    cursor from state-step matching: a windowed frequency classifier's
    segment boundaries lag the raw edges it was computed from (see
    :class:`~led_observer.models.PulsePattern`), so a segment-based time
    cursor can land *after* a burst's true pulses -- e.g. a preceding
    ``SLOW_BLINK`` segment's reported end time can extend well past where a
    later burst actually happened, if the classifier's window still had a
    couple of the burst's pulses mixed in with the tail of the slow blink's
    own pulses. Each pulse_pattern step's search always starts from the end
    of the *previous* pulse_pattern step's match (or the start of ``pulses``
    for the first one) -- independent of intervening state steps.
    """
    pulse_count = len(pattern.pulses)
    for i in range(start_index, len(pulses) - pulse_count + 1):
        if _pulses_match(pattern, pulses[i : i + pulse_count]):
            return i, i + pulse_count
    return None


def verify_sequence(
    expected_sequence: list[SequenceStep],
    actual_sequence: list[ObservedSegment],
    tolerance_pct: float = 0.20,
    pulses: list[Pulse] | None = None,
) -> VerificationReport:
    """Compares an observed segment timeline against an expected schema.

    State-based steps are matched strictly in order, one expected step per
    actual segment. Trailing actual segments beyond what the schema
    describes are not considered a mismatch -- ``expected_sequence`` may
    describe only a prefix of interest within a longer observation.

    A ``pulse_pattern`` step is matched against ``pulses`` instead (which
    must be supplied whenever ``expected_sequence`` contains one): it
    searches forward for a run of raw pulses matching the pattern (see
    :func:`_find_pulse_pattern_match` for why that search is indexed into
    ``pulses`` rather than gated by the state steps' time cursor), then
    skips the actual-segment cursor past every segment that started before
    the match ended (plus the pattern's ``settle_time_s``). This is
    deliberately more forgiving than the strict per-segment matching used
    for state steps, because a short burst like this is exactly the kind
    of thing a windowed frequency classifier reports as a handful of noisy,
    transient segments (see :class:`~led_observer.models.PulsePattern`) --
    those transient segments are consumed by the pulse-pattern step rather
    than needing to be named individually in the schema.
    """
    actual_index = 0
    pulse_index = 0

    for index, expected in enumerate(expected_sequence):
        if expected.pulse_pattern is not None:
            if pulses is None:
                raise ValueError("verify_sequence() needs `pulses` to match a PulsePattern step")
            match = _find_pulse_pattern_match(expected.pulse_pattern, pulses, pulse_index)
            if match is None:
                reason = (
                    f"Expected a {len(expected.pulse_pattern.pulses)}-pulse pattern "
                    f"starting at/after pulse #{pulse_index}, but no matching pulses "
                    "were observed"
                )
                return VerificationReport(
                    matched=False,
                    expected_sequence=expected_sequence,
                    actual_sequence=actual_sequence,
                    divergence_point=DivergencePoint(index=index, reason=reason),
                )
            _, end_index = match
            pulse_index = end_index
            skip_before = pulses[end_index - 1].fall_time + expected.pulse_pattern.settle_time_s
            while (
                actual_index < len(actual_sequence)
                and actual_sequence[actual_index].start_time < skip_before
            ):
                actual_index += 1
            continue

        assert expected.state is not None  # enforced by SequenceStep.__post_init__

        if actual_index >= len(actual_sequence):
            reason = (
                f"Expected {expected.state.value} at step {index}, "
                "but the observation ended before it occurred"
            )
            return VerificationReport(
                matched=False,
                expected_sequence=expected_sequence,
                actual_sequence=actual_sequence,
                divergence_point=DivergencePoint(index=index, reason=reason),
            )

        actual = actual_sequence[actual_index]

        if expected.state != actual.state:
            reason = (
                f"Expected {expected.state.value} at t={actual.start_time:.2f}s, "
                f"but observed {actual.state.value}"
            )
            return VerificationReport(
                matched=False,
                expected_sequence=expected_sequence,
                actual_sequence=actual_sequence,
                divergence_point=DivergencePoint(index=index, reason=reason),
            )

        if expected.duration_s is not None:
            step_tolerance = (
                expected.tolerance_pct if expected.tolerance_pct is not None else tolerance_pct
            )
            allowed_delta = step_tolerance * expected.duration_s
            actual_delta = abs(actual.duration - expected.duration_s)
            if actual_delta > allowed_delta:
                reason = (
                    f"Expected {expected.state.value} for {expected.duration_s:.2f}s "
                    f"(±{step_tolerance * 100:.0f}%) at t={actual.start_time:.2f}s, "
                    f"but observed duration {actual.duration:.2f}s"
                )
                return VerificationReport(
                    matched=False,
                    expected_sequence=expected_sequence,
                    actual_sequence=actual_sequence,
                    divergence_point=DivergencePoint(index=index, reason=reason),
                )

        actual_index += 1

    return VerificationReport(
        matched=True,
        expected_sequence=expected_sequence,
        actual_sequence=actual_sequence,
        divergence_point=None,
    )
