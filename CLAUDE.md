# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Setup (project targets Python >=3.10,<3.13; opencv-python needs a released CPython)
uv venv --python 3.10 .venv && source .venv/bin/activate
uv pip install -e ".[dev]"

ruff check .                       # lint
ruff format .                      # format
mypy                                # strict type check (covers src, tests, examples)
pytest                              # full test suite (~6s; test_service.py is the slow ~5s of it)
pytest --cov                        # with coverage (90% floor enforced, see [tool.coverage.report])
pytest tests/test_detector.py -k test_blink_classifier_detects_slow_blink  # single test
python examples/hil_integration_demo.py   # runnable demo, no camera hardware needed (~28s, uses real sleeps)
python scripts/record_frames.py --duration 60 --out recordings/run1 --autofocus  # record real hardware for offline tuning
python scripts/auto_calibrate.py --camera 0 --duration 6.0   # auto-locate + calibrate against a live, already-blinking LED
```

`tests/test_service.py` drives the real background thread against a
`SyntheticFrameSource` over real wall-clock time (short sleeps, ~5s total)
rather than mocking time, so timing constants in that file are load-bearing
-- see the "Windowed cadence classification" note below before changing
`blink_window_s` or sleep durations there.

## Architecture

Layered, each layer independently unit-testable without a camera:

```
autofocus.autofocus_sweep               -- sharpness-maximizing focus search (camera-agnostic, callable-based)
        |
capture.FrameSource (protocol)          -- OpenCVCameraSource | test's SyntheticFrameSource
        |  OpenCVCameraSource.autofocus() wraps autofocus_sweep against a real camera
calibration.CalibrationProfile          -- ROI + LAB-luminosity hysteresis thresholds + HSV color gate
        |
detector.InstantClassifier              -- one frame -> bool (ON/OFF), via calibration.compute_frame_metrics
        |
detector.BlinkClassifier                -- stream of (timestamp, bool) -> LEDState, sliding time-domain window
        |
sequence.TimelineRecorder               -- LEDState stream -> list[ObservedSegment] (collapses repeats)
sequence.PulseRecorder                  -- (timestamp, bool) stream -> list[Pulse] (full, un-windowed history)
        |
sequence.verify_sequence                -- list[SequenceStep] x (list[ObservedSegment], list[Pulse]) -> VerificationReport
        |
service.LEDObserverService              -- owns a background thread wiring all of the above together;
                                            start_observation/stop_observation/get_current_state
```

`models.py` holds every cross-layer dataclass/enum and has no other
dependencies, so it's safe to import from any layer (including tests)
without pulling in cv2/numpy transitively beyond what numpy itself needs.

### Software autofocus

Continuous hardware autofocus tends to "breathe" (keep hunting/refocusing)
when aimed at a small, high-contrast target like an LED against a dark PCB
-- this library's whole use case -- which drifts a calibrated ROI out of
focus over time. `autofocus.autofocus_sweep` fixes this the standard
embedded-vision way: disable the camera's own autofocus and drive focus to
one fixed position chosen by maximizing variance-of-Laplacian sharpness
(coarse linear scan, then a localized hill-climb). It's decoupled from any
camera API via three callables (`get_frame`/`get_focus`/`set_focus`), so
it's unit-tested against a synthetic focus-vs-sharpness curve with no real
camera or `time.sleep` needed (`tests/test_autofocus.py`).
`OpenCVCameraSource.autofocus()` is the real-camera wrapper; it also
re-asserts hardware-autofocus-off plus the chosen focus a few times
afterward, since some cameras drift back into continuous AF otherwise, and
`OpenCVCameraSource.disable_hardware_autofocus()` additionally writes back
the *current* focus value immediately after setting `CAP_PROP_AUTOFOCUS=0`
-- some UVC drivers need an explicit focus write to actually leave
continuous-AF mode, not just the property set to 0.

Typical workflow once a camera is open: `disable_hardware_autofocus()` (or
just call `autofocus()`, which does this first) before calibrating, ideally
passing `roi=` once the LED's ROI is known so sharpness is scored there
rather than over background clutter.

### Recording real hardware for offline tuning

`scripts/record_frames.py` + `playback.py` exist because repeating a real,
slow, manually-triggered DUT sequence for every calibration/threshold tweak
doesn't scale. The script records lossless per-frame PNGs plus a
`manifest.json` of each frame's true capture-relative timestamp --
deliberately *not* a compressed video: video codecs' lossy temporal
prediction and chroma subsampling are both actively harmful to a small,
fast, low-contrast LED signal that occupies only a few pixels for one frame
at a time, and a video's nominal frame rate isn't the same as true
per-frame timestamps (which matters for a fast, tens-of-ms blink cadence).

`playback.load_recording` reads one back; `playback.replay_recording` feeds
it through the real `InstantClassifier`/`BlinkClassifier`/`TimelineRecorder`/
`PulseRecorder` pipeline using each frame's *original recorded timestamp*
directly (no real-time waiting, no threads) -- the same pattern
`test_real_dut_sequences.py` uses for synthetic data, here applied to real
captured frames, so a different `CalibrationProfile` or `blink_window_s` can
be tried against the same footage near-instantly.
`playback.RecordedFrameSource` is the complementary `FrameSource`
implementation for the rarer case of wanting the real threaded
`LEDObserverService` running against recorded data at (approximately) its
original real-time pacing instead (e.g. to reproduce a service-level issue).

A rolling-shutter camera can partially expose a fast pulse shorter than a
region's row-readout window, producing a dimmer-than-true-ON reading rather
than a clean binary sample -- recording real footage is how to actually
check whether this is happening, rather than assuming idealized levels.

### Windowed cadence classification is causal, not retroactive

`BlinkClassifier.classify()` only ever sees samples up to *now* -- it cannot
know in advance whether an LED that just turned on will turn back off
within `single_flash_max_duration_s` (default 1.0s, making it a
`SINGLE_FLASH`) or stay on. This has two consequences that show up
throughout the detector, service, and test code:

- **A still-open pulse shorter than the threshold reports `UNKNOWN`**, not
  `ON`, until it's either fallen (resolving to `SINGLE_FLASH`) or outlasted
  the threshold (resolving to `ON`). `blink_window_s` must therefore be
  comfortably larger than `single_flash_max_duration_s` for steady-`ON` to
  ever be confidently reported at all -- otherwise the window can never
  retain a sample old enough to prove the pulse has been on that long. See
  `BlinkClassifier._pulses`/`classify` and the `window_s` vs
  `single_flash_max_duration_s` tests in `test_detector.py`.
- **The first cycle of any new blink cadence is briefly seen as
  `SINGLE_FLASH`-then-`ON`** before two completed pulses establish a
  measurable period (frequency classification needs `len(pulses) >= 2`).
  `examples/hil_integration_demo.py`'s `EXPECTED_SEQUENCE` names these
  transient steps explicitly rather than treating them as noise to filter
  -- that's a real property of a live/causal detector, reproduced
  deterministically by that example's synthetic schedule.
- Classification also lags a real state transition by up to
  `blink_window_s` (old samples linger in the window), which is why
  `service.py`'s `TimelineRecorder` filtering only drops `UNKNOWN`
  specifically -- a lingering stale classification is still a real,
  intentional value, not noise -- and why `SequenceStep.duration_s` checks
  need either "any duration" (`None`) or a tolerance sized relative to
  `blink_window_s` for segments shortly after a transition.
- **`blink_window_s` needs roughly 2.5x+ the slowest expected blink
  *period* (1/frequency), not just "more than one period".** A window only
  ~1.5x the period (e.g. `window_s=3.0` for a 2.0s-period/0.5Hz blink)
  empirically *oscillates* between the correct `SLOW_BLINK`/`FAST_BLINK`
  reading and a wrong one, because whether the window's completed-pulse
  count is 1 or 2 at any given moment becomes phase-dependent. Traced and
  regression-tested in `test_real_dut_sequences.py`
  (`test_double_press_commissioning_blink_is_unstable_below_2_5x_period_window`).
  This is separate from, and in addition to, the `single_flash_max_duration_s`
  constraint above.

### Fixed pulse bursts need `PulsePattern`, not `LEDState`

`BlinkClassifier` estimates a frequency by averaging periods between
completed pulses -- fine for a genuine repeating cadence, but a **short,
fixed-count burst** (e.g. a firmware's "2-blink confirmation" code) doesn't
have enough pulses to average a stable period from, and gets reported as a
handful of noisy, transient `LEDState`s that are actively misleading (e.g.
two ~100ms pulses ~0.2s apart can transiently average out to a plausible
but wrong `FAST_BLINK` reading once the window trims down to just those two
pulses). Trying to name those transient states individually in an expected
schema is fragile -- which transient states appear, and for how long,
depends on exact window/pulse timing.

Instead, give the `SequenceStep` a `pulse_pattern: PulsePattern` (a fixed,
ordered list of `(on_duration_s, off_duration_s)` pairs) instead of a
`state`. It's matched against `PulseRecorder`'s raw, un-windowed pulse log
directly -- bypassing `BlinkClassifier` entirely for that step -- so the
burst's own noisy `LEDState` segments never need to be named. Two important
follow-on details, both in `sequence.py`:

- The pulse search is **indexed into the pulse list, not gated by the
  preceding state steps' time cursor.** A classifier's segment boundaries
  lag the raw edges (previous bullet), so a preceding segment's reported
  end time can land *after* the burst's true pulses already happened (the
  window can keep mixing a burst's pulses into an unrelated adjacent
  segment's frequency average for a while before finally reporting
  something clearly wrong). Gating the pulse search by that inflated time
  would make it search right past the real pulses.
- After a match, `PulsePattern.settle_time_s` (roughly `blink_window_s`,
  tuned empirically -- see its docstring) tells `verify_sequence` how much
  longer to keep skipping actual segments before resuming ordinary state
  matching for the *next* step, since the misleading transient segment
  lingers for a while after the burst's last pulse physically falls.

See `test_real_dut_sequences.py` for a full worked example traced from real
firmware documentation (two related sequences that share this exact burst).

### Sequence verification semantics

`verify_sequence` matches `expected_sequence` against `actual_sequence`
strictly in order, one expected step per actual segment; trailing actual
segments beyond `len(expected_sequence)` are not a failure (the expected
schema may describe only a prefix of interest). `SequenceStep.tolerance_pct`
overrides the function's `tolerance_pct` default per-step when set. A
`pulse_pattern` step is the one exception to "strict" -- see above.

### Service threading model

`LEDObserverService` runs one background daemon thread (`_run`) that
continuously reads/classifies frames and updates `get_current_state()`
*regardless* of whether an observation window is open; `start_observation`/
`stop_observation` only bracket what additionally gets fed into the
`TimelineRecorder`. All shared state is guarded by a single `threading.Lock`
inside `_run`, `start_observation`, `stop_observation`, and
`get_current_state`. A per-frame exception (bad decode, wrong ROI/frame
shape) is caught inside `_run` and counted in `dropped_frames` rather than
killing the worker thread.

`ObservationResult.verification` is populated automatically on
`stop_observation()` only when `expected_sequence` was passed to
`start_observation()`; otherwise call `sequence.verify_sequence` (or the
equivalent `LEDObserverService.verify_sequence` static method) directly
against `result.segments`.
