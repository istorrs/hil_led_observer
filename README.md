# led-observer

Optical LED state/cadence observer for Hardware-In-The-Loop (HIL) test
harnesses. Watches a DUT's status LED through a USB camera (via OpenCV),
classifies its ON/OFF cadence, and verifies an observed sequence of states
against an expected schema with timing tolerances -- no ML frameworks or GPU
required.

## Install

Install a release from GitHub without publishing to PyPI. Pinning a full commit
SHA gives the most reproducible result; a release tag is also supported.

In another project's `requirements.txt`:

```text
led-observer @ git+https://github.com/istorrs/hil_led_observer.git@v0.1.0
```

Or in its `pyproject.toml`:

```toml
[project]
dependencies = [
    "led-observer @ git+https://github.com/istorrs/hil_led_observer.git@v0.1.0",
]
```

For a public repository, the prebuilt wheel attached to the GitHub Release can
also be installed directly:

```text
led-observer @ https://github.com/istorrs/hil_led_observer/releases/download/v0.1.0/led_observer-0.1.0-py3-none-any.whl
```

The distribution name is `led-observer`; import it as `led_observer`. The Git
forms require Git and repository access. The wheel URL requires access to the
release asset. All forms install the runtime dependencies declared in
`pyproject.toml`.

For development in this checkout:

```bash
uv venv --python 3.10 .venv && source .venv/bin/activate
uv pip install -e ".[dev]"
```

## Releases

1. Update `project.version` in `pyproject.toml` to the next `MAJOR.MINOR.PATCH`
   version and commit the change.
2. Push the commit, then create and push a matching tag, for example
   `git tag v0.1.0` and `git push origin v0.1.0`.
3. The GitHub Actions workflow runs lint, type checks, and tests on Python
   3.10–3.12. It verifies that the tag matches `project.version`, builds a
   wheel and source archive, checks the wheel import, and attaches both files
   to a GitHub Release. No PyPI account or publishing credentials are needed.

The release files are available under the repository's **Releases** page. Use
the new tag or a full commit SHA in consuming projects when upgrading.

## Develop

```bash
ruff check .            # lint
ruff format .           # format
mypy                    # strict type check
pytest                  # unit + integration tests
pytest --cov            # with coverage report
python examples/hil_integration_demo.py   # runnable demo, no hardware needed
python scripts/record_frames.py --duration 60 --out recordings/run1 --autofocus  # record real hardware
python scripts/auto_calibrate.py --camera 0 --duration 6.0   # auto-locate + calibrate a live blinking LED
```

## Architecture

- `autofocus.py` -- `autofocus_sweep`: camera-agnostic software autofocus
  (maximizes variance-of-Laplacian sharpness via coarse scan + hill-climb),
  for cameras whose hardware autofocus "breathes" on a small LED target.
- `calibration.py` -- define an ROI, sample ON/OFF reference frames, derive
  luminosity (LAB-L) hysteresis thresholds and an HSV color-match gate, and
  persist the result as a `CalibrationProfile` (JSON).
- `detector.py` -- `InstantClassifier` turns one frame into an ON/OFF
  reading; `BlinkClassifier` turns a stream of readings into a
  `LEDState` (`OFF`/`ON`/`SLOW_BLINK`/`FAST_BLINK`/`SINGLE_FLASH`) over a
  sliding time-domain window. `estimate_frequency_fft` is the
  frequency-domain alternative for cross-checking.
- `capture.py` -- `FrameSource` protocol; `OpenCVCameraSource` wraps a real
  USB camera (tests substitute a synthetic source instead), with
  `disable_hardware_autofocus()` / `autofocus()` for real-camera focus
  control.
- `sequence.py` -- `TimelineRecorder` collapses a classification stream into
  contiguous segments; `PulseRecorder` keeps the full raw pulse (edge)
  history; `verify_sequence` diffs an observed timeline against an expected
  schema with per-step duration tolerance, or against a fixed
  `PulsePattern` (e.g. a firmware's "N-blink" confirmation code) for a
  short burst too brief for frequency classification to read reliably.
- `service.py` -- `LEDObserverService`: a thread-safe façade running the
  camera-read/classify loop on a background thread, exposing
  `start_observation` / `stop_observation` / `get_current_state`.
- `playback.py` -- record real hardware once (`scripts/record_frames.py`,
  lossless PNGs + a timestamp manifest, not lossy/temporal-compressed
  video), then `load_recording` + `replay_recording` feed it through the
  real classification pipeline offline, near-instantly, for tuning
  calibration/thresholds without repeating a live DUT trigger sequence.
  `RecordedFrameSource` replays a recording through the real threaded
  service instead, at its original pacing.

See `examples/hil_integration_demo.py` for the intended integration shape.
