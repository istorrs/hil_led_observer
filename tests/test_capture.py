from __future__ import annotations

from unittest.mock import MagicMock, patch

import cv2
import pytest

from led_observer.capture import CameraUnavailableError, OpenCVCameraSource
from tests.mocks import make_frame


def test_missing_camera_device_raises_camera_unavailable_error() -> None:
    # No physical camera exists in CI; an implausible device index exercises
    # the same "device could not be opened" path a missing/unplugged camera
    # would hit, without needing hardware.
    with pytest.raises(CameraUnavailableError):
        OpenCVCameraSource(device_index=9999)


def _mock_capture(focus_start: float = 42.0) -> MagicMock:
    mock = MagicMock()
    mock.isOpened.return_value = True
    state = {"focus": focus_start, "autofocus": 1}

    def get(prop: int) -> float:
        if prop == cv2.CAP_PROP_FOCUS:
            return state["focus"]
        if prop == cv2.CAP_PROP_AUTOFOCUS:
            return state["autofocus"]
        return 0.0

    def set_(prop: int, value: float) -> bool:
        if prop == cv2.CAP_PROP_FOCUS:
            state["focus"] = value
        elif prop == cv2.CAP_PROP_AUTOFOCUS:
            state["autofocus"] = value
        return True

    mock.get.side_effect = get
    mock.set.side_effect = set_
    mock.read.return_value = (True, make_frame(True))
    return mock


def test_disable_hardware_autofocus_turns_off_and_rewrites_focus() -> None:
    with patch("cv2.VideoCapture", return_value=_mock_capture(focus_start=17.0)):
        source = OpenCVCameraSource(0)

    source.disable_hardware_autofocus()

    mock_capture = source._capture  # a MagicMock at runtime, despite the declared type
    calls = mock_capture.set.call_args_list  # type: ignore[attr-defined]
    assert (cv2.CAP_PROP_AUTOFOCUS, 0) in [c.args for c in calls]
    assert (cv2.CAP_PROP_FOCUS, 17.0) in [c.args for c in calls]


def test_autofocus_disables_hardware_af_and_returns_a_result() -> None:
    with patch("cv2.VideoCapture", return_value=_mock_capture()):
        source = OpenCVCameraSource(0)

    result = source.autofocus(focus_range=(0, 20), settle_s=0.0)

    assert result.focus == source._capture.get(cv2.CAP_PROP_FOCUS)  # noqa: SLF001
    assert source._capture.get(cv2.CAP_PROP_AUTOFOCUS) == 0  # noqa: SLF001
