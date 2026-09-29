"""Regression tests for public model behavior."""

from led_observer.models import LEDState


def test_led_state_string_representation() -> None:
    assert str(LEDState.OFF) == "LEDState.OFF"
