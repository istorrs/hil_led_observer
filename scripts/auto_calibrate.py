#!/usr/bin/env python3
"""Compatibility wrapper for the installed calibration command."""

from led_observer.auto_calibrate import main

if __name__ == "__main__":
    raise SystemExit(main())
