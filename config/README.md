# config

Saved `CalibrationProfile` JSON files (ROI + luminosity thresholds + HSV
bounds), one per DUT/rig setup where the camera framing or LED color
differs. Produced by `CalibrationProfile.save(path)` after running
`build_calibration_profile` against ON/OFF reference frames captured on that
rig; loaded back with `CalibrationProfile.load(path)`.

`calibration.example.json` was generated from synthetic frames purely to
show the file's shape -- it is not a real bench calibration and shouldn't be
loaded against actual hardware.
