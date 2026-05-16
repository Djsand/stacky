from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from stacky.body.calibration import BodyCalibration, load_body_calibration, save_body_calibration


class BodyCalibrationTest(unittest.TestCase):
    def test_load_defaults_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            calibration = load_body_calibration(Path(tmp))

        self.assertEqual(calibration.center_yaw, 90)
        self.assertEqual(calibration.center_pitch, 260)

    def test_save_and_load_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            save_body_calibration(root, BodyCalibration(center_yaw=130, center_pitch=245))

            calibration = load_body_calibration(root)

        self.assertEqual(calibration.center_yaw, 130)
        self.assertEqual(calibration.center_pitch, 245)

    def test_nudge_clamps_to_firmware_limits(self) -> None:
        calibration = BodyCalibration(center_yaw=1270, center_pitch=40).nudge(yaw_delta=50, pitch_delta=-50)

        self.assertEqual(calibration.center_yaw, 1280)
        self.assertEqual(calibration.center_pitch, 30)


if __name__ == "__main__":
    unittest.main()
