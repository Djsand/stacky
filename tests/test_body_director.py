from __future__ import annotations

import unittest

from stacky.body.calibration import BodyCalibration
from stacky.body.director import BodyDirector


class FakeDirectorController:
    def __init__(self) -> None:
        self.configs: list[dict[str, int]] = []
        self.expressions: list[str] = []
        self.gestures: list[tuple[str, float, int]] = []

    def configure_motion(
        self,
        *,
        center_yaw: int,
        center_pitch: int,
        yaw_range: int = 720,
        look_up_range: int = 520,
        look_down_range: int = 220,
    ) -> bool:
        self.configs.append(
            {
                "center_yaw": center_yaw,
                "center_pitch": center_pitch,
                "yaw_range": yaw_range,
                "look_up_range": look_up_range,
                "look_down_range": look_down_range,
            }
        )
        return True

    def set_expression(self, name: str, *, intensity: float = 1.0) -> bool:
        self.expressions.append(name)
        return True

    def gesture(self, name: str, *, intensity: float = 1.0, speed: int = 500) -> bool:
        self.gestures.append((name, intensity, speed))
        return True


class BodyDirectorTest(unittest.TestCase):
    def test_apply_calibration_sends_motion_config(self) -> None:
        fake = FakeDirectorController()
        director = BodyDirector(fake, BodyCalibration(center_yaw=125, center_pitch=255))  # type: ignore[arg-type]

        self.assertTrue(director.apply_calibration())

        self.assertEqual(fake.configs[0]["center_yaw"], 125)
        self.assertEqual(fake.configs[0]["center_pitch"], 255)

    def test_set_state_updates_expression_and_small_motion(self) -> None:
        fake = FakeDirectorController()
        director = BodyDirector(fake, BodyCalibration())  # type: ignore[arg-type]

        self.assertTrue(director.set_state("thinking"))

        self.assertEqual(fake.expressions, ["thinking"])
        self.assertEqual(fake.gestures[0][0], "look_up")


if __name__ == "__main__":
    unittest.main()
