from __future__ import annotations

import unittest

from stacky.body.calibration import BodyCalibration
from stacky.body.director import BodyDirector


class FakeDirectorController:
    def __init__(self) -> None:
        self.configs: list[dict[str, int]] = []
        self.expressions: list[str] = []
        self.gestures: list[tuple[str, float, int]] = []
        self.looks: list[tuple[float, float, int]] = []

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

    def look_at(self, x: float, y: float, *, speed: int = 500) -> bool:
        self.looks.append((x, y, speed))
        return True


class BodyDirectorTest(unittest.TestCase):
    def test_apply_calibration_sends_motion_config(self) -> None:
        fake = FakeDirectorController()
        director = BodyDirector(fake, BodyCalibration(center_yaw=125, center_pitch=255))  # type: ignore[arg-type]

        self.assertTrue(director.apply_calibration())

        self.assertEqual(fake.configs[0]["center_yaw"], 125)
        self.assertEqual(fake.configs[0]["center_pitch"], 255)

    def test_thinking_updates_expression_without_twitch_motion(self) -> None:
        fake = FakeDirectorController()
        director = BodyDirector(fake, BodyCalibration())  # type: ignore[arg-type]

        self.assertTrue(director.set_state("thinking"))

        self.assertEqual(fake.expressions, ["thinking"])
        self.assertEqual(fake.gestures, [])

    def test_happy_only_updates_expression(self) -> None:
        fake = FakeDirectorController()
        director = BodyDirector(fake, BodyCalibration())  # type: ignore[arg-type]

        self.assertTrue(director.set_state("happy"))

        self.assertEqual(fake.expressions, ["happy"])
        self.assertEqual(fake.gestures, [])

    def test_reply_started_uses_small_contextual_motion(self) -> None:
        fake = FakeDirectorController()
        director = BodyDirector(fake, BodyCalibration())  # type: ignore[arg-type]

        self.assertTrue(director.reply_started("Det giver mening, jeg gør det."))

        self.assertEqual(fake.gestures, [("nod", 0.18, 220)])

    def test_reply_started_can_signal_uncertainty(self) -> None:
        fake = FakeDirectorController()
        director = BodyDirector(fake, BodyCalibration())  # type: ignore[arg-type]

        self.assertTrue(director.reply_started("Beklager, jeg misforstod dig."))

        self.assertEqual(fake.gestures, [("shake", 0.14, 210)])

    def test_reply_started_can_signal_question(self) -> None:
        fake = FakeDirectorController()
        director = BodyDirector(fake, BodyCalibration())  # type: ignore[arg-type]

        self.assertTrue(director.reply_started("Skal jeg gøre det sådan?"))

        self.assertEqual(fake.gestures, [("look_up", 0.14, 190)])


    def test_track_face_moves_gently_toward_off_center_face(self) -> None:
        fake = FakeDirectorController()
        director = BodyDirector(fake, BodyCalibration())  # type: ignore[arg-type]

        self.assertTrue(director.track_face(0.5, -0.4, confidence=0.8, now=10.0))

        self.assertEqual(fake.looks, [(0.35, -0.22000000000000003, 145)])

    def test_track_face_ignores_centered_face(self) -> None:
        fake = FakeDirectorController()
        director = BodyDirector(fake, BodyCalibration())  # type: ignore[arg-type]

        self.assertTrue(director.track_face(0.04, -0.03, confidence=0.8, now=10.0))

        self.assertEqual(fake.looks, [])


if __name__ == "__main__":
    unittest.main()
