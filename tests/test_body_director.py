from __future__ import annotations

import unittest

from stacky.body.calibration import BodyCalibration
from stacky.body.director import BodyDirector


class FakeDirectorController:
    def __init__(self) -> None:
        self.configs: list[dict[str, int]] = []
        self.expressions: list[str] = []
        self.gestures: list[tuple[str, float, int]] = []
        self.gesture_bases: list[tuple[float | None, float | None]] = []
        self.looks: list[tuple[float, float, int]] = []
        self.leds: list[dict[str, object]] = []

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

    def gesture(
        self,
        name: str,
        *,
        intensity: float = 1.0,
        speed: int = 500,
        base_x: float | None = None,
        base_y: float | None = None,
    ) -> bool:
        self.gestures.append((name, intensity, speed))
        self.gesture_bases.append((base_x, base_y))
        return True

    def look_at(self, x: float, y: float, *, speed: int = 500) -> bool:
        self.looks.append((x, y, speed))
        return True

    def set_leds(
        self,
        *,
        r: int = 0,
        g: int = 0,
        b: int = 0,
        brightness: float = 1.0,
        duration_ms: int = 300,
        side: str = "both",
        mode: str = "solid",
    ) -> bool:
        self.leds.append(
            {
                "r": r,
                "g": g,
                "b": b,
                "brightness": brightness,
                "duration_ms": duration_ms,
                "side": side,
                "mode": mode,
            }
        )
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
        self.assertEqual(fake.leds[0]["r"], 160)

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
        self.assertEqual(fake.leds[0]["mode"], "solid")

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

    def test_reply_started_anchors_question_motion_to_recent_face_lock(self) -> None:
        fake = FakeDirectorController()
        director = BodyDirector(fake, BodyCalibration())  # type: ignore[arg-type]

        self.assertTrue(director.track_face(0.5, -0.4, confidence=0.8, now=10.0))
        self.assertTrue(director.reply_started("Skal jeg goere det saadan?", now=12.0))

        self.assertEqual(fake.looks, [(0.31, -0.192, 105)])
        self.assertEqual(fake.gestures, [("look_up", 0.14, 190)])
        self.assertEqual(fake.gesture_bases, [(0.31, -0.192)])
        self.assertEqual(fake.leds[-1]["r"], 130)

    def test_reply_started_anchors_agreement_nod_to_recent_face_lock(self) -> None:
        fake = FakeDirectorController()
        director = BodyDirector(fake, BodyCalibration())  # type: ignore[arg-type]

        self.assertTrue(director.track_face(-0.5, 0.2, confidence=0.8, now=10.0))
        self.assertTrue(director.reply_started("Okay, det giver mening.", now=13.0))

        self.assertEqual(fake.looks, [(-0.31, 0.096, 105)])
        self.assertEqual(fake.gestures, [("nod", 0.18, 220)])
        self.assertEqual(fake.gesture_bases, [(-0.31, 0.096)])
        self.assertEqual(fake.leds[-1]["mode"], "solid")

    def test_reply_started_uses_motion_after_face_lock_expires(self) -> None:
        fake = FakeDirectorController()
        director = BodyDirector(fake, BodyCalibration())  # type: ignore[arg-type]

        self.assertTrue(director.track_face(0.5, -0.4, confidence=0.8, now=10.0))
        self.assertTrue(director.reply_started("Skal jeg goere det saadan?", now=15.0))

        self.assertEqual(fake.gestures, [("look_up", 0.14, 190)])
        self.assertEqual(fake.gesture_bases, [(None, None)])

    def test_reply_completed_can_do_semantic_motion_without_speaking_led(self) -> None:
        fake = FakeDirectorController()
        director = BodyDirector(fake, BodyCalibration())  # type: ignore[arg-type]

        self.assertTrue(director.reply_completed("Det giver mening."))

        self.assertEqual(fake.gestures, [("nod", 0.18, 220)])
        self.assertEqual(fake.leds, [])


    def test_track_face_moves_gently_toward_off_center_face(self) -> None:
        fake = FakeDirectorController()
        director = BodyDirector(fake, BodyCalibration())  # type: ignore[arg-type]

        self.assertTrue(director.track_face(0.5, -0.4, confidence=0.8, now=10.0))
        self.assertEqual(fake.looks, [(0.31, -0.192, 105)])
        self.assertEqual(director.last_motion_at, 10.0)

    def test_track_face_ignores_centered_face(self) -> None:
        fake = FakeDirectorController()
        director = BodyDirector(fake, BodyCalibration())  # type: ignore[arg-type]

        self.assertTrue(director.track_face(0.04, -0.03, confidence=0.8, now=10.0))

        self.assertEqual(fake.looks, [])
        self.assertEqual(director.last_motion_at, 0.0)

    def test_track_face_uses_slow_cooldown(self) -> None:
        fake = FakeDirectorController()
        director = BodyDirector(fake, BodyCalibration())  # type: ignore[arg-type]

        self.assertTrue(director.track_face(0.42, 0.12, confidence=0.84, now=10.0))
        self.assertTrue(director.track_face(-0.82, -0.55, confidence=0.84, now=10.9))

        self.assertEqual(len(fake.looks), 1)
        self.assertGreater(fake.looks[0][0], 0.0)

    def test_track_face_allows_next_move_after_cooldown(self) -> None:
        fake = FakeDirectorController()
        director = BodyDirector(fake, BodyCalibration())  # type: ignore[arg-type]

        self.assertTrue(director.track_face(0.42, 0.12, confidence=0.84, now=10.0))
        self.assertTrue(director.track_face(-0.82, -0.55, confidence=0.84, now=11.5))

        self.assertEqual(len(fake.looks), 2)

    def test_presence_tick_adds_small_thinking_motion(self) -> None:
        fake = FakeDirectorController()
        director = BodyDirector(fake, BodyCalibration())  # type: ignore[arg-type]

        self.assertTrue(director.presence_tick("thinking", now=10.0))

        self.assertEqual(fake.gestures, [("look_up", 0.10, 135)])

    def test_presence_tick_defers_to_recent_face_tracking(self) -> None:
        fake = FakeDirectorController()
        director = BodyDirector(fake, BodyCalibration())  # type: ignore[arg-type]

        self.assertTrue(director.track_face(0.4, -0.2, confidence=0.84, now=10.0))
        self.assertTrue(director.presence_tick("thinking", now=10.4))

        self.assertEqual(fake.gestures, [])
        self.assertEqual(len(fake.looks), 1)

    def test_presence_tick_uses_led_only_for_listening_idle(self) -> None:
        fake = FakeDirectorController()
        director = BodyDirector(fake, BodyCalibration())  # type: ignore[arg-type]

        self.assertTrue(director.presence_tick("listening", now=10.0))

        self.assertEqual(fake.gestures, [])
        self.assertEqual(fake.leds[-1]["r"], 35)

    def test_touch_reaction_uses_small_feedback(self) -> None:
        fake = FakeDirectorController()
        director = BodyDirector(fake, BodyCalibration())  # type: ignore[arg-type]

        self.assertTrue(director.handle_touch({"gesture": "press"}, now=10.0))

        self.assertEqual(fake.expressions, ["happy"])
        self.assertEqual(fake.gestures, [("nod", 0.18, 150)])
        self.assertEqual(fake.leds[0]["mode"], "pulse")


if __name__ == "__main__":
    unittest.main()
