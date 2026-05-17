from __future__ import annotations

import time
from dataclasses import dataclass

from .calibration import BodyCalibration
from .controller import StackChanBodyController


@dataclass(frozen=True)
class MotionPlan:
    name: str
    intensity: float
    speed: int


class BodyDirector:
    """Small motion layer for a present but restrained Stacky body."""

    def __init__(self, controller: StackChanBodyController, calibration: BodyCalibration) -> None:
        self.controller = controller
        self.calibration = calibration.clamp()
        self._last_motion_at = 0.0
        self._last_face_track_at = 0.0

    @property
    def last_motion_at(self) -> float:
        return self._last_motion_at

    def apply_calibration(self) -> bool:
        return self.controller.configure_motion(
            center_yaw=self.calibration.center_yaw,
            center_pitch=self.calibration.center_pitch,
            yaw_range=self.calibration.yaw_range,
            look_up_range=self.calibration.look_up_range,
            look_down_range=self.calibration.look_down_range,
        )

    def update_calibration(self, calibration: BodyCalibration) -> bool:
        self.calibration = calibration.clamp()
        return self.apply_calibration()

    def set_state(self, name: str) -> bool:
        ok = self.controller.set_expression(name)
        now = time.monotonic()
        if name == "listening":
            ok = self._motion("center", intensity=0.14, speed=170, cooldown=4.0, now=now) and ok
        return ok

    def reply_started(self, text: str) -> bool:
        plan = self.plan_reply_motion(text)
        if plan is None:
            return True
        return self._motion(plan.name, intensity=plan.intensity, speed=plan.speed, cooldown=0.0)

    def plan_reply_motion(self, text: str) -> MotionPlan | None:
        lowered = text.lower()
        if any(token in lowered for token in ("beklager", "ikke helt", "kan ikke", "misforstod", "ikke sikker")):
            return MotionPlan("shake", intensity=0.14, speed=210)
        if any(
            token in lowered
            for token in (
                "det giver mening",
                "du har ret",
                "enig",
                "klart",
                "okay",
                "fedt",
                "godt",
                "modtaget",
            )
        ):
            return MotionPlan("nod", intensity=0.18, speed=220)
        if "?" in text:
            return MotionPlan("look_up", intensity=0.14, speed=190)
        if len(lowered) > 80 and time.monotonic() - self._last_motion_at > 3.0:
            return MotionPlan("nod", intensity=0.12, speed=180)
        return None

    def gesture(self, name: str, *, intensity: float = 1.0, speed: int = 500) -> bool:
        self._last_motion_at = time.monotonic()
        return self.controller.gesture(name, intensity=intensity, speed=speed)

    def track_face(
        self,
        x: float,
        y: float,
        *,
        confidence: float = 1.0,
        speed: int = 105,
        now: float | None = None,
    ) -> bool:
        """Gently keep the head oriented toward a detected face."""

        if confidence < 0.50:
            return True
        now = now if now is not None else time.monotonic()
        if now - self._last_face_track_at < 1.35:
            return True
        if max(abs(x), abs(y)) < 0.14:
            return True
        self._last_face_track_at = now
        self._last_motion_at = now
        return self.controller.look_at(
            max(-0.56, min(0.56, float(x) * 0.62)),
            max(-0.38, min(0.38, float(y) * 0.48)),
            speed=max(70, min(150, int(speed))),
        )

    def _motion(
        self,
        name: str,
        *,
        intensity: float,
        speed: int,
        cooldown: float,
        now: float | None = None,
    ) -> bool:
        now = now if now is not None else time.monotonic()
        if now - self._last_motion_at < cooldown:
            return True
        self._last_motion_at = now
        return self.controller.gesture(name, intensity=intensity, speed=speed)
