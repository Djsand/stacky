from __future__ import annotations

import time

from .calibration import BodyCalibration
from .controller import StackChanBodyController


class BodyDirector:
    """Small motion layer for a present but restrained Stacky body."""

    def __init__(self, controller: StackChanBodyController, calibration: BodyCalibration) -> None:
        self.controller = controller
        self.calibration = calibration.clamp()
        self._last_motion_at = 0.0

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
        if name == "thinking":
            ok = self._motion("look_up", intensity=0.22, speed=220, cooldown=0.7, now=now) and ok
        elif name == "listening":
            ok = self._motion("center", intensity=0.45, speed=220, cooldown=2.5, now=now) and ok
        elif name == "happy":
            ok = self._motion("nod", intensity=0.28, speed=320, cooldown=1.2, now=now) and ok
        return ok

    def reply_started(self, text: str) -> bool:
        if len(text.strip()) < 12:
            return True
        return self._motion("nod", intensity=0.22, speed=260, cooldown=1.4)

    def gesture(self, name: str, *, intensity: float = 1.0, speed: int = 500) -> bool:
        self._last_motion_at = time.monotonic()
        return self.controller.gesture(name, intensity=intensity, speed=speed)

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
