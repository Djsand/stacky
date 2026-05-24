from __future__ import annotations

import time
from dataclasses import dataclass

from .calibration import BodyCalibration
from .controller import StackChanBodyController


FACE_LOCK_REPLY_HOLD_SECONDS = 4.0


@dataclass(frozen=True)
class MotionPlan:
    name: str
    intensity: float
    speed: int


@dataclass(frozen=True)
class LedPlan:
    r: int
    g: int
    b: int
    brightness: float
    duration_ms: int = 350
    mode: str = "solid"


class BodyDirector:
    """Small motion layer for a present but restrained Stacky body."""

    def __init__(self, controller: StackChanBodyController, calibration: BodyCalibration) -> None:
        self.controller = controller
        self.calibration = calibration.clamp()
        self._last_motion_at = 0.0
        self._last_face_track_at = 0.0
        self._face_x: float | None = None
        self._face_y: float | None = None
        self._face_command_x = 0.0
        self._face_command_y = 0.0
        self._face_lock_count = 0
        self._last_touch_at = 0.0
        self._last_presence_at = 0.0
        self._presence_index = 0
        self._presence_mode = "stille_ven"
        self._stacky_mood = "rolig"

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
        ok = self._led_for_state(name) and ok
        now = time.monotonic()
        if name == "listening":
            ok = self._motion("center", intensity=0.14, speed=170, cooldown=4.0, now=now) and ok
        return ok

    def set_presence_mode(self, mode: str) -> None:
        clean = mode.strip().lower().replace("-", "_").replace(" ", "_")
        self._presence_mode = clean or "stille_ven"

    def set_stacky_mood(self, mood: str) -> None:
        self._stacky_mood = mood.strip().lower() or "rolig"

    def reply_started(self, text: str, *, now: float | None = None) -> bool:
        now = now if now is not None else time.monotonic()
        self._led_for_state("speaking")
        plan = self.plan_reply_motion(text, now=now)
        if plan is None:
            return True
        base_x, base_y = self._face_motion_base(now)
        return self._motion(
            plan.name,
            intensity=plan.intensity,
            speed=plan.speed,
            cooldown=0.0,
            now=now,
            base_x=base_x,
            base_y=base_y,
        )

    def reply_completed(self, text: str, *, now: float | None = None) -> bool:
        now = now if now is not None else time.monotonic()
        plan = self.plan_reply_motion(text, now=now)
        if plan is None:
            return True
        base_x, base_y = self._face_motion_base(now)
        return self._motion(
            plan.name,
            intensity=plan.intensity,
            speed=plan.speed,
            cooldown=0.0,
            now=now,
            base_x=base_x,
            base_y=base_y,
        )

    def plan_reply_motion(self, text: str, *, now: float | None = None) -> MotionPlan | None:
        now = now if now is not None else time.monotonic()
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
        if any(token in lowered for token in ("hm", "nå", "naa", "windows", "maskinrummet", "forhænget", "forhaenget")):
            return MotionPlan("look_up", intensity=0.10, speed=165)
        if len(lowered) > 80 and now - self._last_motion_at > 3.0:
            return MotionPlan("nod", intensity=0.12, speed=180)
        return None

    def gesture(self, name: str, *, intensity: float = 1.0, speed: int = 500) -> bool:
        self._last_motion_at = time.monotonic()
        return self.controller.gesture(name, intensity=intensity, speed=speed)

    def handle_touch(self, payload: dict[str, object] | None = None, *, now: float | None = None) -> bool:
        now = now if now is not None else time.monotonic()
        if now - self._last_touch_at < 0.45:
            return True
        self._last_touch_at = now
        kind = str((payload or {}).get("gesture", (payload or {}).get("kind", "press"))).lower()
        ok = self.controller.set_expression("happy")
        ok = self._set_leds(r=120, g=255, b=170, brightness=0.55, duration_ms=220, mode="pulse") and ok
        if "swipe" in kind:
            ok = self._motion("look_up", intensity=0.18, speed=150, cooldown=0.0, now=now) and ok
        elif kind != "release":
            ok = self._motion("nod", intensity=0.18, speed=150, cooldown=0.0, now=now) and ok
        return ok

    def handle_proximity(self, payload: dict[str, object] | None = None, *, now: float | None = None) -> bool:
        now = now if now is not None else time.monotonic()
        distance = _payload_float(payload or {}, "distanceMm")
        if distance is None:
            distance = _payload_float(payload or {}, "distance")
        if distance is None or distance > 220:
            return True
        ok = self._set_leds(r=100, g=170, b=255, brightness=0.40, duration_ms=260, mode="pulse")
        return self._motion("look_up", intensity=0.14, speed=145, cooldown=1.2, now=now) and ok

    def presence_tick(self, state: str, *, now: float | None = None) -> bool:
        """Small autonomous body presence when no fresher face tracking is driving the head."""

        now = now if now is not None else time.monotonic()
        state = state.lower().strip()
        if now - self._last_face_track_at < 0.90:
            return True
        if now - self._last_motion_at < 0.75:
            return True
        if self._presence_mode == "ikke_forstyr":
            if state == "listening" and now - self._last_presence_at >= 8.0:
                self._last_presence_at = now
                return self._led_for_state("listening")
            return True

        if state == "thinking":
            if now - self._last_presence_at < 1.25:
                return True
            self._last_presence_at = now
            self._presence_index += 1
            if self._presence_index % 3 == 0:
                return self._motion("nod", intensity=0.10, speed=135, cooldown=0.0, now=now)
            return self._motion("look_up", intensity=0.10, speed=135, cooldown=0.0, now=now)

        if state in {"happy", "speaking"}:
            if now - self._last_presence_at < 1.80:
                return True
            self._last_presence_at = now
            self._presence_index += 1
            if self._presence_index % 2 == 0:
                return self._motion("nod", intensity=0.08, speed=125, cooldown=0.0, now=now)
            return self._set_leds(r=120, g=180, b=110, brightness=0.24, duration_ms=260, mode="solid")

        if state == "listening" and now - self._last_presence_at >= 4.0:
            self._last_presence_at = now
            if self._presence_mode == "vaagen_makker":
                self._presence_index += 1
                if self._presence_index % 3 == 0:
                    return self._motion("look_up", intensity=0.08, speed=120, cooldown=0.0, now=now)
            return self._led_for_state("listening")
        return True

    def track_face(
        self,
        x: float,
        y: float,
        *,
        confidence: float = 1.0,
        area: float | None = None,
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
        command_x = max(-0.56, min(0.56, float(x) * 0.62))
        command_y = max(-0.38, min(0.38, float(y) * 0.48))
        self._last_face_track_at = now
        self._last_motion_at = now
        self._face_command_x = command_x
        self._face_command_y = command_y
        return self.controller.look_at(command_x, command_y, speed=max(70, min(150, int(speed))))

    def _motion(
        self,
        name: str,
        *,
        intensity: float,
        speed: int,
        cooldown: float,
        now: float | None = None,
        base_x: float | None = None,
        base_y: float | None = None,
    ) -> bool:
        now = now if now is not None else time.monotonic()
        if now - self._last_motion_at < cooldown:
            return True
        self._last_motion_at = now
        return self.controller.gesture(name, intensity=intensity, speed=speed, base_x=base_x, base_y=base_y)

    def _led_for_state(self, name: str) -> bool:
        plans = {
            "listening": LedPlan(35, 80, 130, 0.22, duration_ms=550),
            "thinking": LedPlan(160, 120, 35, 0.32, duration_ms=300),
            "happy": LedPlan(70, 190, 120, 0.34, duration_ms=320),
            "speaking": LedPlan(130, 180, 120, 0.30, duration_ms=220),
            "connected": LedPlan(70, 140, 255, 0.30, duration_ms=400),
            "error": LedPlan(220, 50, 45, 0.45, duration_ms=250, mode="pulse"),
        }
        if self._presence_mode == "agent_vagt" and name == "thinking":
            plans["thinking"] = LedPlan(145, 95, 190, 0.34, duration_ms=300)
        if self._stacky_mood == "vagt" and name in {"listening", "thinking"}:
            plans[name] = LedPlan(190, 125, 55, 0.32, duration_ms=320, mode="pulse")
        if self._presence_mode == "ikke_forstyr" and name == "listening":
            plans["listening"] = LedPlan(25, 45, 70, 0.12, duration_ms=700)
        plan = plans.get(name)
        if plan is None:
            return True
        return self._set_leds(
            r=plan.r,
            g=plan.g,
            b=plan.b,
            brightness=plan.brightness,
            duration_ms=plan.duration_ms,
            mode=plan.mode,
        )

    def _set_leds(
        self,
        *,
        r: int,
        g: int,
        b: int,
        brightness: float,
        duration_ms: int,
        mode: str = "solid",
    ) -> bool:
        set_leds = getattr(self.controller, "set_leds", None)
        if set_leds is None:
            return True
        return bool(
            set_leds(
                r=r,
                g=g,
                b=b,
                brightness=brightness,
                duration_ms=duration_ms,
                mode=mode,
            )
        )

    def _has_recent_face_lock(self, now: float) -> bool:
        return self._last_face_track_at > 0.0 and now - self._last_face_track_at < FACE_LOCK_REPLY_HOLD_SECONDS

    def _face_motion_base(self, now: float) -> tuple[float | None, float | None]:
        if not self._has_recent_face_lock(now):
            return None, None
        return self._face_command_x, self._face_command_y


def _payload_float(payload: dict[str, object], key: str) -> float | None:
    value = payload.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
