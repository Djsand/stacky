from __future__ import annotations

import base64
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any


BODY_COMMAND_TYPES = {
    "audio.end",
    "audio.hold",
    "audio.input_gain",
    "audio.raw",
    "audio.stop",
    "audio.start",
    "audio.chunk",
    "audio.out",
    "audio.tone",
    "audio.volume",
    "display.brightness",
    "body.set_expression",
    "body.look_at",
    "body.gesture",
    "body.i2c_scan",
    "body.leds",
    "body.motion_config",
    "body.status",
    "vision.capture",
    "mobility.intent",
}

BODY_EVENT_TYPES = {
    "audio.chunk_ack",
    "audio.play_done",
    "audio.in",
    "touch",
    "battery",
    "imu",
    "i2c.scan",
    "proximity",
    "status",
    "vision.frame",
}


@dataclass(frozen=True)
class BodyCommand:
    type: str
    payload: dict[str, Any] = field(default_factory=dict)
    command_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    ts: float = field(default_factory=time.time)

    def to_json(self) -> str:
        if self.type not in BODY_COMMAND_TYPES:
            raise ValueError(f"Unknown body command type: {self.type}")
        return json.dumps(
            {
                "type": self.type,
                "commandId": self.command_id,
                "ts": self.ts,
                "payload": self.payload,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )


@dataclass(frozen=True)
class BodyEvent:
    type: str
    payload: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)

    @classmethod
    def from_json(cls, raw: str) -> "BodyEvent":
        data = json.loads(raw)
        event_type = str(data.get("type", ""))
        if event_type not in BODY_EVENT_TYPES:
            raise ValueError(f"Unknown body event type: {event_type}")
        payload = data.get("payload", {})
        if not isinstance(payload, dict):
            raise ValueError("Body event payload must be an object")
        return cls(event_type, payload, float(data.get("ts", time.time())))


def expression(name: str, *, intensity: float = 1.0) -> BodyCommand:
    return BodyCommand("body.set_expression", {"name": name, "intensity": intensity})


def look_at(x: float, y: float, *, speed: int = 500) -> BodyCommand:
    return BodyCommand(
        "body.look_at",
        {
            "x": max(-1.0, min(1.0, float(x))),
            "y": max(-1.0, min(1.0, float(y))),
            "speed": max(0, min(1000, int(speed))),
        },
    )


def gesture(
    name: str,
    *,
    intensity: float = 1.0,
    speed: int = 500,
    base_x: float | None = None,
    base_y: float | None = None,
) -> BodyCommand:
    payload: dict[str, Any] = {
        "name": name,
        "intensity": max(0.0, min(1.0, float(intensity))),
        "speed": max(0, min(1000, int(speed))),
    }
    if base_x is not None:
        payload["baseX"] = max(-1.0, min(1.0, float(base_x)))
    if base_y is not None:
        payload["baseY"] = max(-1.0, min(1.0, float(base_y)))
    return BodyCommand("body.gesture", payload)


def leds(
    *,
    r: int = 0,
    g: int = 0,
    b: int = 0,
    brightness: float = 1.0,
    duration_ms: int = 300,
    side: str = "both",
    mode: str = "solid",
) -> BodyCommand:
    return BodyCommand(
        "body.leds",
        {
            "r": max(0, min(255, int(r))),
            "g": max(0, min(255, int(g))),
            "b": max(0, min(255, int(b))),
            "brightness": max(0.0, min(1.0, float(brightness))),
            "durationMs": max(0, min(5000, int(duration_ms))),
            "side": side if side in {"left", "right", "both"} else "both",
            "mode": mode if mode in {"solid", "pulse", "off"} else "solid",
        },
    )


def motion_config(
    *,
    center_yaw: int,
    center_pitch: int,
    yaw_range: int = 720,
    look_up_range: int = 520,
    look_down_range: int = 220,
) -> BodyCommand:
    return BodyCommand(
        "body.motion_config",
        {
            "centerYaw": max(-1280, min(1280, int(center_yaw))),
            "centerPitch": max(30, min(870, int(center_pitch))),
            "yawRange": max(0, min(1280, int(yaw_range))),
            "lookUpRange": max(0, min(870, int(look_up_range))),
            "lookDownRange": max(0, min(870, int(look_down_range))),
        },
    )


def speak_audio(pcm: bytes, *, sample_rate: int = 24000, channels: int = 1) -> BodyCommand:
    return BodyCommand(
        "audio.out",
        {
            "encoding": "pcm16le",
            "sampleRate": sample_rate,
            "channels": channels,
            "data": base64.b64encode(pcm).decode("ascii"),
        },
    )


def audio_start(*, sample_rate: int = 24000, channels: int = 1, total_bytes: int = 0) -> BodyCommand:
    return BodyCommand(
        "audio.start",
        {
            "encoding": "pcm16le",
            "sampleRate": sample_rate,
            "channels": channels,
            "totalBytes": total_bytes,
        },
    )


def audio_chunk(pcm: bytes, *, seq: int) -> BodyCommand:
    return BodyCommand(
        "audio.chunk",
        {
            "encoding": "pcm16le",
            "seq": seq,
            "data": base64.b64encode(pcm).decode("ascii"),
        },
    )


def audio_end() -> BodyCommand:
    return BodyCommand("audio.end", {})


def stop_audio() -> BodyCommand:
    return BodyCommand("audio.stop", {})


def hold_audio(*, active: bool) -> BodyCommand:
    return BodyCommand("audio.hold", {"active": active})


def speaker_tone(*, frequency: int = 880, duration_ms: int = 180) -> BodyCommand:
    return BodyCommand(
        "audio.tone",
        {
            "frequency": frequency,
            "durationMs": duration_ms,
        },
    )


def speaker_volume(level: int) -> BodyCommand:
    return BodyCommand("audio.volume", {"level": max(0, min(100, int(level)))})


def mic_input_gain(level: int) -> BodyCommand:
    return BodyCommand("audio.input_gain", {"level": max(0, min(100, int(level)))})


def display_brightness(level: int, *, permanent: bool = True) -> BodyCommand:
    return BodyCommand(
        "display.brightness",
        {
            "level": max(1, min(100, int(level))),
            "permanent": bool(permanent),
        },
    )


def body_status() -> BodyCommand:
    return BodyCommand("body.status", {})


def i2c_scan() -> BodyCommand:
    return BodyCommand("body.i2c_scan", {})


def vision_capture(
    *,
    width: int = 320,
    height: int = 240,
    format: str = "jpeg",
    quality: int = 50,
    discard_frames: int = 4,
    settle_ms: int = 30,
    ae_level: int = 2,
    sensor_gain: int | None = 30,
    sensor_exposure: int | None = 1200,
) -> BodyCommand:
    payload: dict[str, Any] = {
        "width": max(64, min(1280, int(width))),
        "height": max(64, min(720, int(height))),
        "format": format,
        "quality": max(5, min(80, int(quality))),
        "discardFrames": max(0, min(12, int(discard_frames))),
        "settleMs": max(0, min(250, int(settle_ms))),
        "aeLevel": max(-2, min(2, int(ae_level))),
    }
    if sensor_gain is not None:
        payload["sensorGain"] = max(0, min(30, int(sensor_gain)))
    if sensor_exposure is not None:
        payload["sensorExposure"] = max(0, min(1200, int(sensor_exposure)))
    return BodyCommand("vision.capture", payload)


def decode_pcm_payload(payload: dict[str, Any]) -> tuple[bytes, int, int]:
    encoding = str(payload.get("encoding", ""))
    if encoding != "pcm16le":
        raise ValueError(f"Unsupported audio encoding: {encoding}")
    sample_rate = int(payload.get("sampleRate", 0))
    channels = int(payload.get("channels", 1))
    pcm = payload.get("pcm")
    if isinstance(pcm, bytes):
        return pcm, sample_rate, channels
    if isinstance(pcm, bytearray):
        return bytes(pcm), sample_rate, channels
    data = payload.get("data", "")
    if not isinstance(data, str):
        raise ValueError("Audio payload must contain raw PCM bytes or base64 text")
    return base64.b64decode(data), sample_rate, channels


def decode_vision_frame_payload(payload: dict[str, Any]) -> bytes:
    if not bool(payload.get("available", False)):
        reason = str(payload.get("reason", "camera_unavailable"))
        raise ValueError(f"Vision frame unavailable: {reason}")
    encoding = str(payload.get("encoding", ""))
    if encoding != "base64":
        raise ValueError(f"Unsupported vision frame encoding: {encoding}")
    data = payload.get("data", "")
    if not isinstance(data, str):
        raise ValueError("Vision frame payload must contain base64 text")
    return base64.b64decode(data)


def mobility_intent(direction: str, *, speed: float = 0.0, enabled: bool = False) -> BodyCommand:
    return BodyCommand(
        "mobility.intent",
        {
            "direction": direction,
            "speed": speed,
            "enabled": enabled,
            "reason": "Wheels are protocol-reserved but disabled until physical calibration.",
        },
    )
