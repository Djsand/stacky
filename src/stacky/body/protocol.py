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
    "audio.raw",
    "audio.stop",
    "audio.start",
    "audio.chunk",
    "audio.out",
    "audio.tone",
    "body.set_expression",
    "body.look_at",
    "body.gesture",
    "body.leds",
    "body.status",
    "mobility.intent",
}

BODY_EVENT_TYPES = {
    "audio.chunk_ack",
    "audio.play_done",
    "audio.in",
    "touch",
    "battery",
    "imu",
    "proximity",
    "status",
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
