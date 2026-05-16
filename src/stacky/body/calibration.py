from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class BodyCalibration:
    center_yaw: int = 90
    center_pitch: int = 260
    yaw_range: int = 720
    look_up_range: int = 520
    look_down_range: int = 220

    def clamp(self) -> "BodyCalibration":
        return BodyCalibration(
            center_yaw=_clamp(self.center_yaw, -1280, 1280),
            center_pitch=_clamp(self.center_pitch, 30, 870),
            yaw_range=_clamp(self.yaw_range, 0, 1280),
            look_up_range=_clamp(self.look_up_range, 0, 870),
            look_down_range=_clamp(self.look_down_range, 0, 870),
        )

    def nudge(self, *, yaw_delta: int = 0, pitch_delta: int = 0) -> "BodyCalibration":
        return BodyCalibration(
            center_yaw=self.center_yaw + yaw_delta,
            center_pitch=self.center_pitch + pitch_delta,
            yaw_range=self.yaw_range,
            look_up_range=self.look_up_range,
            look_down_range=self.look_down_range,
        ).clamp()


def calibration_path(data_dir: Path) -> Path:
    return data_dir / "body_calibration.json"


def load_body_calibration(data_dir: Path) -> BodyCalibration:
    path = calibration_path(data_dir)
    if not path.exists():
        return BodyCalibration()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return BodyCalibration()
    if not isinstance(raw, dict):
        return BodyCalibration()
    return BodyCalibration(
        center_yaw=int(raw.get("center_yaw", BodyCalibration.center_yaw)),
        center_pitch=int(raw.get("center_pitch", BodyCalibration.center_pitch)),
        yaw_range=int(raw.get("yaw_range", BodyCalibration.yaw_range)),
        look_up_range=int(raw.get("look_up_range", BodyCalibration.look_up_range)),
        look_down_range=int(raw.get("look_down_range", BodyCalibration.look_down_range)),
    ).clamp()


def save_body_calibration(data_dir: Path, calibration: BodyCalibration) -> Path:
    path = calibration_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(calibration.clamp()), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, int(value)))
