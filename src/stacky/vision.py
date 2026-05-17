from __future__ import annotations

import time
import base64
from dataclasses import dataclass
from io import BytesIO
from typing import Any

from .body.protocol import decode_vision_frame_payload


@dataclass(frozen=True)
class FaceObservation:
    """Normalized face position from StackChan's camera.

    x is left/right in [-1, 1], where +1 is image right.
    y is up/down in [-1, 1], where +1 is image top.
    """

    x: float
    y: float
    width: float
    height: float
    confidence: float = 1.0
    identity: str | None = None

    @property
    def area(self) -> float:
        return max(0.0, min(1.0, self.width * self.height))


@dataclass(frozen=True)
class VisionSnapshot:
    captured_at: float
    width: int = 0
    height: int = 0
    faces: tuple[FaceObservation, ...] = ()
    detector: str = "none"
    error: str | None = None

    @property
    def primary_face(self) -> FaceObservation | None:
        return self.faces[0] if self.faces else None

    def prompt_context(self, *, max_age_seconds: float = 8.0, now: float | None = None) -> str:
        if self.error:
            return ""
        now = time.monotonic() if now is None else now
        if now - self.captured_at > max_age_seconds:
            return ""
        face = self.primary_face
        if face is None:
            return (
                "Visuel kontekst fra Stackys kamera: Jeg kan ikke se et tydeligt ansigt lige nu. "
                "Brug det kun som baggrund; det er ikke noget Nicolai har sagt."
            )
        subject = face.identity or "et ansigt"
        return (
            "Visuel kontekst fra Stackys kamera: Jeg kan se "
            f"{subject} {self._position_text(face)} og {self._distance_text(face)}. "
            "Identitet er ukendt medmindre den er eksplicit angivet. "
            "Brug det kun lavmaelt som baggrund; det er ikke noget Nicolai har sagt."
        )

    @staticmethod
    def _position_text(face: FaceObservation) -> str:
        horizontal = ""
        vertical = ""
        if face.x < -0.35:
            horizontal = "til venstre i billedet"
        elif face.x > 0.35:
            horizontal = "til hojre i billedet"
        elif face.x < -0.12:
            horizontal = "lidt til venstre"
        elif face.x > 0.12:
            horizontal = "lidt til hojre"
        else:
            horizontal = "omtrent midt i billedet"

        if face.y > 0.25:
            vertical = "lidt hojt"
        elif face.y < -0.25:
            vertical = "lidt lavt"
        if not vertical:
            return horizontal
        return f"{horizontal}, {vertical}"

    @staticmethod
    def _distance_text(face: FaceObservation) -> str:
        if face.area >= 0.16:
            return "taet paa kameraet"
        if face.area >= 0.05:
            return "i normal afstand"
        return "ret langt fra kameraet"


class HaarFaceDetector:
    def __init__(self) -> None:
        self._cv2: Any | None = None
        self._np: Any | None = None
        self._cascades: list[tuple[str, Any]] = []
        self.name = "opencv-haar"
        self.error: str | None = None
        try:
            import cv2  # type: ignore[import-not-found]
            import numpy as np  # type: ignore[import-not-found]

            cascade_names = (
                "haarcascade_frontalface_alt2.xml",
                "haarcascade_frontalface_alt.xml",
                "haarcascade_frontalface_default.xml",
                "haarcascade_profileface.xml",
            )
            for cascade_name in cascade_names:
                cascade_path = cv2.data.haarcascades + cascade_name
                cascade = cv2.CascadeClassifier(cascade_path)
                if not cascade.empty():
                    self._cascades.append((cascade_name, cascade))
            if self._cascades:
                self._cv2 = cv2
                self._np = np
                self.name = "opencv-haar"
                return
            self.error = "opencv_haar_cascade_missing"
        except Exception as exc:
            self.error = f"opencv_unavailable:{exc.__class__.__name__}"

    @property
    def available(self) -> bool:
        return self._cv2 is not None and self._np is not None and bool(self._cascades)

    def analyze_jpeg(self, jpeg: bytes, *, captured_at: float | None = None) -> VisionSnapshot:
        captured_at = time.monotonic() if captured_at is None else captured_at
        if not self.available:
            return VisionSnapshot(captured_at=captured_at, detector=self.name, error=self.error or "detector_unavailable")
        cv2 = self._cv2
        np = self._np
        try:
            encoded = np.frombuffer(_enhance_for_detection(jpeg), dtype=np.uint8)
            image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if image is None:
                return VisionSnapshot(captured_at=captured_at, detector=self.name, error="jpeg_decode_failed")
            height, width = image.shape[:2]
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            equalized = cv2.equalizeHist(gray)
            boxes = []
            for cascade_name, cascade in self._cascades:
                is_profile = "profile" in cascade_name
                frames = [(gray, False), (equalized, False)]
                if is_profile:
                    frames.extend([(cv2.flip(gray, 1), True), (cv2.flip(equalized, 1), True)])
                for frame, flipped in frames:
                    found = cascade.detectMultiScale(
                        frame,
                        scaleFactor=1.02 if is_profile else 1.08,
                        minNeighbors=3 if is_profile else 4,
                        minSize=(36, 36),
                        flags=cv2.CASCADE_SCALE_IMAGE,
                    )
                    for box in found:
                        x, y, box_width, box_height = [int(value) for value in box]
                        if flipped:
                            x = int(width) - x - box_width
                        boxes.append((x, y, box_width, box_height))
            faces = tuple(_face_from_box(box, image_width=width, image_height=height) for box in boxes)
            faces = _dedupe_faces(tuple(sorted(faces, key=lambda face: face.area, reverse=True)))
            return VisionSnapshot(
                captured_at=captured_at,
                width=int(width),
                height=int(height),
                faces=faces,
                detector=self.name,
            )
        except Exception as exc:
            return VisionSnapshot(captured_at=captured_at, detector=self.name, error=f"analysis_failed:{exc.__class__.__name__}")


class VisionState:
    def __init__(self, detector: HaarFaceDetector | None = None) -> None:
        self.detector = detector or HaarFaceDetector()
        self.latest: VisionSnapshot | None = None
        self.latest_jpeg: bytes | None = None
        self.latest_jpeg_at: float = 0.0

    @property
    def detector_status(self) -> str:
        if self.detector.available:
            return self.detector.name
        return self.detector.error or "detector_unavailable"

    def observe_payload(self, payload: dict[str, Any]) -> VisionSnapshot:
        try:
            jpeg = decode_vision_frame_payload(payload)
        except ValueError as exc:
            snapshot = VisionSnapshot(captured_at=time.monotonic(), detector=self.detector.name, error=str(exc))
            self.latest = snapshot
            return snapshot
        self.latest_jpeg = jpeg
        self.latest_jpeg_at = time.monotonic()
        snapshot = self.detector.analyze_jpeg(_enhance_for_detection(jpeg))
        self.latest = snapshot
        return snapshot

    def prompt_context(self, *, max_age_seconds: float = 8.0) -> str:
        if self.latest is None:
            return ""
        return self.latest.prompt_context(max_age_seconds=max_age_seconds)

    def image_base64(self, *, max_age_seconds: float = 8.0) -> str | None:
        if self.latest_jpeg is None:
            return None
        if time.monotonic() - self.latest_jpeg_at > max_age_seconds:
            return None
        return base64.b64encode(self.latest_jpeg).decode("ascii")


def _face_from_box(box: Any, *, image_width: int, image_height: int) -> FaceObservation:
    x, y, width, height = [float(value) for value in box]
    cx = (x + width / 2.0) / max(1.0, float(image_width))
    cy = (y + height / 2.0) / max(1.0, float(image_height))
    normalized_x = max(-1.0, min(1.0, (cx - 0.5) * 2.0))
    normalized_y = max(-1.0, min(1.0, (0.5 - cy) * 2.0))
    width_fraction = max(0.0, min(1.0, width / max(1.0, float(image_width))))
    height_fraction = max(0.0, min(1.0, height / max(1.0, float(image_height))))
    confidence = max(0.35, min(1.0, 0.55 + (width_fraction * height_fraction * 2.0)))
    return FaceObservation(
        x=normalized_x,
        y=normalized_y,
        width=width_fraction,
        height=height_fraction,
        confidence=confidence,
    )


def _dedupe_faces(faces: tuple[FaceObservation, ...]) -> tuple[FaceObservation, ...]:
    result: list[FaceObservation] = []
    for face in faces:
        if any(abs(face.x - existing.x) < 0.18 and abs(face.y - existing.y) < 0.18 for existing in result):
            continue
        result.append(face)
    return tuple(result)


def _enhance_for_detection(jpeg: bytes, *, target_mean: float = 130.0) -> bytes:
    try:
        from PIL import Image, ImageEnhance, ImageStat

        with Image.open(BytesIO(jpeg)) as image:
            rgb = image.convert("RGB")
        stat = ImageStat.Stat(rgb.convert("L"))
        luma = float(stat.mean[0]) if stat.mean else 0.0
        if luma <= 0.0 or luma >= target_mean:
            return jpeg
        factor = min(4.0, max(1.0, target_mean / luma))
        enhanced = ImageEnhance.Brightness(rgb).enhance(factor)
        enhanced = ImageEnhance.Contrast(enhanced).enhance(1.08)
        output = BytesIO()
        enhanced.save(output, format="JPEG", quality=85, optimize=True)
        return output.getvalue()
    except Exception:
        return jpeg
