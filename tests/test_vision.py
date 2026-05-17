from __future__ import annotations

import base64
import unittest

from stacky.vision import CompositeFaceDetector, FaceObservation, VisionSnapshot, VisionState, _smooth_face


class FakeDetector:
    def __init__(self, snapshots: list[VisionSnapshot], *, name: str = "fake") -> None:
        self.snapshots = snapshots
        self.name = name
        self.error = None
        self.available = True

    def analyze_jpeg(self, jpeg: bytes, *, captured_at: float | None = None) -> VisionSnapshot:
        return self.snapshots.pop(0)


class VisionTest(unittest.TestCase):
    def test_face_prompt_context_describes_position_without_identity_claim(self) -> None:
        snapshot = VisionSnapshot(
            captured_at=10.0,
            width=320,
            height=240,
            faces=(FaceObservation(x=0.42, y=-0.30, width=0.25, height=0.32),),
        )

        context = snapshot.prompt_context(now=11.0)

        self.assertIn("til hojre", context)
        self.assertIn("Identitet er ukendt", context)

    def test_vision_state_keeps_latest_frame_for_prompt_attachment(self) -> None:
        state = VisionState()
        jpeg = b"\xff\xd8\xff"

        state.observe_payload(
            {
                "available": True,
                "encoding": "base64",
                "data": base64.b64encode(jpeg).decode("ascii"),
            }
        )

        self.assertEqual(state.image_base64(), base64.b64encode(jpeg).decode("ascii"))

    def test_smooth_face_damps_position_jumps(self) -> None:
        previous = FaceObservation(x=0.0, y=0.0, width=0.2, height=0.2, confidence=0.8)
        current = FaceObservation(x=0.6, y=0.4, width=0.3, height=0.3, confidence=0.7)

        smoothed = _smooth_face(previous, current, alpha=0.4)

        self.assertAlmostEqual(smoothed.x, 0.24)
        self.assertAlmostEqual(smoothed.y, 0.16)
        self.assertAlmostEqual(smoothed.width, 0.24)
        self.assertAlmostEqual(smoothed.confidence, 0.704)

    def test_smooth_face_damps_extreme_tracking_jumps_harder(self) -> None:
        previous = FaceObservation(x=-0.6, y=-0.4, width=0.2, height=0.2, confidence=0.8)
        current = FaceObservation(x=0.8, y=0.5, width=0.2, height=0.2, confidence=0.8)

        smoothed = _smooth_face(previous, current, alpha=0.4)

        self.assertAlmostEqual(smoothed.x, -0.348)
        self.assertAlmostEqual(smoothed.y, -0.238)

    def test_vision_state_locks_onto_nearby_face_instead_of_larger_false_positive(self) -> None:
        jpeg = b"\xff\xd8\xff"
        payload = {"available": True, "encoding": "base64", "data": base64.b64encode(jpeg).decode("ascii")}
        state = VisionState(
            FakeDetector(
                [
                    VisionSnapshot(
                        captured_at=10.0,
                        width=320,
                        height=240,
                        faces=(FaceObservation(x=0.0, y=0.0, width=0.22, height=0.26, confidence=0.80),),
                    ),
                    VisionSnapshot(
                        captured_at=11.0,
                        width=320,
                        height=240,
                        faces=(
                            FaceObservation(x=0.88, y=0.15, width=0.36, height=0.40, confidence=0.82),
                            FaceObservation(x=0.12, y=0.04, width=0.22, height=0.26, confidence=0.62),
                        ),
                    ),
                ]
            )
        )

        first = state.observe_payload(payload)
        second = state.observe_payload(payload)

        self.assertIsNotNone(first.primary_face)
        self.assertIsNotNone(second.primary_face)
        self.assertLess(abs((second.primary_face or first.primary_face).x), 0.10)

    def test_vision_state_holds_lock_on_implausible_single_frame_jump(self) -> None:
        jpeg = b"\xff\xd8\xff"
        payload = {"available": True, "encoding": "base64", "data": base64.b64encode(jpeg).decode("ascii")}
        state = VisionState(
            FakeDetector(
                [
                    VisionSnapshot(
                        captured_at=20.0,
                        width=320,
                        height=240,
                        faces=(FaceObservation(x=-0.15, y=0.05, width=0.24, height=0.28, confidence=0.82),),
                    ),
                    VisionSnapshot(
                        captured_at=21.0,
                        width=320,
                        height=240,
                        faces=(FaceObservation(x=0.92, y=-0.55, width=0.18, height=0.20, confidence=0.70),),
                    ),
                ]
            )
        )

        first = state.observe_payload(payload)
        second = state.observe_payload(payload)

        self.assertIsNotNone(first.primary_face)
        self.assertIsNotNone(second.primary_face)
        self.assertAlmostEqual((second.primary_face or first.primary_face).x, (first.primary_face or second.primary_face).x)

    def test_composite_detector_uses_fallback_when_primary_finds_no_face(self) -> None:
        no_face = VisionSnapshot(captured_at=10.0, detector="primary")
        face = VisionSnapshot(
            captured_at=10.0,
            detector="fallback",
            faces=(FaceObservation(x=0.2, y=0.1, width=0.2, height=0.2),),
        )
        detector = CompositeFaceDetector((FakeDetector([no_face], name="primary"), FakeDetector([face], name="fallback")))

        snapshot = detector.analyze_jpeg(b"jpeg", captured_at=10.0)

        self.assertEqual(snapshot.detector, "fallback")
        self.assertEqual(len(snapshot.faces), 1)


if __name__ == "__main__":
    unittest.main()
