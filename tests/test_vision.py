from __future__ import annotations

import base64
import unittest

from stacky.vision import FaceObservation, VisionSnapshot, VisionState


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


if __name__ == "__main__":
    unittest.main()
