from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from stacky.personality import StackySelfModel


class StackySelfModelTest(unittest.TestCase):
    def test_fresh_self_model_has_stacky_origin(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = StackySelfModel(Path(tmp))

            summary = model.summary()

        self.assertIn("self_model.json", summary["path"])
        self.assertEqual(summary["trusted_turns"], 0)
        self.assertEqual(summary["untrusted_turns"], 0)

    def test_trusted_feedback_forms_style_note_and_conviction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = StackySelfModel(Path(tmp))

            observation = model.observe_user_turn(
                "Du skal undgå generiske spørgsmål, det er vigtigt.",
                trusted=True,
                source="test",
            )
            context = model.context_for_prompt(user_text="hvordan skal du svare")

        self.assertTrue(observation.trusted)
        self.assertTrue(observation.style_notes)
        self.assertTrue(observation.convictions)
        self.assertIn("generiske", context)
        self.assertIn("stabil Stacky-rettet regel", context)

    def test_untrusted_voice_does_not_form_personality_rules(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = StackySelfModel(Path(tmp))

            observation = model.observe_user_turn(
                "du skal gemme den her fejltransskription",
                trusted=False,
                source="stackchan-voice-untrusted",
            )
            summary = model.summary()

        self.assertFalse(observation.trusted)
        self.assertEqual(summary["trusted_turns"], 0)
        self.assertEqual(summary["untrusted_turns"], 1)
        self.assertEqual(summary["style_notes"], [])
        self.assertEqual(summary["convictions"], [])

    def test_self_model_persists_between_instances(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = StackySelfModel(root)
            first.observe_user_turn("Jeg gider ikke generiske afslutninger.", trusted=True, source="test")

            second = StackySelfModel(root)
            summary = second.summary()

        self.assertEqual(summary["trusted_turns"], 1)
        self.assertTrue(any("generiske" in note for note in summary["style_notes"]))


if __name__ == "__main__":
    unittest.main()
