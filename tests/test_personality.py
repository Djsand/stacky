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

    def test_testing_and_polish_feedback_forms_style_note(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = StackySelfModel(Path(tmp))

            observation = model.observe_user_turn(
                "Jeg tester bare lige nu, jeg vil jo gerne have du er perfekt.",
                trusted=True,
                source="stackchan-voice",
            )

        self.assertTrue(any("tester" in note.lower() for note in observation.style_notes))
        self.assertTrue(any("finpudser" in note.lower() for note in observation.style_notes))
        self.assertTrue(observation.convictions)

    def test_personality_feedback_forms_anti_llm_style_notes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = StackySelfModel(Path(tmp))

            observation = model.observe_user_turn(
                "Du mangler personlighed, svarene bliver lange og ligegyldige og meget LLM agtige. "
                "Den er for stiv og robotagtig, og den maa gerne kunne grine og bruge lidt humor.",
                trusted=True,
                source="test",
            )

        notes = " ".join(observation.style_notes).lower()
        self.assertIn("tydeligere egen stemme", notes)
        self.assertIn("konkret værdi", notes)
        self.assertIn("generisk chatbot", notes)
        self.assertIn("kort grin", notes)
        self.assertIn("tør humor", notes)

    def test_persona_tuning_persists_anti_assistant_dark_humor_feedback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = StackySelfModel(root)
            before = model.summary()["persona_tuning"]

            observation = model.observe_user_turn(
                "Vi skal have en mere persistent personlighed, ingen assistent adfærd, "
                "og humor - gerne dark humor og galgenhumor.",
                trusted=True,
                source="test",
            )
            after = model.summary()["persona_tuning"]
            context = model.context_for_prompt(user_text="hvem er du")
            reloaded = StackySelfModel(root).summary()["persona_tuning"]

        self.assertTrue(observation.persona_adjustments)
        self.assertGreater(after["assistant_suppression"], before["assistant_suppression"])
        self.assertGreater(after["dark_humor"], before["dark_humor"])
        self.assertGreater(after["dry_humor"], before["dry_humor"])
        self.assertEqual(reloaded, after)
        self.assertIn("Persistent persona-tuning", context)
        self.assertIn("anti-assistent", context)
        self.assertIn("Mørk humor", context)
        self.assertTrue(any("persistent" in note.lower() for note in model.summary()["style_notes"]))


if __name__ == "__main__":
    unittest.main()
