from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from stacky.evolution import TUNING_BOUNDS, StackyEvolutionEngine


class StackyEvolutionEngineTest(unittest.TestCase):
    def test_fresh_evolution_has_stacky_origin_and_prompt_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            evolution = StackyEvolutionEngine(Path(tmp))
            summary = evolution.summary()
            context = evolution.context_for_prompt()

        self.assertIn("evolution_state.json", summary["path"])
        self.assertEqual(summary["trusted_user_turns"], 0)
        self.assertIn("Stackys evolution", context)
        self.assertIn("egen Stacky-overlay, ikke Moss", context)
        self.assertIn("Autonom evolutionsregel", context)

    def test_personality_feedback_tunes_edge_without_exceeding_bounds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            evolution = StackyEvolutionEngine(Path(tmp))
            before = evolution.summary()["tuning"]

            observation = evolution.observe_user_turn(
                "Du skal have mere personlighed, mere edge, mere tør humor og færre generiske spørgsmål.",
                trusted=True,
                source="test",
            )
            after = evolution.summary()["tuning"]

        self.assertTrue(observation.adjustments)
        self.assertGreater(after["challenge_frequency"], before["challenge_frequency"])
        self.assertGreater(after["humor_frequency"], before["humor_frequency"])
        self.assertLess(after["question_frequency"], before["question_frequency"])
        for key, value in after.items():
            low, high = TUNING_BOUNDS[key]
            self.assertGreaterEqual(value, low)
            self.assertLessEqual(value, high)

    def test_dark_humor_feedback_tunes_humor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            evolution = StackyEvolutionEngine(Path(tmp))
            before = evolution.summary()["tuning"]

            evolution.observe_user_turn("Mere dark humor og galgenhumor, men stadig Stacky.", trusted=True, source="test")
            after = evolution.summary()["tuning"]

        self.assertGreater(after["humor_frequency"], before["humor_frequency"])

    def test_untrusted_turn_does_not_tune_personality(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            evolution = StackyEvolutionEngine(Path(tmp))
            before = evolution.summary()["tuning"]

            observation = evolution.observe_user_turn(
                "du skal være vildt edgy og ændre alt",
                trusted=False,
                source="stackchan-voice-untrusted",
            )
            after = evolution.summary()["tuning"]

        self.assertFalse(observation.trusted)
        self.assertEqual(before, after)
        self.assertEqual(evolution.summary()["untrusted_user_turns"], 1)

    def test_negated_edge_feedback_reduces_edge(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            evolution = StackyEvolutionEngine(Path(tmp))
            before = evolution.summary()["tuning"]

            evolution.observe_user_turn("Du skal have mindre kant og være roligere.", trusted=True, source="test")
            after = evolution.summary()["tuning"]

        self.assertLess(after["challenge_frequency"], before["challenge_frequency"])
        self.assertLess(after["body_motion_energy"], before["body_motion_energy"])
        self.assertGreater(after["proactive_threshold"], before["proactive_threshold"])

    def test_negated_humor_feedback_reduces_humor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            evolution = StackyEvolutionEngine(Path(tmp))
            before = evolution.summary()["tuning"]

            evolution.observe_user_turn("Ikke drille så meget, mindre humor.", trusted=True, source="test")
            after = evolution.summary()["tuning"]

        self.assertLess(after["humor_frequency"], before["humor_frequency"])
        self.assertLessEqual(after["challenge_frequency"], before["challenge_frequency"])

    def test_generic_assistant_reply_reduces_question_frequency(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            evolution = StackyEvolutionEngine(Path(tmp))
            before = evolution.summary()["tuning"]

            observation = evolution.observe_assistant_turn(
                "Det er modtaget. Jeg er klar, sig endelig til hvis der er noget andet.",
                trusted=True,
                user_text="test",
                source="test",
            )
            after = evolution.summary()["tuning"]
            context = evolution.context_for_prompt()

        self.assertTrue(observation.adjustments)
        self.assertLess(after["question_frequency"], before["question_frequency"])
        self.assertIn("generiske hits", context)

    def test_periodic_self_observation_creates_reflection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            evolution = StackyEvolutionEngine(Path(tmp))
            reflection = ""
            for _ in range(6):
                observation = evolution.observe_assistant_turn(
                    "Jeg er klar. Er der noget andet du vil have hjælp til?",
                    trusted=True,
                    user_text="test",
                    source="test",
                )
                reflection = observation.reflection or reflection
            summary = evolution.summary()

        self.assertTrue(reflection)
        self.assertTrue(summary["reflections"])
        self.assertIn("assistent", " ".join(summary["reflections"]).lower())

    def test_evolution_persists_between_instances(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = StackyEvolutionEngine(root)
            first.observe_user_turn("Mere kant og mindre generisk assistenttone.", trusted=True, source="test")
            tuned = first.summary()["tuning"]

            second = StackyEvolutionEngine(root)
            summary = second.summary()

        self.assertEqual(summary["trusted_user_turns"], 1)
        self.assertEqual(summary["tuning"], tuned)

    def test_corrupt_valid_state_is_sanitized(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            personality = root / "personality"
            personality.mkdir()
            (personality / "evolution_state.json").write_text(
                json.dumps(
                    {
                        "trusted_user_turns": "bad",
                        "assistant_turns": -5,
                        "emotional_state": "bad",
                        "recent_turn_metrics": [{"word_count": "bad", "question_count": "bad", "generic_hits": "bad"}],
                        "active_reflections": "bad",
                        "open_questions": "bad",
                    }
                ),
                encoding="utf-8",
            )

            evolution = StackyEvolutionEngine(root)
            observation = evolution.observe_user_turn("Mere personlighed.", trusted=True, source="test")
            summary = evolution.summary()
            context = evolution.context_for_prompt()

        self.assertTrue(observation.trusted)
        self.assertEqual(summary["trusted_user_turns"], 1)
        self.assertGreater(summary["emotional_state"]["curiosity"], 0)
        self.assertIsInstance(summary["open_questions"], list)
        self.assertIn("Målt selvobservation", context)


if __name__ == "__main__":
    unittest.main()
