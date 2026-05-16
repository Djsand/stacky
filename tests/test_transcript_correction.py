from __future__ import annotations

import unittest

from stacky.voice.transcript_correction import correct_danish_transcript


class TranscriptCorrectionTest(unittest.TestCase):
    def test_corrects_stacky_name_variants(self) -> None:
        correction = correct_danish_transcript("hej stakke")

        self.assertEqual(correction.text, "Hej Stacky")
        self.assertTrue(correction.changed)

    def test_corrects_nicolai_name_variant(self) -> None:
        correction = correct_danish_transcript("nikolaj siger hej til stakki")

        self.assertEqual(correction.text, "Nicolai siger hej til Stacky")

    def test_corrects_known_volume_command_failure(self) -> None:
        correction = correct_danish_transcript("oligopoly")

        self.assertEqual(correction.text, "Skru lidt op for lyden.")

    def test_corrects_partial_right_motion_command(self) -> None:
        correction = correct_danish_transcript("lidt til hojre")

        self.assertEqual(correction.text, "Kig lidt til højre.")

    def test_corrects_fast_right_motion_failure(self) -> None:
        correction = correct_danish_transcript("lidt for her")

        self.assertEqual(correction.text, "Kig lidt til højre.")

    def test_corrects_mumbled_greeting_failure(self) -> None:
        correction = correct_danish_transcript("hej op i")

        self.assertEqual(correction.text, "Hej Stacky")

    def test_does_not_force_unrelated_chat_into_command(self) -> None:
        correction = correct_danish_transcript("regnede dig")

        self.assertEqual(correction.text, "regnede dig")
        self.assertFalse(correction.changed)


if __name__ == "__main__":
    unittest.main()
