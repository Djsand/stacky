from __future__ import annotations

import unittest

from stacky.voice.speech_adapter import adapt_for_danish_speech, split_for_speech


class SpeechAdapterTest(unittest.TestCase):
    def test_pronunciation_fixes(self) -> None:
        spoken = adapt_for_danish_speech("Nicolai, Stacky siger at Sandcode er faerdig.")

        self.assertIn("Nikolai", spoken)
        self.assertIn("Stækki", spoken)
        self.assertIn("Sand-kode", spoken)

    def test_voice_labels_are_spoken_as_danish_words(self) -> None:
        spoken = adapt_for_danish_speech("F2 og M3 lyder bedre end F1.")

        self.assertIn("eff to", spoken)
        self.assertIn("em tre", spoken)
        self.assertIn("eff en", spoken)

    def test_leading_name_greeting_is_kept_short(self) -> None:
        spoken = adapt_for_danish_speech("Hej Nicolai, det her er Stacky.")

        self.assertEqual(spoken, "Hej. Det er Stækki.")

    def test_repeated_long_words_are_collapsed(self) -> None:
        spoken = adapt_for_danish_speech("Det skal være dansk dansk og roligt roligt.")

        self.assertEqual(spoken, "Det skal være dansk og roligt.")

    def test_split_for_speech(self) -> None:
        chunks = split_for_speech("Hej. " * 100, max_chars=40)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 45 for chunk in chunks))

    def test_split_for_speech_breaks_long_sentence(self) -> None:
        text = (
            "Jeg kan godt gore det, men jeg tager det i sma bidder, "
            "sa den forste lyd kommer hurtigt og resten folger efter."
        )

        chunks = split_for_speech(text, max_chars=48)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 52 for chunk in chunks))


if __name__ == "__main__":
    unittest.main()
