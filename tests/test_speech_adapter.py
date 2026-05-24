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

    def test_dig_is_pronounced_without_touching_longer_words(self) -> None:
        spoken = adapt_for_danish_speech("Det er godt at høre fra dig. Jeg er færdig.")

        self.assertIn("fra dej", spoken)
        self.assertIn("færdig", spoken)

    def test_leading_name_greeting_is_kept_short(self) -> None:
        spoken = adapt_for_danish_speech("Hej Nicolai, det her er Stacky.")

        self.assertEqual(spoken, "Hej, det er Stækki.")

    def test_repeated_long_words_are_collapsed(self) -> None:
        spoken = adapt_for_danish_speech("Det skal være dansk dansk og roligt roligt.")

        self.assertEqual(spoken, "Det skal være dansk og roligt.")

    def test_rhythm_punctuation_for_live_speech(self) -> None:
        spoken = adapt_for_danish_speech("Okay det giver mening men jeg venter hvis du tester.")

        self.assertEqual(spoken, "Okay, det giver mening, men jeg venter, hvis du tester.")

    def test_rhythm_keeps_phrase_marker_as_comma(self) -> None:
        spoken = adapt_for_danish_speech("Det giver mening jeg venter.")

        self.assertEqual(spoken, "Det giver mening, jeg venter.")

    def test_rhythm_does_not_break_intensifier_saa(self) -> None:
        spoken = adapt_for_danish_speech("Det er så fedt.")

        self.assertEqual(spoken, "Det er så fedt.")

    def test_rhythm_adds_pause_before_saa_clause(self) -> None:
        spoken = adapt_for_danish_speech("Det giver mening så jeg venter.")

        self.assertEqual(spoken, "Det giver mening, så jeg venter.")

    def test_laughter_is_spoken_as_short_sound(self) -> None:
        spoken = adapt_for_danish_speech("Haha, den var ny. (griner) Det kan jeg godt lide.")

        self.assertIn("ha ha", spoken)
        self.assertIn("ha, Det", spoken)

    def test_assistant_stock_phrases_are_softened(self) -> None:
        spoken = adapt_for_danish_speech("Det er modtaget. Jeg afventer dit næste signal.")

        self.assertEqual(spoken, "Okay, jeg venter på dit næste signal.")

    def test_split_for_speech(self) -> None:
        chunks = split_for_speech("Hej. " * 100, max_chars=40)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 45 for chunk in chunks))

    def test_rhythmic_split_keeps_short_speech_together(self) -> None:
        chunks = split_for_speech("Okay. Det giver mening, men jeg venter.", max_chars=160, rhythmic=True)

        self.assertEqual(chunks, ["Okay, det giver mening, men jeg venter."])

    def test_rhythmic_split_keeps_comma_pause_inside_short_speech(self) -> None:
        chunks = split_for_speech("Det giver mening, jeg venter.", max_chars=160, rhythmic=True)

        self.assertEqual(chunks, ["Det giver mening, jeg venter."])

    def test_rhythmic_split_does_not_merge_short_sentences_when_text_is_long(self) -> None:
        chunks = split_for_speech("Okay. Jeg tester rytmen. Den skal ikke flyde sammen.", max_chars=42, rhythmic=True)

        self.assertEqual(chunks, ["Okay, jeg tester rytmen.", "Den skal ikke flyde sammen."])

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
