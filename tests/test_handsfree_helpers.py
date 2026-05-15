from __future__ import annotations

import unittest

from stacky.cli import _accept_stt_result, _clean_transcript, _is_likely_hallucination, _transcript_key
from stacky.voice.stt import AudioStats, STTResult


class HandsfreeHelpersTest(unittest.TestCase):
    def test_clean_transcript_collapses_repeated_sentence(self) -> None:
        self.assertEqual(_clean_transcript("Hej! Hej!"), "Hej!")

    def test_clean_transcript_collapses_repeated_words(self) -> None:
        self.assertEqual(_clean_transcript("hej stacky hej stacky"), "hej stacky")

    def test_transcript_key_ignores_punctuation(self) -> None:
        self.assertEqual(_transcript_key("Hej, Stacky!"), "hejstacky")

    def test_clean_transcript_normalizes_short_danish_greeting(self) -> None:
        self.assertEqual(_clean_transcript("haj"), "Hej!")

    def test_known_short_audio_hallucination_is_rejected(self) -> None:
        self.assertTrue(_is_likely_hallucination("Det er det, jeg har været på."))
        self.assertFalse(_is_likely_hallucination("Hej!"))

    def test_accepts_short_clear_greeting(self) -> None:
        result = STTResult(
            text="Hej!",
            audio=AudioStats(duration_seconds=0.8, rms=720, peak=5900, sample_rate=16000, channels=1),
            avg_logprob=-1.35,
            no_speech_prob=0.58,
            compression_ratio=0.4,
        )

        accepted, _ = _accept_stt_result(result)

        self.assertTrue(accepted)

    def test_rejects_quiet_short_whisper_guess(self) -> None:
        result = STTResult(
            text="Det var jo fin, det var mellem.",
            audio=AudioStats(duration_seconds=0.9, rms=480, peak=2500, sample_rate=16000, channels=1),
            avg_logprob=-1.2,
            no_speech_prob=0.4,
            compression_ratio=0.8,
        )

        accepted, reason = _accept_stt_result(result)

        self.assertFalse(accepted)
        self.assertIn(reason, {"for lavt mic-niveau", "lav STT confidence"})

    def test_accepts_soft_high_confidence_danish_phrase(self) -> None:
        result = STTResult(
            text="hej med dig",
            audio=AudioStats(duration_seconds=1.2, rms=294, peak=1954, sample_rate=16000, channels=1),
            avg_logprob=-0.32,
            no_speech_prob=0.0,
            compression_ratio=0.0,
        )

        accepted, _ = _accept_stt_result(result)

        self.assertTrue(accepted)


if __name__ == "__main__":
    unittest.main()
