from __future__ import annotations

import unittest

from stacky.cli import (
    _accept_stt_result,
    _capture_prompt_for_style,
    _clean_transcript,
    _is_likely_hallucination,
    _parse_calibration_command,
    _parse_local_realtime_reply,
    _parse_motion_command,
    _parse_stt_bench_spec,
    _parse_volume_command,
    _resolve_capture_speech_styles,
    _resolve_stt_bench_specs,
    _transcript_key,
    _word_error_rate,
)
from stacky.voice.stt import AudioStats, STTResult
from stacky.voice.transcript_correction import correct_danish_transcript
from stacky.voice.turn_detection import TurnSignalQuality


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

    def test_rejects_short_uncertain_uncorrected_fragment(self) -> None:
        result = STTResult(
            text="hej op i",
            audio=AudioStats(duration_seconds=1.4, rms=900, peak=3200, sample_rate=16000, channels=1),
            avg_logprob=-0.66,
            no_speech_prob=0.0,
            compression_ratio=0.0,
        )

        accepted, reason = _accept_stt_result(result, text="hej op i")

        self.assertFalse(accepted)
        self.assertEqual(reason, "kort usikkert STT-fragment")

    def test_accepts_trusted_transcript_correction_even_when_raw_is_odd(self) -> None:
        result = STTResult(
            text="oligopoly",
            audio=AudioStats(duration_seconds=2.5, rms=850, peak=3400, sample_rate=16000, channels=1),
            avg_logprob=-0.64,
            no_speech_prob=0.0,
            compression_ratio=0.0,
        )

        accepted, reason = _accept_stt_result(result, text="Skru lidt op for lyden.", trusted_transcript=True)

        self.assertTrue(accepted)
        self.assertEqual(reason, "trusted transcript correction")

    def test_parse_volume_command_absolute_percent(self) -> None:
        self.assertEqual(_parse_volume_command("sæt din volumen til 60 procent", current_level=80), (60, "Okay, min volumen er nu 60 procent."))

    def test_parse_volume_command_relative_up(self) -> None:
        self.assertEqual(_parse_volume_command("skru op", current_level=80), (95, "Okay, jeg skruer op til 95 procent."))

    def test_corrected_bad_stt_volume_phrase_reaches_parser(self) -> None:
        text = correct_danish_transcript("oligopoly").text

        self.assertEqual(_parse_volume_command(text, current_level=80), (95, "Okay, jeg skruer op til 95 procent."))

    def test_parse_volume_command_relative_down(self) -> None:
        self.assertEqual(_parse_volume_command("skru ned", current_level=10), (0, "Okay, jeg skruer ned til 0 procent."))

    def test_rejects_short_unclear_confirmation_from_noisy_signal(self) -> None:
        result = STTResult(
            text="ja",
            audio=AudioStats(duration_seconds=1.0, rms=1500, peak=32767, sample_rate=24000, channels=1),
            avg_logprob=-1.3,
            no_speech_prob=0.0,
            compression_ratio=0.0,
        )
        quality = TurnSignalQuality(
            duration_seconds=1.0,
            median_rms=300,
            p80_rms=450,
            p95_rms=1900,
            peak=32767,
            active_ratio=0.10,
            active_ms=100,
            max_active_run_ms=80,
            crest_factor=58.0,
            active_threshold=650,
        )

        accepted, reason = _accept_stt_result(result, signal_quality=quality)

        self.assertFalse(accepted)
        self.assertIn(reason, {"kort uklart svar", "klik/percussiv støj"})

    def test_rejects_incomplete_sparse_stt_fragment(self) -> None:
        result = STTResult(
            text="det er",
            audio=AudioStats(duration_seconds=1.28, rms=492, peak=8356, sample_rate=24000, channels=1),
            avg_logprob=-0.85,
            no_speech_prob=0.0,
            compression_ratio=0.0,
        )
        quality = TurnSignalQuality(
            duration_seconds=1.28,
            median_rms=292,
            p80_rms=500,
            p95_rms=1169,
            peak=8356,
            active_ratio=0.14,
            active_ms=180,
            max_active_run_ms=40,
            crest_factor=20.5,
            active_threshold=650,
        )

        accepted, reason = _accept_stt_result(result, signal_quality=quality)

        self.assertFalse(accepted)
        self.assertIn(reason, {"typisk STT-støjfragment", "ufærdigt STT-fragment", "for lidt sammenhængende tale"})

    def test_local_realtime_reply_bypasses_brain_for_wait_commands(self) -> None:
        self.assertEqual(_parse_local_realtime_reply("vent lige"), "Jeg venter.")
        self.assertEqual(_parse_local_realtime_reply("stop lige"), "Jeg venter.")
        self.assertIsNone(_parse_local_realtime_reply("hvad laver du"))

    def test_parse_motion_command(self) -> None:
        self.assertEqual((_parse_motion_command("kig til venstre") or None).gesture, "look_left")
        self.assertEqual((_parse_motion_command("kig til højre") or None).gesture, "look_right")
        self.assertEqual((_parse_motion_command("gik til venstre") or None).gesture, "look_left")
        self.assertEqual((_parse_motion_command("gik op") or None).gesture, "look_up")
        self.assertEqual((_parse_motion_command("nik med hovedet") or None).gesture, "nod")
        self.assertEqual((_parse_motion_command("prøv en bevægelse") or None).gesture, "demo")
        self.assertIsNone(_parse_motion_command("skru op"))

    def test_corrected_partial_motion_phrase_reaches_parser(self) -> None:
        text = correct_danish_transcript("lidt til hojre").text

        self.assertEqual((_parse_motion_command(text) or None).gesture, "look_right")

    def test_parse_calibration_command(self) -> None:
        right = _parse_calibration_command("lidt mere til højre")
        left = _parse_calibration_command("lidt mere til venstre")
        up = _parse_calibration_command("lidt op")
        save = _parse_calibration_command("gem den her position som center")

        self.assertEqual(right.yaw_delta if right else None, 30)
        self.assertEqual(left.yaw_delta if left else None, -30)
        self.assertEqual(up.pitch_delta if up else None, 30)
        self.assertTrue(save.save_current if save else False)
        self.assertIsNone(_parse_calibration_command("skru op"))

    def test_parse_stt_bench_aliases(self) -> None:
        self.assertEqual(_parse_stt_bench_spec("roest"), ("wav2vec2", "roest"))
        self.assertEqual(_parse_stt_bench_spec("roest-v2-2b"), ("wav2vec2", "roest-v2-2b"))
        self.assertEqual(_parse_stt_bench_spec("qwen3"), ("qwen3", "qwen3-0.6b"))
        self.assertEqual(_parse_stt_bench_spec("wav2vec2:custom/model"), ("wav2vec2", "custom/model"))

    def test_stt_bench_specs_default_to_low_latency_models(self) -> None:
        specs = _resolve_stt_bench_specs([], include_heavy=False)

        self.assertEqual(specs, [("wav2vec2", "roest-v3"), ("wav2vec2", "roest-v2")])

    def test_capture_speech_styles_default_and_dedupe(self) -> None:
        self.assertEqual(_resolve_capture_speech_styles([]), ["normal"])
        self.assertEqual(_resolve_capture_speech_styles(["fast", "fast", "mumble"]), ["fast", "mumble"])

    def test_capture_prompt_for_fast_and_mumbled_speech(self) -> None:
        self.assertIn("hurtigt", _capture_prompt_for_style("Hej Stacky.", "fast"))
        self.assertIn("Muml", _capture_prompt_for_style("Hej Stacky.", "mumble"))

    def test_word_error_rate(self) -> None:
        self.assertEqual(_word_error_rate("hej med dig", "hej med dig"), 0.0)
        self.assertAlmostEqual(_word_error_rate("hej med dig", "hej dig"), 1 / 3)


if __name__ == "__main__":
    unittest.main()
