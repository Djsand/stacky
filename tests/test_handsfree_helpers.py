from __future__ import annotations

import unittest

from stacky.cli import (
    _accept_stt_result,
    _capture_prompt_for_style,
    _clean_transcript,
    _is_likely_hallucination,
    _parse_calibration_command,
    _format_battery_status_reply,
    _parse_battery_status_command,
    _parse_display_brightness_command,
    _parse_local_realtime_reply,
    _parse_motion_command,
    _parse_stt_bench_spec,
    _parse_volume_command,
    _resolve_capture_speech_styles,
    _resolve_stt_bench_specs,
    _run_motion_gesture,
    _transcript_key,
    _voice_memory_policy,
    _word_error_rate,
)
from stacky.voice.stt import AudioStats, STTResult
from stacky.voice.transcript_correction import correct_danish_transcript
from stacky.voice.turn_detection import TurnSignalQuality


class FakeMotionActor:
    def __init__(self) -> None:
        self.gestures: list[tuple[str, float, int]] = []

    def gesture(self, name: str, *, intensity: float = 1.0, speed: int = 500) -> bool:
        self.gestures.append((name, intensity, speed))
        return True


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

    def test_accepts_short_greeting_with_clean_signal_quality(self) -> None:
        result = STTResult(
            text="hej",
            audio=AudioStats(duration_seconds=1.20, rms=900, peak=6500, sample_rate=24000, channels=1),
            avg_logprob=-0.45,
            no_speech_prob=0.0,
            compression_ratio=0.0,
        )
        quality = TurnSignalQuality(
            duration_seconds=1.20,
            median_rms=260,
            p80_rms=900,
            p95_rms=2200,
            peak=6500,
            active_ratio=0.36,
            active_ms=420,
            max_active_run_ms=340,
            crest_factor=8.0,
            active_threshold=420,
            zero_crossing_rate=0.16,
            speech_band_ms=420,
            max_speech_band_run_ms=340,
        )

        accepted, reason = _accept_stt_result(result, signal_quality=quality)

        self.assertTrue(accepted)
        self.assertEqual(reason, "kort hilsen")

    def test_rejects_noisy_short_greeting_guess(self) -> None:
        result = STTResult(
            text="hej",
            audio=AudioStats(duration_seconds=1.86, rms=2183, peak=21352, sample_rate=24000, channels=1),
            avg_logprob=-0.89,
            no_speech_prob=0.0,
            compression_ratio=0.0,
        )
        quality = TurnSignalQuality(
            duration_seconds=1.86,
            median_rms=220,
            p80_rms=1100,
            p95_rms=2600,
            peak=21352,
            active_ratio=0.22,
            active_ms=400,
            max_active_run_ms=240,
            crest_factor=10.0,
            active_threshold=420,
            zero_crossing_rate=0.17,
            speech_band_ms=360,
            max_speech_band_run_ms=300,
        )

        accepted, reason = _accept_stt_result(result, signal_quality=quality)

        self.assertFalse(accepted)
        self.assertEqual(reason, "kort hilsen fra støj")

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

    def test_parse_volume_command_followup_adjust_to_number(self) -> None:
        self.assertEqual(_parse_volume_command("justerer til 50", current_level=80), (50, "Okay, min volumen er nu 50 procent."))

    def test_parse_volume_command_soft_request_with_words_between(self) -> None:
        self.assertEqual(
            _parse_volume_command("lige inden vi går videre kan du så ikke lige skrue lyden ned", current_level=80),
            (65, "Okay, jeg skruer ned til 65 procent."),
        )

    def test_parse_volume_command_directional_absolute_level(self) -> None:
        self.assertEqual(_parse_volume_command("du ned til 65", current_level=80), (65, "Okay, min volumen er nu 65 procent."))

    def test_parse_volume_command_from_live_stt_mishearings(self) -> None:
        self.assertEqual(
            _parse_volume_command("det ser bedre udbrede at skole lydstyrken ned", current_level=80),
            (65, "Okay, jeg skruer ned til 65 procent."),
        )
        self.assertEqual(
            _parse_volume_command("nej det virkede ikke kronet til 65", current_level=80),
            (65, "Okay, min volumen er nu 65 procent."),
        )

    def test_parse_volume_command_much_further_down(self) -> None:
        self.assertEqual(
            _parse_volume_command("for at skrue meget længere ned", current_level=65),
            (30, "Okay, jeg skruer ned til 30 procent."),
        )

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

    def test_rejects_low_confidence_transcript_from_noisy_high_zcr_turn(self) -> None:
        result = STTResult(
            text="den her den den den den",
            audio=AudioStats(duration_seconds=9.0, rms=870, peak=6108, sample_rate=24000, channels=1),
            avg_logprob=-1.14,
            no_speech_prob=0.0,
            compression_ratio=0.0,
        )
        quality = TurnSignalQuality(
            duration_seconds=9.0,
            median_rms=868,
            p80_rms=900,
            p95_rms=937,
            peak=6108,
            active_ratio=0.91,
            active_ms=8180,
            max_active_run_ms=7660,
            crest_factor=7.4,
            active_threshold=498,
            zero_crossing_rate=0.49,
            speech_band_ms=420,
            max_speech_band_run_ms=200,
        )

        accepted, reason = _accept_stt_result(result, signal_quality=quality)

        self.assertFalse(accepted)
        self.assertEqual(reason, "støjfyldt højfrekvent transcript")

    def test_rejects_short_trusted_correction_from_high_frequency_noise(self) -> None:
        result = STTResult(
            text="ej",
            audio=AudioStats(duration_seconds=1.12, rms=1185, peak=8748, sample_rate=24000, channels=1),
            avg_logprob=-0.10,
            no_speech_prob=0.0,
            compression_ratio=0.0,
        )
        quality = TurnSignalQuality(
            duration_seconds=1.12,
            median_rms=654,
            p80_rms=1100,
            p95_rms=2525,
            peak=8748,
            active_ratio=0.21,
            active_ms=240,
            max_active_run_ms=240,
            crest_factor=9.0,
            active_threshold=1136,
            zero_crossing_rate=0.43,
            speech_band_ms=240,
            max_speech_band_run_ms=240,
        )

        accepted, reason = _accept_stt_result(
            result,
            text="Hej.",
            signal_quality=quality,
            trusted_transcript=True,
        )

        self.assertFalse(accepted)
        self.assertEqual(reason, "kort højfrekvent STT-fragment")

    def test_rejects_noisy_jeg_kan_fragment(self) -> None:
        result = STTResult(
            text="jeg kan",
            audio=AudioStats(duration_seconds=1.58, rms=1893, peak=25610, sample_rate=24000, channels=1),
            avg_logprob=-0.52,
            no_speech_prob=0.0,
            compression_ratio=0.0,
        )
        quality = TurnSignalQuality(
            duration_seconds=1.58,
            median_rms=128,
            p80_rms=900,
            p95_rms=5332,
            peak=25610,
            active_ratio=0.27,
            active_ms=420,
            max_active_run_ms=240,
            crest_factor=26.4,
            active_threshold=420,
            zero_crossing_rate=0.20,
            speech_band_ms=420,
            max_speech_band_run_ms=240,
        )

        accepted, reason = _accept_stt_result(result, signal_quality=quality)

        self.assertFalse(accepted)
        self.assertEqual(reason, "kort højfrekvent STT-fragment")

    def test_rejects_medium_confidence_filler_noise_turn(self) -> None:
        result = STTResult(
            text="den her du den",
            audio=AudioStats(duration_seconds=3.46, rms=3085, peak=30000, sample_rate=24000, channels=1),
            avg_logprob=-0.67,
            no_speech_prob=0.0,
            compression_ratio=0.0,
        )
        quality = TurnSignalQuality(
            duration_seconds=3.46,
            median_rms=342,
            p80_rms=1500,
            p95_rms=3017,
            peak=7774,
            active_ratio=0.48,
            active_ms=840,
            max_active_run_ms=500,
            crest_factor=10.6,
            active_threshold=420,
            zero_crossing_rate=0.11,
            speech_band_ms=840,
            max_speech_band_run_ms=500,
        )

        accepted, reason = _accept_stt_result(result, signal_quality=quality)

        self.assertFalse(accepted)
        self.assertEqual(reason, "repetitivt filler-støjfragment")

    def test_rejects_repeated_den_er_noise_turn_even_with_moderate_confidence(self) -> None:
        result = STTResult(
            text="den er den den",
            audio=AudioStats(duration_seconds=7.70, rms=2150, peak=30000, sample_rate=24000, channels=1),
            avg_logprob=-0.47,
            no_speech_prob=0.0,
            compression_ratio=0.0,
        )
        quality = TurnSignalQuality(
            duration_seconds=7.70,
            median_rms=500,
            p80_rms=1800,
            p95_rms=3400,
            peak=30000,
            active_ratio=0.45,
            active_ms=3460,
            max_active_run_ms=520,
            crest_factor=11.0,
            active_threshold=420,
            zero_crossing_rate=0.16,
            speech_band_ms=3400,
            max_speech_band_run_ms=520,
        )

        accepted, reason = _accept_stt_result(result, signal_quality=quality)

        self.assertFalse(accepted)
        self.assertEqual(reason, "repetitivt filler-støjfragment")

    def test_rejects_ack_prefixed_repeated_den_noise_turn(self) -> None:
        result = STTResult(
            text="ja den er den",
            audio=AudioStats(duration_seconds=8.90, rms=1712, peak=30000, sample_rate=24000, channels=1),
            avg_logprob=-0.58,
            no_speech_prob=0.0,
            compression_ratio=0.0,
        )
        quality = TurnSignalQuality(
            duration_seconds=8.90,
            median_rms=520,
            p80_rms=1700,
            p95_rms=3300,
            peak=30000,
            active_ratio=0.44,
            active_ms=3920,
            max_active_run_ms=540,
            crest_factor=12.0,
            active_threshold=420,
            zero_crossing_rate=0.15,
            speech_band_ms=3800,
            max_speech_band_run_ms=520,
        )

        accepted, reason = _accept_stt_result(result, signal_quality=quality)

        self.assertFalse(accepted)
        self.assertEqual(reason, "repetitivt filler-støjfragment")

    def test_rejects_den_her_jeg_kan_noise_turn(self) -> None:
        result = STTResult(
            text="den her jeg kan",
            audio=AudioStats(duration_seconds=5.28, rms=3092, peak=30722, sample_rate=24000, channels=1),
            avg_logprob=-0.89,
            no_speech_prob=0.0,
            compression_ratio=0.0,
        )
        quality = TurnSignalQuality(
            duration_seconds=5.28,
            median_rms=620,
            p80_rms=2400,
            p95_rms=4200,
            peak=30722,
            active_ratio=0.48,
            active_ms=2540,
            max_active_run_ms=560,
            crest_factor=14.0,
            active_threshold=420,
            zero_crossing_rate=0.18,
            speech_band_ms=2500,
            max_speech_band_run_ms=500,
        )

        accepted, reason = _accept_stt_result(result, signal_quality=quality)

        self.assertFalse(accepted)
        self.assertEqual(reason, "repetitivt filler-støjfragment")

    def test_rejects_i_prefixed_den_her_jeg_kan_noise_turn(self) -> None:
        result = STTResult(
            text="jeg kan i den her den her",
            audio=AudioStats(duration_seconds=9.0, rms=3099, peak=32768, sample_rate=24000, channels=1),
            avg_logprob=-0.65,
            no_speech_prob=0.0,
            compression_ratio=0.0,
        )
        quality = TurnSignalQuality(
            duration_seconds=9.0,
            median_rms=700,
            p80_rms=2500,
            p95_rms=4300,
            peak=32768,
            active_ratio=0.50,
            active_ms=4500,
            max_active_run_ms=560,
            crest_factor=14.0,
            active_threshold=420,
            zero_crossing_rate=0.18,
            speech_band_ms=4400,
            max_speech_band_run_ms=500,
        )

        accepted, reason = _accept_stt_result(result, signal_quality=quality)

        self.assertFalse(accepted)
        self.assertEqual(reason, "repetitivt filler-støjfragment")

    def test_rejects_clipped_sparse_repeating_noise_turn(self) -> None:
        result = STTResult(
            text="den her den her til for",
            audio=AudioStats(duration_seconds=9.0, rms=1335, peak=32768, sample_rate=24000, channels=1),
            avg_logprob=-1.04,
            no_speech_prob=0.0,
            compression_ratio=0.0,
        )
        quality = TurnSignalQuality(
            duration_seconds=9.0,
            median_rms=242,
            p80_rms=900,
            p95_rms=2233,
            peak=32768,
            active_ratio=0.28,
            active_ms=2560,
            max_active_run_ms=280,
            crest_factor=56.7,
            active_threshold=420,
            zero_crossing_rate=0.15,
            speech_band_ms=2540,
            max_speech_band_run_ms=240,
        )

        accepted, reason = _accept_stt_result(result, signal_quality=quality)

        self.assertFalse(accepted)
        self.assertEqual(reason, "clippet støj uden sammenhængende tale")

    def test_rejects_repeating_filler_noise_even_with_moderate_confidence(self) -> None:
        result = STTResult(
            text="jeg kan den her den her den her den her den her du den",
            audio=AudioStats(duration_seconds=9.0, rms=1200, peak=25000, sample_rate=24000, channels=1),
            avg_logprob=-0.40,
            no_speech_prob=0.0,
            compression_ratio=0.0,
        )
        quality = TurnSignalQuality(
            duration_seconds=9.0,
            median_rms=220,
            p80_rms=900,
            p95_rms=1500,
            peak=25000,
            active_ratio=0.22,
            active_ms=1980,
            max_active_run_ms=260,
            crest_factor=38.0,
            active_threshold=420,
            zero_crossing_rate=0.13,
            speech_band_ms=1960,
            max_speech_band_run_ms=220,
        )

        accepted, reason = _accept_stt_result(result, signal_quality=quality)

        self.assertFalse(accepted)
        self.assertEqual(reason, "clippet støj uden sammenhængende tale")

    def test_voice_memory_policy_defaults_to_trusted_session(self) -> None:
        policy = _voice_memory_policy("trusted")

        self.assertTrue(policy.persist_session)
        self.assertTrue(policy.allow_memory_writes)
        self.assertTrue(policy.remember_recent)
        self.assertEqual(policy.session_source, "stackchan-voice")

    def test_voice_memory_policy_can_disable_writes(self) -> None:
        policy = _voice_memory_policy("off")

        self.assertFalse(policy.persist_session)
        self.assertFalse(policy.allow_memory_writes)
        self.assertFalse(policy.remember_recent)

    def test_local_realtime_reply_bypasses_brain_for_wait_commands(self) -> None:
        self.assertEqual(_parse_local_realtime_reply("vent lige"), "Jeg venter.")
        self.assertEqual(_parse_local_realtime_reply("stop lige"), "Jeg venter.")
        self.assertIsNone(_parse_local_realtime_reply("hvad laver du"))

    def test_parse_motion_command(self) -> None:
        self.assertEqual((_parse_motion_command("kig til venstre") or None).gesture, "look_left")
        self.assertEqual((_parse_motion_command("kig til højre") or None).gesture, "look_right")
        self.assertEqual((_parse_motion_command("gik til venstre") or None).gesture, "look_left")
        self.assertEqual((_parse_motion_command("gik op") or None).gesture, "look_up")
        self.assertEqual((_parse_motion_command("kigger lidt mig op ad") or None).gesture, "look_up")
        self.assertEqual((_parse_motion_command("rest paa hovedet") or None).gesture, "shake")
        self.assertEqual((_parse_motion_command("kan du danse") or None).gesture, "dance")
        self.assertEqual((_parse_motion_command("nik med hovedet") or None).gesture, "nod")
        self.assertEqual((_parse_motion_command("prøv en bevægelse") or None).gesture, "demo")
        self.assertIsNone(_parse_motion_command("skru op"))
        self.assertIsNone(_parse_motion_command("hvordan skulle den funktion fungere"))

    def test_motion_gesture_uses_restrained_physical_profile(self) -> None:
        actor = FakeMotionActor()

        self.assertTrue(_run_motion_gesture(actor, "shake", speed=900))

        self.assertEqual(actor.gestures, [("shake", 0.10, 190)])

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
        self.assertIsNone(_parse_calibration_command("kig lidt op ad"))
        self.assertIsNone(_parse_calibration_command("kan du justere skaermlysestyrken lidt ned"))

    def test_parse_display_brightness_command(self) -> None:
        down = _parse_display_brightness_command("kan du justere skaermlysestyrken lidt ned", current_level=80)
        absolute = _parse_display_brightness_command("saet skaermen til 35", current_level=80)
        followup = _parse_display_brightness_command("lidt mere", current_level=65, previous_direction=-1)

        self.assertEqual(down.level if down else None, 65)
        self.assertEqual(down.direction if down else None, -1)
        self.assertEqual(absolute.level if absolute else None, 35)
        self.assertEqual(followup.level if followup else None, 50)

    def test_parse_battery_status_command(self) -> None:
        self.assertTrue(_parse_battery_status_command("batteri status="))
        self.assertTrue(_parse_battery_status_command("hvor meget strom er der"))
        self.assertFalse(_parse_battery_status_command("status på lyden"))

    def test_format_battery_status_reply(self) -> None:
        self.assertEqual(
            _format_battery_status_reply({"batteryLevel": 78, "batteryCharging": True}),
            "Mit batteri er på 78 procent og jeg oplader.",
        )
        self.assertEqual(
            _format_battery_status_reply({"batteryLevel": 15, "batteryCharging": False}),
            "Mit batteri er på 15 procent og jeg kører på batteri, så det er lavt.",
        )
        self.assertEqual(
            _format_battery_status_reply({}),
            "Jeg har ikke fået batteridata fra firmware endnu.",
        )

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
