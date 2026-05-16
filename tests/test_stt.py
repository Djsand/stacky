from __future__ import annotations

import tempfile
import unittest
import wave
from pathlib import Path

from stacky.voice.stt import (
    Qwen3DanishSTT,
    Wav2Vec2DanishSTT,
    apply_stt_agc,
    create_danish_stt,
    resolve_stt_model_name,
    wav_audio_stats,
    wav_to_mono_float32,
    write_pcm_wav,
)


class STTTest(unittest.TestCase):
    def test_write_pcm_wav(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = write_pcm_wav(Path(tmp) / "sample.wav", b"\x00\x00" * 160, sample_rate=16000)

            with wave.open(str(path), "rb") as wav_file:
                self.assertEqual(wav_file.getframerate(), 16000)
                self.assertEqual(wav_file.getnchannels(), 1)
                self.assertEqual(wav_file.getsampwidth(), 2)

    def test_wav_audio_stats_reports_level(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pcm = int(1000).to_bytes(2, "little", signed=True) * 160
            path = write_pcm_wav(Path(tmp) / "sample.wav", pcm, sample_rate=16000)

            stats = wav_audio_stats(path)

            self.assertAlmostEqual(stats.duration_seconds, 0.01, places=3)
            self.assertGreater(stats.rms, 900)
            self.assertEqual(stats.peak, 1000)

    def test_wav_to_mono_float32(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pcm = int(3276).to_bytes(2, "little", signed=True) * 160
            path = write_pcm_wav(Path(tmp) / "sample.wav", pcm, sample_rate=16000)

            samples = wav_to_mono_float32(path)

            self.assertEqual(len(samples), 160)
            self.assertAlmostEqual(float(samples[0]), 0.1, places=3)

    def test_apply_stt_agc_raises_quiet_active_speech(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pcm = int(500).to_bytes(2, "little", signed=True) * 160
            path = write_pcm_wav(Path(tmp) / "sample.wav", pcm, sample_rate=16000)

            raw = wav_to_mono_float32(path)
            boosted = apply_stt_agc(raw)

            self.assertGreater(float(abs(boosted[0])), float(abs(raw[0])) * 2)
            self.assertLessEqual(float(abs(boosted[0])), 0.98)

    def test_create_danish_stt_defaults_to_wav2vec2(self) -> None:
        stt = create_danish_stt("wav2vec2")

        self.assertIsInstance(stt, Wav2Vec2DanishSTT)

    def test_resolve_stt_model_aliases(self) -> None:
        self.assertEqual(resolve_stt_model_name("wav2vec2", "roest"), "CoRal-project/roest-v3-wav2vec2-315m")
        self.assertEqual(resolve_stt_model_name("wav2vec2", "ftspeech"), "saattrupdan/wav2vec2-xls-r-300m-ftspeech")
        self.assertEqual(resolve_stt_model_name("qwen3", "saga"), "capacit-ai/saga")

    def test_create_qwen3_stt_without_loading_optional_dependency(self) -> None:
        stt = create_danish_stt("qwen3", "qwen3-0.6b")

        self.assertIsInstance(stt, Qwen3DanishSTT)


if __name__ == "__main__":
    unittest.main()
