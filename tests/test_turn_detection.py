from __future__ import annotations

import math
import unittest

from stacky.voice.turn_detection import EnergyTurnDetector, analyze_turn_signal, pcm16_rms


def pcm_sample(value: int, count: int) -> bytes:
    return int(value).to_bytes(2, "little", signed=True) * count


def pcm_pulse(value: int, count: int, *, every: int) -> bytes:
    samples = []
    for index in range(count):
        sample = value if index % every == 0 else 120
        samples.append(int(sample).to_bytes(2, "little", signed=True))
    return b"".join(samples)


def pcm_tone(frequency: int, seconds: float, *, sample_rate: int = 16000, amplitude: int = 4000) -> bytes:
    samples = []
    sample_count = int(seconds * sample_rate)
    for index in range(sample_count):
        sample = int(math.sin(2.0 * math.pi * frequency * index / sample_rate) * amplitude)
        samples.append(sample.to_bytes(2, "little", signed=True))
    return b"".join(samples)


class TurnDetectionTest(unittest.TestCase):
    def test_rms_detects_pcm_energy(self) -> None:
        self.assertEqual(pcm16_rms(pcm_sample(0, 1600)), 0)
        self.assertGreater(pcm16_rms(pcm_sample(2000, 1600)), 1000)

    def test_detector_returns_turn_after_silence(self) -> None:
        detector = EnergyTurnDetector(threshold=500, min_speech_ms=100, end_silence_ms=200, preroll_ms=0)
        sample_rate = 16000

        self.assertIsNone(detector.push(pcm_sample(1800, 1600), sample_rate=sample_rate))
        self.assertIsNone(detector.push(pcm_sample(1800, 1600), sample_rate=sample_rate))
        self.assertIsNone(detector.push(pcm_sample(0, 1600), sample_rate=sample_rate))
        turn = detector.push(pcm_sample(0, 1600), sample_rate=sample_rate)

        self.assertIsNotNone(turn)
        self.assertGreater(len(turn.pcm), 0)

    def test_signal_quality_rejects_sparse_keyboard_clicks(self) -> None:
        quality = analyze_turn_signal(pcm_pulse(14000, 16000, every=1600), sample_rate=16000)

        self.assertFalse(quality.speech_like)
        self.assertIn(quality.reason, {"klik/percussiv støj", "for lidt sammenhængende tale"})

    def test_signal_quality_rejects_high_frequency_noise(self) -> None:
        quality = analyze_turn_signal(pcm_tone(5000, 1.0), sample_rate=16000)

        self.assertFalse(quality.speech_like)
        self.assertEqual(quality.reason, "højfrekvent støj")

    def test_signal_quality_accepts_sustained_voice_like_audio(self) -> None:
        pcm = pcm_sample(1500, 6400) + pcm_sample(1800, 6400) + pcm_sample(100, 3200)

        quality = analyze_turn_signal(pcm, sample_rate=16000)

        self.assertTrue(quality.speech_like)


if __name__ == "__main__":
    unittest.main()
