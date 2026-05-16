from __future__ import annotations

import math
import unittest

from stacky.voice.turn_detection import EnergyTurnDetector, TurnSignalQuality, analyze_turn_signal, pcm16_rms


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


def pcm_speech_like_tone(frequency: int, seconds: float, *, sample_rate: int = 16000, amplitude: int = 1200) -> bytes:
    samples = []
    sample_count = int(seconds * sample_rate)
    for index in range(sample_count):
        carrier = math.sin(2.0 * math.pi * frequency * index / sample_rate)
        envelope = 0.65 + 0.35 * math.sin(2.0 * math.pi * 5 * index / sample_rate)
        sample = int(carrier * envelope * amplitude)
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

    def test_detector_treats_learned_noise_floor_as_silence_after_speech(self) -> None:
        detector = EnergyTurnDetector(
            threshold=280,
            min_speech_ms=100,
            start_speech_ms=80,
            end_silence_ms=220,
            preroll_ms=0,
        )
        sample_rate = 16000

        for _ in range(20):
            self.assertIsNone(detector.push(pcm_sample(340, 320), sample_rate=sample_rate))
        self.assertIsNone(detector.push(pcm_sample(1800, 1600), sample_rate=sample_rate))
        turn = None
        for _ in range(12):
            turn = detector.push(pcm_sample(340, 320), sample_rate=sample_rate)
            if turn is not None:
                break

        self.assertIsNotNone(turn)
        self.assertLess((len(turn.pcm) // 2) / sample_rate, 2.0)

    def test_detector_starts_on_soft_speech_after_noisy_room_floor(self) -> None:
        detector = EnergyTurnDetector(
            threshold=280,
            min_speech_ms=100,
            start_speech_ms=80,
            end_silence_ms=220,
            preroll_ms=0,
        )
        sample_rate = 16000

        for _ in range(30):
            self.assertIsNone(detector.push(pcm_sample(300, 320), sample_rate=sample_rate))
        self.assertIsNone(detector.push(pcm_sample(460, 640), sample_rate=sample_rate))
        self.assertIsNone(detector.push(pcm_sample(460, 640), sample_rate=sample_rate))
        self.assertIsNone(detector.push(pcm_sample(460, 640), sample_rate=sample_rate))
        turn = None
        for _ in range(12):
            turn = detector.push(pcm_sample(300, 320), sample_rate=sample_rate)
            if turn is not None:
                break

        self.assertIsNotNone(turn)

    def test_signal_quality_rejects_sparse_keyboard_clicks(self) -> None:
        quality = analyze_turn_signal(pcm_pulse(14000, 16000, every=1600), sample_rate=16000)

        self.assertFalse(quality.speech_like)
        self.assertIn(quality.reason, {"klik/percussiv støj", "for lidt sammenhængende tale"})

    def test_signal_quality_rejects_clipped_percussive_noise(self) -> None:
        quality = TurnSignalQuality(
            duration_seconds=4.82,
            median_rms=250,
            p80_rms=520,
            p95_rms=2200,
            peak=32768,
            active_ratio=0.29,
            active_ms=1400,
            max_active_run_ms=120,
            crest_factor=40.0,
            active_threshold=520,
            zero_crossing_rate=0.2,
        )

        self.assertFalse(quality.speech_like)
        self.assertEqual(quality.reason, "klik/percussiv støj")

    def test_signal_quality_accepts_soft_short_speech_run(self) -> None:
        quality = TurnSignalQuality(
            duration_seconds=1.10,
            median_rms=67,
            p80_rms=350,
            p95_rms=914,
            peak=2387,
            active_ratio=0.18,
            active_ms=200,
            max_active_run_ms=200,
            crest_factor=11.3,
            active_threshold=420,
            zero_crossing_rate=0.12,
        )

        self.assertTrue(quality.speech_like)

    def test_signal_quality_rejects_high_frequency_noise(self) -> None:
        quality = analyze_turn_signal(pcm_tone(5000, 1.0), sample_rate=16000)

        self.assertFalse(quality.speech_like)
        self.assertEqual(quality.reason, "højfrekvent støj")

    def test_signal_quality_accepts_noisy_speech_with_voice_band_frames(self) -> None:
        quality = TurnSignalQuality(
            duration_seconds=1.8,
            median_rms=860,
            p80_rms=1200,
            p95_rms=3100,
            peak=12000,
            active_ratio=0.62,
            active_ms=1100,
            max_active_run_ms=340,
            crest_factor=12.0,
            active_threshold=700,
            zero_crossing_rate=0.47,
            speech_band_ms=520,
            max_speech_band_run_ms=260,
        )

        self.assertTrue(quality.speech_like)

    def test_detector_does_not_lock_on_high_frequency_noise_floor(self) -> None:
        detector = EnergyTurnDetector(
            threshold=280,
            min_speech_ms=100,
            start_speech_ms=80,
            end_silence_ms=220,
            preroll_ms=0,
        )
        sample_rate = 16000

        for _ in range(60):
            self.assertIsNone(detector.push(pcm_tone(5000, 0.02, amplitude=900), sample_rate=sample_rate))
        self.assertIsNone(detector.push(pcm_speech_like_tone(180, 0.08, amplitude=850), sample_rate=sample_rate))
        self.assertIsNone(detector.push(pcm_speech_like_tone(220, 0.08, amplitude=900), sample_rate=sample_rate))
        turn = None
        for _ in range(12):
            turn = detector.push(pcm_tone(5000, 0.02, amplitude=900), sample_rate=sample_rate)
            if turn is not None:
                break

        self.assertIsNotNone(turn)
        self.assertLess((len(turn.pcm) // 2) / sample_rate, 2.0)

    def test_signal_quality_accepts_sustained_voice_like_audio(self) -> None:
        pcm = pcm_sample(1500, 6400) + pcm_sample(1800, 6400) + pcm_sample(100, 3200)

        quality = analyze_turn_signal(pcm, sample_rate=16000)

        self.assertTrue(quality.speech_like)


if __name__ == "__main__":
    unittest.main()
