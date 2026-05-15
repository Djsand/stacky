from __future__ import annotations

import unittest

from stacky.voice.turn_detection import EnergyTurnDetector, pcm16_rms


def pcm_sample(value: int, count: int) -> bytes:
    return int(value).to_bytes(2, "little", signed=True) * count


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


if __name__ == "__main__":
    unittest.main()
