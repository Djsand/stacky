from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class AudioTurn:
    pcm: bytes
    sample_rate: int
    channels: int = 1


class EnergyTurnDetector:
    """Small PCM16 voice turn detector for StackChan mic chunks."""

    def __init__(
        self,
        *,
        threshold: int = 520,
        min_speech_ms: int = 150,
        end_silence_ms: int = 450,
        preroll_ms: int = 120,
        max_utterance_ms: int = 9000,
    ) -> None:
        self.threshold = threshold
        self.min_speech_ms = min_speech_ms
        self.end_silence_ms = end_silence_ms
        self.preroll_ms = preroll_ms
        self.max_utterance_ms = max_utterance_ms
        self._preroll: deque[tuple[bytes, int, int]] = deque()
        self._frames: list[bytes] = []
        self._active = False
        self._speech_ms = 0
        self._silence_ms = 0
        self._utterance_ms = 0

    def push(self, pcm: bytes, *, sample_rate: int, channels: int = 1) -> AudioTurn | None:
        if not pcm:
            return None
        duration_ms = _duration_ms(pcm, sample_rate=sample_rate, channels=channels)
        rms = pcm16_rms(pcm)
        is_voice = rms >= self.threshold

        if not self._active:
            self._remember_preroll(pcm, duration_ms, sample_rate)
            if not is_voice:
                return None
            self._active = True
            self._frames = [frame for frame, _, _ in self._preroll]
            self._speech_ms = duration_ms
            self._silence_ms = 0
            self._utterance_ms = sum(ms for _, ms, _ in self._preroll)
            return None

        self._frames.append(pcm)
        self._utterance_ms += duration_ms
        if is_voice:
            self._speech_ms += duration_ms
            self._silence_ms = 0
        else:
            self._silence_ms += duration_ms

        if self._utterance_ms >= self.max_utterance_ms:
            return self._finish(sample_rate=sample_rate, channels=channels)
        if self._speech_ms >= self.min_speech_ms and self._silence_ms >= self.end_silence_ms:
            return self._finish(sample_rate=sample_rate, channels=channels)
        return None

    def _remember_preroll(self, pcm: bytes, duration_ms: int, sample_rate: int) -> None:
        self._preroll.append((pcm, duration_ms, sample_rate))
        while sum(ms for _, ms, _ in self._preroll) > self.preroll_ms:
            self._preroll.popleft()

    def _finish(self, *, sample_rate: int, channels: int) -> AudioTurn | None:
        pcm = b"".join(self._frames)
        enough_speech = self._speech_ms >= self.min_speech_ms
        self.reset()
        if not enough_speech:
            return None
        return AudioTurn(pcm=pcm, sample_rate=sample_rate, channels=channels)

    def reset(self) -> None:
        self._preroll.clear()
        self._frames = []
        self._active = False
        self._speech_ms = 0
        self._silence_ms = 0
        self._utterance_ms = 0


def pcm16_rms(pcm: bytes) -> int:
    if len(pcm) < 2:
        return 0
    count = len(pcm) // 2
    total = 0
    for index in range(0, count * 2, 2):
        sample = int.from_bytes(pcm[index : index + 2], "little", signed=True)
        total += sample * sample
    return int(math.sqrt(total / count))


def _duration_ms(pcm: bytes, *, sample_rate: int, channels: int) -> int:
    if sample_rate <= 0 or channels <= 0:
        return 0
    samples = len(pcm) // (2 * channels)
    return int((samples / sample_rate) * 1000)
