from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from statistics import median, quantiles


@dataclass(frozen=True)
class AudioTurn:
    pcm: bytes
    sample_rate: int
    channels: int = 1


@dataclass(frozen=True)
class TurnSignalQuality:
    duration_seconds: float
    median_rms: int
    p80_rms: int
    p95_rms: int
    peak: int
    active_ratio: float
    active_ms: int
    max_active_run_ms: int
    crest_factor: float
    active_threshold: int
    zero_crossing_rate: float = 0.0

    @property
    def speech_like(self) -> bool:
        if self.duration_seconds < 0.65:
            return False
        if self.p95_rms < 420 and self.peak < 1400:
            return False
        if self.zero_crossing_rate >= 0.45:
            return False
        if self.zero_crossing_rate >= 0.32 and self.active_ratio <= 0.75:
            return False
        if self.percussive_noise_like:
            return False
        if self.active_ratio <= 0.12 and self.max_active_run_ms < 180:
            return False
        return self.active_ms >= 220 or self.max_active_run_ms >= 220

    @property
    def percussive_noise_like(self) -> bool:
        if self.peak >= 32000 and self.crest_factor >= 20.0 and self.p95_rms >= 2200:
            return True
        if self.peak >= 32000 and self.crest_factor >= 24.0 and self.max_active_run_ms <= 160:
            return True
        return self.crest_factor >= 28.0 and (self.active_ratio <= 0.35 or self.max_active_run_ms <= 160)

    @property
    def reason(self) -> str:
        if self.speech_like:
            return "speech-like"
        if self.duration_seconds < 0.65:
            return "for kort signal"
        if self.p95_rms < 420 and self.peak < 1400:
            return "lavt signal"
        if self.zero_crossing_rate >= 0.45 or (self.zero_crossing_rate >= 0.32 and self.active_ratio <= 0.75):
            return "højfrekvent støj"
        if self.percussive_noise_like:
            return "klik/percussiv støj"
        if self.active_ratio <= 0.12 and self.max_active_run_ms < 180:
            return "for lidt sammenhængende tale"
        if self.active_ms < 220 and self.max_active_run_ms < 220:
            return "for lidt sammenhængende tale"
        return "ikke tale-lignende"


class EnergyTurnDetector:
    """Small PCM16 voice turn detector for StackChan mic chunks."""

    def __init__(
        self,
        *,
        threshold: int = 520,
        min_speech_ms: int = 150,
        end_silence_ms: int = 450,
        start_speech_ms: int = 120,
        preroll_ms: int = 120,
        max_utterance_ms: int = 9000,
    ) -> None:
        self.threshold = threshold
        self.min_speech_ms = min_speech_ms
        self.end_silence_ms = end_silence_ms
        self.start_speech_ms = start_speech_ms
        self.preroll_ms = preroll_ms
        self.max_utterance_ms = max_utterance_ms
        self._preroll: deque[tuple[bytes, int, int]] = deque()
        self._frames: list[bytes] = []
        self._active = False
        self._candidate_voice_ms = 0
        self._speech_ms = 0
        self._silence_ms = 0
        self._utterance_ms = 0
        self._noise_rms = float(threshold)

    def push(self, pcm: bytes, *, sample_rate: int, channels: int = 1) -> AudioTurn | None:
        if not pcm:
            return None
        duration_ms = _duration_ms(pcm, sample_rate=sample_rate, channels=channels)
        rms = pcm16_rms(pcm)
        voice_threshold = self._active_threshold() if self._active else self._start_threshold()
        is_voice = rms >= voice_threshold

        if not self._active:
            if not is_voice:
                self._update_noise_floor(rms)
                self._candidate_voice_ms = 0
            self._remember_preroll(pcm, duration_ms, sample_rate)
            if not is_voice:
                return None
            self._candidate_voice_ms += duration_ms
            if self._candidate_voice_ms < self.start_speech_ms:
                return None
            self._active = True
            self._frames = [frame for frame, _, _ in self._preroll]
            self._speech_ms = self._candidate_voice_ms
            self._silence_ms = 0
            self._utterance_ms = sum(ms for _, ms, _ in self._preroll)
            self._candidate_voice_ms = 0
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
        self._candidate_voice_ms = 0
        self._speech_ms = 0
        self._silence_ms = 0
        self._utterance_ms = 0
        self._noise_rms = max(self._noise_rms, 1.0)

    def _start_threshold(self) -> int:
        return int(max(self.threshold, self._noise_rms * 2.0, self._noise_rms + 170))

    def _active_threshold(self) -> int:
        return int(max(self.threshold, self._noise_rms * 1.35, self._noise_rms + 120))

    def _update_noise_floor(self, rms: int) -> None:
        if rms <= 0:
            return
        if rms > self._start_threshold():
            return
        self._noise_rms = self._noise_rms * 0.96 + float(rms) * 0.04


def pcm16_rms(pcm: bytes) -> int:
    if len(pcm) < 2:
        return 0
    count = len(pcm) // 2
    total = 0
    for index in range(0, count * 2, 2):
        sample = int.from_bytes(pcm[index : index + 2], "little", signed=True)
        total += sample * sample
    return int(math.sqrt(total / count))


def analyze_turn_signal(pcm: bytes, *, sample_rate: int, channels: int = 1, frame_ms: int = 20) -> TurnSignalQuality:
    if not pcm or sample_rate <= 0 or channels <= 0:
        return TurnSignalQuality(0.0, 0, 0, 0, 0, 0.0, 0, 0, 0.0, 0)

    frame_samples = max(1, int(sample_rate * frame_ms / 1000))
    values = _pcm16_mono_samples(pcm, channels=channels)
    frame_rms: list[int] = []
    frame_peak: list[int] = []
    for offset in range(0, len(values), frame_samples):
        frame = values[offset : offset + frame_samples]
        if not frame:
            continue
        frame_rms.append(int(math.sqrt(sum(sample * sample for sample in frame) / len(frame))))
        frame_peak.append(max(abs(sample) for sample in frame))

    if not frame_rms:
        return TurnSignalQuality(0.0, 0, 0, 0, 0, 0.0, 0, 0, 0.0, 0)

    med = int(median(frame_rms))
    p80 = _percentile(frame_rms, 80)
    p95 = _percentile(frame_rms, 95)
    peak = max(frame_peak)
    sorted_rms = sorted(frame_rms)
    quiet_count = max(1, len(sorted_rms) // 5)
    quiet_floor = int(median(sorted_rms[:quiet_count]))
    noise_based_threshold = max(420, quiet_floor * 2.2, quiet_floor + 220)
    signal_based_cap = max(420, p80 * 0.55, p95 * 0.45)
    active_threshold = int(min(noise_based_threshold, signal_based_cap))
    active = [rms >= active_threshold for rms in frame_rms]
    active_count = sum(1 for item in active if item)
    active_ratio = active_count / len(active)
    active_ms = active_count * frame_ms
    max_active_run_ms = _max_true_run(active) * frame_ms
    avg_rms = sum(frame_rms) / len(frame_rms)
    crest_factor = peak / max(avg_rms, 1.0)
    zero_crossing_rate = _zero_crossing_rate(values)
    duration_seconds = len(values) / sample_rate
    return TurnSignalQuality(
        duration_seconds=duration_seconds,
        median_rms=med,
        p80_rms=p80,
        p95_rms=p95,
        peak=peak,
        active_ratio=active_ratio,
        active_ms=active_ms,
        max_active_run_ms=max_active_run_ms,
        crest_factor=crest_factor,
        active_threshold=active_threshold,
        zero_crossing_rate=zero_crossing_rate,
    )


def _pcm16_mono_samples(pcm: bytes, *, channels: int) -> list[int]:
    frame_bytes = max(2, channels * 2)
    values: list[int] = []
    for index in range(0, len(pcm) - frame_bytes + 1, frame_bytes):
        values.append(int.from_bytes(pcm[index : index + 2], "little", signed=True))
    return values


def _percentile(values: list[int], percentile: int) -> int:
    if not values:
        return 0
    if len(values) < 5:
        return int(max(values))
    index = max(0, min(99, percentile)) - 1
    return int(quantiles(values, n=100)[index])


def _max_true_run(values: list[bool]) -> int:
    best = 0
    current = 0
    for value in values:
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def _zero_crossing_rate(values: list[int]) -> float:
    if len(values) < 2:
        return 0.0
    crossings = 0
    previous = values[0]
    for current in values[1:]:
        if previous == 0:
            previous = current
            continue
        if current != 0 and ((previous < 0 < current) or (previous > 0 > current)):
            crossings += 1
        previous = current
    return crossings / max(1, len(values) - 1)


def _duration_ms(pcm: bytes, *, sample_rate: int, channels: int) -> int:
    if sample_rate <= 0 or channels <= 0:
        return 0
    samples = len(pcm) // (2 * channels)
    return int((samples / sample_rate) * 1000)
