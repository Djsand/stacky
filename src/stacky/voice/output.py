from __future__ import annotations

import asyncio
import contextlib
import math
import shutil
import subprocess
import threading
import time
import wave
from pathlib import Path
from typing import Protocol

from ..body.controller import StackChanBodyController
from ..config import ROOT
from ..danish import add_spoken_question_markers
from .piper_tts import FastPiperTTS, ensure_danish_piper_voice
from .speech_adapter import split_for_speech
from .supertonic_tts import SupertonicTTS, SupertonicVoice


class FileTTS(Protocol):
    def load(self) -> None:
        ...

    def synthesize_to_file(self, text: str, output_path: Path, **kwargs: object) -> Path:
        ...


class PiperSpeechOutput:
    """Low-latency local speech output with cancellable playback."""

    def __init__(
        self,
        tts: FileTTS,
        *,
        output_dir: Path | None = None,
        player: str = "ffplay",
        chunk_chars: int = 95,
    ) -> None:
        self.tts = tts
        self.output_dir = output_dir or ROOT / "artifacts" / "live_speech"
        self.player = player
        self.chunk_chars = chunk_chars
        self._task: asyncio.Task[None] | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self._utterance_id = 0

    async def preload(self) -> None:
        await asyncio.to_thread(self.tts.load)

    async def speak(self, text: str) -> None:
        await self.stop()
        text = add_spoken_question_markers(text)
        print(f"Stacky: {text}", flush=True)
        self._utterance_id += 1
        utterance_id = self._utterance_id
        self._task = asyncio.create_task(self._speak_chunks(text, utterance_id))

    async def stop(self) -> None:
        task = self._task
        self._task = None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._stop_process()

    async def wait(self) -> None:
        task = self._task
        if task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await task
            if self._task is task:
                self._task = None

    async def _speak_chunks(self, text: str, utterance_id: int) -> None:
        chunks = split_for_speech(text, max_chars=self.chunk_chars)
        if not chunks:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for index, chunk in enumerate(chunks, start=1):
            if utterance_id != self._utterance_id:
                return
            output = self.output_dir / f"{utterance_id:04d}-{index:02d}.wav"
            await asyncio.to_thread(self.tts.synthesize_to_file, chunk, output)
            if utterance_id != self._utterance_id:
                return
            self._process = subprocess.Popen(
                [self.player, "-nodisp", "-autoexit", "-loglevel", "quiet", str(output)]
            )
            returncode = await asyncio.to_thread(self._process.wait)
            self._process = None
            if returncode != 0 or utterance_id != self._utterance_id:
                return

    def _stop_process(self) -> None:
        process = self._process
        self._process = None
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=0.7)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=0.7)


def create_fast_piper_output() -> PiperSpeechOutput:
    return PiperSpeechOutput(FastPiperTTS(ensure_danish_piper_voice()))


def create_supertonic_output(voice: SupertonicVoice | None = None) -> PiperSpeechOutput:
    return PiperSpeechOutput(SupertonicTTS(voice), output_dir=ROOT / "artifacts" / "live_speech_supertonic")


class StackChanSpeechOutput:
    """Speak through StackChan speaker by sending PCM audio.out commands."""

    def __init__(
        self,
        tts: FileTTS,
        controller: StackChanBodyController,
        *,
        output_dir: Path | None = None,
        chunk_chars: int = 280,
        stackchan_sample_rate: int = 24000,
        max_stackchan_pcm_bytes: int = 1_020_000,
        rhythm_gap_seconds: float = 0.04,
        rhythmic_chunks: bool = False,
        target_active_rms: int = 9000,
        max_gain: float = 4.0,
        volume_level: int = 80,
    ) -> None:
        self.tts = tts
        self.controller = controller
        self.output_dir = output_dir or ROOT / "artifacts" / "stackchan_speech"
        self.chunk_chars = chunk_chars
        self.stackchan_sample_rate = stackchan_sample_rate
        self.max_stackchan_pcm_bytes = max_stackchan_pcm_bytes
        self.rhythm_gap_seconds = rhythm_gap_seconds
        self.rhythmic_chunks = rhythmic_chunks
        self.target_active_rms = target_active_rms
        self.max_gain = max_gain
        self.volume_level = clamp_volume_level(volume_level)
        self._task: asyncio.Task[None] | None = None
        self._utterance_id = 0
        self._stop_requested = threading.Event()

    async def preload(self) -> None:
        await asyncio.to_thread(self.tts.load)
        self.controller.set_volume(self.volume_level)

    def set_volume(self, level: int) -> bool:
        self.volume_level = clamp_volume_level(level)
        self.target_active_rms = target_rms_for_stackchan_volume(self.volume_level)
        self.max_gain = max_gain_for_stackchan_volume(self.volume_level)
        return self.controller.set_volume(self.volume_level)

    async def speak(self, text: str) -> None:
        await self.stop()
        self._stop_requested.clear()
        text = add_spoken_question_markers(text)
        print(f"Stacky: {text}", flush=True)
        self._utterance_id += 1
        utterance_id = self._utterance_id
        self._task = asyncio.create_task(self._speak_chunks(text, utterance_id))

    async def stop(self) -> None:
        task = self._task
        if task is None and not self._stop_requested.is_set():
            return
        self._utterance_id += 1
        self._stop_requested.set()
        self.controller.interrupt_audio()
        self._task = None
        if task is not None:
            task.cancel()
            try:
                await asyncio.wait_for(task, timeout=0.8)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            except Exception as exc:
                print(f"[speech] StackChan output stopped after task error: {exc}", flush=True)
        self.controller.interrupt_audio()

    async def wait(self) -> None:
        task = self._task
        if task is not None:
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                print(f"[speech] StackChan output failed: {exc}", flush=True)
            finally:
                if self._task is task:
                    self._task = None

    async def _speak_chunks(self, text: str, utterance_id: int) -> None:
        chunks = split_for_speech(text, max_chars=self.chunk_chars, rhythmic=self.rhythmic_chunks)
        if not chunks:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        pcm_chunks: list[bytes] = []
        sample_rate: int | None = None
        channels: int | None = None
        for index, chunk in enumerate(chunks, start=1):
            if utterance_id != self._utterance_id:
                return
            output = self.output_dir / f"{utterance_id:04d}-{index:02d}.wav"
            await asyncio.to_thread(self.tts.synthesize_to_file, chunk, output)
            if utterance_id != self._utterance_id:
                return
            chunk_channels, chunk_sample_rate, pcm = await asyncio.to_thread(self._read_stackchan_pcm, output)
            if channels is None:
                channels = chunk_channels
                sample_rate = chunk_sample_rate
            elif channels != chunk_channels or sample_rate != chunk_sample_rate:
                raise ValueError("StackChan TTS chunks must share one WAV format")
            pcm_chunks.append(pcm)

        if utterance_id != self._utterance_id or not pcm_chunks or sample_rate is None or channels is None:
            return
        pcm = join_pcm16_chunks(
            pcm_chunks,
            sample_rate=sample_rate,
            channels=channels,
            gap_seconds=self.rhythm_gap_seconds,
        )
        duration = await asyncio.to_thread(
            self._send_pcm_to_stackchan,
            pcm,
            sample_rate=sample_rate,
            channels=channels,
        )
        if duration > 0:
            await asyncio.sleep(duration)

    def _send_wav_to_stackchan(self, wav_path: Path) -> float:
        channels, sample_rate, pcm = self._read_stackchan_pcm(wav_path)
        return self._send_pcm_to_stackchan(pcm, sample_rate=sample_rate, channels=channels)

    def _send_pcm_to_stackchan(self, pcm: bytes, *, sample_rate: int, channels: int) -> float:
        started = time.perf_counter()
        pcm = boost_pcm16_for_stackchan(
            pcm,
            target_active_rms=self.target_active_rms,
            max_gain=self.max_gain,
        )
        self.controller.hold_audio(True)
        try:
            for segment in split_pcm16_segments(
                pcm,
                sample_rate=sample_rate,
                channels=channels,
                max_bytes=self.max_stackchan_pcm_bytes,
            ):
                if self._stop_requested.is_set():
                    return 0.0
                segment_duration = len(segment) / max(1, sample_rate * channels * 2)
                self.controller.speak_audio_chunks(
                    segment,
                    sample_rate=sample_rate,
                    channels=channels,
                    chunk_bytes=16384,
                    chunk_delay_seconds=0.0,
                    wait_for_ack=False,
                    playback_timeout_seconds=segment_duration + 6.0,
                )
                if self._stop_requested.is_set():
                    return 0.0
                time.sleep(0.04)
        finally:
            self.controller.hold_audio(False)
        elapsed = time.perf_counter() - started
        return max(0.0, 0.05 - elapsed)

    def _read_stackchan_pcm(self, wav_path: Path) -> tuple[int, int, bytes]:
        stackchan_wav = self._prepare_stackchan_wav(wav_path)
        with wave.open(str(stackchan_wav), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()
            frames = wav_file.getnframes()
            pcm = wav_file.readframes(frames)
        if sample_width != 2:
            raise ValueError("StackChan speaker output expects PCM16 WAV data")
        return channels, sample_rate, pcm

    def _prepare_stackchan_wav(self, wav_path: Path) -> Path:
        with wave.open(str(wav_path), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()
        if channels == 1 and sample_width == 2 and sample_rate == self.stackchan_sample_rate:
            return wav_path

        output_path = wav_path.with_suffix(".stackchan.wav")
        if sample_width == 2 and channels in {1, 2}:
            return _write_stackchan_pcm16_wav(
                wav_path,
                output_path,
                target_sample_rate=self.stackchan_sample_rate,
            )

        ffmpeg_exe = shutil.which("ffmpeg")
        if not ffmpeg_exe:
            for candidate in (
                Path(r"C:\Users\nicol\miniconda3\Library\bin\ffmpeg.exe"),
                Path("/opt/homebrew/bin/ffmpeg"),
                Path("/usr/local/bin/ffmpeg"),
                Path("/usr/bin/ffmpeg"),
            ):
                if candidate.exists():
                    ffmpeg_exe = str(candidate)
                    break
        if not ffmpeg_exe:
            raise RuntimeError("ffmpeg not found on PATH or known install locations")
        temp_output = output_path.with_suffix(output_path.suffix + ".tmp")
        with contextlib.suppress(FileNotFoundError):
            temp_output.unlink()
        subprocess.run(
            [
                ffmpeg_exe,
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(wav_path),
                "-ac",
                "1",
                "-ar",
                str(self.stackchan_sample_rate),
                "-sample_fmt",
                "s16",
                str(temp_output),
            ],
            check=True,
            timeout=10.0,
        )
        if not temp_output.exists() or temp_output.stat().st_size <= 44:
            raise RuntimeError(f"ffmpeg produced empty StackChan WAV: {temp_output}")
        temp_output.replace(output_path)
        return output_path


def create_stackchan_piper_output(
    controller: StackChanBodyController,
    *,
    target_active_rms: int = 9000,
    max_gain: float = 4.0,
    volume_level: int = 80,
) -> StackChanSpeechOutput:
    return StackChanSpeechOutput(
        FastPiperTTS(ensure_danish_piper_voice()),
        controller,
        target_active_rms=target_active_rms,
        max_gain=max_gain,
        volume_level=volume_level,
    )


def create_stackchan_supertonic_output(
    controller: StackChanBodyController,
    voice: SupertonicVoice | None = None,
    *,
    target_active_rms: int = 9000,
    max_gain: float = 4.0,
    volume_level: int = 80,
) -> StackChanSpeechOutput:
    return StackChanSpeechOutput(
        SupertonicTTS(voice),
        controller,
        output_dir=ROOT / "artifacts" / "stackchan_speech_supertonic",
        chunk_chars=200,
        rhythm_gap_seconds=0.035,
        rhythmic_chunks=True,
        target_active_rms=target_active_rms,
        max_gain=max_gain,
        volume_level=volume_level,
    )


def _write_stackchan_pcm16_wav(
    input_path: Path,
    output_path: Path,
    *,
    target_sample_rate: int,
) -> Path:
    with wave.open(str(input_path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frames = wav_file.getnframes()
        raw = wav_file.readframes(frames)
    if sample_width != 2 or channels not in {1, 2}:
        raise ValueError("pure StackChan WAV conversion expects mono/stereo PCM16")

    samples = [
        int.from_bytes(raw[index : index + 2], "little", signed=True)
        for index in range(0, len(raw), 2)
    ]
    if channels == 2:
        samples = [
            int((samples[index] + samples[index + 1]) / 2)
            for index in range(0, len(samples) - 1, 2)
        ]

    resampled = _resample_pcm16_mono(samples, source_rate=sample_rate, target_rate=target_sample_rate)
    temp_output = output_path.with_suffix(output_path.suffix + ".tmp")
    with contextlib.suppress(FileNotFoundError):
        temp_output.unlink()
    with wave.open(str(temp_output), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(target_sample_rate)
        wav_file.writeframes(
            b"".join(int(sample).to_bytes(2, "little", signed=True) for sample in resampled)
        )
    if temp_output.stat().st_size <= 44:
        raise RuntimeError(f"empty StackChan WAV conversion: {temp_output}")
    temp_output.replace(output_path)
    return output_path


def _resample_pcm16_mono(samples: list[int], *, source_rate: int, target_rate: int) -> list[int]:
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("sample rates must be positive")
    if not samples or source_rate == target_rate:
        return samples

    target_count = max(1, round(len(samples) * target_rate / source_rate))
    ratio = source_rate / target_rate
    result: list[int] = []
    last_index = len(samples) - 1
    for out_index in range(target_count):
        source_position = out_index * ratio
        left_index = min(last_index, int(source_position))
        right_index = min(last_index, left_index + 1)
        fraction = source_position - left_index
        value = samples[left_index] * (1.0 - fraction) + samples[right_index] * fraction
        if value > 32767:
            value = 32767
        elif value < -32768:
            value = -32768
        result.append(int(round(value)))
    return result


def clamp_volume_level(level: int) -> int:
    return max(0, min(100, int(level)))


def target_rms_for_stackchan_volume(level: int) -> int:
    return 1800 + clamp_volume_level(level) * 90


def max_gain_for_stackchan_volume(level: int) -> float:
    return 2.0 + clamp_volume_level(level) * 0.04


def boost_pcm16_for_stackchan(
    pcm: bytes,
    *,
    target_active_rms: int = 4500,
    max_gain: float = 2.2,
    active_floor: int = 200,
) -> bytes:
    """Make local TTS audible on the tiny CoreS3 speaker."""

    sample_count = len(pcm) // 2
    if sample_count == 0:
        return pcm

    samples = [int.from_bytes(pcm[index : index + 2], "little", signed=True) for index in range(0, sample_count * 2, 2)]
    active = [sample for sample in samples if abs(sample) >= active_floor]
    if not active:
        return pcm

    active_rms = math.sqrt(sum(sample * sample for sample in active) / len(active))
    if active_rms <= 0:
        return pcm
    gain = min(max_gain, max(1.0, target_active_rms / active_rms))
    if gain <= 1.01:
        return pcm

    out = bytearray(len(pcm))
    for sample_index, sample in enumerate(samples):
        boosted = int(sample * gain)
        if boosted > 32767:
            boosted = 32767
        elif boosted < -32768:
            boosted = -32768
        out[sample_index * 2 : sample_index * 2 + 2] = boosted.to_bytes(2, "little", signed=True)
    return bytes(out)


def join_pcm16_chunks(
    chunks: list[bytes],
    *,
    sample_rate: int,
    channels: int,
    gap_seconds: float = 0.04,
) -> bytes:
    if not chunks:
        return b""
    frame_bytes = max(2, channels * 2)
    gap_frames = max(0, int(sample_rate * gap_seconds))
    gap = b"\x00" * gap_frames * frame_bytes
    return gap.join(chunks)


def split_pcm16_segments(
    pcm: bytes,
    *,
    sample_rate: int,
    channels: int,
    max_bytes: int = 160_000,
) -> list[bytes]:
    frame_bytes = max(2, channels * 2)
    max_bytes = max(frame_bytes, (max_bytes // frame_bytes) * frame_bytes)
    pcm = pcm[: len(pcm) - (len(pcm) % frame_bytes)]
    if len(pcm) <= max_bytes:
        return [pcm] if pcm else []
    return [pcm[offset : offset + max_bytes] for offset in range(0, len(pcm), max_bytes)]
