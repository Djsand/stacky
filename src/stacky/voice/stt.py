from __future__ import annotations

import asyncio
import math
import wave
from dataclasses import dataclass
from pathlib import Path

from .turn_detection import pcm16_rms


@dataclass(frozen=True)
class AudioStats:
    duration_seconds: float
    rms: int
    peak: int
    sample_rate: int
    channels: int


@dataclass(frozen=True)
class STTResult:
    text: str
    audio: AudioStats
    avg_logprob: float
    no_speech_prob: float
    compression_ratio: float


class FasterWhisperDanishSTT:
    def __init__(self, model_size: str = "base", *, device: str = "cpu", compute_type: str = "int8") -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model = None

    async def preload(self) -> None:
        await asyncio.to_thread(self.load)

    def load(self) -> None:
        if self._model is not None:
            return
        from faster_whisper import WhisperModel

        self._model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)

    async def transcribe_wav(self, wav_path: Path) -> str:
        result = await self.transcribe_wav_result(wav_path)
        return result.text

    async def transcribe_wav_result(self, wav_path: Path) -> STTResult:
        return await asyncio.to_thread(self._transcribe_wav_result_sync, wav_path)

    def _transcribe_wav_result_sync(self, wav_path: Path) -> STTResult:
        audio = wav_audio_stats(wav_path)
        self.load()
        segments, _ = self._model.transcribe(
            str(wav_path),
            language="da",
            beam_size=1,
            vad_filter=False,
            condition_on_previous_text=False,
            initial_prompt="Dansk samtale mellem Nicolai og Stacky. Korte danske sætninger.",
        )
        texts: list[str] = []
        weighted_logprob = 0.0
        total_weight = 0.0
        no_speech_prob = 0.0
        compression_ratio = 0.0
        for segment in segments:
            text = segment.text.strip()
            if text:
                texts.append(text)
            weight = max(float(segment.end - segment.start), 0.01)
            weighted_logprob += float(getattr(segment, "avg_logprob", -10.0)) * weight
            total_weight += weight
            no_speech_prob = max(no_speech_prob, float(getattr(segment, "no_speech_prob", 0.0)))
            compression_ratio = max(compression_ratio, float(getattr(segment, "compression_ratio", 0.0)))

        if total_weight <= 0:
            return STTResult(
                text="",
                audio=audio,
                avg_logprob=-10.0,
                no_speech_prob=1.0,
                compression_ratio=0.0,
            )
        return STTResult(
            text=" ".join(texts).strip(),
            audio=audio,
            avg_logprob=weighted_logprob / total_weight,
            no_speech_prob=no_speech_prob,
            compression_ratio=compression_ratio,
        )


class Wav2Vec2DanishSTT:
    DEFAULT_MODEL = "CoRal-project/roest-v3-wav2vec2-315m"

    def __init__(self, model_id: str = DEFAULT_MODEL, *, device: str | None = None) -> None:
        self.model_id = model_id
        self.device = device
        self._processor = None
        self._model = None
        self._torch = None

    async def preload(self) -> None:
        await asyncio.to_thread(self.load)

    def load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCTC, AutoProcessor
        from transformers.utils import logging as transformers_logging

        self._torch = torch
        transformers_logging.set_verbosity_error()
        transformers_logging.disable_progress_bar()
        try:
            from huggingface_hub.utils import disable_progress_bars

            disable_progress_bars()
        except ImportError:
            pass
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = AutoModelForCTC.from_pretrained(self.model_id)
        if self.device == "cpu":
            self._model = self._model.float()
        self._model.to(self.device)
        self._model.eval()

    async def transcribe_wav(self, wav_path: Path) -> str:
        result = await self.transcribe_wav_result(wav_path)
        return result.text

    async def transcribe_wav_result(self, wav_path: Path) -> STTResult:
        return await asyncio.to_thread(self._transcribe_wav_result_sync, wav_path)

    def _transcribe_wav_result_sync(self, wav_path: Path) -> STTResult:
        audio = wav_audio_stats(wav_path)
        samples = wav_to_mono_float32(wav_path, target_sample_rate=16000, apply_agc=True)
        self.load()
        assert self._model is not None
        assert self._processor is not None
        assert self._torch is not None

        inputs = self._processor(samples, sampling_rate=16000, return_tensors="pt", padding=True)
        model_inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
            if hasattr(value, "to")
        }
        with self._torch.inference_mode():
            logits = self._model(**model_inputs).logits
        predicted_ids = self._torch.argmax(logits, dim=-1)
        text = _decode_ctc_text(self._processor, logits, predicted_ids)
        avg_logprob = _ctc_avg_logprob(
            logits,
            predicted_ids,
            blank_id=_ctc_blank_id(self._processor, self._model),
            torch_module=self._torch,
        )
        return STTResult(
            text=text,
            audio=audio,
            avg_logprob=avg_logprob,
            no_speech_prob=1.0 if not text else 0.0,
            compression_ratio=0.0,
        )


def create_danish_stt(engine: str, model_name: str | None = None):
    if engine == "whisper":
        return FasterWhisperDanishSTT(model_name or "small")
    if engine == "wav2vec2":
        return Wav2Vec2DanishSTT(model_name or Wav2Vec2DanishSTT.DEFAULT_MODEL)
    raise ValueError(f"Unknown STT engine: {engine}")


def write_pcm_wav(path: Path, pcm: bytes, *, sample_rate: int, channels: int = 1) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm)
    return path


def wav_audio_stats(path: Path) -> AudioStats:
    with wave.open(str(path), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        frame_count = wav_file.getnframes()
        pcm = wav_file.readframes(frame_count)

    duration_seconds = frame_count / sample_rate if sample_rate else 0.0
    if sample_width != 2:
        return AudioStats(
            duration_seconds=duration_seconds,
            rms=0,
            peak=0,
            sample_rate=sample_rate,
            channels=channels,
        )
    return AudioStats(
        duration_seconds=duration_seconds,
        rms=pcm16_rms(pcm),
        peak=_pcm16_peak(pcm),
        sample_rate=sample_rate,
        channels=channels,
    )


def wav_to_mono_float32(path: Path, *, target_sample_rate: int = 16000, apply_agc: bool = False):
    import numpy as np

    with wave.open(str(path), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        frame_count = wav_file.getnframes()
        pcm = wav_file.readframes(frame_count)

    if sample_width != 2:
        raise ValueError("Only PCM16 WAV audio is supported for local Danish STT.")
    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    if sample_rate != target_sample_rate:
        samples = _resample_linear(samples, sample_rate, target_sample_rate)
    if apply_agc:
        samples = apply_stt_agc(samples)
    return samples.astype(np.float32, copy=False)


def apply_stt_agc(samples, *, target_rms: float = 0.12, max_gain: float = 12.0, active_floor: float = 0.004):
    import numpy as np

    if len(samples) == 0:
        return samples
    active = samples[np.abs(samples) >= active_floor]
    if len(active) == 0:
        return samples.astype(np.float32, copy=False)
    rms = float(np.sqrt(np.mean(active.astype(np.float64) ** 2)))
    if rms <= 0:
        return samples.astype(np.float32, copy=False)
    gain = min(max_gain, max(1.0, target_rms / rms))
    if gain <= 1.01:
        return samples.astype(np.float32, copy=False)
    return np.clip(samples * gain, -0.98, 0.98).astype(np.float32, copy=False)


def _pcm16_peak(pcm: bytes) -> int:
    peak = 0
    for index in range(0, len(pcm) - 1, 2):
        sample = int.from_bytes(pcm[index : index + 2], "little", signed=True)
        peak = max(peak, abs(sample))
    return peak


def _resample_linear(samples, source_sample_rate: int, target_sample_rate: int):
    import numpy as np

    if source_sample_rate <= 0 or target_sample_rate <= 0 or len(samples) == 0:
        return samples
    if source_sample_rate == target_sample_rate:
        return samples
    duration = len(samples) / source_sample_rate
    target_count = max(1, int(duration * target_sample_rate))
    source_x = np.linspace(0.0, duration, num=len(samples), endpoint=False)
    target_x = np.linspace(0.0, duration, num=target_count, endpoint=False)
    return np.interp(target_x, source_x, samples).astype(np.float32)


def _ctc_blank_id(processor, model) -> int:
    tokenizer = getattr(processor, "tokenizer", None)
    blank_id = getattr(tokenizer, "pad_token_id", None)
    if blank_id is None:
        blank_id = getattr(model.config, "pad_token_id", 0)
    return int(blank_id)


def _decode_ctc_text(processor, logits, predicted_ids) -> str:
    if hasattr(processor, "decoder"):
        decoded = processor.batch_decode(logits.detach().cpu().numpy())
        return decoded.text[0].strip()
    return processor.batch_decode(predicted_ids)[0].strip()


def _ctc_avg_logprob(logits, predicted_ids, *, blank_id: int, torch_module) -> float:
    probs = torch_module.softmax(logits, dim=-1)
    max_probs = probs.max(dim=-1).values[0]
    ids = predicted_ids[0]
    non_blank = ids != blank_id
    if int(non_blank.sum().item()) == 0:
        return -10.0
    selected = max_probs[non_blank].clamp_min(1e-8)
    value = float(torch_module.log(selected).mean().item())
    if math.isnan(value) or math.isinf(value):
        return -10.0
    return value
