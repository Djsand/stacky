from __future__ import annotations

import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download

from ..config import ROOT
from .speech_adapter import adapt_for_danish_speech


VOICE_REPO = "rhasspy/piper-voices"
VOICE_MODEL_REPO_PATH = "da/da_DK/talesyntese/medium/da_DK-talesyntese-medium.onnx"
VOICE_CONFIG_REPO_PATH = "da/da_DK/talesyntese/medium/da_DK-talesyntese-medium.onnx.json"


@dataclass(frozen=True)
class PiperVoice:
    model_path: Path
    config_path: Path


def default_voice_dir() -> Path:
    return ROOT / "models" / "tts" / "piper" / "da_DK-talesyntese-medium"


def ensure_danish_piper_voice(voice_dir: Path | None = None) -> PiperVoice:
    voice_dir = voice_dir or default_voice_dir()
    voice_dir.mkdir(parents=True, exist_ok=True)
    local_model = voice_dir / VOICE_MODEL_REPO_PATH
    local_config = voice_dir / VOICE_CONFIG_REPO_PATH
    if local_model.exists() and local_config.exists():
        return PiperVoice(model_path=local_model, config_path=local_config)
    model_path = Path(hf_hub_download(VOICE_REPO, VOICE_MODEL_REPO_PATH, repo_type="model", local_dir=voice_dir))
    config_path = Path(hf_hub_download(VOICE_REPO, VOICE_CONFIG_REPO_PATH, repo_type="model", local_dir=voice_dir))
    return PiperVoice(model_path=model_path, config_path=config_path)


class PiperTTS:
    def __init__(self, voice: PiperVoice, *, piper_exe: Path | None = None) -> None:
        self.voice = voice
        self.piper_exe = piper_exe or ROOT / ".venv" / "Scripts" / "piper.exe"

    def synthesize_to_file(
        self,
        text: str,
        output_path: Path,
        *,
        length_scale: float = 1.03,
        sentence_silence: float = 0.18,
        volume: float = 1.0,
    ) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        spoken = adapt_for_danish_speech(text)
        command = [
            str(self.piper_exe),
            "--model",
            str(self.voice.model_path),
            "--config",
            str(self.voice.config_path),
            "--output_file",
            str(output_path),
            "--length-scale",
            str(length_scale),
            "--sentence-silence",
            str(sentence_silence),
            "--volume",
            str(volume),
        ]
        subprocess.run(command, input=spoken, text=True, check=True)
        return output_path


class FastPiperTTS:
    """In-process Piper runtime for low-latency local speech."""

    def __init__(self, voice: PiperVoice, *, use_cuda: bool = False) -> None:
        self.voice = voice
        self.use_cuda = use_cuda
        self._runtime_voice = None

    def load(self) -> None:
        if self._runtime_voice is not None:
            return
        from piper.voice import PiperVoice as RuntimePiperVoice

        self._runtime_voice = RuntimePiperVoice.load(
            self.voice.model_path,
            config_path=self.voice.config_path,
            use_cuda=self.use_cuda,
        )

    def synthesize_to_file(
        self,
        text: str,
        output_path: Path,
        *,
        length_scale: float = 1.03,
        sentence_silence: float = 0.18,
        volume: float = 1.0,
    ) -> Path:
        import wave

        from piper.config import SynthesisConfig

        self.load()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        spoken = adapt_for_danish_speech(text)
        syn_config = SynthesisConfig(length_scale=length_scale, volume=volume)
        silence_frames = 0

        with wave.open(str(output_path), "wb") as wav_file:
            first_chunk = True
            for chunk in self._runtime_voice.synthesize(spoken, syn_config=syn_config):
                if first_chunk:
                    wav_file.setframerate(chunk.sample_rate)
                    wav_file.setsampwidth(chunk.sample_width)
                    wav_file.setnchannels(chunk.sample_channels)
                    silence_frames = int(chunk.sample_rate * sentence_silence)
                    first_chunk = False
                wav_file.writeframes(chunk.audio_int16_bytes)
                if silence_frames > 0:
                    wav_file.writeframes(b"\x00" * silence_frames * chunk.sample_width * chunk.sample_channels)

        return output_path


def pitch_shift_wav(input_path: Path, output_path: Path, *, factor: float = 1.12) -> Path:
    """Pitch-shift a WAV locally with ffmpeg while roughly preserving duration."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(input_path), "rb") as wav:
        sample_rate = wav.getframerate()
    tempo = 1.0 / factor
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-af",
        f"asetrate={int(sample_rate * factor)},aresample={sample_rate},atempo={tempo:.6f}",
        str(output_path),
    ]
    subprocess.run(command, check=True)
    return output_path
