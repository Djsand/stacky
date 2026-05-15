from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import snapshot_download

from ..config import ROOT
from .speech_adapter import adapt_for_danish_speech


ROEST_REPO_ID = "CoRal-project/roest-v3-chatterbox-350m"
ROEST_ALLOW_PATTERNS = (
    "*.safetensors",
    "*.json",
    "*.txt",
    "*.pt",
    "*.model",
    "audio_samples/*.wav",
)
ROEST_PROMPTS = {
    "nic": "audio_samples/01_nic_00_t0.7_p0.95_k600_r1.2.wav",
    "mic": "audio_samples/01_mic_00_t0.7_p0.95_k600_r1.2.wav",
}


@dataclass(frozen=True)
class RoestVoice:
    model_dir: Path
    prompt_path: Path
    speaker: str


def default_roest_dir() -> Path:
    return ROOT / "models" / "tts" / "roest-v3-chatterbox-350m"


def ensure_roest_model(model_dir: Path | None = None) -> Path:
    model_dir = model_dir or default_roest_dir()
    required = model_dir / "t3_turbo_v1.safetensors"
    if required.exists():
        return model_dir
    model_dir.mkdir(parents=True, exist_ok=True)
    return Path(
        snapshot_download(
            repo_id=ROEST_REPO_ID,
            local_dir=model_dir,
            allow_patterns=list(ROEST_ALLOW_PATTERNS),
        )
    )


def roest_voice(speaker: str = "nic", model_dir: Path | None = None) -> RoestVoice:
    if speaker not in ROEST_PROMPTS:
        choices = ", ".join(sorted(ROEST_PROMPTS))
        raise ValueError(f"Unknown Roest speaker {speaker!r}. Choose one of: {choices}.")
    model_dir = ensure_roest_model(model_dir)
    prompt_path = model_dir / ROEST_PROMPTS[speaker]
    if not prompt_path.exists():
        raise FileNotFoundError(f"Missing Roest voice prompt: {prompt_path}")
    return RoestVoice(model_dir=model_dir, prompt_path=prompt_path, speaker=speaker)


class RoestTTS:
    def __init__(self, voice: RoestVoice, *, device: str | None = None) -> None:
        self.voice = voice
        self.device = device or _default_device()
        self._model = None

    def load(self) -> None:
        if self._model is not None:
            return
        from chatterbox.tts_turbo import ChatterboxTurboTTS

        self._model = ChatterboxTurboTTS.from_local(str(self.voice.model_dir), self.device)

    def synthesize_to_file(
        self,
        text: str,
        output_path: Path,
        *,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 600,
        repetition_penalty: float = 1.2,
    ) -> Path:
        import torchaudio as ta

        self.load()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        spoken = adapt_for_danish_speech(text)
        wav = self._model.generate(
            spoken,
            audio_prompt_path=str(self.voice.prompt_path),
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )
        ta.save(str(output_path), wav, self._model.sr)
        return output_path


def _default_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"
