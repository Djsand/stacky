from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .speech_adapter import adapt_for_danish_speech


@dataclass(frozen=True)
class SupertonicVoice:
    voice_name: str = "F2"
    language: str = "da"
    speed: float = 1.18
    total_steps: int = 8
    max_chunk_length: int = 240
    silence_duration: float = 0.035


SUPERTONIC_VOICE_PRESETS: dict[str, SupertonicVoice] = {
    "stacky": SupertonicVoice(),
    "calm": SupertonicVoice(speed=1.08, total_steps=10, max_chunk_length=240, silence_duration=0.055),
    "clear": SupertonicVoice(voice_name="F1", speed=1.16, total_steps=10, max_chunk_length=220, silence_duration=0.04),
    "quick": SupertonicVoice(speed=1.25, total_steps=6, max_chunk_length=220, silence_duration=0.025),
}


def supertonic_voice_preset(
    profile: str = "stacky",
    *,
    voice_name: str | None = None,
    speed: float | None = None,
    total_steps: int | None = None,
    max_chunk_length: int | None = None,
    silence_duration: float | None = None,
) -> SupertonicVoice:
    base = SUPERTONIC_VOICE_PRESETS.get(profile, SUPERTONIC_VOICE_PRESETS["stacky"])
    return SupertonicVoice(
        voice_name=voice_name or base.voice_name,
        language=base.language,
        speed=speed if speed is not None else base.speed,
        total_steps=total_steps if total_steps is not None else base.total_steps,
        max_chunk_length=max_chunk_length if max_chunk_length is not None else base.max_chunk_length,
        silence_duration=silence_duration if silence_duration is not None else base.silence_duration,
    )


class SupertonicTTS:
    """Local Supertonic 3 TTS for more natural Danish voice experiments."""

    def __init__(self, voice: SupertonicVoice | None = None) -> None:
        self.voice = voice or SupertonicVoice()
        self._tts = None
        self._style = None

    def load(self) -> None:
        if self._tts is not None:
            return
        from supertonic import TTS

        self._tts = TTS(auto_download=True)
        self._style = self._tts.get_voice_style(voice_name=self.voice.voice_name)

    def synthesize_to_file(
        self,
        text: str,
        output_path: Path,
        *,
        speed: float | None = None,
        silence_duration: float | None = None,
        total_steps: int | None = None,
        max_chunk_length: int | None = None,
    ) -> Path:
        self.load()
        assert self._tts is not None
        assert self._style is not None

        output_path.parent.mkdir(parents=True, exist_ok=True)
        spoken = adapt_for_danish_speech(text)
        wav, _ = self._tts.synthesize(
            spoken,
            voice_style=self._style,
            lang=self.voice.language,
            total_steps=total_steps if total_steps is not None else self.voice.total_steps,
            speed=speed if speed is not None else self.voice.speed,
            max_chunk_length=max_chunk_length if max_chunk_length is not None else self.voice.max_chunk_length,
            silence_duration=silence_duration if silence_duration is not None else self.voice.silence_duration,
        )
        self._tts.save_audio(wav, str(output_path))
        return output_path
