from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT / "configs" / "stacky.toml"
DEFAULT_DATA_DIR = ROOT / "data" / "stacky"


@dataclass(frozen=True)
class LMStudioConfig:
    base_url: str = "http://127.0.0.1:1234/v1"
    api_key: str = ""
    model: str = "qwen3.6-35b"
    timeout_seconds: float = 90.0


@dataclass(frozen=True)
class VoiceConfig:
    language: str = "da-DK"
    stt_provider: str = "local"
    tts_provider: str = "local"
    tts_engine: str = "piper"
    allow_language_switch: bool = False
    barge_in: bool = True
    sample_rate_in: int = 16000
    sample_rate_out: int = 24000


@dataclass(frozen=True)
class StackChanConfig:
    host: str = "127.0.0.1"
    port: int = 8765
    device_name: str = "stackchan-cores3"
    wheels_enabled: bool = False


@dataclass(frozen=True)
class SandcodeConfig:
    repo_root: Path = Path("C:/Users/nicol/SANDCODE")
    host_script: Path = Path("C:/Users/nicol/SANDCODE/ios/host/sandcode-mobile-host.mjs")
    host: str = "127.0.0.1"
    port: int = 7390
    token: str = "stacky-local-change-me"
    provider: str = "ChatGPT Codex"
    model: str = "gpt-5.5"
    effort: str = "max"
    permission_mode: str = "autonomousAgent"


@dataclass(frozen=True)
class HomeAssistantConfig:
    base_url: str = "http://homeassistant.local:8123"
    token: str = ""
    suggest_first: bool = True


@dataclass(frozen=True)
class StackyConfig:
    name: str = "Stacky"
    data_dir: Path = DEFAULT_DATA_DIR
    timezone: str = "Europe/Copenhagen"
    lmstudio: LMStudioConfig = LMStudioConfig()
    voice: VoiceConfig = VoiceConfig()
    stackchan: StackChanConfig = StackChanConfig()
    sandcode: SandcodeConfig = SandcodeConfig()
    home_assistant: HomeAssistantConfig = HomeAssistantConfig()

    @property
    def soul_path(self) -> Path:
        return self.data_dir / "soul.yaml"

    @property
    def memory_path(self) -> Path:
        return self.data_dir / "memory.sqlite"


def load_config(path: str | Path | None = None) -> StackyConfig:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    raw: dict[str, object] = {}
    if config_path.exists():
        with config_path.open("rb") as handle:
            raw = tomllib.load(handle)

    stacky_raw = _section(raw, "stacky")
    lm_raw = _section(raw, "lmstudio")
    voice_raw = _section(raw, "voice")
    stackchan_raw = _section(raw, "stackchan")
    sandcode_raw = _section(raw, "sandcode")
    ha_raw = _section(raw, "home_assistant")

    data_dir = Path(str(stacky_raw.get("data_dir", os.getenv("STACKY_DATA_DIR", DEFAULT_DATA_DIR))))
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir

    return StackyConfig(
        name=str(stacky_raw.get("name", os.getenv("STACKY_NAME", "Stacky"))),
        data_dir=data_dir,
        timezone=str(stacky_raw.get("timezone", os.getenv("STACKY_TIMEZONE", "Europe/Copenhagen"))),
        lmstudio=LMStudioConfig(
            base_url=str(lm_raw.get("base_url", os.getenv("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1"))),
            api_key=str(lm_raw.get("api_key", os.getenv("LMSTUDIO_API_KEY", ""))),
            model=str(lm_raw.get("model", os.getenv("LMSTUDIO_MODEL", "qwen3.6-35b"))),
            timeout_seconds=float(lm_raw.get("timeout_seconds", os.getenv("LMSTUDIO_TIMEOUT", "90"))),
        ),
        voice=VoiceConfig(
            language=str(voice_raw.get("language", "da-DK")),
            stt_provider=str(voice_raw.get("stt_provider", "local")),
            tts_provider=str(voice_raw.get("tts_provider", "local")),
            tts_engine=str(voice_raw.get("tts_engine", "piper")),
            allow_language_switch=bool(voice_raw.get("allow_language_switch", False)),
            barge_in=bool(voice_raw.get("barge_in", True)),
            sample_rate_in=int(voice_raw.get("sample_rate_in", 16000)),
            sample_rate_out=int(voice_raw.get("sample_rate_out", 24000)),
        ),
        stackchan=StackChanConfig(
            host=str(stackchan_raw.get("host", "127.0.0.1")),
            port=int(stackchan_raw.get("port", 8765)),
            device_name=str(stackchan_raw.get("device_name", "stackchan-cores3")),
            wheels_enabled=bool(stackchan_raw.get("wheels_enabled", False)),
        ),
        sandcode=SandcodeConfig(
            repo_root=Path(str(sandcode_raw.get("repo_root", "C:/Users/nicol/SANDCODE"))),
            host_script=Path(str(sandcode_raw.get("host_script", "C:/Users/nicol/SANDCODE/ios/host/sandcode-mobile-host.mjs"))),
            host=str(sandcode_raw.get("host", "127.0.0.1")),
            port=int(sandcode_raw.get("port", 7390)),
            token=str(sandcode_raw.get("token", os.getenv("SANDCODE_MOBILE_TOKEN", "stacky-local-change-me"))),
            provider=str(sandcode_raw.get("provider", "ChatGPT Codex")),
            model=str(sandcode_raw.get("model", "gpt-5.5")),
            effort=str(sandcode_raw.get("effort", "max")),
            permission_mode=str(sandcode_raw.get("permission_mode", "autonomousAgent")),
        ),
        home_assistant=HomeAssistantConfig(
            base_url=str(ha_raw.get("base_url", os.getenv("HOME_ASSISTANT_URL", "http://homeassistant.local:8123"))),
            token=str(ha_raw.get("token", os.getenv("HOME_ASSISTANT_TOKEN", ""))),
            suggest_first=bool(ha_raw.get("suggest_first", True)),
        ),
    )


def _section(raw: dict[str, object], name: str) -> dict[str, object]:
    value = raw.get(name, {})
    return value if isinstance(value, dict) else {}
