from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from stacky.config import load_config


class ConfigTest(unittest.TestCase):
    def test_voice_engine_defaults_to_realtime_piper(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "stacky.toml"
            config = load_config(config_path)

        self.assertEqual(config.voice.tts_engine, "piper")

    def test_gemini_provider_uses_gemini_env_over_lmstudio_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "stacky.toml"
            config_path.write_text(
                "\n".join(
                    [
                        "[lmstudio]",
                        'base_url = "http://127.0.0.1:1234/v1"',
                        'api_key = "lm-key"',
                        'model = "local-model"',
                    ]
                ),
                encoding="utf-8",
            )
            env = {
                "STACKY_BRAIN_PROVIDER": "gemini",
                "GEMINI_API_KEY": "gemini-key",
                "GEMINI_MODEL": "gemini-3.1-flash-lite-preview",
            }
            with patch.dict("os.environ", env, clear=False):
                config = load_config(config_path)

        self.assertEqual(config.lmstudio.provider, "gemini")
        self.assertEqual(config.lmstudio.api_key, "gemini-key")
        self.assertEqual(config.lmstudio.model, "gemini-3.1-flash-lite-preview")
        self.assertEqual(config.lmstudio.base_url, "https://generativelanguage.googleapis.com/v1beta")


if __name__ == "__main__":
    unittest.main()
