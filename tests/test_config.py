from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from stacky.config import load_config


class ConfigTest(unittest.TestCase):
    def test_voice_engine_defaults_to_livelier_supertonic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "stacky.toml"
            config = load_config(config_path)

        self.assertEqual(config.voice.tts_engine, "supertonic")
        self.assertTrue(config.websearch.enabled)
        self.assertEqual(config.websearch.provider, "duckduckgo_lite")
        self.assertTrue(config.websearch.allow_insecure_tls_fallback)
        self.assertTrue(config.computer.enabled)

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

    def test_websearch_can_be_disabled_by_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "stacky.toml"
            with patch.dict("os.environ", {"STACKY_WEBSEARCH_ENABLED": "false"}, clear=False):
                config = load_config(config_path)

        self.assertFalse(config.websearch.enabled)

    def test_websearch_tls_fallback_can_be_disabled_by_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "stacky.toml"
            with patch.dict(
                "os.environ",
                {"STACKY_WEBSEARCH_ALLOW_INSECURE_TLS_FALLBACK": "false"},
                clear=False,
            ):
                config = load_config(config_path)

        self.assertFalse(config.websearch.allow_insecure_tls_fallback)

    def test_computer_context_can_be_disabled_by_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "stacky.toml"
            with patch.dict("os.environ", {"STACKY_COMPUTER_ENABLED": "false"}, clear=False):
                config = load_config(config_path)

        self.assertFalse(config.computer.enabled)


if __name__ == "__main__":
    unittest.main()
