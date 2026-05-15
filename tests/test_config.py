from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from stacky.config import load_config


class ConfigTest(unittest.TestCase):
    def test_voice_engine_defaults_to_realtime_piper(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "stacky.toml"
            config = load_config(config_path)

        self.assertEqual(config.voice.tts_engine, "piper")


if __name__ == "__main__":
    unittest.main()
