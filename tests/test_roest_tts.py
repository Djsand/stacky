from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from stacky.voice.roest_tts import roest_voice


class RoestTTSTest(unittest.TestCase):
    def test_roest_voice_uses_local_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            (model_dir / "t3_turbo_v1.safetensors").write_bytes(b"")
            prompt = model_dir / "audio_samples" / "01_nic_00_t0.7_p0.95_k600_r1.2.wav"
            prompt.parent.mkdir(parents=True)
            prompt.write_bytes(b"RIFF")

            voice = roest_voice("nic", model_dir=model_dir)

        self.assertEqual(voice.speaker, "nic")
        self.assertEqual(voice.model_dir, model_dir)
        self.assertEqual(voice.prompt_path, prompt)


if __name__ == "__main__":
    unittest.main()
