from __future__ import annotations

import wave
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from stacky.voice.piper_tts import ensure_danish_piper_voice, pitch_shift_wav


class PiperTTSTest(unittest.TestCase):
    def test_ensure_voice_uses_local_files_without_download(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            voice_dir = Path(tmp)
            model = voice_dir / "da" / "da_DK" / "talesyntese" / "medium" / "da_DK-talesyntese-medium.onnx"
            config = voice_dir / "da" / "da_DK" / "talesyntese" / "medium" / "da_DK-talesyntese-medium.onnx.json"
            model.parent.mkdir(parents=True)
            model.write_bytes(b"model")
            config.write_text("{}", encoding="utf-8")

            with patch("stacky.voice.piper_tts.hf_hub_download") as download:
                voice = ensure_danish_piper_voice(voice_dir)

        download.assert_not_called()
        self.assertEqual(voice.model_path, model)
        self.assertEqual(voice.config_path, config)

    def test_pitch_shift_uses_input_sample_rate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.wav"
            target = Path(tmp) / "target.wav"
            with wave.open(str(source), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(22050)
                wav.writeframes(b"\x00\x00" * 100)

            with patch("stacky.voice.piper_tts.subprocess.run") as run:
                result = pitch_shift_wav(source, target, factor=1.1)

            self.assertEqual(result, target)
            command = run.call_args.args[0]
            self.assertTrue(any("asetrate=24255,aresample=22050" in part for part in command))


if __name__ == "__main__":
    unittest.main()
