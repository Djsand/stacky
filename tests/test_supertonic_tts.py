from __future__ import annotations

import unittest

from stacky.voice.supertonic_tts import SUPERTONIC_VOICE_PRESETS, supertonic_voice_preset


class SupertonicVoiceTest(unittest.TestCase):
    def test_stacky_preset_is_the_default_danish_voice(self) -> None:
        voice = supertonic_voice_preset("stacky")

        self.assertEqual(voice.voice_name, "F2")
        self.assertEqual(voice.language, "da")
        self.assertGreaterEqual(voice.speed, 1.15)

    def test_preset_allows_explicit_overrides(self) -> None:
        voice = supertonic_voice_preset("calm", voice_name="F1", speed=1.21, total_steps=7)

        self.assertEqual(voice.voice_name, "F1")
        self.assertEqual(voice.speed, 1.21)
        self.assertEqual(voice.total_steps, 7)
        self.assertEqual(voice.silence_duration, SUPERTONIC_VOICE_PRESETS["calm"].silence_duration)


if __name__ == "__main__":
    unittest.main()
