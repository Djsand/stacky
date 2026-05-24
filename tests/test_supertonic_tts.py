from __future__ import annotations

import unittest

from stacky.voice.supertonic_tts import SUPERTONIC_VOICE_PRESETS, supertonic_voice_preset


class SupertonicVoiceTest(unittest.TestCase):
    def test_stacky_preset_is_the_default_danish_voice(self) -> None:
        voice = supertonic_voice_preset("stacky")

        self.assertEqual(voice.voice_name, "F2")
        self.assertEqual(voice.language, "da")
        self.assertGreaterEqual(voice.speed, 1.08)
        self.assertLessEqual(voice.speed, 1.12)
        self.assertGreaterEqual(voice.silence_duration, 0.045)

    def test_preset_allows_explicit_overrides(self) -> None:
        voice = supertonic_voice_preset("calm", voice_name="F1", speed=1.21, total_steps=7)

        self.assertEqual(voice.voice_name, "F1")
        self.assertEqual(voice.speed, 1.21)
        self.assertEqual(voice.total_steps, 7)
        self.assertEqual(voice.silence_duration, SUPERTONIC_VOICE_PRESETS["calm"].silence_duration)

    def test_quick_profile_keeps_natural_rhythm(self) -> None:
        voice = supertonic_voice_preset("quick")

        self.assertGreaterEqual(voice.speed, 1.08)
        self.assertLessEqual(voice.speed, 1.12)
        self.assertGreaterEqual(voice.silence_duration, 0.045)
        self.assertLessEqual(voice.max_chunk_length, 160)

    def test_alive_profile_is_default_live_personality_tuning(self) -> None:
        voice = supertonic_voice_preset("alive")

        self.assertGreaterEqual(voice.speed, 1.06)
        self.assertLessEqual(voice.speed, 1.10)
        self.assertGreaterEqual(voice.silence_duration, 0.065)
        self.assertLessEqual(voice.max_chunk_length, 150)


if __name__ == "__main__":
    unittest.main()
