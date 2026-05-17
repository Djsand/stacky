from __future__ import annotations

import unittest

from stacky.danish import (
    assert_danish_voice_config,
    compact_for_speech,
    live_speech_style_prompt,
    spoken_danish_system_prompt,
)


class DanishContractTest(unittest.TestCase):
    def test_danish_voice_is_hard_requirement(self) -> None:
        prompt = spoken_danish_system_prompt()
        self.assertIn("dansk", prompt.lower())
        self.assertIn("må kun skifte", prompt.lower())
        assert_danish_voice_config("da-DK", False)
        with self.assertRaises(ValueError):
            assert_danish_voice_config("en-US", False)
        with self.assertRaises(ValueError):
            assert_danish_voice_config("da-DK", True)
        self.assertIn("1-2 korte", live_speech_style_prompt())
        self.assertIn("2-5 sætninger", live_speech_style_prompt())
        self.assertIn("jordbundet", live_speech_style_prompt())
        self.assertIn("tom begejstring", live_speech_style_prompt())
        self.assertIn("kommende Stacky-feature", live_speech_style_prompt())
        self.assertIn("generiske afslutninger", live_speech_style_prompt())
        self.assertIn("tester dig", live_speech_style_prompt())

    def test_compact_for_speech_limits_long_text(self) -> None:
        text = "Første sætning. " + ("meget lang tekst " * 80)
        compact = compact_for_speech(text, max_chars=80)
        self.assertLessEqual(len(compact), 80)


if __name__ == "__main__":
    unittest.main()
