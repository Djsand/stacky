from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from stacky.soul import StackySoul, load_soul, write_default_soul


class SoulTest(unittest.TestCase):
    def test_default_prompt_includes_personality_quirks(self) -> None:
        prompt = StackySoul().to_system_prompt()

        self.assertIn("små vaner", prompt)
        self.assertIn("spørgsmål", prompt)
        self.assertIn("nuttede vendinger", prompt)

    def test_speech_quirks_can_be_loaded_from_soul_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "soul.yaml"
            path.write_text(
                """
name: Stacky
speech_quirks:
  - Sig bip når noget lykkes.
""".strip(),
                encoding="utf-8",
            )

            soul = load_soul(path)

        self.assertEqual(soul.speech_quirks, ("Sig bip når noget lykkes.",))

    def test_default_soul_file_writes_speech_quirks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "soul.yaml"

            write_default_soul(path)
            text = path.read_text(encoding="utf-8")

        self.assertIn("speech_quirks:", text)
        self.assertIn("spørgsmål", text)


if __name__ == "__main__":
    unittest.main()
