from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from stacky.computer import (
    LocalComputerActions,
    LocalComputerContext,
    extract_computer_search_query,
    parse_local_computer_action,
    wants_computer_context,
)


class ComputerContextTest(unittest.TestCase):
    def test_requires_explicit_computer_intent(self) -> None:
        self.assertFalse(wants_computer_context("hvordan har du det"))
        self.assertTrue(wants_computer_context("hvad siger git status"))
        self.assertTrue(wants_computer_context("kan du kigge i din egen kode"))

    def test_extracts_simple_search_query(self) -> None:
        self.assertEqual(extract_computer_search_query("grep efter StackChan i koden"), "stackchan")
        self.assertEqual(extract_computer_search_query("find hvor websearch ligger"), "websearch ligger")

    def test_returns_read_only_project_file_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src").mkdir()
            (root / "src" / "app.py").write_text("print('hej')\n", encoding="utf-8")

            context = LocalComputerContext(root).context_for("vis terminal status for repo")

        self.assertIn("Computer-kontekst", context)
        self.assertIn("workspace:", context)
        self.assertIn("src/app.py", context)
        self.assertIn("read-only", context)

    def test_returns_empty_context_without_trigger(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            context = LocalComputerContext(Path(tmp)).context_for("bare snak lidt")

        self.assertEqual(context, "")

    def test_parses_and_creates_text_file_on_desktop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as desktop:
            root = Path(tmp)
            desktop_path = Path(desktop)
            action = parse_local_computer_action(
                "lav en tekstfil på mit skrivebord med en hilsen",
                root=root,
                desktop=desktop_path,
            )
            self.assertIsNotNone(action)
            assert action is not None
            result = LocalComputerActions(root, desktop=desktop_path).run(action)

            self.assertTrue(result.ok)
            self.assertTrue((desktop_path / "hilsen.txt").exists())
            self.assertIn("skrivebordet", result.spoken)

    def test_parses_git_status_action(self) -> None:
        action = parse_local_computer_action("hvad siger git status", root=Path.cwd())

        self.assertIsNotNone(action)
        assert action is not None
        self.assertEqual(action.kind, "git_status")

    def test_parses_dir_as_safe_workspace_listing(self) -> None:
        action = parse_local_computer_action("kør dir", root=Path.cwd())

        self.assertIsNotNone(action)
        assert action is not None
        self.assertEqual(action.kind, "list_workspace")

    def test_does_not_parse_vague_file_talk_as_action(self) -> None:
        action = parse_local_computer_action("vi skal snakke om filer senere", root=Path.cwd())

        self.assertIsNone(action)

    def test_web_search_request_is_not_local_computer_search(self) -> None:
        text = "søg på nettet om der er noget nyt om texttospeech modeller"

        self.assertFalse(wants_computer_context(text))
        self.assertIsNone(parse_local_computer_action(text, root=Path.cwd()))


if __name__ == "__main__":
    unittest.main()
