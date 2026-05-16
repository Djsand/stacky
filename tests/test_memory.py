from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from stacky.memory import MemoryStore


class MemoryStoreTest(unittest.TestCase):
    def test_fresh_db_starts_empty_and_can_recall_correct_forget(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp) / "memory.sqlite")

            self.assertEqual(store.count(), 0)
            memory = store.remember(
                "Nicol foretrækker at Stacky taler dansk.",
                kind="preference",
                tags=("voice",),
            )

            self.assertEqual(store.count(), 1)
            recalled = store.recall("hvilket sprog skal Stacky tale?")
            self.assertEqual(recalled[0].id, memory.id)

            corrected = store.correct(memory.id, "Nicol kræver dansk stemme fra Stacky.")
            self.assertIn("dansk", corrected.text)

            self.assertTrue(store.forget(memory.id))
            self.assertEqual(store.count(), 0)

    def test_forget_by_tag_removes_matching_memories_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp) / "memory.sqlite")
            store.remember("dialogue noise", tags=("dialogue",))
            keep = store.remember("stable preference", kind="preference", tags=("fresh-stacky",))

            deleted = store.forget_by_tag("dialogue")

            self.assertEqual(deleted, 1)
            self.assertEqual([memory.id for memory in store.all()], [keep.id])

    def test_recall_excludes_dialogue_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp) / "memory.sqlite")
            store.remember(
                "Samtale: Nicolai sagde noget fejltransskriberet.",
                kind="episode",
                source="conversation",
                tags=("dialogue",),
            )
            fact = store.remember(
                "Nicolai kræver dansk stemme fra Stacky.",
                kind="preference",
                source="conversation",
                tags=("fresh-stacky",),
            )

            recalled = store.recall("hvilket sprog skal stacky tale?")

        self.assertEqual([memory.id for memory in recalled], [fact.id])


if __name__ == "__main__":
    unittest.main()
