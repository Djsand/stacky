from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from stacky.memory_map import MemoryMapStore


class MemoryMapStoreTest(unittest.TestCase):
    def test_core_memory_map_includes_sandcode_capability(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryMapStore(Path(tmp) / "memory_map.json")

            context = store.context_for_prompt(user_text="kan du bruge agenten")

        self.assertIn("Sandcode-agent", context)
        self.assertIn("uden triggerord", context)
        self.assertIn("tydelig handlingsintention", context)

    def test_observe_turn_stores_agent_reporting_preference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryMapStore(Path(tmp) / "memory_map.json")

            stored = store.observe_turn(
                "agent rapportering er dårlig, jeg vil have proaktive status beskeder",
                source="test",
            )
            reply = MemoryMapStore(Path(tmp) / "memory_map.json").recall_reply("agent status")

        self.assertTrue(stored)
        self.assertIn("proaktive status", reply)
        self.assertIn("Sandcode", reply)

    def test_remember_text_writes_only_short_curated_note(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryMapStore(Path(tmp) / "memory_map.json")

            entry = store.remember_text("Stacky skal huske den røde tråd.", tags=("test",), source="test")
            summary = MemoryMapStore(Path(tmp) / "memory_map.json").summary()

        self.assertIsNotNone(entry)
        self.assertGreaterEqual(summary["count"], 3)
        self.assertIn("memory_map.json", summary["path"])


if __name__ == "__main__":
    unittest.main()
