from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from stacky.memory import Memory
from stacky.sessions import InfiniteSessionStore, read_jsonl_messages


class InfiniteSessionStoreTest(unittest.TestCase):
    def test_append_and_stitch_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = InfiniteSessionStore(Path(tmp) / "data")
            store.append_message("user", "hej", meta={"timestamp": "2026-05-16T10:00:00+00:00"})
            store.append_message("assistant", "hej Nicolai", meta={"timestamp": "2026-05-16T10:00:01+00:00"})

            messages, meta = store.stitch_context()

        self.assertEqual([message["role"] for message in messages], ["user", "assistant"])
        self.assertEqual(meta.total_messages, 2)
        self.assertEqual(messages[0]["content"], "hej")

    def test_rolls_active_thread_when_limit_is_reached(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = InfiniteSessionStore(Path(tmp) / "data", roll_tokens=3)
            store.append_message("user", "første besked er lang nok")
            store.append_message("assistant", "andet svar")

            rolled = sorted(store.session_dir.glob("stacky-infinite-thread.*.jsonl"))

        self.assertEqual(len(rolled), 1)

    def test_stitch_injects_recalled_memory_as_system_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = InfiniteSessionStore(Path(tmp) / "data")
            memory = Memory(
                id="m1",
                kind="preference",
                text="<Stacky skal tale dansk>",
                importance=1.0,
                source="test",
                tags=("voice",),
                created_at="now",
                updated_at="now",
                score=0.9,
            )

            messages, meta = store.stitch_context(recalled_memories=[memory])

        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("&lt;Stacky skal tale dansk&gt;", messages[0]["content"])
        self.assertEqual(meta.recalled_memory_count, 1)

    def test_read_jsonl_messages_ignores_non_message_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "thread.jsonl"
            path.write_text('{"type":"status"}\nnot json\n{"type":"message","message":{"role":"user","content":"hej"}}\n', encoding="utf-8")

            messages = read_jsonl_messages(path)

        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["content"], "hej")


if __name__ == "__main__":
    unittest.main()
