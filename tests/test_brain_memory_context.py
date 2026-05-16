from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from stacky.brain import StackyBrain
from stacky.llm import ChatMessage
from stacky.memory import MemoryStore
from stacky.soul import StackySoul


class FakeLLM:
    def __init__(self) -> None:
        self.messages: list[list[ChatMessage]] = []

    async def chat(self, messages: list[ChatMessage], *, temperature: float = 0.4, max_tokens: int | None = None) -> str:
        self.messages.append(messages)
        return messages[0].content


class LongFakeLLM:
    async def chat(self, messages: list[ChatMessage], *, temperature: float = 0.4, max_tokens: int | None = None) -> str:
        return "Første korte svar. " + ("Mere forklaring. " * 40)


class BrainMemoryContextTest(unittest.IsolatedAsyncioTestCase):
    async def test_pinned_identity_fact_is_included_even_when_query_does_not_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(Path(tmp) / "memory.sqlite")
            memory.remember(
                "Brugerens navn er Nicolai.",
                kind="identity_fact",
                importance=1.0,
                source="test",
                tags=("name",),
            )
            brain = StackyBrain(StackySoul(created_for="Nicolai"), memory, FakeLLM())  # type: ignore[arg-type]

            reply = await brain.respond("Hej")

            self.assertIn("Brugerens navn er Nicolai.", reply.text)

    async def test_spoken_reply_is_compact_for_live_speech(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(Path(tmp) / "memory.sqlite")
            brain = StackyBrain(StackySoul(created_for="Nicolai"), memory, LongFakeLLM())  # type: ignore[arg-type]

            reply = await brain.respond("Hej")

        self.assertIsNotNone(reply.spoken_text)
        self.assertLessEqual(len(reply.spoken_text or ""), 240)

    async def test_recent_live_context_is_included_on_next_turn(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(Path(tmp) / "memory.sqlite")
            llm = FakeLLM()
            brain = StackyBrain(StackySoul(created_for="Nicolai"), memory, llm)  # type: ignore[arg-type]

            await brain.respond("vi taler om stacky")
            await brain.respond("hvad sagde jeg")

        self.assertGreaterEqual(len(llm.messages), 2)
        second_system = llm.messages[1][0].content
        self.assertIn("Seneste live-kontekst", second_system)
        self.assertIn("vi taler om stacky", second_system)


if __name__ == "__main__":
    unittest.main()
