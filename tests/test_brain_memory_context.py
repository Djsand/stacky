from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from stacky.brain import StackyBrain
from stacky.llm import ChatMessage, LLMError
from stacky.memory import MemoryStore
from stacky.sessions import InfiniteSessionStore, read_jsonl_messages
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


class FailingFakeLLM:
    async def chat(self, messages: list[ChatMessage], *, temperature: float = 0.4, max_tokens: int | None = None) -> str:
        raise LLMError("connection refused")


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

    async def test_dialogue_is_not_written_to_long_term_memory_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(Path(tmp) / "memory.sqlite")
            brain = StackyBrain(StackySoul(created_for="Nicolai"), memory, LongFakeLLM())  # type: ignore[arg-type]

            await brain.respond("hej")

            self.assertEqual(memory.count(), 0)

    async def test_memory_writes_can_be_disabled_for_untrusted_voice(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(Path(tmp) / "memory.sqlite")
            brain = StackyBrain(StackySoul(created_for="Nicolai"), memory, LongFakeLLM())  # type: ignore[arg-type]

            await brain.respond("mit navn er forkert transcript", allow_memory_writes=False)

            self.assertEqual(memory.count(), 0)

    async def test_degraded_brain_reply_has_short_spoken_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(Path(tmp) / "memory.sqlite")
            brain = StackyBrain(StackySoul(created_for="Nicolai"), memory, FailingFakeLLM())  # type: ignore[arg-type]

            reply = await brain.respond("hej")

        self.assertTrue(reply.degraded)
        self.assertIn("connection refused", reply.text)
        self.assertEqual(reply.spoken_text, "Min brain-model svarer ikke lige nu. Jeg lytter stadig.")

    async def test_session_store_persists_trusted_turns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            memory = MemoryStore(root / "memory.sqlite")
            session_store = InfiniteSessionStore(root / "data")
            brain = StackyBrain(StackySoul(created_for="Nicolai"), memory, LongFakeLLM(), session_store)  # type: ignore[arg-type]

            await brain.respond("vi bygger stacky")

            messages = read_jsonl_messages(session_store.active_path)

        self.assertEqual([message["role"] for message in messages], ["user", "assistant"])

    async def test_untrusted_voice_does_not_persist_session_turn(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            memory = MemoryStore(root / "memory.sqlite")
            session_store = InfiniteSessionStore(root / "data")
            brain = StackyBrain(StackySoul(created_for="Nicolai"), memory, LongFakeLLM(), session_store)  # type: ignore[arg-type]

            await brain.respond("forkert stt", persist_session=False, allow_memory_writes=False, remember_recent=False)

            self.assertFalse(session_store.active_path.exists())


if __name__ == "__main__":
    unittest.main()
