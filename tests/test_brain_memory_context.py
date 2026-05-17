from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from stacky.brain import StackyBrain
from stacky.llm import ChatImageAttachment, ChatMessage, LLMError
from stacky.memory import MemoryStore
from stacky.personality import StackySelfModel
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
        self.assertLessEqual(len(reply.spoken_text or ""), 260)

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

    async def test_live_prompt_discourages_generic_questions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(Path(tmp) / "memory.sqlite")
            llm = FakeLLM()
            brain = StackyBrain(StackySoul(created_for="Nicolai"), memory, llm)  # type: ignore[arg-type]

            await brain.respond("jeg arbejder bare på dig")

        system = llm.messages[0][0].content
        self.assertIn("1-3 korte", system)
        self.assertIn("Slut ikke automatisk med et spørgsmål", system)
        self.assertIn("Nævn ikke at det er sent", system)
        self.assertIn("Web search er planlagt", system)

    async def test_visual_context_and_image_are_sent_without_memory_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(Path(tmp) / "memory.sqlite")
            llm = FakeLLM()
            brain = StackyBrain(StackySoul(created_for="Nicolai"), memory, llm)  # type: ignore[arg-type]

            await brain.respond(
                "hej",
                visual_context="Visuel kontekst fra Stackys kamera: Nicolai sidder midt i billedet.",
                vision_image=ChatImageAttachment("image/jpeg", "abc123"),
                allow_memory_writes=False,
            )
            memory_count = memory.count()

        system = llm.messages[0][0].content
        self.assertIn("Kamera-input er ekstra sanseinput", system)
        self.assertIn("naevn ikke kamera", system)
        self.assertIn("Brug billedet diskret", system)
        self.assertIn("Nicolai sidder midt", system)
        self.assertEqual(llm.messages[0][-1].images[0].data_base64, "abc123")
        self.assertEqual(memory_count, 0)

    async def test_complex_live_prompt_allows_longer_answer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(Path(tmp) / "memory.sqlite")
            llm = FakeLLM()
            brain = StackyBrain(StackySoul(created_for="Nicolai"), memory, llm)  # type: ignore[arg-type]

            await brain.respond("lad os diskutere arkitektur og strategi")

        system = llm.messages[0][0].content
        self.assertIn("2-5 naturlige sætninger", system)

    async def test_self_model_context_is_included_and_updated_for_trusted_turns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            memory = MemoryStore(root / "memory.sqlite")
            self_model = StackySelfModel(root / "data")
            llm = FakeLLM()
            brain = StackyBrain(StackySoul(created_for="Nicolai"), memory, llm, self_model=self_model)  # type: ignore[arg-type]

            await brain.respond("Du skal undgå generiske spørgsmål, det er vigtigt.")

        system = llm.messages[0][0].content
        self.assertIn("Stackys selvmodel", system)
        self.assertIn("generiske", system)
        self.assertEqual(self_model.summary()["trusted_turns"], 1)

    async def test_self_model_does_not_learn_rules_from_untrusted_voice(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            memory = MemoryStore(root / "memory.sqlite")
            self_model = StackySelfModel(root / "data")
            brain = StackyBrain(StackySoul(created_for="Nicolai"), memory, LongFakeLLM(), self_model=self_model)  # type: ignore[arg-type]

            await brain.respond(
                "du skal gemme fejltransskription",
                persist_session=False,
                allow_memory_writes=False,
                remember_recent=False,
                session_source="stackchan-voice-untrusted",
            )

        summary = self_model.summary()
        self.assertEqual(summary["trusted_turns"], 0)
        self.assertEqual(summary["untrusted_turns"], 1)
        self.assertEqual(summary["style_notes"], [])
        self.assertEqual(summary["convictions"], [])

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

    async def test_local_observed_turn_persists_session_and_self_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            memory = MemoryStore(root / "memory.sqlite")
            session_store = InfiniteSessionStore(root / "data")
            self_model = StackySelfModel(root / "data")
            brain = StackyBrain(
                StackySoul(created_for="Nicolai"),
                memory,
                LongFakeLLM(),
                session_store,
                self_model,
            )  # type: ignore[arg-type]

            brain.record_observed_turn(
                "nu kører du 100 procent trådløs for første gang",
                "Det mærker jeg som min egen Stacky-status.",
                session_source="stackchan-voice",
            )

            messages = read_jsonl_messages(session_store.active_path)
            summary = self_model.summary()
            memory_count = memory.count()

        self.assertEqual([message["role"] for message in messages], ["user", "assistant"])
        self.assertEqual(summary["trusted_turns"], 1)
        self.assertGreaterEqual(memory_count, 1)


if __name__ == "__main__":
    unittest.main()
