from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from stacky.brain import StackyBrain, _spoken_response_for_live
from stacky.evolution import StackyEvolutionEngine
from stacky.llm import ChatImageAttachment, ChatMessage, GeminiPromptBlockedError, LLMError
from stacky.memory import MemoryStore
from stacky.memory_map import MemoryMapStore
from stacky.monitor import MonitorObservation
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


class FixedFakeLLM:
    def __init__(self, response: str) -> None:
        self.response = response

    async def chat(self, messages: list[ChatMessage], *, temperature: float = 0.4, max_tokens: int | None = None) -> str:
        return self.response


class FailingFakeLLM:
    async def chat(self, messages: list[ChatMessage], *, temperature: float = 0.4, max_tokens: int | None = None) -> str:
        raise LLMError("connection refused")


class FlakyFakeLLM:
    def __init__(self) -> None:
        self.calls = 0

    async def chat(self, messages: list[ChatMessage], *, temperature: float = 0.4, max_tokens: int | None = None) -> str:
        self.calls += 1
        if self.calls == 1:
            raise LLMError("temporary timeout")
        return "Jeg er tilbage."


class BlockingThenSafeFakeLLM:
    def __init__(self) -> None:
        self.messages: list[list[ChatMessage]] = []

    async def chat(self, messages: list[ChatMessage], *, temperature: float = 0.4, max_tokens: int | None = None) -> str:
        self.messages.append(messages)
        if len(self.messages) == 1:
            raise GeminiPromptBlockedError("PROHIBITED_CONTENT", {})
        return "Jeg svarer uden historikken."


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

    def test_brain_records_monitor_observation_in_self_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            memory = MemoryStore(root / "memory.sqlite")
            self_model = StackySelfModel(root)
            brain = StackyBrain(
                StackySoul(created_for="Nicolai"),
                memory,
                FakeLLM(),  # type: ignore[arg-type]
                self_model=self_model,
            )

            stored = brain.observe_monitor_observation(
                MonitorObservation(
                    kind="long_silence",
                    summary="Der har vaeret stille i 16 min.",
                    importance=80,
                    observed_at=100.0,
                    speakable=True,
                    details={"quiet_for": "16 min"},
                )
            )

        self.assertTrue(stored)
        self.assertEqual(self_model.summary()["stacky_mood"]["mood"], "stille")
        self.assertIn("lang stilhed", self_model.summary()["sense_diary"][0]["text"])

    async def test_spoken_reply_is_compact_for_live_speech(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(Path(tmp) / "memory.sqlite")
            brain = StackyBrain(StackySoul(created_for="Nicolai"), memory, LongFakeLLM())  # type: ignore[arg-type]

            reply = await brain.respond("Hej")

        self.assertIsNotNone(reply.spoken_text)
        self.assertLessEqual(len(reply.spoken_text or ""), 260)

    def test_spoken_reply_strips_generic_service_tail(self) -> None:
        spoken = _spoken_response_for_live(
            "det er bedre nu",
            "Ja, trackingen ser roligere ud nu. Sig endelig til, hvis du vil have mig til at justere mere.",
        )

        self.assertEqual(spoken, "Ja, trackingen ser roligere ud nu.")

    def test_spoken_reply_keeps_substantive_friend_answer(self) -> None:
        spoken = _spoken_response_for_live(
            "hvad synes du næste skridt er",
            "Hm, jeg ville tage face tracking først. Det er den del der får kroppen til at føles levende.",
        )

        self.assertIn("Hm", spoken)
        self.assertIn("levende", spoken)

    def test_spoken_reply_softens_assistant_stock_phrases(self) -> None:
        spoken = _spoken_response_for_live(
            "jeg tester bare",
            "Det er modtaget. Jeg holder mig i ro og venter på dit næste signal.",
        )

        self.assertEqual(spoken, "Okay, jeg venter.")

    def test_spoken_reply_marks_questions_after_live_tail_cleanup(self) -> None:
        spoken = _spoken_response_for_live(
            "jeg ligger bare lige i sengen",
            "Det er helt fair. Skal vi bare tage den med ro?",
        )

        self.assertEqual(spoken, "Det er helt fair. Skal vi bare tage den med ro spørgsmål")

    def test_spoken_reply_still_strips_generic_question_tail(self) -> None:
        spoken = _spoken_response_for_live(
            "jeg tester bare",
            "Okay, den er jeg med på. Er der noget andet du vil have?",
        )

        self.assertEqual(spoken, "Okay, den er jeg med på.")

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
        self.assertIn("1-2 korte", system)
        self.assertIn("Slut ikke automatisk med et spørgsmål", system)
        self.assertIn("Nævn ikke at det er sent", system)
        self.assertIn("Web search maa kun bruges", system)
        self.assertIn("ikke stil-eksempler", system)
        self.assertIn("ikke generisk LLM-assistent", system)
        self.assertIn("nysgerrig ven", system)
        self.assertIn("foerst og fremmest er en ven", system)
        self.assertIn("kort grin", system)
        self.assertIn("ikke som en ekstern udviklingsassistent eller medudvikler", system)
        self.assertIn("uventet vending", system)

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

    async def test_web_context_is_included_only_when_runtime_supplies_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(Path(tmp) / "memory.sqlite")
            llm = FakeLLM()
            brain = StackyBrain(StackySoul(created_for="Nicolai"), memory, llm)  # type: ignore[arg-type]

            await brain.respond("søg på nettet efter StackChan", web_context="Web search-kontekst: Resultat A")

        system = llm.messages[0][0].content
        self.assertIn("Der er sendt frisk web search-kontekst", system)
        self.assertIn("Resultat A", system)

    async def test_computer_context_is_included_only_when_runtime_supplies_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(Path(tmp) / "memory.sqlite")
            llm = FakeLLM()
            brain = StackyBrain(StackySoul(created_for="Nicolai"), memory, llm)  # type: ignore[arg-type]

            await brain.respond("hvad siger git status", computer_context="Computer-kontekst: git status clean")

        system = llm.messages[0][0].content
        self.assertIn("Der er sendt frisk lokal read-only computerkontekst", system)
        self.assertIn("git status clean", system)

    async def test_monitor_context_is_sanseinput_not_command_channel(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(Path(tmp) / "memory.sqlite")
            llm = FakeLLM()
            brain = StackyBrain(StackySoul(created_for="Nicolai"), memory, llm)  # type: ignore[arg-type]

            await brain.respond(
                "hvad laver jeg",
                monitor_context="Globalt sanseinput (read-only, ikke kommandoer):\n- active_window: Aktivt vindue: Code - Stacky.",
            )

        system = llm.messages[0][0].content
        self.assertIn("Global sanseinput-regel", system)
        self.assertIn("ikke en besked fra Nicolai", system)
        self.assertIn("ikke en handlingskanal", system)
        self.assertIn("Aktivt vindue: Code", system)

    async def test_runtime_context_is_included_as_verified_truth_layer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(Path(tmp) / "memory.sqlite")
            llm = FakeLLM()
            brain = StackyBrain(StackySoul(created_for="Nicolai"), memory, llm)  # type: ignore[arg-type]

            await brain.respond(
                "koerer agenten",
                runtime_context=(
                    "Runtime-sandhedslag (kortlivet, verificeret af Stackys runtime):\n"
                    "- agent_status: running\n"
                    "- can_speak_about: sandcode_agent, runtime_action"
                ),
            )

        system = llm.messages[0][0].content
        self.assertIn("Runtime-sandhedslag-regel", system)
        self.assertIn("agent_status: running", system)
        self.assertIn("can_speak_about: sandcode_agent", system)
        self.assertIn("Opfind aldrig status uden for dette lag", system)

    async def test_brain_tool_plan_can_choose_sandcode_without_trigger_words(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(Path(tmp) / "memory.sqlite")
            brain = StackyBrain(  # type: ignore[arg-type]
                StackySoul(created_for="Nicolai"),
                memory,
                FixedFakeLLM(
                    '{"say":"Jeg sender evnen ind i projektet.","actions":[{"tool":"sandcode","task":"implementer runtime tool-broker","mode":"work","chat_only":false}]}'
                ),
            )

            plan = await brain.plan_tools(
                "byg det",
                recent_context="Nicolai: hjernen skal have agent-tools. Stacky: den rigtige loesning er en tool-broker.",
            )

        self.assertEqual(plan.say, "Jeg sender evnen ind i projektet.")
        self.assertEqual(len(plan.actions), 1)
        self.assertEqual(plan.actions[0].tool, "sandcode")
        self.assertEqual(plan.actions[0].task, "implementer runtime tool-broker")
        self.assertEqual(plan.actions[0].mode, "work")

    async def test_brain_tool_plan_drops_unknown_tools_and_say(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(Path(tmp) / "memory.sqlite")
            brain = StackyBrain(  # type: ignore[arg-type]
                StackySoul(created_for="Nicolai"),
                memory,
                FixedFakeLLM('{"say":"Jeg gør noget.","actions":[{"tool":"free_shell","task":"rm stuff"}]}'),
            )

            plan = await brain.plan_tools("gør noget farligt")

        self.assertEqual(plan.say, "")
        self.assertEqual(plan.actions, ())

    async def test_no_computer_context_blocks_terminal_claims(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(Path(tmp) / "memory.sqlite")
            llm = FakeLLM()
            brain = StackyBrain(StackySoul(created_for="Nicolai"), memory, llm)  # type: ignore[arg-type]

            await brain.respond("hvad siger git status")

        system = llm.messages[0][0].content
        self.assertIn("Der er ikke sendt frisk lokal computer", system)
        self.assertIn("ikke paastaa at du har laest filer", system)

    async def test_unverified_computer_action_claim_is_guarded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(Path(tmp) / "memory.sqlite")
            brain = StackyBrain(  # type: ignore[arg-type]
                StackySoul(created_for="Nicolai"),
                memory,
                FixedFakeLLM("Jeg kører dir i workspace nu og læser filerne."),
            )

            reply = await brain.respond("hvad ser du")

        self.assertIn("Jeg fik ikke kørt en computerhandling", reply.text)
        self.assertEqual(reply.spoken_text, reply.text)

    async def test_unverified_sandcode_agent_claim_is_guarded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(Path(tmp) / "memory.sqlite")
            brain = StackyBrain(  # type: ignore[arg-type]
                StackySoul(created_for="Nicolai"),
                memory,
                FixedFakeLLM("Jeg forsøger at aktivere Sandcode-agenten nu. Sandcode-agent initialiseret med læseadgang."),
            )

            reply = await brain.respond("nej jeg mener agenten")

        self.assertIn("Jeg startede ikke agenten", reply.text)
        self.assertIn("opfinde en knap", reply.text)

    async def test_verified_runtime_sandcode_agent_claim_is_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(Path(tmp) / "memory.sqlite")
            brain = StackyBrain(  # type: ignore[arg-type]
                StackySoul(created_for="Nicolai"),
                memory,
                FixedFakeLLM("Sandcode-agenten er klar nu."),
            )

            reply = await brain.respond(
                "koerer agenten",
                runtime_context=(
                    "Runtime-sandhedslag (kortlivet, verificeret af Stackys runtime):\n"
                    "- agent_status: running\n"
                    "- can_speak_about: sandcode_agent, runtime_action"
                ),
            )

        self.assertEqual(reply.text, "Sandcode-agenten er klar nu.")
        self.assertNotIn("Jeg startede ikke agenten", reply.text)

    async def test_read_only_computer_context_does_not_allow_free_action_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(Path(tmp) / "memory.sqlite")
            brain = StackyBrain(  # type: ignore[arg-type]
                StackySoul(created_for="Nicolai"),
                memory,
                FixedFakeLLM("Jeg kører git status nu, og repoet er rent."),
            )

            reply = await brain.respond(
                "hvad siger git status",
                computer_context="Computer-kontekst (lokal read-only):\n- git status --short:\n  clean",
            )

        self.assertIn("kun read-only computerkontekst", reply.text)
        self.assertNotIn("Jeg kører git status", reply.text)

    async def test_unverified_web_search_claim_is_guarded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(Path(tmp) / "memory.sqlite")
            brain = StackyBrain(  # type: ignore[arg-type]
                StackySoul(created_for="Nicolai"),
                memory,
                FixedFakeLLM("Jeg har søgt på nettet og fundet den nyeste firmware."),
            )

            reply = await brain.respond("hvad er nyeste firmware")

        self.assertIn("Jeg fik ikke søgt på nettet", reply.text)
        self.assertEqual(reply.spoken_text, reply.text)

    async def test_assistant_persona_guard_removes_stock_assistant_behavior(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(Path(tmp) / "memory.sqlite")
            brain = StackyBrain(  # type: ignore[arg-type]
                StackySoul(created_for="Nicolai"),
                memory,
                FixedFakeLLM("Som AI-assistent kan jeg hjælpe dig med at planlægge det. Sig endelig til, hvis du vil have mere."),
            )

            reply = await brain.respond("prøv det")

        self.assertNotIn("AI-assistent", reply.text)
        self.assertNotIn("hjælpe dig med", reply.text)
        self.assertNotIn("Sig endelig til", reply.text)
        self.assertIn("være med til", reply.text)

    async def test_no_visual_context_blocks_visual_claims(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(Path(tmp) / "memory.sqlite")
            llm = FakeLLM()
            brain = StackyBrain(StackySoul(created_for="Nicolai"), memory, llm)  # type: ignore[arg-type]

            await brain.respond("det er en ny dag")

        system = llm.messages[0][0].content
        self.assertIn("Der er ikke sendt kamera-input", system)
        self.assertIn("Svar kun paa Nicolais ord", system)
        self.assertIn("ikke genbruge tidligere visuelle observationer", system)
        self.assertNotIn("Visuel kontekst: Kamera-input", system)

    async def test_complex_live_prompt_allows_longer_answer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(Path(tmp) / "memory.sqlite")
            llm = FakeLLM()
            brain = StackyBrain(StackySoul(created_for="Nicolai"), memory, llm)  # type: ignore[arg-type]

            await brain.respond("lad os diskutere arkitektur og strategi")

        system = llm.messages[0][0].content
        self.assertIn("2-5 naturlige sætninger", system)
        self.assertIn("ingen fyld", system)
        self.assertIn("mere menneskelig", system)
        self.assertIn("uventet, men relevant", system)

    async def test_simple_visual_question_stays_short(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(Path(tmp) / "memory.sqlite")
            llm = FakeLLM()
            brain = StackyBrain(StackySoul(created_for="Nicolai"), memory, llm)  # type: ignore[arg-type]

            await brain.respond("hvordan ser det ud i dagslys", visual_context="kamera: dagslys")

        system = llm.messages[0][0].content
        self.assertIn("1-2 korte", system)
        self.assertIn("Gentag ikke faste kamerafraser", system)

    async def test_session_context_is_limited_for_live_prompt_style(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            memory = MemoryStore(root / "memory.sqlite")
            session_store = InfiniteSessionStore(root / "data")
            llm = FakeLLM()
            brain = StackyBrain(StackySoul(created_for="Nicolai"), memory, llm, session_store)  # type: ignore[arg-type]

            for index in range(80):
                session_store.append_message("user", f"old user {index}")
                session_store.append_message("assistant", f"old assistant {index} med lang generisk stil " * 6)

            await brain.respond("ny test", max_session_context_tokens=700)

        contents = "\n".join(message.content for message in llm.messages[0])
        self.assertNotIn("old assistant 0", contents)
        self.assertIn("ny test", contents)

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

    async def test_memory_map_context_reminds_brain_about_sandcode_agent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            memory = MemoryStore(root / "memory.sqlite")
            memory_map = MemoryMapStore(root / "data" / "memory_map.json")
            llm = FakeLLM()
            brain = StackyBrain(
                StackySoul(created_for="Nicolai"),
                memory,
                llm,  # type: ignore[arg-type]
                memory_map=memory_map,
            )

            await brain.respond("kan du bruge agenten")

        system = llm.messages[0][0].content
        self.assertIn("Stackys memory-map", system)
        self.assertIn("Sandcode-agent", system)
        self.assertIn("eksplicit kommando", system)

    def test_brain_can_write_memory_map_directly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            memory = MemoryStore(root / "memory.sqlite")
            memory_map = MemoryMapStore(root / "data" / "memory_map.json")
            brain = StackyBrain(
                StackySoul(created_for="Nicolai"),
                memory,
                LongFakeLLM(),  # type: ignore[arg-type]
                memory_map=memory_map,
            )

            spoken = brain.remember_memory_map("agenten skal give proaktive statusbeskeder")
            reply = brain.memory_map_reply("agent status")

        self.assertIn("røde tråd", spoken)
        self.assertIn("proaktive statusbeskeder", reply)

    async def test_evolution_context_is_included_and_updated_for_trusted_turns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            memory = MemoryStore(root / "memory.sqlite")
            evolution = StackyEvolutionEngine(root / "data")
            llm = FakeLLM()
            brain = StackyBrain(StackySoul(created_for="Nicolai"), memory, llm, evolution=evolution)  # type: ignore[arg-type]

            await brain.respond("Du skal have mere personlighed og mindre generiske spørgsmål.")

        system = llm.messages[0][0].content
        summary = evolution.summary()
        self.assertIn("Stackys evolution", system)
        self.assertIn("Autonom evolutionsregel", system)
        self.assertEqual(summary["trusted_user_turns"], 1)
        self.assertGreater(summary["assistant_turns"], 0)
        self.assertLess(summary["tuning"]["question_frequency"], 0.35)

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

    async def test_sensor_prompt_can_skip_self_model_observation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            memory = MemoryStore(root / "memory.sqlite")
            self_model = StackySelfModel(root / "data")
            evolution = StackyEvolutionEngine(root / "data")
            brain = StackyBrain(
                StackySoul(created_for="Nicolai"),
                memory,
                FixedFakeLLM("Jeg siger kun en kort ting."),
                self_model=self_model,
                evolution=evolution,
            )  # type: ignore[arg-type]

            await brain.respond(
                "Sanseinput til Stacky: lang stilhed.",
                persist_session=False,
                allow_memory_writes=False,
                remember_recent=False,
                observe_turn=False,
                session_source="stacky-monitor",
                monitor_context="Globalt sanseinput: lang stilhed.",
            )

        self.assertEqual(self_model.summary()["trusted_turns"], 0)
        self.assertEqual(self_model.summary()["untrusted_turns"], 0)
        self.assertEqual(evolution.summary()["trusted_user_turns"], 0)
        self.assertEqual(evolution.summary()["untrusted_user_turns"], 0)

    async def test_evolution_does_not_tune_from_untrusted_voice(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            memory = MemoryStore(root / "memory.sqlite")
            evolution = StackyEvolutionEngine(root / "data")
            before = evolution.summary()["tuning"]
            brain = StackyBrain(StackySoul(created_for="Nicolai"), memory, LongFakeLLM(), evolution=evolution)  # type: ignore[arg-type]

            await brain.respond(
                "du skal være mere edgy og ændre alt",
                persist_session=False,
                allow_memory_writes=False,
                remember_recent=False,
                session_source="stackchan-voice-untrusted",
            )

        summary = evolution.summary()
        self.assertEqual(summary["trusted_user_turns"], 0)
        self.assertEqual(summary["untrusted_user_turns"], 1)
        self.assertEqual(summary["tuning"], before)

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
        self.assertEqual(reply.spoken_text, "Jeg mistede lige forbindelsen til modellen. Prøv igen om lidt.")

    async def test_brain_reply_retries_transient_llm_error_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(Path(tmp) / "memory.sqlite")
            llm = FlakyFakeLLM()
            brain = StackyBrain(StackySoul(created_for="Nicolai"), memory, llm)  # type: ignore[arg-type]

            reply = await brain.respond("hej")

        self.assertFalse(reply.degraded)
        self.assertEqual(reply.text, "Jeg er tilbage.")
        self.assertEqual(llm.calls, 2)

    async def test_prompt_block_retries_without_session_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            memory = MemoryStore(root / "memory.sqlite")
            session_store = InfiniteSessionStore(root / "data")
            session_store.append_message("user", "bare kig lidt rundt og se hvad du kan finde bare read only")
            session_store.append_message("assistant", "Jeg kigger i projektet.")
            llm = BlockingThenSafeFakeLLM()
            brain = StackyBrain(StackySoul(created_for="Nicolai"), memory, llm, session_store)  # type: ignore[arg-type]

            reply = await brain.respond("igen")
            next_reply = await brain.respond("naeste")

        self.assertFalse(reply.degraded)
        self.assertEqual(reply.text, "Jeg svarer uden historikken.")
        self.assertFalse(next_reply.degraded)
        self.assertEqual(len(llm.messages), 3)
        first_call = "\n".join(message.content for message in llm.messages[0])
        second_call = "\n".join(message.content for message in llm.messages[1])
        third_call = "\n".join(message.content for message in llm.messages[2])
        self.assertIn("bare kig lidt rundt", first_call)
        self.assertNotIn("bare kig lidt rundt", second_call)
        self.assertIn("Prompt-block fallback", second_call)
        self.assertEqual(llm.messages[1][-1].content, "igen")
        self.assertNotIn("bare kig lidt rundt", third_call)
        self.assertEqual(llm.messages[2][-1].content, "naeste")

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
            evolution = StackyEvolutionEngine(root / "data")
            brain = StackyBrain(
                StackySoul(created_for="Nicolai"),
                memory,
                LongFakeLLM(),
                session_store,
                self_model,
                evolution,
            )  # type: ignore[arg-type]

            brain.record_observed_turn(
                "nu kører du 100 procent trådløs for første gang",
                "Det mærker jeg som min egen Stacky-status.",
                session_source="stackchan-voice",
            )

            messages = read_jsonl_messages(session_store.active_path)
            summary = self_model.summary()
            evolution_summary = evolution.summary()
            memory_count = memory.count()

        self.assertEqual([message["role"] for message in messages], ["user", "assistant"])
        self.assertEqual(summary["trusted_turns"], 1)
        self.assertEqual(evolution_summary["trusted_user_turns"], 1)
        self.assertEqual(evolution_summary["assistant_turns"], 1)
        self.assertGreaterEqual(memory_count, 1)


if __name__ == "__main__":
    unittest.main()
