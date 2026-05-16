from __future__ import annotations

from dataclasses import dataclass

from .danish import compact_for_speech, live_speech_style_prompt, spoken_danish_system_prompt
from .llm import ChatClient, ChatMessage, LLMError
from .memory import Memory, MemoryStore
from .personality import StackySelfModel
from .sessions import InfiniteSessionStore
from .soul import StackySoul


@dataclass(frozen=True)
class BrainReply:
    text: str
    spoken_text: str | None = None
    remembered: tuple[Memory, ...] = ()
    used_memories: tuple[Memory, ...] = ()
    degraded: bool = False


class StackyBrain:
    def __init__(
        self,
        soul: StackySoul,
        memory: MemoryStore,
        lmstudio: ChatClient,
        session_store: InfiniteSessionStore | None = None,
        self_model: StackySelfModel | None = None,
    ) -> None:
        self.soul = soul
        self.memory = memory
        self.lmstudio = lmstudio
        self.session_store = session_store
        self.self_model = self_model
        self._recent_turns: list[tuple[str, str]] = []

    async def respond(
        self,
        user_text: str,
        *,
        max_spoken_chars: int = 260,
        detail_spoken_chars: int = 420,
        use_session_context: bool = True,
        persist_session: bool = True,
        allow_memory_writes: bool = True,
        remember_dialogue: bool = False,
        remember_recent: bool = True,
        session_source: str = "conversation",
    ) -> BrainReply:
        trusted_self_update = bool(allow_memory_writes and persist_session)
        if self.self_model is not None:
            self.self_model.observe_user_turn(user_text, trusted=trusted_self_update, source=session_source)
        memories = tuple(_dedupe_memories([*self.memory.pinned(limit=6), *self.memory.recall(user_text, limit=5)]))
        stitched_messages: list[dict[str, str]] = []
        session_user_persisted = False
        if self.session_store is not None and use_session_context:
            if persist_session:
                self.session_store.append_message("user", user_text, meta={"source": session_source})
                session_user_persisted = True
            stitched_messages, _ = self.session_store.stitch_context(recalled_memories=memories)
        messages = self._messages(
            user_text,
            memories,
            max_spoken_chars=max_spoken_chars,
            stitched_messages=stitched_messages,
            include_current_user=not session_user_persisted,
        )
        remembered: list[Memory] = []
        try:
            response = await self.lmstudio.chat(messages)
        except LLMError as exc:
            text = f"Jeg kan ikke få fat i min brain-model lige nu: {exc}"
            spoken = "Min brain-model svarer ikke lige nu. Jeg lytter stadig."
            return BrainReply(text, spoken_text=spoken, degraded=True, used_memories=memories)

        if allow_memory_writes:
            for candidate in self._candidate_memories(user_text):
                remembered.append(
                    self.memory.remember(
                        candidate,
                        kind="preference" if "foretrækker" in candidate or "kan lide" in candidate else "episode",
                        importance=0.7,
                        source="conversation",
                        tags=("fresh-stacky",),
                    )
                )
        spoken_response = _spoken_response_for_live(
            user_text,
            response,
            max_chars=max_spoken_chars,
            detail_chars=detail_spoken_chars,
        )
        if self.session_store is not None and persist_session:
            self.session_store.append_message("assistant", response, meta={"source": "stacky"})
        if self.self_model is not None:
            self.self_model.observe_assistant_turn(response, trusted=trusted_self_update, source="stacky")
        if allow_memory_writes and remember_dialogue:
            self.memory.remember(
                f"Samtale: {self.soul.created_for} sagde: {user_text} | Stacky svarede: {response}",
                kind="episode",
                importance=0.35,
                source="conversation",
                tags=("dialogue",),
            )
        if remember_recent:
            self._remember_recent_turn(user_text, response)
        return BrainReply(response, spoken_text=spoken_response, remembered=tuple(remembered), used_memories=memories)

    def record_observed_turn(
        self,
        user_text: str,
        assistant_text: str,
        *,
        persist_session: bool = True,
        allow_memory_writes: bool = True,
        remember_dialogue: bool = False,
        remember_recent: bool = True,
        session_source: str = "conversation",
    ) -> tuple[Memory, ...]:
        """Record a completed local turn without calling the brain model."""

        user_text = user_text.strip()
        assistant_text = assistant_text.strip()
        if not user_text and not assistant_text:
            return ()

        trusted_self_update = bool(allow_memory_writes and persist_session)
        if self.self_model is not None and user_text:
            self.self_model.observe_user_turn(user_text, trusted=trusted_self_update, source=session_source)

        if self.session_store is not None and persist_session:
            if user_text:
                self.session_store.append_message("user", user_text, meta={"source": session_source})
            if assistant_text:
                self.session_store.append_message("assistant", assistant_text, meta={"source": "stacky"})

        remembered: list[Memory] = []
        if allow_memory_writes and user_text:
            for candidate in self._candidate_memories(user_text):
                remembered.append(
                    self.memory.remember(
                        candidate,
                        kind="preference" if "foretrækker" in candidate or "kan lide" in candidate else "episode",
                        importance=0.7,
                        source="conversation",
                        tags=("fresh-stacky",),
                    )
                )

        if self.self_model is not None and assistant_text:
            self.self_model.observe_assistant_turn(assistant_text, trusted=trusted_self_update, source="stacky")
        if allow_memory_writes and remember_dialogue and user_text and assistant_text:
            self.memory.remember(
                f"Samtale: {self.soul.created_for} sagde: {user_text} | Stacky svarede: {assistant_text}",
                kind="episode",
                importance=0.35,
                source="conversation",
                tags=("dialogue",),
            )
        if remember_recent and user_text and assistant_text:
            self._remember_recent_turn(user_text, assistant_text)
        return tuple(remembered)

    def _messages(
        self,
        user_text: str,
        memories: tuple[Memory, ...],
        *,
        max_spoken_chars: int = 260,
        stitched_messages: list[dict[str, str]] | None = None,
        include_current_user: bool = True,
    ) -> list[ChatMessage]:
        memory_text = "\n".join(f"- {memory.text}" for memory in memories) or "- Ingen relevante friske minder endnu."
        recent_text = self._recent_context_text()
        self_context = self.self_model.context_for_prompt(user_text=user_text) if self.self_model is not None else ""
        system = "\n\n".join(
            [
                self.soul.to_system_prompt(),
                spoken_danish_system_prompt(),
                live_speech_style_prompt(),
                _live_answer_rule(user_text, max_chars=max_spoken_chars),
                self_context,
                "Seneste live-kontekst i denne session:\n" + recent_text,
                "Relevante friske Stacky-minder:\n" + memory_text,
                (
                    "Kropslig situationsregel: Du er Stacky i StackChan-kroppen på bordet hos Nicolai. "
                    "Når Nicolai siger at du, Stacky, kroppen, lyden, mikrofonen, batteriet eller trådløs drift virker, "
                    "så svar fra din egen Stacky-status. Tolk ikke automatisk ordene som om Nicolai selv er apparatet."
                ),
                (
                    "Hvis voice-transcriptet virker afbrudt eller semantisk tyndt, så sig kort at du ikke fik fat i det, "
                    "i stedet for at opfinde et nyt emne."
                ),
                "Svar som en nærværende ven, ikke som et kæledyr eller en assistent med marketingtone.",
            ]
        )
        messages = [ChatMessage("system", system)]
        for message in stitched_messages or []:
            role = message.get("role", "user")
            if role not in {"system", "user", "assistant"}:
                role = "user"
            messages.append(ChatMessage(role, message.get("content", "")))
        if include_current_user:
            messages.append(ChatMessage("user", user_text))
        return messages

    def _candidate_memories(self, user_text: str) -> list[str]:
        lowered = user_text.lower()
        triggers = (
            "husk",
            "jeg foretrækker",
            "jeg kan lide",
            "jeg hader",
            "mit navn",
            "jeg hedder",
            "mit hjem",
            "min pc",
            "for første gang",
            "trådløs",
            "wireless",
            "batteri",
            "opgradering",
            "firmware",
            "hjul",
        )
        if any(trigger in lowered for trigger in triggers):
            return [f"{self.soul.created_for} fortalte Stacky: {user_text.strip()}"]
        return []

    def _remember_recent_turn(self, user_text: str, response: str) -> None:
        self._recent_turns.append((user_text.strip(), response.strip()))
        del self._recent_turns[:-6]

    def _recent_context_text(self) -> str:
        if not self._recent_turns:
            return "- Ingen endnu."
        lines: list[str] = []
        for user_text, response in self._recent_turns[-6:]:
            lines.append(f"- {self.soul.created_for}: {user_text}")
            lines.append(f"- Stacky: {response}")
        return "\n".join(lines)


def _dedupe_memories(memories: list[Memory]) -> list[Memory]:
    seen: set[str] = set()
    result: list[Memory] = []
    for memory in memories:
        if memory.id in seen:
            continue
        seen.add(memory.id)
        result.append(memory)
    return result


def _live_answer_rule(user_text: str, *, max_chars: int = 260) -> str:
    if _wants_detail(user_text):
        return (
            "Brugeren beder sandsynligvis om detaljer eller diskuterer noget komplekst; "
            "giv en kort konklusion først, og uddyb det nødvendige i 2-5 naturlige sætninger. "
            "Det må gerne fylde mere end et hurtigt live-svar, men undgå lange monologer."
        )
    return (
        "Dette er live samtale: svar med 1-3 korte, konkrete sætninger som default, "
        f"helst under cirka {max_chars} tegn. Slut ikke automatisk med et spørgsmål. "
        "Spørg kun hvis Nicolai tydeligt mangler en afklaring for at komme videre. "
        "Nævn ikke at det er sent, aften, nat eller sengetid, medmindre Nicolai spørger om tid eller søvn. "
        "Når Nicolai siger at han tester dig, så anerkend testen kort og vent på næste observation. "
        "Web search er planlagt som en tidlig feature, men er ikke aktiv i runtime endnu; "
        "du skal ikke påstå at du har søgt på nettet."
    )


def _spoken_response_for_live(user_text: str, response: str, *, max_chars: int = 260, detail_chars: int = 420) -> str:
    limit = detail_chars if _wants_detail(user_text) else max_chars
    return compact_for_speech(response, max_chars=limit)


def _wants_detail(user_text: str) -> bool:
    lowered = user_text.lower()
    triggers = (
        "forklar",
        "detaljer",
        "uddyb",
        "trin",
        "plan",
        "kode",
        "implement",
        "debug",
        "fejl",
        "hvorfor",
        "hvordan",
        "diskuter",
        "kompleks",
        "komplekst",
        "arkitektur",
        "strategi",
        "research",
        "lav det",
        "gør det",
    )
    return any(trigger in lowered for trigger in triggers)
