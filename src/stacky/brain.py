from __future__ import annotations

from dataclasses import dataclass

from .danish import compact_for_speech, live_speech_style_prompt, spoken_danish_system_prompt
from .llm import ChatClient, ChatMessage, LLMError
from .memory import Memory, MemoryStore
from .soul import StackySoul


@dataclass(frozen=True)
class BrainReply:
    text: str
    spoken_text: str | None = None
    remembered: tuple[Memory, ...] = ()
    used_memories: tuple[Memory, ...] = ()
    degraded: bool = False


class StackyBrain:
    def __init__(self, soul: StackySoul, memory: MemoryStore, lmstudio: ChatClient) -> None:
        self.soul = soul
        self.memory = memory
        self.lmstudio = lmstudio
        self._recent_turns: list[tuple[str, str]] = []

    async def respond(
        self,
        user_text: str,
        *,
        max_spoken_chars: int = 150,
        detail_spoken_chars: int = 260,
    ) -> BrainReply:
        memories = tuple(_dedupe_memories([*self.memory.pinned(limit=6), *self.memory.recall(user_text, limit=5)]))
        messages = self._messages(user_text, memories, max_spoken_chars=max_spoken_chars)
        remembered: list[Memory] = []
        try:
            response = await self.lmstudio.chat(messages)
        except LLMError as exc:
            spoken = (
                "Jeg kan ikke få fat i min brain-model lige nu. "
                f"Fejlen er: {exc}. Jeg bliver stadig her og lytter."
            )
            return BrainReply(spoken, degraded=True, used_memories=memories)

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
        self.memory.remember(
            f"Samtale: {self.soul.created_for} sagde: {user_text} | Stacky svarede: {response}",
            kind="episode",
            importance=0.35,
            source="conversation",
            tags=("dialogue",),
        )
        self._remember_recent_turn(user_text, response)
        return BrainReply(response, spoken_text=spoken_response, remembered=tuple(remembered), used_memories=memories)

    def _messages(self, user_text: str, memories: tuple[Memory, ...], *, max_spoken_chars: int = 150) -> list[ChatMessage]:
        memory_text = "\n".join(f"- {memory.text}" for memory in memories) or "- Ingen relevante friske minder endnu."
        recent_text = self._recent_context_text()
        system = "\n\n".join(
            [
                self.soul.to_system_prompt(),
                spoken_danish_system_prompt(),
                live_speech_style_prompt(),
                _live_answer_rule(user_text, max_chars=max_spoken_chars),
                "Seneste live-kontekst i denne session:\n" + recent_text,
                "Relevante friske Stacky-minder:\n" + memory_text,
                "Svar som en nærværende ven, ikke som et kæledyr eller en assistent med marketingtone.",
            ]
        )
        return [
            ChatMessage("system", system),
            ChatMessage("user", user_text),
        ]

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


def _live_answer_rule(user_text: str, *, max_chars: int = 150) -> str:
    if _wants_detail(user_text):
        return "Brugeren beder sandsynligvis om detaljer; giv stadig en kort konklusion først."
    return f"Dette er live samtale: svar med 1 kort sætning som default, helst under cirka {max_chars} tegn."


def _spoken_response_for_live(user_text: str, response: str, *, max_chars: int = 150, detail_chars: int = 260) -> str:
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
        "lav det",
        "gør det",
    )
    return any(trigger in lowered for trigger in triggers)
