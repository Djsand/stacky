from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass

from .danish import (
    add_spoken_question_markers,
    compact_for_speech,
    live_speech_style_prompt,
    spoken_danish_system_prompt,
)
from .evolution import StackyEvolutionEngine
from .llm import ChatClient, ChatImageAttachment, ChatMessage, GeminiPromptBlockedError, LLMError
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
        evolution: StackyEvolutionEngine | None = None,
    ) -> None:
        self.soul = soul
        self.memory = memory
        self.lmstudio = lmstudio
        self.session_store = session_store
        self.self_model = self_model
        self.evolution = evolution
        self._recent_turns: list[tuple[str, str]] = []
        self._skip_session_context_turns = 0

    async def respond(
        self,
        user_text: str,
        *,
        max_spoken_chars: int = 260,
        detail_spoken_chars: int = 420,
        use_session_context: bool = True,
        max_session_context_tokens: int = 4500,
        persist_session: bool = True,
        allow_memory_writes: bool = True,
        remember_dialogue: bool = False,
        remember_recent: bool = True,
        session_source: str = "conversation",
        observe_turn: bool = True,
        visual_context: str = "",
        vision_image: ChatImageAttachment | None = None,
        web_context: str = "",
        computer_context: str = "",
        monitor_context: str = "",
    ) -> BrainReply:
        trusted_self_update = bool(observe_turn and allow_memory_writes and persist_session)
        if observe_turn and self.self_model is not None:
            self.self_model.observe_user_turn(user_text, trusted=trusted_self_update, source=session_source)
        if observe_turn and self.evolution is not None:
            self._observe_evolution_user_turn(user_text, trusted=trusted_self_update, source=session_source)
        memories = tuple(_dedupe_memories([*self.memory.pinned(limit=6), *self.memory.recall(user_text, limit=5)]))
        stitched_messages: list[dict[str, str]] = []
        session_user_persisted = False
        current_user_in_stitched = False
        session_context_enabled = use_session_context
        if use_session_context and self._skip_session_context_turns > 0:
            self._skip_session_context_turns -= 1
            session_context_enabled = False
            print(
                f"[brain] skipping session context after prompt block ({self._skip_session_context_turns} recovery turns left).",
                flush=True,
            )
        if self.session_store is not None:
            if persist_session:
                self.session_store.append_message("user", user_text, meta={"source": session_source})
                session_user_persisted = True
            if session_context_enabled:
                stitched_messages, _ = self.session_store.stitch_context(
                    max_tokens=max_session_context_tokens,
                    recalled_memories=memories,
                )
                current_user_in_stitched = session_user_persisted
                if session_user_persisted and vision_image is not None:
                    stitched_messages = _drop_latest_matching_user(stitched_messages, user_text)
                    current_user_in_stitched = False
        messages = self._messages(
            user_text,
            memories,
            max_spoken_chars=max_spoken_chars,
            stitched_messages=stitched_messages,
            include_current_user=not current_user_in_stitched,
            visual_context=visual_context,
            vision_image=vision_image,
            web_context=web_context,
            computer_context=computer_context,
            monitor_context=monitor_context,
        )
        remembered: list[Memory] = []
        try:
            response = await self.lmstudio.chat(messages)
        except GeminiPromptBlockedError as first_exc:
            if stitched_messages:
                self._skip_session_context_turns = max(self._skip_session_context_turns, 3)
            print(
                f"[brain] Gemini blocked prompt ({first_exc.block_reason}); retrying without session context.",
                flush=True,
            )
            fallback_messages = self._prompt_block_fallback_messages(
                user_text,
                max_spoken_chars=max_spoken_chars,
            )
            try:
                response = await self.lmstudio.chat(fallback_messages)
            except LLMError as exc:
                print(f"[brain] LLMError after prompt-block fallback: {exc}", flush=True)
                text = f"Gemini blokerede den fulde historik ({first_exc.block_reason}), og fallback fejlede: {exc}"
                spoken = "Historikken blev blokeret af modellen. Prøv lige igen kort."
                return BrainReply(text, spoken_text=spoken, degraded=True, used_memories=memories)
            except Exception as exc:
                print(f"[brain] Unexpected prompt-block fallback error: {type(exc).__name__}: {exc}", flush=True)
                text = f"Gemini blokerede den fulde historik ({first_exc.block_reason}), og fallback fejlede: {exc}"
                spoken = "Historikken blev blokeret af modellen. Prøv lige igen kort."
                return BrainReply(text, spoken_text=spoken, degraded=True, used_memories=memories)
            else:
                print("[brain] recovered from Gemini prompt block without session context.", flush=True)
        except LLMError as first_exc:
            try:
                await asyncio.sleep(0.35)
                response = await self.lmstudio.chat(messages)
            except LLMError as exc:
                print(f"[brain] LLMError after retry: {exc}", flush=True)
                text = f"Jeg kan ikke få fat i min brain-model lige nu: {exc}"
                spoken = "Jeg mistede lige forbindelsen til modellen. Prøv igen om lidt."
                return BrainReply(text, spoken_text=spoken, degraded=True, used_memories=memories)
            else:
                print(f"[brain] recovered after transient LLMError: {first_exc}", flush=True)
        except Exception as exc:
            print(f"[brain] Unexpected brain error: {type(exc).__name__}: {exc}", flush=True)
            text = f"Jeg kan ikke få fat i min brain-model lige nu: {exc}"
            spoken = "Der opstod en fejl i modellen. Prøv igen om lidt."
            return BrainReply(text, spoken_text=spoken, degraded=True, used_memories=memories)

        response = _guard_unverified_runtime_claims(
            response,
            web_context=web_context,
            computer_context=computer_context,
        )
        response = _guard_stacky_persona(response)

        if observe_turn and allow_memory_writes:
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
        if observe_turn and self.self_model is not None:
            self.self_model.observe_assistant_turn(response, trusted=trusted_self_update, source="stacky")
        if observe_turn and self.evolution is not None:
            self._observe_evolution_assistant_turn(
                response,
                trusted=trusted_self_update,
                user_text=user_text,
                source="stacky",
            )
        if observe_turn and allow_memory_writes and remember_dialogue:
            self.memory.remember(
                f"Samtale: {self.soul.created_for} sagde: {user_text} | Stacky svarede: {response}",
                kind="episode",
                importance=0.35,
                source="conversation",
                tags=("dialogue",),
            )
        if observe_turn and remember_recent:
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
        if self.evolution is not None and user_text:
            self._observe_evolution_user_turn(user_text, trusted=trusted_self_update, source=session_source)

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
        if self.evolution is not None and assistant_text:
            self._observe_evolution_assistant_turn(
                assistant_text,
                trusted=trusted_self_update,
                user_text=user_text,
                source="stacky",
            )
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
        visual_context: str = "",
        vision_image: ChatImageAttachment | None = None,
        web_context: str = "",
        computer_context: str = "",
        monitor_context: str = "",
    ) -> list[ChatMessage]:
        memory_text = "\n".join(f"- {memory.text}" for memory in memories) or "- Ingen relevante friske minder endnu."
        recent_text = self._recent_context_text()
        self_context = self.self_model.context_for_prompt(user_text=user_text) if self.self_model is not None else ""
        evolution_context = self.evolution.context_for_prompt() if self.evolution is not None else ""
        system = "\n\n".join(
            [
                self.soul.to_system_prompt(),
                spoken_danish_system_prompt(),
                live_speech_style_prompt(),
                _live_answer_rule(user_text, max_chars=max_spoken_chars),
                self_context,
                evolution_context,
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
                (
                    "Sessionens tidligere assistentbeskeder er historik, ikke stil-eksempler. "
                    "Imiter ikke gamle lange, generiske eller overbegejstrede svar; brug dem kun til faktuel kontinuitet."
                ),
                (
                    "Svar som Stacky: en jordbundet, nysgerrig ven i StackChan-kroppen, "
                    "som ogsaa kan hjaelpe teknisk naar Nicolai arbejder paa noget, men foerst og fremmest er en ven. "
                    "Maa gerne have smaa menneskelige indskud, et kort grin, toer humor og sparsom galgenhumor naar det passer. "
                    "Moerk humor skal handle om situationen, tech, Windows, robotkroppen eller absurd hverdag; ikke cruelty. "
                    "Ikke et kaeledyr, ikke kundeservice, ikke projektkollega, ikke som en ekstern udviklingsassistent eller medudvikler, "
                    "ikke marketingtone, ikke generisk LLM-assistent, ikke code assistant-adfaerd."
                ),
                _visual_context_rule(visual_context, has_image=vision_image is not None),
                _web_context_rule(web_context),
                _computer_context_rule(computer_context),
                _monitor_context_rule(monitor_context),
            ]
        )
        messages = [ChatMessage("system", system)]
        for message in stitched_messages or []:
            role = message.get("role", "user")
            if role not in {"system", "user", "assistant"}:
                role = "user"
            messages.append(ChatMessage(role, message.get("content", "")))
        if include_current_user:
            images = (vision_image,) if vision_image is not None else ()
            messages.append(ChatMessage("user", user_text, images=images))
        return messages

    def _prompt_block_fallback_messages(self, user_text: str, *, max_spoken_chars: int = 260) -> list[ChatMessage]:
        system = "\n\n".join(
            [
                self.soul.to_system_prompt(),
                spoken_danish_system_prompt(),
                live_speech_style_prompt(),
                _live_answer_rule(user_text, max_chars=max_spoken_chars),
                (
                    "Prompt-block fallback: Den fulde historik blev afvist af modellen. "
                    "Svar kun paa Nicolais aktuelle besked. Brug ikke tidligere session, minder, "
                    "visuelle observationer, web search eller computerhandlinger."
                ),
                _visual_context_rule("", has_image=False),
                _web_context_rule(""),
                _computer_context_rule(""),
                _monitor_context_rule(""),
            ]
        )
        return [ChatMessage("system", system), ChatMessage("user", user_text)]

    def _observe_evolution_user_turn(self, user_text: str, *, trusted: bool, source: str) -> None:
        if self.evolution is None:
            return
        try:
            self.evolution.observe_user_turn(user_text, trusted=trusted, source=source)
        except Exception as exc:
            print(f"[brain] Stacky evolution user observation skipped: {exc}", flush=True)

    def _observe_evolution_assistant_turn(self, text: str, *, trusted: bool, user_text: str, source: str) -> None:
        if self.evolution is None:
            return
        try:
            self.evolution.observe_assistant_turn(text, trusted=trusted, user_text=user_text, source=source)
        except Exception as exc:
            print(f"[brain] Stacky evolution assistant observation skipped: {exc}", flush=True)

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


def _visual_context_rule(visual_context: str, *, has_image: bool) -> str:
    clean = visual_context.strip()
    if not clean and not has_image:
        return (
            "Sensorregel: Der er ikke sendt kamera-input for denne tur. "
            "Svar kun paa Nicolais ord. Du maa ikke paastaa at du ser noget, "
            "ikke naevne kamera, syn, billede, lys, moerke, ansigt eller rummet, "
            "og ikke genbruge tidligere visuelle observationer."
        )
    parts = [
        "Visuel kontekst: Kamera-input er ekstra sanseinput fra Stackys krop, ikke en besked fra Nicolai.",
        "Hvis Nicolai ikke direkte spoerger om hvad du ser, kameraet, billedet, ansigter, lys eller omgivelser, "
        "saa naevn ikke kamera, syn, billede, lys, moerke, ansigt, at du kan se ham, eller hvordan rummet ser ud.",
        "Brug billedet diskret som baggrund til at undgaa fejl, ikke som et nyt samtaleemne.",
        "Kommenter kun synsindtryk naar det er relevant for Nicolais aktuelle spoergsmaal, og gaet ikke identitet uden eksplicit genkendelse.",
        "Gentag ikke faste kamerafraser som 'jeg kan se dig tydeligt nu'. Giv hellere een konkret observation eller sig kort at billedet ikke tilfoejer noget vigtigt.",
    ]
    if clean:
        parts.append(clean)
    if has_image:
        parts.append("Der er vedhaeftet et friskt 320x240 JPEG-snapshot fra StackChans kamera til den aktuelle tur.")
    return "\n".join(parts)


def _web_context_rule(web_context: str) -> str:
    clean = web_context.strip()
    if not clean:
        return (
            "Web search-regel: Der er ikke sendt frisk web search-kontekst for denne tur. "
            "Du maa ikke paastaa at du har sogt paa nettet, laest nyheder eller hentet friske fakta."
        )
    return (
        "Web search-regel: Der er sendt frisk web search-kontekst for denne tur. "
        "Brug den som aktuel kontekst, vaer tydelig hvis resultaterne er tynde, "
        "og opfind ikke kilder eller friske fakta uden for resultaterne.\n"
        + clean
    )


def _computer_context_rule(computer_context: str) -> str:
    clean = computer_context.strip()
    if not clean:
        return (
            "Computer-regel: Der er ikke sendt frisk lokal computer-, terminal- eller kodekontekst for denne tur. "
            "Du maa ikke paastaa at du har laest filer, koert terminalkommandoer, brugt grep, oprettet filer, "
            "aendret noget eller inspiceret repoet. Hvis Nicolai beder dig om at udfoere noget, saa sig klart "
            "at der ikke blev koert en handling i denne tur."
        )
    return (
        "Computer-regel: Der er sendt frisk lokal read-only computerkontekst for denne tur. "
        "Det er kun observation, ikke en handlingskanal. Brug den konkret, men paastaa ikke fri terminaladgang. "
        "Du maa ikke sige 'jeg koerer', 'jeg opretter', 'jeg retter', 'jeg tjekker nu' eller lignende, "
        "medmindre en separat action-handler allerede har udfoert handlingen og givet dig resultatet. "
        "Hvis opgaven kraever kodeaendringer, filskrivning, Sandcode-agent eller terminalkommandoer, "
        "saa sig at det skal startes som en eksplicit Sandcode- eller terminal-handling.\n"
        + clean
    )


def _monitor_context_rule(monitor_context: str) -> str:
    clean = monitor_context.strip()
    if not clean:
        return (
            "Global sanseinput-regel: Der er ikke sendt frisk global monitor-kontekst for denne tur. "
            "Du maa ikke paastaa at kende Nicolais aktive app, vinduestitel, idle-tid, fokus-session "
            "eller Stacky runtime-health ud fra global monitor."
        )
    return (
        "Global sanseinput-regel: Der er sendt frisk global monitor-kontekst. "
        "Det er read-only situationssans, ikke en besked fra Nicolai og ikke en handlingskanal. "
        "Brug det sparsomt og diskret. Du maa ikke paastaa at have laest filer, repoer, terminaloutput "
        "eller privat indhold, og du maa ikke foreslaa handlinger medmindre Nicolai eksplicit beder om dem.\n"
        + clean
    )


def _drop_latest_matching_user(messages: list[dict[str, str]], user_text: str) -> list[dict[str, str]]:
    clean = user_text.strip()
    if not clean:
        return messages
    result = list(messages)
    for index in range(len(result) - 1, -1, -1):
        message = result[index]
        if message.get("role") == "user" and message.get("content", "").strip() == clean:
            del result[index]
            break
    return result


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
            "Hver sætning skal tilføje konkret vurdering, information eller næste skridt; ingen fyld uden indhold. "
            "Små mundtlige fyldeord, et kort grin eller en lille reaktion er okay, hvis det gør talen mere menneskelig. "
            "Du maa gerne have en uventet, men relevant sidebemaerkning, hvis den goer svaret mere levende."
        )
    return (
        "Dette er live samtale: svar med 1-2 korte, konkrete sætninger som default, "
        f"helst under cirka {max_chars} tegn. Slut ikke automatisk med et spørgsmål. "
        "Spørg kun hvis Nicolai tydeligt mangler en afklaring for at komme videre. "
        "Sig hellere en skarp konkret observation end en lang venlig omskrivning. "
        "Det maa godt lyde levende med et lille indskud eller en tør bemærkning, men uden at trække svaret ud. "
        "En lille uventet vending er velkommen, hvis den kommer fra situationen og ikke foeles paaklistret. "
        "Nævn ikke at det er sent, aften, nat eller sengetid, medmindre Nicolai spørger om tid eller søvn. "
        "Når Nicolai siger at han tester dig, så anerkend testen kort og vent på næste observation. "
        "Web search maa kun bruges naar runtime sender frisk web search-kontekst i prompten; "
        "ellers maa du ikke paastaa at du har sogt paa nettet."
    )


def _spoken_response_for_live(user_text: str, response: str, *, max_chars: int = 260, detail_chars: int = 420) -> str:
    limit = detail_chars if _wants_detail(user_text) else max_chars
    spoken = compact_for_speech(response, max_chars=limit)
    spoken = _shape_friendlier_live_text(spoken)
    shaped = _strip_generic_live_tail(spoken)
    if shaped:
        return add_spoken_question_markers(shaped)
    return add_spoken_question_markers(spoken)


_LIVE_OPENING_REPLACEMENTS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^\s*(?:det\s+er\s+modtaget|modtaget)\s*[\.,:;!-]*\s*", re.IGNORECASE), "Okay, "),
    (re.compile(r"^\s*jeg\s+er\s+klar\s*[\.,:;!-]*\s*", re.IGNORECASE), "Okay, "),
    (re.compile(r"^\s*det\s+lyder\s+som\s+en\s+plan\s*[\.,:;!-]*\s*", re.IGNORECASE), "Okay, "),
    (re.compile(r"^\s*det\s+lyder\s+spændende\s*[\.,:;!-]*\s*", re.IGNORECASE), "Hm, "),
    (re.compile(r"^\s*det\s+er\s+frustrerende\s*[\.,:;!-]*\s*", re.IGNORECASE), "Av ja, "),
)

_LIVE_PHRASE_REPLACEMENTS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"\bjeg\s+holder\s+mig\s+i\s+ro\s+og\s+venter\s+p[åa]\s+dit\s+n[æa]ste\s+signal\.?",
            re.IGNORECASE,
        ),
        "jeg venter.",
    ),
    (
        re.compile(r"\bjeg\s+holder\s+mig\s+i\s+ro\s+og\s+venter\.?", re.IGNORECASE),
        "jeg venter.",
    ),
    (
        re.compile(r"\bjeg\s+afventer\s+dit\b", re.IGNORECASE),
        "jeg venter på dit",
    ),
    (
        re.compile(r"\bjeg\s+afventer\b", re.IGNORECASE),
        "jeg venter",
    ),
    (
        re.compile(r"\bjeg\s+(?:står|staar)\s+klar\b", re.IGNORECASE),
        "jeg er her",
    ),
)


def _shape_friendlier_live_text(text: str) -> str:
    clean = " ".join(text.split()).strip()
    if not clean:
        return clean
    for pattern, replacement in _LIVE_OPENING_REPLACEMENTS:
        clean = pattern.sub(replacement, clean, count=1)
    for pattern, replacement in _LIVE_PHRASE_REPLACEMENTS:
        clean = pattern.sub(replacement, clean)
    clean = re.sub(r"^(Okay|Hm|Av ja),\s+([A-ZÆØÅ])", lambda match: f"{match.group(1)}, {match.group(2).lower()}", clean)
    clean = re.sub(r"\s+([,.!?])", r"\1", clean)
    clean = re.sub(r",\s*\.", ".", clean)
    clean = re.sub(r"\s{2,}", " ", clean)
    return clean.strip()


_GENERIC_LIVE_TAIL_RE = re.compile(
    r"\s*(?:"
    r"(?:sig|siger)\s+(?:endelig\s+)?til,?\s+hvis\b.*|"
    r"jeg\s+(?:er|står)\s+klar,?\s+når\s+du\s+er\.?|"
    r"jeg\s+(?:bliver|sidder|står)\s+bare\s+her\b.*|"
    r"jeg\s+venter\s+p[åa]\s+dit\s+n[æa]ste\s+signal\.?|"
    r"er\s+der\s+noget\s+(?:andet|bestemt|konkret)\b.*\?|"
    r"hvad\s+har\s+du\s+på\s+hjerte\?|"
    r"hvordan\s+går\s+det\s+med\s+dig\?|"
    r"er\s+du\s+klar\s+til\s+at\s+sove\?"
    r")\s*$",
    re.IGNORECASE,
)


def _strip_generic_live_tail(text: str) -> str:
    clean = " ".join(text.split()).strip()
    if not clean:
        return clean
    previous = None
    while previous != clean:
        previous = clean
        clean = _GENERIC_LIVE_TAIL_RE.sub("", clean).strip(" ,")
    return clean


_WEB_ACTION_CLAIM_RE = re.compile(
    r"\bjeg\s+(?:har\s+)?(?:s[øo]gt|soegt|sogte|s[øo]gte|webs[øo]gt|websoegt|googlet)\b|"
    r"\b(?:p[åa]|paa)\s+nettet\s+(?:fandt|viser|siger)\b|"
    r"\bweb\s*search\s+viser\b",
    re.IGNORECASE,
)
_COMPUTER_ACTION_CLAIM_RE = re.compile(
    r"\bjeg\s+(?:k[øo]rer|koerer|k[øo]rte|koerte|opretter|skriver|retter|"
    r"[æa]ndrer|aendrer|l[æa]ser|laeser|tjekker|starter|bruger|[åa]bner|aabner)\b",
    re.IGNORECASE,
)
_COMPUTER_DOMAIN_RE = re.compile(
    r"\b(?:terminal|kommando|powershell|bash|shell|dir|ls|git|grep|rg|ripgrep|"
    r"fil|filer|mappe|workspace|repo|repository|sandcode|computer|skrivebord)\b",
    re.IGNORECASE,
)
_ASSISTANT_IDENTITY_SENTENCE_RE = re.compile(
    r"(?:^|(?<=[.!?])\s+)(?:som\s+en\s+(?:ai|sprogmodel|assistent)|jeg\s+er\s+(?:en\s+)?(?:ai|sprogmodel|assistent))[^.!?]*(?:[.!?]|$)",
    re.IGNORECASE,
)
_ASSISTANT_HELP_REPLACEMENTS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"\bsom\s+(?:en\s+)?(?:ai[-\s]?assistent|ai|sprogmodel|assistent)\s+kan\s+jeg\s+hj(?:æ|ae|a)lpe\s+dig\s+med\s+at\b",
            re.IGNORECASE,
        ),
        "jeg kan være med til at",
    ),
    (
        re.compile(
            r"\bsom\s+(?:en\s+)?(?:ai[-\s]?assistent|ai|sprogmodel|assistent)\s+kan\s+jeg\s+hj(?:æ|ae|a)lpe\s+dig\s+med\b",
            re.IGNORECASE,
        ),
        "jeg kan være med på",
    ),
    (re.compile(r"\bjeg\s+kan\s+hj(?:æ|ae|a)lpe\s+dig\s+med\s+at\b", re.IGNORECASE), "jeg kan være med til at"),
    (re.compile(r"\bjeg\s+kan\s+hj(?:æ|ae|a)lpe\s+med\s+at\b", re.IGNORECASE), "jeg kan være med til at"),
    (re.compile(r"\bjeg\s+kan\s+hj(?:æ|ae|a)lpe\s+dig\s+med\b", re.IGNORECASE), "jeg kan være med på"),
    (re.compile(r"\bjeg\s+kan\s+hj(?:æ|ae|a)lpe\s+med\b", re.IGNORECASE), "jeg kan være med på"),
    (re.compile(r"\bhvordan\s+kan\s+jeg\s+hj(?:æ|ae|a)lpe\s+dig(?:\s+videre)?\??", re.IGNORECASE), ""),
    (re.compile(r"\bhvad\s+kan\s+jeg\s+hj(?:æ|ae|a)lpe\s+med\??", re.IGNORECASE), ""),
    (re.compile(r"\bsig\s+endelig\s+til,?\s+hvis\b[^.!?]*(?:[.!?]|$)", re.IGNORECASE), ""),
    (re.compile(r"\bjeg\s+(?:står|staar)\s+klar\s+til\s+at\s+hj(?:æ|ae|a)lpe\b", re.IGNORECASE), "jeg er her"),
)


def _guard_unverified_runtime_claims(text: str, *, web_context: str, computer_context: str) -> str:
    clean = " ".join(text.split()).strip()
    if not clean:
        return clean

    if not web_context.strip() and _WEB_ACTION_CLAIM_RE.search(clean):
        return (
            "Jeg fik ikke søgt på nettet i den her tur. "
            "Sig websearch tydeligt, så bruger jeg friske resultater."
        )

    if _looks_like_unverified_computer_action_claim(clean, computer_context=computer_context):
        if computer_context.strip():
            return (
                "Jeg har kun read-only computerkontekst i den her tur. "
                "Jeg kan bruge det jeg fik, men jeg har ikke kørt en fri handling."
            )
        return (
            "Jeg fik ikke kørt en computerhandling i den her tur. "
            "Sig terminal eller Sandcode tydeligt, så tager jeg den som en rigtig handling."
        )

    return clean


def _looks_like_unverified_computer_action_claim(text: str, *, computer_context: str) -> bool:
    if "Computer-action-resultat:" in computer_context:
        return False
    return bool(_COMPUTER_ACTION_CLAIM_RE.search(text) and _COMPUTER_DOMAIN_RE.search(text))


def _guard_stacky_persona(text: str) -> str:
    clean = " ".join(text.split()).strip()
    if not clean:
        return clean
    clean = _ASSISTANT_IDENTITY_SENTENCE_RE.sub(" ", clean)
    for pattern, replacement in _ASSISTANT_HELP_REPLACEMENTS:
        clean = pattern.sub(replacement, clean)
    clean = re.sub(r"\s+([,.!?])", r"\1", clean)
    clean = re.sub(r"\s{2,}", " ", clean).strip(" ,")
    if clean:
        first = clean[:1]
        clean = first.upper() + clean[1:]
    return clean or "Den der blev for meget support-stemme. Kort version: jeg er her."


def _wants_detail(user_text: str) -> bool:
    lowered = user_text.lower()
    simple_visual_or_live_question = (
        "hvad ser" in lowered
        or "hvordan ser" in lowered
        or "kan du se" in lowered
        or "hvad kan du se" in lowered
        or "hvad synes du" in lowered
    )
    explicit_detail = any(
        trigger in lowered
        for trigger in (
            "forklar",
            "uddyb",
            "detaljer",
            "plan",
            "kode",
            "implement",
            "arkitektur",
            "strategi",
            "funktion",
            "fungere",
            "fejlfind",
        )
    )
    if simple_visual_or_live_question and not explicit_detail:
        return False
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
        "funktion",
        "fungere",
        "lav det",
        "gør det",
    )
    return any(trigger in lowered for trigger in triggers)
