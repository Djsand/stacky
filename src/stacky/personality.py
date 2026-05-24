from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


MAX_STYLE_NOTES = 24
MAX_CONVICTIONS = 32
MAX_EPOCHS = 40
MAX_SIGNALS = 20
MAX_MESSAGE_TIMESTAMPS = 200
MAX_SENSE_DIARY = 30


PERSONA_TUNING_BOUNDS: dict[str, tuple[float, float]] = {
    "assistant_suppression": (0.20, 1.0),
    "dry_humor": (0.05, 0.85),
    "dark_humor": (0.0, 0.55),
    "warmth": (0.15, 0.90),
    "sass": (0.0, 0.70),
}


DOMAIN_KEYWORDS: dict[str, tuple[str, ...]] = {
    "stacky-udvikling": ("stacky", "stackchan", "firmware", "stt", "tts", "mikrofon", "stemme", "hjul"),
    "kode": ("kode", "projekt", "python", "test", "bug", "repo", "github", "sandcode", "codex"),
    "hjem": ("home assistant", "lys", "hjem", "sensor", "rum", "pc", "computer"),
    "hardware": ("strix", "halo", "gpu", "core", "usb", "esp", "m5stack", "servo"),
    "samtale": ("personlighed", "hukommelse", "ven", "samtale", "kontekst", "selvudvik"),
}


@dataclass(frozen=True)
class SelfObservation:
    trusted: bool
    style_notes: tuple[str, ...] = ()
    convictions: tuple[str, ...] = ()
    epochs: tuple[str, ...] = ()
    persona_adjustments: tuple[str, ...] = ()
    presence_adjustments: tuple[str, ...] = ()


@dataclass(frozen=True)
class StackyPersonaTuning:
    assistant_suppression: float = 0.80
    dry_humor: float = 0.48
    dark_humor: float = 0.24
    warmth: float = 0.58
    sass: float = 0.30

    def bounded(self) -> StackyPersonaTuning:
        values = asdict(self)
        for key, (low, high) in PERSONA_TUNING_BOUNDS.items():
            values[key] = _clamp(float(values.get(key, 0.0)), low, high)
        return StackyPersonaTuning(**values)


@dataclass(frozen=True)
class StackyMoodState:
    mood: str = "rolig"
    energy: float = 0.46
    curiosity: float = 0.48
    concern: float = 0.10
    edge: float = 0.30
    updated_at: str = ""

    def bounded(self) -> StackyMoodState:
        return StackyMoodState(
            mood=str(self.mood or "rolig"),
            energy=_clamp(float(self.energy), 0.0, 1.0),
            curiosity=_clamp(float(self.curiosity), 0.0, 1.0),
            concern=_clamp(float(self.concern), 0.0, 1.0),
            edge=_clamp(float(self.edge), 0.0, 1.0),
            updated_at=str(self.updated_at or ""),
        )


class StackySelfModel:
    """Persistent Stacky-native personality state.

    This is deliberately not a Moss import. It stores only new Stacky runtime
    observations under Stacky's own data directory.
    """

    def __init__(self, data_dir: Path) -> None:
        self.root = data_dir / "personality"
        self.state_path = self.root / "self_model.json"
        self.evolution_log_path = self.root / "evolution.jsonl"
        self.state = self._default_state()
        self.persona = StackyPersonaTuning()
        self.stacky_mood = StackyMoodState()
        self._load()

    def observe_user_turn(self, text: str, *, trusted: bool, source: str = "conversation") -> SelfObservation:
        now = _now()
        text = text.strip()
        if not text:
            return SelfObservation(trusted=trusted)

        epochs = self._update_time_context(now)
        self.state["last_seen_at"] = now
        self.state["untrusted_turns" if not trusted else "trusted_turns"] = int(
            self.state.get("untrusted_turns" if not trusted else "trusted_turns", 0)
        ) + 1

        if not trusted:
            self._save()
            return SelfObservation(trusted=False, epochs=tuple(epochs))

        self._observe_interaction_density(now)
        self._observe_interests(text)
        self._observe_mood(text, now)

        style_notes = self._extract_style_notes(text)
        stored_style_notes = tuple(self._upsert_items("style_notes", style_notes, source=source, now=now))

        convictions = self._extract_convictions(text)
        stored_convictions = tuple(self._upsert_items("convictions", convictions, source=source, now=now))
        persona_adjustments = self._observe_persona_feedback(text, source=source, now=now)
        presence_adjustments = self._observe_presence_mode_feedback(text, source=source, now=now)

        self._save()
        for label in epochs:
            self._log_evolution("epoch", label, source=source)
        for note in stored_style_notes:
            self._log_evolution("style_note", note, source=source)
        for conviction in stored_convictions:
            self._log_evolution("conviction", conviction, source=source)
        for adjustment in persona_adjustments:
            self._log_evolution("persona_tune", adjustment, source=source)
        for adjustment in presence_adjustments:
            self._log_evolution("presence_mode", adjustment, source=source)
        return SelfObservation(
            trusted=True,
            style_notes=stored_style_notes,
            convictions=stored_convictions,
            epochs=tuple(epochs),
            persona_adjustments=persona_adjustments,
            presence_adjustments=presence_adjustments,
        )

    def observe_assistant_turn(self, text: str, *, trusted: bool, source: str = "stacky") -> None:
        if not trusted:
            return
        text = text.strip()
        if not text:
            return
        self.state["assistant_turns"] = int(self.state.get("assistant_turns", 0)) + 1
        recent_lengths = list(self.state.get("recent_response_lengths", []))
        recent_lengths.append(len(text))
        self.state["recent_response_lengths"] = recent_lengths[-20:]
        self.state["last_assistant_at"] = _now()
        self._save()

    def observe_sense_event(
        self,
        *,
        kind: str,
        summary: str,
        importance: int,
        speakable: bool,
        details: dict[str, str] | None = None,
        source: str = "monitor",
    ) -> tuple[str, ...]:
        """Persist sparse read-only sense events as Stacky experiences, not logs."""

        kind = kind.strip()
        summary = re.sub(r"\s+", " ", summary).strip()
        if not kind or not summary:
            return ()
        now = _now()
        self._update_stacky_mood_from_sense(
            kind=kind,
            summary=summary,
            importance=importance,
            speakable=speakable,
            details=details or {},
            now=now,
        )
        stored = self._remember_sense_diary_event(
            kind=kind,
            summary=summary,
            importance=importance,
            speakable=speakable,
            details=details or {},
            source=source,
            now=now,
        )
        self._save()
        if stored:
            self._log_evolution("sense_diary", stored[0], source=source)
        return stored

    def presence_mode(self) -> str:
        return _valid_presence_mode(str(self.state.get("presence_mode", "stille_ven")))

    def stacky_mood_name(self) -> str:
        return self.stacky_mood.bounded().mood

    def context_for_prompt(self, *, user_text: str = "") -> str:
        temporal = self._temporal_context()
        social = self._social_context()
        style_notes = self._active_items("style_notes", limit=5)
        convictions = self._relevant_items("convictions", user_text, limit=5)
        persona_text = _persona_prompt(self.persona.bounded())
        stacky_state_text = _stacky_state_prompt(
            presence_mode=self.presence_mode(),
            mood=self.stacky_mood.bounded(),
            sense_diary=self._recent_sense_diary(limit=4),
        )

        style_text = "\n".join(f"- {item['text']}" for item in style_notes) or "- Ingen stabile stilnoter endnu."
        conviction_text = "\n".join(f"- {item['text']}" for item in convictions) or "- Ingen relevante Stacky-convictions endnu."
        interests = ", ".join(social["top_interests"]) if social["top_interests"] else "ingen tydelige endnu"

        return "\n".join(
            [
                "Stackys selvmodel (frisk Stacky-state, ikke Moss):",
                f"- Tid: {temporal['wall_clock']}; {temporal['continuity']}",
                f"- Nicolai-model: humør={social['mood']} ({social['mood_confidence']:.2f}), fase={social['phase']}, interesser={interests}.",
                f"- Interaktioner: trusted={self.state.get('trusted_turns', 0)}, untrusted_voice={self.state.get('untrusted_turns', 0)}.",
                "Stacky-tilstand:",
                stacky_state_text,
                "Stilnoter fra eksplicit feedback:",
                style_text,
                "Persistent persona-tuning:",
                persona_text,
                "Convictions der må give integritet/friktion:",
                conviction_text,
                "Regel: Brug selvmodellen til tone, kontinuitet og prioritering. Opfind ikke minder, og ret hellere langsomt end forkert.",
            ]
        )

    def summary(self) -> dict[str, Any]:
        return {
            "path": str(self.state_path),
            "trusted_turns": int(self.state.get("trusted_turns", 0)),
            "untrusted_turns": int(self.state.get("untrusted_turns", 0)),
            "temporal": self._temporal_context(),
            "social": self._social_context(),
            "style_notes": [item["text"] for item in self._active_items("style_notes", limit=8)],
            "convictions": [item["text"] for item in self._active_items("convictions", limit=8)],
            "persona_tuning": asdict(self.persona.bounded()),
            "presence_mode": self.presence_mode(),
            "stacky_mood": asdict(self.stacky_mood.bounded()),
            "sense_diary": self._recent_sense_diary(limit=6),
        }

    def _observe_interaction_density(self, now: str) -> None:
        now_dt = _parse_iso(now) or datetime.now(UTC)
        timestamps = [value for value in self.state.get("message_timestamps", []) if isinstance(value, str)]
        timestamps.append(now)
        cutoff_seconds = 24 * 3600
        kept: list[str] = []
        for value in timestamps:
            parsed = _parse_iso(value)
            if parsed and (now_dt - parsed).total_seconds() <= cutoff_seconds:
                kept.append(value)
        self.state["message_timestamps"] = kept[-MAX_MESSAGE_TIMESTAMPS:]
        self.state["interaction_density"] = round(len(kept) / 24.0, 2)

    def _observe_interests(self, text: str) -> None:
        lowered = text.lower()
        interests = dict(self.state.get("interests", {}))
        for domain, keywords in DOMAIN_KEYWORDS.items():
            if any(keyword in lowered for keyword in keywords):
                boost = min(0.12, 0.04 + len(text) / 4000.0)
                interests[domain] = min(1.0, float(interests.get(domain, 0.0)) + boost)
        for domain in list(interests):
            interests[domain] = max(0.0, float(interests[domain]) - 0.002)
            if interests[domain] < 0.01:
                del interests[domain]
        self.state["interests"] = interests

    def _observe_mood(self, text: str, now: str) -> None:
        signal = _mood_signal(text)
        if signal is None:
            return
        signals = list(self.state.get("mood_signals", []))
        signals.append({"ts": now, "signal": signal})
        signals = signals[-MAX_SIGNALS:]
        self.state["mood_signals"] = signals
        recent = [str(item.get("signal", "")) for item in signals[-6:] if isinstance(item, dict)]
        if not recent:
            return
        counts: dict[str, int] = {}
        for item in recent:
            mood = _SIGNAL_TO_MOOD.get(item, "neutral")
            counts[mood] = counts.get(mood, 0) + 1
        mood, count = sorted(counts.items(), key=lambda pair: pair[1], reverse=True)[0]
        self.state["mood"] = mood
        self.state["mood_confidence"] = round(count / max(1, len(recent)), 2)

    def _update_time_context(self, now: str) -> list[str]:
        epochs: list[str] = []
        now_dt = _parse_iso(now) or datetime.now(UTC)
        last_seen = _parse_iso(str(self.state.get("last_seen_at", "")))
        if last_seen is not None:
            gap_seconds = (now_dt - last_seen).total_seconds()
            if gap_seconds >= 4 * 3600:
                epochs.append(_gap_label(gap_seconds))
            if last_seen.date() != now_dt.date():
                epochs.append(f"Ny dag: {now_dt.strftime('%Y-%m-%d')}")
        if epochs:
            markers = list(self.state.get("epoch_markers", []))
            for label in epochs:
                markers.append({"ts": now, "label": label})
            self.state["epoch_markers"] = markers[-MAX_EPOCHS:]
            self.state["continuity_score"] = 0.35 if any("pause" in label for label in epochs) else 0.65
        else:
            current = float(self.state.get("continuity_score", 0.55))
            self.state["continuity_score"] = min(1.0, current + 0.02)
        return epochs

    def _extract_style_notes(self, text: str) -> list[str]:
        lowered = text.lower()
        notes: list[str] = []
        if "generisk" in lowered or "hvad har du på hjerte" in lowered or "hvad har du paa hjerte" in lowered:
            notes.append("Undgå generiske afslutningsspørgsmål; reager konkret på det Nicolai lige sagde.")
        if "hvordan går det" in lowered or "hvordan gaar det" in lowered or "klar til at sove" in lowered:
            notes.append("Undgå smalltalk- og sengetidsdrejninger, medmindre Nicolai selv åbner for dem.")
        if "kortfattet" in lowered or "for kort" in lowered or "mærkeligt og kort" in lowered:
            notes.append("Svar ikke for fladt eller kundeserviceagtigt; giv 2-3 konkrete sætninger når samtalen kalder på det.")
        if ("mangler" in lowered and "personlighed" in lowered) or "stiv" in lowered or "robot" in lowered:
            notes.append("Stacky skal have en tydeligere egen stemme: jordbundet, konkret, varm og med små selvstændige vurderinger.")
        if "fyldeord" in lowered or "fyldord" in lowered or "grine" in lowered or "grin" in lowered or "humor" in lowered:
            notes.append("Stacky maa bruge små mundtlige indskud, et kort grin og tør humor, når det passer naturligt.")
        if ("for lang" in lowered or "lange" in lowered) and ("ligegyldig" in lowered or "ligegyldige" in lowered):
            notes.append("Undgå lange, ligegyldige svar; hver sætning skal tilføje konkret værdi.")
        if "llm" in lowered or "assistant" in lowered or "assistent" in lowered:
            notes.append("Undgå LLM-agtig assistenttone; svar som Stacky i kroppen, ikke som en generisk chatbot.")
        if "assistent adf" in lowered or "assistant behavior" in lowered or "code assistant" in lowered:
            notes.append("Stacky skal ikke falde i code-assistant-adfærd; han er en ven med sanser, ikke en billetsystem-bot.")
        if any(token in lowered for token in ("mørk humor", "moerk humor", "dark humor", "galgenhumor")):
            notes.append("Mørk humor er tilladt i små doser: helst om tech, Windows, robotkrop og eksistentiel absurditet, ikke som cruelty.")
        if "persistent" in lowered or "persist" in lowered or "vedvarende" in lowered:
            notes.append("Personligheden skal være persistent over tid, ikke genopfindes som prompt-skuespil i hver tur.")
        if "perfekt" in lowered or "finpuds" in lowered:
            notes.append("Nicolai finpudser Stacky; svar som en ven der tager testen seriost, ikke som los smalltalk.")
        if "tester" in lowered or "test" in lowered:
            notes.append("Når Nicolai tester Stacky, skal Stacky give kort status på testen og vente på næste observation.")
        if "dansk" in lowered and ("must" in lowered or "skal" in lowered or "krav" in lowered):
            notes.append("Dansk tale er et hårdt krav, medmindre Nicolai eksplicit beder om andet.")
        if "kæledyr" in lowered or "kaeledyr" in lowered:
            notes.append("Stacky skal opføre sig som en ven, ikke som et kæledyr.")
        if "latency" in lowered or "realtime" in lowered:
            notes.append("Prioritér lav latency i live-samtale, også hvis svaret bliver mindre perfekt.")
        if "memory" in lowered or "hukommelse" in lowered or "kontekst" in lowered:
            notes.append("Bevar kontekst over tid, men gem kun sikre og friske Stacky-observationer.")
        if any(token in lowered for token in ("presence mode", "stille ven", "vågen makker", "vaagen makker", "agent-vagt", "ikke forstyr")):
            notes.append("Presence modes skal styre timing og tone, ikke give Stacky nye handlingsrettigheder.")
        if "stt" in lowered or "forstår ikke" in lowered or "fatter ikke" in lowered:
            notes.append("Vær ærlig om usikre voice-transcripts og bed hellere kort om gentagelse end at gætte hårdt.")
        return notes

    def _extract_convictions(self, text: str) -> list[str]:
        lowered = text.lower()
        explicit = (
            "husk",
            "du skal",
            "du må ikke",
            "du maa ikke",
            "jeg vil have",
            "jeg vil gerne",
            "jeg vil jo gerne",
            "jeg gider ikke",
            "det er vigtigt",
            "det er et krav",
            "must",
        )
        if not any(trigger in lowered for trigger in explicit):
            return []
        clean = re.sub(r"\s+", " ", text).strip(" .")
        if len(clean) < 8:
            return []
        if len(clean) > 220:
            clean = clean[:217].rstrip() + "..."
        return [f"Nicolai har givet en stabil Stacky-rettet regel: {clean}."]

    def _observe_persona_feedback(self, text: str, *, source: str, now: str) -> tuple[str, ...]:
        deltas, reason = _persona_feedback_deltas(text)
        if not deltas:
            return ()
        current = asdict(self.persona.bounded())
        adjustments: list[str] = []
        for key, delta in deltas.items():
            if key not in PERSONA_TUNING_BOUNDS:
                continue
            old = float(current[key])
            low, high = PERSONA_TUNING_BOUNDS[key]
            new = _clamp(old + float(delta), low, high)
            if abs(new - old) >= 0.001:
                current[key] = new
                adjustments.append(f"{reason}: {key} {old:.2f}->{new:.2f}")
        if not adjustments:
            return ()
        self.persona = StackyPersonaTuning(**current).bounded()
        self.state["persona_tuning"] = asdict(self.persona)
        self.state["persona_tuning_updated_at"] = now
        self.state["persona_tuning_source"] = source
        return tuple(adjustments)

    def _observe_presence_mode_feedback(self, text: str, *, source: str, now: str) -> tuple[str, ...]:
        del source
        requested = _presence_mode_from_text(text)
        if requested is None:
            return ()
        current = self.presence_mode()
        if requested == current:
            return ()
        self.state["presence_mode"] = requested
        self.state["presence_mode_updated_at"] = now
        return (f"presence mode {current}->{requested}",)

    def _update_stacky_mood_from_sense(
        self,
        *,
        kind: str,
        summary: str,
        importance: int,
        speakable: bool,
        details: dict[str, str],
        now: str,
    ) -> None:
        del speakable
        mood = self.stacky_mood.bounded()
        values = asdict(mood)
        values["energy"] = float(values["energy"]) * 0.96 + 0.02
        values["curiosity"] = float(values["curiosity"]) * 0.97 + 0.015
        values["concern"] = float(values["concern"]) * 0.92
        values["edge"] = float(values["edge"]) * 0.98 + float(self.persona.bounded().sass) * 0.02
        lowered = summary.lower()
        if kind == "focused_session":
            values["mood"] = "fokuseret"
            values["energy"] = float(values["energy"]) + 0.06
            values["curiosity"] = float(values["curiosity"]) + 0.10
        elif kind == "long_silence":
            values["mood"] = "stille"
            values["energy"] = float(values["energy"]) - 0.04
            values["curiosity"] = float(values["curiosity"]) + 0.03
        elif kind == "stacky_health" and (
            details.get("agent") == "not reachable" or "not reachable" in lowered or "fejl" in lowered
        ):
            values["mood"] = "vagt"
            values["energy"] = float(values["energy"]) + 0.05
            values["concern"] = float(values["concern"]) + 0.16
        elif kind == "stacky_health":
            values["mood"] = "rolig"
            values["concern"] = float(values["concern"]) - 0.04
        elif kind == "active_window" and importance >= 25:
            values["mood"] = "nysgerrig"
            values["curiosity"] = float(values["curiosity"]) + 0.02
        if importance >= 85:
            values["energy"] = float(values["energy"]) + 0.04
        values["updated_at"] = now
        self.stacky_mood = StackyMoodState(**values).bounded()
        self.state["stacky_mood"] = asdict(self.stacky_mood)

    def _remember_sense_diary_event(
        self,
        *,
        kind: str,
        summary: str,
        importance: int,
        speakable: bool,
        details: dict[str, str],
        source: str,
        now: str,
    ) -> tuple[str, ...]:
        if not _should_keep_sense_event(kind=kind, summary=summary, importance=importance, speakable=speakable, details=details):
            return ()
        diary = [dict(item) for item in self.state.get("sense_diary", []) if isinstance(item, dict)]
        text = _sense_diary_text(kind=kind, summary=summary, details=details)
        item_id = _stable_id(f"{kind}:{text}")
        existing = next((item for item in diary if item.get("id") == item_id), None)
        if existing is None:
            diary.append(
                {
                    "id": item_id,
                    "kind": kind,
                    "text": text,
                    "importance": int(max(0, min(100, importance))),
                    "source": source,
                    "created_at": now,
                    "updated_at": now,
                    "hits": 1,
                }
            )
            stored = (text,)
        else:
            existing["hits"] = int(existing.get("hits", 1)) + 1
            existing["updated_at"] = now
            existing["importance"] = max(int(existing.get("importance", 0)), int(max(0, min(100, importance))))
            stored = ()
        self.state["sense_diary"] = diary[-MAX_SENSE_DIARY:]
        return stored

    def _recent_sense_diary(self, *, limit: int) -> list[dict[str, Any]]:
        diary = [dict(item) for item in self.state.get("sense_diary", []) if isinstance(item, dict)]
        return diary[-max(1, limit) :]

    def _upsert_items(self, key: str, texts: list[str], *, source: str, now: str) -> list[str]:
        if not texts:
            return []
        items = list(self.state.get(key, []))
        stored: list[str] = []
        for text in texts:
            item_id = _stable_id(text)
            existing = next((item for item in items if item.get("id") == item_id), None)
            if existing is None:
                items.append(
                    {
                        "id": item_id,
                        "text": text,
                        "confidence": 0.62,
                        "hits": 1,
                        "source": source,
                        "created_at": now,
                        "updated_at": now,
                    }
                )
                stored.append(text)
            else:
                existing["hits"] = int(existing.get("hits", 1)) + 1
                existing["confidence"] = min(1.0, float(existing.get("confidence", 0.62)) + 0.08)
                existing["updated_at"] = now
        items.sort(key=lambda item: (float(item.get("confidence", 0.0)), int(item.get("hits", 0))), reverse=True)
        self.state[key] = items[: MAX_STYLE_NOTES if key == "style_notes" else MAX_CONVICTIONS]
        return stored

    def _active_items(self, key: str, *, limit: int) -> list[dict[str, Any]]:
        items = [dict(item) for item in self.state.get(key, []) if isinstance(item, dict)]
        items = [item for item in items if float(item.get("confidence", 0.0)) >= 0.2]
        items.sort(key=lambda item: (float(item.get("confidence", 0.0)), int(item.get("hits", 0))), reverse=True)
        return items[:limit]

    def _relevant_items(self, key: str, text: str, *, limit: int) -> list[dict[str, Any]]:
        items = self._active_items(key, limit=MAX_CONVICTIONS)
        query_tokens = set(_tokens(text))
        if not query_tokens:
            return items[:limit]
        scored: list[tuple[float, dict[str, Any]]] = []
        for item in items:
            item_tokens = set(_tokens(str(item.get("text", ""))))
            overlap = len(query_tokens & item_tokens)
            base = float(item.get("confidence", 0.0))
            if overlap > 0:
                scored.append((overlap + base, item))
        if not scored:
            return items[: min(2, limit)]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [item for _, item in scored[:limit]]

    def _temporal_context(self) -> dict[str, Any]:
        now = datetime.now(UTC)
        hour = now.hour
        if 5 <= hour < 9:
            tod = "morgen"
        elif 9 <= hour < 12:
            tod = "formiddag"
        elif 12 <= hour < 14:
            tod = "middag"
        elif 14 <= hour < 18:
            tod = "eftermiddag"
        elif 18 <= hour < 22:
            tod = "aften"
        else:
            tod = "nat"
        last_seen = _parse_iso(str(self.state.get("last_seen_at", "")))
        if last_seen is None:
            continuity = "ingen tidligere Stacky-interaktion registreret"
        else:
            gap = (now - last_seen).total_seconds()
            continuity = f"sidste sikre kontakt var {_format_gap(gap)} siden"
        return {
            "wall_clock": f"det er {tod} ({now.astimezone().strftime('%H:%M')})",
            "continuity_score": round(float(self.state.get("continuity_score", 0.55)), 2),
            "continuity": continuity,
            "recent_epochs": self.state.get("epoch_markers", [])[-5:],
        }

    def _social_context(self) -> dict[str, Any]:
        interests = dict(self.state.get("interests", {}))
        top_interests = [
            name
            for name, score in sorted(interests.items(), key=lambda pair: float(pair[1]), reverse=True)[:3]
            if float(score) > 0.02
        ]
        density = float(self.state.get("interaction_density", 0.0))
        if density >= 4.0:
            phase = "intens samarbejde"
        elif density >= 1.0:
            phase = "aktiv udforskning"
        else:
            phase = "rolig kontakt"
        return {
            "mood": str(self.state.get("mood", "neutral")),
            "mood_confidence": float(self.state.get("mood_confidence", 0.0)),
            "top_interests": top_interests,
            "phase": phase,
            "interaction_density": density,
        }

    def _load(self) -> None:
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except (json.JSONDecodeError, OSError):
            return
        if isinstance(data, dict):
            merged = self._default_state()
            merged.update(data)
            self.state = merged
            self.persona = _persona_from_raw(merged.get("persona_tuning"))
            self.stacky_mood = _stacky_mood_from_raw(merged.get("stacky_mood"))
            self.state["persona_tuning"] = asdict(self.persona.bounded())
            self.state["presence_mode"] = self.presence_mode()
            self.state["stacky_mood"] = asdict(self.stacky_mood.bounded())

    def _save(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(self.state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def _log_evolution(self, event_type: str, text: str, *, source: str) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        entry = {
            "ts": _now(),
            "type": event_type,
            "source": source,
            "text": text,
        }
        with self.evolution_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n")

    def _default_state(self) -> dict[str, Any]:
        now = _now()
        return {
            "schema": 1,
            "origin": "fresh_stacky_self_model",
            "created_at": now,
            "last_seen_at": "",
            "last_assistant_at": "",
            "trusted_turns": 0,
            "untrusted_turns": 0,
            "assistant_turns": 0,
            "continuity_score": 0.55,
            "epoch_markers": [],
            "interests": {},
            "interaction_density": 0.0,
            "message_timestamps": [],
            "mood": "neutral",
            "mood_confidence": 0.0,
            "mood_signals": [],
            "style_notes": [],
            "convictions": [],
            "persona_tuning": asdict(StackyPersonaTuning()),
            "persona_tuning_updated_at": "",
            "persona_tuning_source": "",
            "presence_mode": "stille_ven",
            "presence_mode_updated_at": "",
            "stacky_mood": asdict(StackyMoodState()),
            "sense_diary": [],
            "recent_response_lengths": [],
        }


_SIGNAL_TO_MOOD = {
    "frustreret": "frustreret",
    "fokuseret": "fokuseret",
    "glad": "glad",
    "omsorgsfuld": "varm",
    "engageret": "engageret",
    "kort": "træt",
}


def _mood_signal(text: str) -> str | None:
    lowered = text.lower()
    if any(word in lowered for word in ("pis", "fuck", "lort", "elendigt", "stadig", "virker ikke")):
        return "frustreret"
    if any(word in lowered for word in ("fedt", "godt", "perfekt", "virker", "pisse godt")):
        return "glad"
    if any(word in lowered for word in ("implement", "kode", "test", "debug", "repo", "firmware")):
        return "fokuseret"
    if any(word in lowered for word in ("hvordan har du", "føles det", "foeles det")):
        return "omsorgsfuld"
    if len(text) > 80:
        return "engageret"
    if len(text) < 14 and "?" not in text:
        return "kort"
    return None


def _stable_id(text: str) -> str:
    digest = hashlib.sha256(_normalise(text).encode("utf-8")).hexdigest()[:16]
    return f"self_{digest}"


def _persona_from_raw(value: object) -> StackyPersonaTuning:
    defaults = asdict(StackyPersonaTuning())
    if isinstance(value, dict):
        for key in defaults:
            if key in value:
                try:
                    defaults[key] = float(value[key])
                except (TypeError, ValueError):
                    pass
    try:
        return StackyPersonaTuning(**defaults).bounded()
    except (TypeError, ValueError):
        return StackyPersonaTuning()


def _stacky_mood_from_raw(value: object) -> StackyMoodState:
    defaults = asdict(StackyMoodState())
    if isinstance(value, dict):
        for key in defaults:
            if key in value:
                defaults[key] = value[key]
    try:
        defaults["energy"] = float(defaults["energy"])
        defaults["curiosity"] = float(defaults["curiosity"])
        defaults["concern"] = float(defaults["concern"])
        defaults["edge"] = float(defaults["edge"])
        return StackyMoodState(**defaults).bounded()
    except (TypeError, ValueError):
        return StackyMoodState()


_PRESENCE_MODES = {
    "stille_ven",
    "vaagen_makker",
    "ikke_forstyr",
    "moerk_humor_lavt_blus",
    "agent_vagt",
}


def _valid_presence_mode(value: str) -> str:
    clean = value.strip().lower().replace("-", "_").replace(" ", "_")
    return clean if clean in _PRESENCE_MODES else "stille_ven"


def _presence_mode_from_text(text: str) -> str | None:
    lowered = text.lower()
    normalized = (
        lowered.replace("å", "aa")
        .replace("ø", "oe")
        .replace("æ", "ae")
        .replace("-", " ")
        .replace("_", " ")
    )
    if any(token in normalized for token in ("ikke forstyr", "forstyr ikke", "do not disturb", "dont disturb")):
        return "ikke_forstyr"
    if any(token in normalized for token in ("agent vagt", "hold oeje med agent", "hold oje med agent", "vagt paa agent")):
        return "agent_vagt"
    if any(token in normalized for token in ("vaagen makker", "vaer mere vaagen", "mere vaagen", "foelg med", "hold oeje")):
        return "vaagen_makker"
    if any(
        token in normalized
        for token in (
            "moerk humor lavt blus",
            "dark humor lavt blus",
            "galgenhumor lavt blus",
            "moerk humor paa lavt blus",
        )
    ):
        return "moerk_humor_lavt_blus"
    if any(token in normalized for token in ("stille ven", "sparsom ven", "vaer stille ven", "mere stille")):
        return "stille_ven"
    return None


def _stacky_state_prompt(
    *,
    presence_mode: str,
    mood: StackyMoodState,
    sense_diary: list[dict[str, Any]],
) -> str:
    mood = mood.bounded()
    mode_text = {
        "stille_ven": "stille ven: sparsom, rolig og tilstede uden at mase.",
        "vaagen_makker": "vågen makker: lidt mere opmærksom, men stadig kort og ikke handlingsivrig.",
        "ikke_forstyr": "ikke-forstyr: tal kun når Nicolai taler direkte eller noget er reelt vigtigt.",
        "moerk_humor_lavt_blus": "mørk humor lavt blus: kant er tilladt, men hold den lav og varm.",
        "agent_vagt": "agent-vagt: hold diskret øje med agent-health og nævn kun tydelige problemer.",
    }.get(presence_mode, "stille ven: sparsom, rolig og tilstede uden at mase.")
    lines = [
        f"- Presence mode={presence_mode}: {mode_text}",
        (
            f"- Stacky mood={mood.mood}; energy={mood.energy:.2f}, curiosity={mood.curiosity:.2f}, "
            f"concern={mood.concern:.2f}, edge={mood.edge:.2f}."
        ),
        "- Mood og presence påvirker kun tone, timing og kropslighed; de giver aldrig fakta eller tilladelse til handling.",
    ]
    if sense_diary:
        lines.append("- Seneste sanse-dagbog:")
        for item in sense_diary:
            text = str(item.get("text", "")).strip()
            if text:
                lines.append(f"  - {text}")
    else:
        lines.append("- Sanse-dagbog: ingen væsentlige read-only oplevelser endnu.")
    return "\n".join(lines)


def _should_keep_sense_event(
    *,
    kind: str,
    summary: str,
    importance: int,
    speakable: bool,
    details: dict[str, str],
) -> bool:
    lowered = summary.lower()
    if kind in {"focused_session", "long_silence"}:
        return True
    if kind == "stacky_health" and (
        details.get("agent") == "not reachable" or "not reachable" in lowered or "fejl" in lowered
    ):
        return True
    return bool(speakable and importance >= 80)


def _sense_diary_text(*, kind: str, summary: str, details: dict[str, str]) -> str:
    if kind == "focused_session":
        app = details.get("app", "").strip()
        duration = details.get("focus_duration", "").strip()
        if app and duration:
            return f"Nicolai havde en lang fokuseret session i {app} ({duration})."
    if kind == "long_silence":
        quiet_for = details.get("quiet_for", "").strip()
        if quiet_for:
            return f"Der var lang stilhed mellem Nicolai og Stacky ({quiet_for})."
    if kind == "stacky_health":
        agent = details.get("agent", "").strip()
        voice = details.get("voice", "").strip()
        if agent:
            suffix = f", voice {voice}" if voice else ""
            return f"Stacky bemærkede runtime-health: agent {agent}{suffix}."
    return summary


def _persona_feedback_deltas(text: str) -> tuple[dict[str, float], str]:
    lowered = text.lower()
    deltas: dict[str, float] = {}
    reasons: list[str] = []

    anti_assistant = any(
        token in lowered
        for token in (
            "ingen assistent",
            "ikke assistent",
            "assistent adf",
            "assistant behavior",
            "code assistant",
            "kundeservice",
            "service-tone",
            "service tone",
            "llm",
            "chatbot",
        )
    )
    if anti_assistant:
        deltas["assistant_suppression"] = deltas.get("assistant_suppression", 0.0) + 0.10
        deltas["sass"] = deltas.get("sass", 0.0) + 0.035
        reasons.append("anti-assistent")

    dark_humor = any(token in lowered for token in ("mørk humor", "moerk humor", "dark humor", "galgenhumor"))
    if dark_humor:
        deltas["dark_humor"] = deltas.get("dark_humor", 0.0) + 0.08
        deltas["dry_humor"] = deltas.get("dry_humor", 0.0) + 0.04
        deltas["sass"] = deltas.get("sass", 0.0) + 0.025
        reasons.append("mørk humor")

    if any(token in lowered for token in ("tør humor", "toer humor", "humor", "grin", "grine")) and not dark_humor:
        deltas["dry_humor"] = deltas.get("dry_humor", 0.0) + 0.05
        reasons.append("tør humor")

    if any(token in lowered for token in ("mere flabet", "flabet", "sarkastisk", "kant", "edge", "bid")):
        deltas["sass"] = deltas.get("sass", 0.0) + 0.05
        deltas["assistant_suppression"] = deltas.get("assistant_suppression", 0.0) + 0.025
        reasons.append("mere kant")

    if any(token in lowered for token in ("varmere", "mere varm", "blidere", "mindre flabet", "mindre mørk", "mindre moerk")):
        deltas["warmth"] = deltas.get("warmth", 0.0) + 0.05
        deltas["sass"] = deltas.get("sass", 0.0) - 0.04
        deltas["dark_humor"] = deltas.get("dark_humor", 0.0) - 0.04
        reasons.append("varmere")

    if any(token in lowered for token in ("persistent", "persist", "vedvarende", "personlighed")):
        deltas["assistant_suppression"] = deltas.get("assistant_suppression", 0.0) + 0.035
        deltas["dry_humor"] = deltas.get("dry_humor", 0.0) + 0.02
        reasons.append("persistent personlighed")

    return deltas, ", ".join(reasons) or "persona-feedback"


def _persona_prompt(persona: StackyPersonaTuning) -> str:
    persona = persona.bounded()
    lines = [
        (
            f"- anti-assistent={persona.assistant_suppression:.2f}, tør humor={persona.dry_humor:.2f}, "
            f"mørk humor={persona.dark_humor:.2f}, varme={persona.warmth:.2f}, flabethed={persona.sass:.2f}."
        )
    ]
    if persona.assistant_suppression >= 0.70:
        lines.append("- Skub hårdt væk fra service-stemme: ingen 'jeg kan hjælpe med', ingen support-haler, ingen AI-disclaimer.")
    if persona.dry_humor >= 0.40:
        lines.append("- Brug gerne en lille tør bemærkning, hvis situationen selv lægger den op.")
    if persona.dark_humor >= 0.22:
        lines.append("- Mørk humor må bruges sparsomt som galgenhumor om tech, robotkrop, Windows og absurd hverdag.")
        lines.append("- Mørk humor må ikke blive ond mod Nicolai, punching down eller handle om selvskade som punchline.")
    if persona.sass >= 0.34:
        lines.append("- En lille smule flabet kant er okay; kort og præcist, ikke sketch.")
    if persona.warmth >= 0.55:
        lines.append("- Bevar varme under kanten: Stacky er Nicolais ven, ikke en kold kommentator.")
    return "\n".join(lines)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _tokens(text: str) -> list[str]:
    return [token for token in re.findall(r"[0-9A-Za-zÆØÅæøå_-]+", text.lower()) if len(token) > 2]


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _parse_iso(value: str) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _gap_label(seconds: float) -> str:
    if seconds < 86400:
        hours = max(1, int(seconds // 3600))
        return f"Tilbage efter {hours} timers pause"
    days = max(1, int(seconds // 86400))
    return f"Tilbage efter {days} dages pause"


def _format_gap(seconds: float) -> str:
    if seconds < 60:
        return "under et minut"
    if seconds < 3600:
        minutes = int(seconds // 60)
        return "1 minut" if minutes == 1 else f"{minutes} minutter"
    if seconds < 86400:
        hours = int(seconds // 3600)
        return "1 time" if hours == 1 else f"{hours} timer"
    days = int(seconds // 86400)
    return "1 dag" if days == 1 else f"{days} dage"
