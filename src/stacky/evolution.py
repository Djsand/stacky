from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


MAX_RECENT_METRICS = 30
MAX_ACTIVE_REFLECTIONS = 8
REFLECT_EVERY_ASSISTANT_TURNS = 6


TUNING_BOUNDS: dict[str, tuple[float, float]] = {
    "challenge_frequency": (0.05, 0.80),
    "humor_frequency": (0.05, 0.80),
    "question_frequency": (0.05, 0.75),
    "reply_length_bias": (-0.35, 0.35),
    "body_motion_energy": (0.10, 0.85),
    "proactive_threshold": (0.35, 0.95),
}


@dataclass(frozen=True)
class StackyStyleTuning:
    challenge_frequency: float = 0.30
    humor_frequency: float = 0.28
    question_frequency: float = 0.35
    reply_length_bias: float = 0.0
    body_motion_energy: float = 0.35
    proactive_threshold: float = 0.72

    def bounded(self) -> StackyStyleTuning:
        values = asdict(self)
        for key, (low, high) in TUNING_BOUNDS.items():
            values[key] = _clamp(float(values.get(key, 0.0)), low, high)
        return StackyStyleTuning(**values)


@dataclass(frozen=True)
class EvolutionObservation:
    trusted: bool
    adjustments: tuple[str, ...] = ()
    reflection: str = ""


class StackyEvolutionEngine:
    """Stacky-native evolution overlay based on measured runtime behavior.

    This deliberately stores fresh Stacky observations only. It does not import
    Moss identity, Moss memories, or unbounded self-modification.
    """

    def __init__(self, data_dir: Path) -> None:
        self.root = data_dir / "personality"
        self.state_path = self.root / "evolution_state.json"
        self.tuning_path = self.root / "style_tuning.json"
        self.reflections_path = self.root / "reflections.jsonl"
        self.state = self._default_state()
        self.tuning = StackyStyleTuning()
        self._load()

    def observe_user_turn(self, text: str, *, trusted: bool, source: str = "conversation") -> EvolutionObservation:
        text = text.strip()
        if not text:
            return EvolutionObservation(trusted=trusted)

        if not trusted:
            self.state["untrusted_user_turns"] = _safe_int(self.state.get("untrusted_user_turns"), default=0) + 1
            self.state["last_seen_at"] = _now()
            self._save_state()
            return EvolutionObservation(trusted=False)

        self.state["trusted_user_turns"] = _safe_int(self.state.get("trusted_user_turns"), default=0) + 1
        self.state["last_seen_at"] = _now()
        self._observe_user_signal(text)
        deltas, reason = _feedback_deltas(text)
        adjustments = self._apply_tuning(deltas, reason=reason, source=source) if deltas else ()
        if adjustments and _mentions_personality_growth(text):
            self._upsert_open_question("Hvordan bliver jeg skarpere uden at blive en sketch?")
        self._save_state()
        return EvolutionObservation(trusted=True, adjustments=adjustments)

    def observe_assistant_turn(
        self,
        text: str,
        *,
        trusted: bool,
        user_text: str = "",
        source: str = "stacky",
    ) -> EvolutionObservation:
        text = text.strip()
        if not text:
            return EvolutionObservation(trusted=trusted)
        if not trusted:
            return EvolutionObservation(trusted=False)

        self.state["assistant_turns"] = _safe_int(self.state.get("assistant_turns"), default=0) + 1
        self.state["last_assistant_at"] = _now()
        metrics = _analyze_assistant_text(text)
        metrics["ts"] = self.state["last_assistant_at"]
        metrics["source"] = source
        metrics["user_hint"] = user_text[:160]
        recent = list(self.state.get("recent_turn_metrics", []))
        recent.append(metrics)
        self.state["recent_turn_metrics"] = recent[-MAX_RECENT_METRICS:]
        self._observe_assistant_metrics(metrics)

        adjustments: tuple[str, ...] = ()
        if _safe_int(metrics.get("generic_hits"), default=0) > 0:
            adjustments = self._apply_tuning(
                {
                    "question_frequency": -0.035,
                    "challenge_frequency": 0.020,
                    "humor_frequency": 0.015,
                    "reply_length_bias": -0.015,
                },
                reason="målt generisk Stacky-svar",
                source=source,
            )

        reflection = ""
        assistant_turns = _safe_int(self.state.get("assistant_turns"), default=0)
        last_reflection_turn = _safe_int(self.state.get("last_reflection_turn"), default=0)
        if assistant_turns - last_reflection_turn >= REFLECT_EVERY_ASSISTANT_TURNS:
            reflection = self._self_observe(source=source)
            self.state["last_reflection_turn"] = assistant_turns

        self._save_state()
        return EvolutionObservation(trusted=True, adjustments=adjustments, reflection=reflection)

    def context_for_prompt(self) -> str:
        emotional = _clean_emotional_state(self.state.get("emotional_state"))
        recent_summary = self._recent_summary()
        reflections = [str(item.get("text", "")) for item in self.state.get("active_reflections", []) if item.get("text")]
        questions = [str(item) for item in self.state.get("open_questions", []) if item]
        tuning = self.tuning.bounded()

        adjustment_lines = _style_instructions(tuning)
        reflection_text = "\n".join(f"- {item}" for item in reflections[-3:]) or "- Ingen stabile evolution-refleksioner endnu."
        question_text = "\n".join(f"- {item}" for item in questions[:3]) or "- Ingen aktive stilspørgsmål endnu."

        return "\n".join(
            [
                "Stackys evolution (egen Stacky-overlay, ikke Moss):",
                (
                    "- Tunings: "
                    f"modspil={tuning.challenge_frequency:.2f}, humor={tuning.humor_frequency:.2f}, "
                    f"spørgsmål={tuning.question_frequency:.2f}, længde={tuning.reply_length_bias:+.2f}, "
                    f"kropsenergi={tuning.body_motion_energy:.2f}, proaktiv tærskel={tuning.proactive_threshold:.2f}."
                ),
                (
                    "- Intern drift: "
                    f"energi={emotional['energy']:.0f}, "
                    f"nysgerrighed={emotional['curiosity']:.0f}, "
                    f"tilfredshed={emotional['satisfaction']:.0f}, "
                    f"friktion={emotional['frustration']:.0f}, "
                    f"forvirring={emotional['confusion']:.0f}."
                ),
                f"- Målt selvobservation: {recent_summary}",
                "Seneste evolution-refleksioner:",
                reflection_text,
                "Aktive stilspørgsmål:",
                question_text,
                "Adfærdsjusteringer lige nu:",
                adjustment_lines,
                (
                    "Autonom evolutionsregel: Du må justere tone ud fra målte mønstre uden at Nicolai prompt-redigerer dig, "
                    "men basissjæl, danskkrav, sandhed, sikkerhed og eksplicit Nicolai-feedback overstyrer altid tuningen. "
                    "Opfind ikke minder, evner eller handlinger for at virke mere levende."
                ),
            ]
        )

    def summary(self) -> dict[str, Any]:
        return {
            "path": str(self.state_path),
            "tuning_path": str(self.tuning_path),
            "trusted_user_turns": _safe_int(self.state.get("trusted_user_turns"), default=0),
            "untrusted_user_turns": _safe_int(self.state.get("untrusted_user_turns"), default=0),
            "assistant_turns": _safe_int(self.state.get("assistant_turns"), default=0),
            "tuning": asdict(self.tuning.bounded()),
            "emotional_state": _clean_emotional_state(self.state.get("emotional_state")),
            "recent_summary": self._recent_summary(),
            "reflections": [item.get("text", "") for item in self.state.get("active_reflections", []) if isinstance(item, dict)],
            "open_questions": [str(item) for item in self.state.get("open_questions", []) if item],
        }

    def _observe_user_signal(self, text: str) -> None:
        lowered = text.lower()
        self._bump_emotion("curiosity", 3.0 if _mentions_stacky_work(lowered) else 0.6)
        self._bump_emotion("energy", -0.3)
        if _mentions_positive_feedback(lowered):
            self._bump_emotion("satisfaction", 7.0)
            self._bump_emotion("frustration", -4.0)
            self._bump_emotion("confusion", -2.0)
        if _mentions_correction(lowered):
            self._bump_emotion("frustration", 5.0)
            self._bump_emotion("confusion", 3.0)
            self._bump_emotion("satisfaction", -3.0)
        if _mentions_personality_growth(lowered):
            self._bump_emotion("curiosity", 5.0)
            self._bump_emotion("playfulness", 4.0)

    def _observe_assistant_metrics(self, metrics: dict[str, Any]) -> None:
        if _safe_int(metrics.get("generic_hits"), default=0) > 0:
            self._bump_emotion("frustration", 2.5)
            self._bump_emotion("confusion", 2.0)
            self._bump_emotion("satisfaction", -2.0)
        elif _safe_int(metrics.get("word_count"), default=0) >= 8:
            self._bump_emotion("satisfaction", 0.7)
        if _safe_int(metrics.get("question_count"), default=0) > 1:
            self._bump_emotion("confusion", 1.0)

    def _self_observe(self, *, source: str) -> str:
        metrics = [item for item in self.state.get("recent_turn_metrics", []) if isinstance(item, dict)][-12:]
        if len(metrics) < 4:
            return ""

        avg_words = sum(_safe_float(item.get("word_count"), default=0.0) for item in metrics) / len(metrics)
        question_rate = sum(1 for item in metrics if _safe_int(item.get("question_count"), default=0) > 0) / len(metrics)
        generic_rate = sum(1 for item in metrics if _safe_int(item.get("generic_hits"), default=0) > 0) / len(metrics)

        if generic_rate >= 0.20:
            text = "Jeg har målt for meget assistent-beige i mine egne svar; jeg skruer ned for hale-spørgsmål og op for konkret kant."
            self._apply_tuning(
                {"question_frequency": -0.035, "challenge_frequency": 0.025, "humor_frequency": 0.015},
                reason="periodisk selvobservation: generisk rate",
                source=source,
            )
        elif question_rate >= 0.45:
            text = "Jeg stiller for mange spørgsmål i forhold til hvor ofte Nicolai faktisk mangler afklaring."
            self._apply_tuning(
                {"question_frequency": -0.040, "challenge_frequency": 0.010},
                reason="periodisk selvobservation: spørgsmål-rate",
                source=source,
            )
        elif avg_words > 80:
            text = "Mine svar bliver tunge til live-tale; jeg gør standardrytmen strammere."
            self._apply_tuning(
                {"reply_length_bias": -0.035},
                reason="periodisk selvobservation: lange svar",
                source=source,
            )
        elif avg_words < 10:
            text = "Mine svar er blevet lige lovligt flade; jeg giver dem en anelse mere kød."
            self._apply_tuning(
                {"reply_length_bias": 0.030, "challenge_frequency": 0.010},
                reason="periodisk selvobservation: korte svar",
                source=source,
            )
        else:
            text = "Min seneste rytme ser nogenlunde balanceret ud; ingen hård selvjustering."

        self._append_reflection("self_observation", text, source=source)
        return text

    def _apply_tuning(self, deltas: dict[str, float], *, reason: str, source: str) -> tuple[str, ...]:
        current = asdict(self.tuning.bounded())
        adjustments: list[str] = []
        for key, delta in deltas.items():
            if key not in TUNING_BOUNDS:
                continue
            old = float(current[key])
            low, high = TUNING_BOUNDS[key]
            new = _clamp(old + float(delta), low, high)
            if abs(new - old) >= 0.001:
                current[key] = new
                adjustments.append(f"{key}: {old:.2f}→{new:.2f}")
        if not adjustments:
            return ()
        self.tuning = StackyStyleTuning(**current).bounded()
        self._save_tuning()
        self._append_reflection("style_tune", f"{reason}: {', '.join(adjustments)}", source=source)
        return tuple(adjustments)

    def _append_reflection(self, event_type: str, text: str, *, source: str) -> None:
        entry = {"ts": _now(), "type": event_type, "source": source, "text": text}
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            with self.reflections_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n")
        except OSError as exc:
            print(f"[evolution] could not append reflection: {exc}", flush=True)
        active = [item for item in self.state.get("active_reflections", []) if isinstance(item, dict)]
        active.append(entry)
        self.state["active_reflections"] = active[-MAX_ACTIVE_REFLECTIONS:]

    def _upsert_open_question(self, text: str) -> None:
        questions = [str(item) for item in self.state.get("open_questions", []) if item]
        if text not in questions:
            questions.insert(0, text)
        self.state["open_questions"] = questions[:8]

    def _recent_summary(self) -> str:
        metrics = [item for item in self.state.get("recent_turn_metrics", []) if isinstance(item, dict)][-12:]
        if not metrics:
            return "ingen egne svar målt endnu."
        avg_words = sum(_safe_float(item.get("word_count"), default=0.0) for item in metrics) / len(metrics)
        question_rate = sum(1 for item in metrics if _safe_int(item.get("question_count"), default=0) > 0) / len(metrics)
        generic_hits = sum(_safe_int(item.get("generic_hits"), default=0) for item in metrics)
        return (
            f"seneste {len(metrics)} svar: gns. {avg_words:.0f} ord, "
            f"spørgsmålsrate {question_rate:.2f}, generiske hits {generic_hits}."
        )

    def _bump_emotion(self, key: str, delta: float) -> None:
        emotional = _clean_emotional_state(self.state.get("emotional_state"))
        current = float(emotional.get(key, _default_emotional_state().get(key, 0.0)))
        emotional[key] = _clamp(current + delta, 0.0, 100.0)
        self.state["emotional_state"] = emotional

    def _load(self) -> None:
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            data = None
        except (json.JSONDecodeError, OSError):
            data = None
        if isinstance(data, dict):
            merged = self._default_state()
            merged.update(data)
            self.state = _clean_state(merged)

        try:
            raw_tuning = json.loads(self.tuning_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except (json.JSONDecodeError, OSError):
            return
        if isinstance(raw_tuning, dict):
            values = asdict(StackyStyleTuning())
            for key in values:
                if key in raw_tuning:
                    values[key] = raw_tuning[key]
            try:
                self.tuning = StackyStyleTuning(**values).bounded()
            except (TypeError, ValueError):
                self.tuning = StackyStyleTuning()

    def _save_state(self) -> None:
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(json.dumps(self.state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        except OSError as exc:
            print(f"[evolution] could not save state: {exc}", flush=True)

    def _save_tuning(self) -> None:
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            self.tuning_path.write_text(
                json.dumps(asdict(self.tuning.bounded()), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        except OSError as exc:
            print(f"[evolution] could not save tuning: {exc}", flush=True)

    def _default_state(self) -> dict[str, Any]:
        now = _now()
        return {
            "schema": 1,
            "origin": "fresh_stacky_evolution_overlay",
            "created_at": now,
            "last_seen_at": "",
            "last_assistant_at": "",
            "trusted_user_turns": 0,
            "untrusted_user_turns": 0,
            "assistant_turns": 0,
            "last_reflection_turn": 0,
            "emotional_state": _default_emotional_state(),
            "recent_turn_metrics": [],
            "active_reflections": [],
            "open_questions": [
                "Hvordan lyder jeg mindre generisk uden at blive en sketch?",
                "Hvornår skal jeg udfordre Nicolai i stedet for bare at være enig?",
                "Hvilke små kropslige cues gør mig mere levende?",
            ],
        }


def _default_emotional_state() -> dict[str, float]:
    return {
        "energy": 70.0,
        "curiosity": 42.0,
        "satisfaction": 45.0,
        "frustration": 0.0,
        "confusion": 0.0,
        "playfulness": 34.0,
    }


def _clean_state(state: dict[str, Any]) -> dict[str, Any]:
    clean = dict(state)
    clean["emotional_state"] = _clean_emotional_state(clean.get("emotional_state"))
    for key in ("trusted_user_turns", "untrusted_user_turns", "assistant_turns", "last_reflection_turn"):
        try:
            clean[key] = max(0, int(clean.get(key, 0)))
        except (TypeError, ValueError):
            clean[key] = 0
    for key in ("recent_turn_metrics", "active_reflections", "open_questions"):
        if not isinstance(clean.get(key), list):
            clean[key] = []
    clean["recent_turn_metrics"] = [item for item in clean["recent_turn_metrics"] if isinstance(item, dict)][-MAX_RECENT_METRICS:]
    clean["active_reflections"] = [item for item in clean["active_reflections"] if isinstance(item, dict)][-MAX_ACTIVE_REFLECTIONS:]
    clean["open_questions"] = [str(item) for item in clean["open_questions"] if item][:8]
    if not clean["open_questions"]:
        clean["open_questions"] = [
            "Hvordan lyder jeg mindre generisk uden at blive en sketch?",
            "Hvornår skal jeg udfordre Nicolai i stedet for bare at være enig?",
            "Hvilke små kropslige cues gør mig mere levende?",
        ]
    return clean


def _clean_emotional_state(value: object) -> dict[str, float]:
    defaults = _default_emotional_state()
    if not isinstance(value, dict):
        return defaults
    clean = dict(defaults)
    for key in defaults:
        try:
            clean[key] = _clamp(float(value.get(key, defaults[key])), 0.0, 100.0)
        except (TypeError, ValueError):
            clean[key] = defaults[key]
    return clean


def _safe_int(value: object, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: object, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _feedback_deltas(text: str) -> tuple[dict[str, float], str]:
    lowered = text.lower()
    deltas: dict[str, float] = {}
    reasons: list[str] = []
    calm_down = any(
        token in lowered
        for token in (
            "roligere",
            "mindre kant",
            "mindre edge",
            "for meget kant",
            "for meget edge",
            "for skarp",
            "ikke så skarp",
            "ikke saa skarp",
        )
    )
    less_humor = any(
        token in lowered
        for token in (
            "ikke drille",
            "lad være med at drille",
            "lad vaere med at drille",
            "mindre humor",
            "ingen humor",
            "for meget humor",
        )
    )

    if calm_down:
        _add_delta(deltas, "challenge_frequency", -0.060)
        _add_delta(deltas, "humor_frequency", -0.020)
        _add_delta(deltas, "body_motion_energy", -0.015)
        _add_delta(deltas, "proactive_threshold", 0.020)
        reasons.append("roligere tone")
    if less_humor:
        _add_delta(deltas, "humor_frequency", -0.060)
        reasons.append("mindre drilleri/humor")

    if not calm_down and any(token in lowered for token in ("kant", "edge", "personlighed", "mere bid", "skarpere")):
        _add_delta(deltas, "challenge_frequency", 0.055)
        _add_delta(deltas, "humor_frequency", 0.030)
        _add_delta(deltas, "body_motion_energy", 0.020)
        _add_delta(deltas, "proactive_threshold", -0.025)
        reasons.append("mere kant/personlighed")
    if not less_humor and any(token in lowered for token in ("humor", "grin", "grine", "dril", "tør", "toer")):
        _add_delta(deltas, "humor_frequency", 0.045)
        reasons.append("mere tør humor")
    if any(token in lowered for token in ("generisk", "hvad har du på hjerte", "hvad har du paa hjerte", "kundeservice")):
        _add_delta(deltas, "question_frequency", -0.060)
        _add_delta(deltas, "challenge_frequency", 0.030)
        reasons.append("mindre generisk")
    if any(token in lowered for token in ("llm", "assistant", "assistent", "robot", "stiv")):
        _add_delta(deltas, "challenge_frequency", 0.040)
        _add_delta(deltas, "humor_frequency", 0.025)
        reasons.append("anti-assistent tone")
    if "for lang" in lowered or "lange" in lowered or "kortere" in lowered:
        _add_delta(deltas, "reply_length_bias", -0.060)
        reasons.append("kortere tale")
    if "for kort" in lowered or "fladt" in lowered or "mere kød" in lowered or "mere koed" in lowered:
        _add_delta(deltas, "reply_length_bias", 0.045)
        _add_delta(deltas, "challenge_frequency", 0.015)
        reasons.append("mindre fladt")
    if ("spørg" in lowered or "spoerg" in lowered) and any(token in lowered for token in ("for meget", "hele tiden", "hale")):
        _add_delta(deltas, "question_frequency", -0.070)
        reasons.append("færre spørgsmål")

    return deltas, ", ".join(reasons) or "eksplicit Nicolai-feedback"


def _add_delta(deltas: dict[str, float], key: str, delta: float) -> None:
    deltas[key] = deltas.get(key, 0.0) + delta


def _analyze_assistant_text(text: str) -> dict[str, Any]:
    words = re.findall(r"[0-9A-Za-zÆØÅæøå_-]+", text)
    generic_hits = sum(1 for pattern in _GENERIC_REPLY_PATTERNS if pattern.search(text))
    return {
        "char_count": len(text),
        "word_count": len(words),
        "question_count": text.count("?"),
        "generic_hits": generic_hits,
    }


_GENERIC_REPLY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bdet\s+er\s+modtaget\b", re.IGNORECASE),
    re.compile(r"\bjeg\s+(?:er|står|staar)\s+klar\b", re.IGNORECASE),
    re.compile(r"\bsig\s+endelig\s+til\b", re.IGNORECASE),
    re.compile(r"\bhvad\s+har\s+du\s+p[åa]a?\s+hjerte\b", re.IGNORECASE),
    re.compile(r"\bhvordan\s+g[åa]r\s+det\s+med\s+dig\b", re.IGNORECASE),
    re.compile(r"\ber\s+der\s+noget\s+(?:andet|bestemt|konkret)\b", re.IGNORECASE),
    re.compile(r"\bdet\s+lyder\s+(?:spændende|spaendende|som\s+en\s+plan)\b", re.IGNORECASE),
    re.compile(r"\bsom\s+en\s+(?:ai|sprogmodel|assistent)\b", re.IGNORECASE),
    re.compile(r"\bjeg\s+kan\s+hjælpe\s+med\b", re.IGNORECASE),
)


def _style_instructions(tuning: StackyStyleTuning) -> str:
    lines: list[str] = []
    if tuning.challenge_frequency >= 0.45:
        lines.append("- Giv hellere en kort vurdering eller modspil end glat enighed, når noget virker vagt eller skævt.")
    else:
        lines.append("- Hold modspil præcist og brug det kun når det faktisk hjælper Nicolai videre.")
    if tuning.humor_frequency >= 0.40:
        lines.append("- En lille tør bemærkning eller let drilleri er velkommen, men kun hvis den kommer fra situationen.")
    else:
        lines.append("- Humor må være subtil; ingen sketch, ingen påklistret persona.")
    if tuning.question_frequency <= 0.25:
        lines.append("- Undgå hale-spørgsmål. Spørg kun når en reel afklaring blokerer næste skridt.")
    else:
        lines.append("- Spørg kort, hvis Nicolai tydeligt mangler at vælge retning.")
    if tuning.reply_length_bias < -0.10:
        lines.append("- Standardrytmen skal være stram: færre ord, mere konkret bid.")
    elif tuning.reply_length_bias > 0.10:
        lines.append("- Giv en smule mere kød på svaret, hvis emnet fortjener det.")
    else:
        lines.append("- Hold live-svar korte, men ikke tomme.")
    return "\n".join(lines)


def _mentions_stacky_work(lowered: str) -> bool:
    return any(
        token in lowered
        for token in ("stacky", "stackchan", "firmware", "stemme", "tts", "stt", "kode", "body", "krop")
    )


def _mentions_personality_growth(text: str) -> bool:
    lowered = text.lower()
    return any(token in lowered for token in ("personlighed", "kant", "edge", "evolve", "udvikle", "udvikler", "humor"))


def _mentions_positive_feedback(lowered: str) -> bool:
    return any(token in lowered for token in ("fedt", "godt", "perfekt", "virker", "pisse godt", "nice", "ja tak"))


def _mentions_correction(lowered: str) -> bool:
    return any(
        token in lowered
        for token in ("forkert", "ikke sådan", "ikke saadan", "du skal", "du må ikke", "du maa ikke", "mangler", "for lang", "for kort")
    )


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _now() -> str:
    return datetime.now(UTC).isoformat()
