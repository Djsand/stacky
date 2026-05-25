from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


MAX_MEMORY_MAP_ENTRIES = 80

_TOKEN_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9_]+")
_STOPWORDS = {
    "og",
    "i",
    "jeg",
    "du",
    "det",
    "der",
    "en",
    "et",
    "at",
    "til",
    "på",
    "med",
    "for",
    "som",
    "er",
    "har",
    "skal",
    "ikke",
}


@dataclass(frozen=True)
class MemoryMapEntry:
    key: str
    title: str
    text: str
    tags: tuple[str, ...]
    importance: float
    source: str
    created_at: str
    updated_at: str
    hits: int = 1
    score: float = 0.0


class MemoryMapStore:
    """Small writeable index for Stacky's red thread.

    This is intentionally not a raw transcript store. It only keeps short,
    curated facts, capabilities, decisions, and active preferences in Stacky's
    own data directory.
    """

    def __init__(self, path: Path) -> None:
        self.path = path if path.suffix.lower() == ".json" else path / "memory_map.json"
        self.state: dict[str, Any] = {"schema": 1, "entries": []}
        self._load()
        self.ensure_core_memories()

    def ensure_core_memories(self) -> None:
        changed = False
        for entry in _core_entries():
            changed = self._upsert(entry, save=False, bump_hits=False) or changed
        if changed:
            self._save()

    def remember_text(
        self,
        text: str,
        *,
        title: str = "",
        key: str = "",
        tags: tuple[str, ...] = (),
        importance: float = 0.72,
        source: str = "stacky",
    ) -> MemoryMapEntry | None:
        clean = _clean_text(text, max_chars=360)
        if len(clean) < 6:
            return None
        entry = MemoryMapEntry(
            key=key or _stable_key(clean),
            title=title.strip() or _title_from_text(clean),
            text=clean,
            tags=tuple(_clean_tag(tag) for tag in tags if _clean_tag(tag)),
            importance=_clamp(float(importance), 0.0, 1.0),
            source=source,
            created_at=_now(),
            updated_at=_now(),
        )
        self._upsert(entry)
        return entry

    def observe_turn(self, user_text: str, *, source: str = "conversation") -> tuple[MemoryMapEntry, ...]:
        """Extract only high-signal red-thread notes from a trusted turn."""

        clean = _clean_text(user_text, max_chars=420)
        if not clean:
            return ()
        lowered = clean.lower()
        stored: list[MemoryMapEntry] = []

        if _mentions_agent(lowered) and any(token in lowered for token in ("glem", "huske", "funktion", "kan lave")):
            entry = self.remember_text(
                "Stacky skal huske, at Sandcode/Codex-agenten er en af hans egne funktioner. "
                "Når Nicolai taler om agenten i Stacky-sammenhæng, skal Stacky ikke opføre sig som om funktionen er ny.",
                title="Sandcode-agent er en Stacky-funktion",
                key="capability.sandcode_agent",
                tags=("capability", "sandcode", "agent"),
                importance=1.0,
                source=source,
            )
            if entry is not None:
                stored.append(entry)

        if _mentions_agent(lowered) and any(token in lowered for token in ("proaktiv", "rapport", "status", "spørg", "spoerg")):
            entry = self.remember_text(
                "Under lange Sandcode-agentkørsler skal Stacky give korte proaktive status-pings og en tydelig slutmelding, "
                "uden at læse fulde logs højt.",
                title="Agent-status skal være proaktiv",
                key="preference.sandcode_proactive_status",
                tags=("preference", "sandcode", "agent", "status"),
                importance=0.96,
                source=source,
            )
            if entry is not None:
                stored.append(entry)

        if any(token in lowered for token in ("husk at", "vigtigt", "beslutning", "konklusion", "røde tråd", "rode traad")):
            entry = self.remember_text(
                clean,
                title="Nicolais røde tråd",
                tags=tuple(_infer_tags(lowered)),
                importance=0.82,
                source=source,
            )
            if entry is not None:
                stored.append(entry)

        return tuple(stored)

    def context_for_prompt(self, *, user_text: str = "", limit: int = 7) -> str:
        entries = self.recall(user_text, limit=limit)
        lines = [
            "Stackys memory-map (egen writeable røde-tråd, ikke rå historik):",
            "Regel: Brug dette til kontinuitet og evner. Opfind ikke minder, og start aldrig handlinger uden tydelig handlingsintention fra Nicolai.",
        ]
        for entry in entries:
            tag_text = ", ".join(entry.tags) if entry.tags else "note"
            lines.append(f"- [{tag_text}] {entry.title}: {entry.text}")
        if len(lines) == 2:
            lines.append("- Ingen relevante røde-tråd-noter endnu.")
        return "\n".join(lines)

    def recall(self, query: str = "", *, limit: int = 6) -> list[MemoryMapEntry]:
        entries = self.all()
        query_tokens = set(_tokens(query))
        scored: list[MemoryMapEntry] = []
        for entry in entries:
            entry_tokens = set(_tokens(" ".join((entry.title, entry.text, " ".join(entry.tags)))))
            overlap = _lexical_overlap(query_tokens, entry_tokens) if query_tokens else 0.0
            tag_boost = 0.20 if "capability" in entry.tags else 0.0
            score = overlap + (entry.importance * 0.35) + (min(entry.hits, 5) * 0.03) + tag_boost
            scored.append(_replace_score(entry, score))
        scored.sort(key=lambda item: (item.score, item.importance, item.updated_at), reverse=True)
        return scored[: max(1, limit)]

    def recall_reply(self, query: str = "", *, limit: int = 5) -> str:
        entries = self.recall(query, limit=limit)
        if not entries:
            return "Jeg har ikke en rød tråd endnu. Lidt tomt, men pænt ryddet op."
        parts = [f"{index}. {entry.title}: {entry.text}" for index, entry in enumerate(entries, start=1)]
        return "Min røde tråd siger: " + " ".join(parts)

    def all(self) -> list[MemoryMapEntry]:
        entries = []
        for raw in self.state.get("entries", []):
            if isinstance(raw, dict):
                entry = _entry_from_raw(raw)
                if entry is not None:
                    entries.append(entry)
        return entries

    def summary(self) -> dict[str, Any]:
        entries = self.all()
        return {
            "path": str(self.path),
            "count": len(entries),
            "entries": [asdict(entry) for entry in entries[:12]],
        }

    def _upsert(self, entry: MemoryMapEntry, *, save: bool = True, bump_hits: bool = True) -> bool:
        entries = [asdict(existing) for existing in self.all()]
        existing = next((item for item in entries if item.get("key") == entry.key), None)
        if existing is None:
            entries.append(asdict(entry))
            changed = True
        else:
            if (
                not bump_hits
                and existing.get("title") == entry.title
                and existing.get("text") == entry.text
                and tuple(existing.get("tags", [])) == entry.tags
                and float(existing.get("importance", 0.0)) >= entry.importance
            ):
                return False
            existing["title"] = entry.title
            existing["text"] = entry.text
            existing["tags"] = list(entry.tags)
            existing["importance"] = max(float(existing.get("importance", 0.0)), entry.importance)
            existing["source"] = entry.source
            existing["updated_at"] = entry.updated_at
            existing["hits"] = int(existing.get("hits", 1)) + (1 if bump_hits else 0)
            changed = True
        entries.sort(key=lambda item: (float(item.get("importance", 0.0)), str(item.get("updated_at", ""))), reverse=True)
        self.state["entries"] = entries[:MAX_MEMORY_MAP_ENTRIES]
        if save:
            self._save()
        return changed

    def _load(self) -> None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except (json.JSONDecodeError, OSError):
            return
        if isinstance(data, dict):
            self.state = {"schema": 1, "entries": data.get("entries", []) if isinstance(data.get("entries"), list) else []}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _core_entries() -> tuple[MemoryMapEntry, ...]:
    now = _now()
    return (
        MemoryMapEntry(
            key="capability.sandcode_agent",
            title="Sandcode-agent",
            text=(
                "Stacky kan starte Sandcode/Codex-agenten som en rigtig runtime-evne, når Nicolai tydeligt beder "
                "om projekt-, kode- eller filarbejde. Det må gerne ske uden triggerord som 'Sandcode' eller 'agent', "
                "fx 'byg det', 'gør det' eller 'fortsæt', hvis seneste kontekst er konkret projektarbejde."
            ),
            tags=("capability", "sandcode", "agent"),
            importance=1.0,
            source="core",
            created_at=now,
            updated_at=now,
        ),
        MemoryMapEntry(
            key="boundary.self_memory_only",
            title="Skriveadgang er afgrænset",
            text=(
                "Stackys egen skriveadgang er til Stackys memory-map og runtime-state. "
                "Fri filændring, kodeændring og repo-handlinger kræver tydelig handlingsintention og skal gå via "
                "Sandcode-agenten eller en terminal-runtime, ikke fri fantasi i svarteksten."
            ),
            tags=("boundary", "memory", "privacy"),
            importance=0.98,
            source="core",
            created_at=now,
            updated_at=now,
        ),
    )


def _entry_from_raw(raw: dict[str, Any]) -> MemoryMapEntry | None:
    try:
        return MemoryMapEntry(
            key=str(raw.get("key") or ""),
            title=str(raw.get("title") or "Note"),
            text=str(raw.get("text") or "").strip(),
            tags=tuple(str(tag) for tag in raw.get("tags", []) if str(tag).strip()),
            importance=_clamp(float(raw.get("importance", 0.5)), 0.0, 1.0),
            source=str(raw.get("source") or "stacky"),
            created_at=str(raw.get("created_at") or ""),
            updated_at=str(raw.get("updated_at") or ""),
            hits=max(1, int(raw.get("hits", 1))),
        )
    except (TypeError, ValueError):
        return None


def _replace_score(entry: MemoryMapEntry, score: float) -> MemoryMapEntry:
    return MemoryMapEntry(
        key=entry.key,
        title=entry.title,
        text=entry.text,
        tags=entry.tags,
        importance=entry.importance,
        source=entry.source,
        created_at=entry.created_at,
        updated_at=entry.updated_at,
        hits=entry.hits,
        score=score,
    )


def _infer_tags(lowered: str) -> tuple[str, ...]:
    tags: list[str] = ["red-thread"]
    if _mentions_agent(lowered):
        tags.extend(["sandcode", "agent"])
    if "personlighed" in lowered or "assistent" in lowered:
        tags.append("personality")
    if "monitor" in lowered or "sanse" in lowered:
        tags.append("senses")
    return tuple(dict.fromkeys(tags))


def _mentions_agent(lowered: str) -> bool:
    return any(token in lowered for token in ("sandcode", "sand code", "agent", "codex"))


def _tokens(text: str) -> list[str]:
    return [
        token.lower()
        for token in _TOKEN_RE.findall(text)
        if len(token) > 1 and token.lower() not in _STOPWORDS
    ]


def _lexical_overlap(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _clean_text(text: str, *, max_chars: int) -> str:
    clean = re.sub(r"\s+", " ", text).strip(" .")
    if len(clean) <= max_chars:
        return clean
    return clean[: max_chars - 1].rstrip() + "…"


def _title_from_text(text: str) -> str:
    words = text.strip().split()
    title = " ".join(words[:7]).strip(" .,:;")
    return title or "Memory-map note"


def _clean_tag(tag: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9_-]+", "-", tag.strip().lower()).strip("-")
    return clean[:32]


def _stable_key(text: str) -> str:
    digest = hashlib.sha256(" ".join(_tokens(text)).encode("utf-8")).hexdigest()[:16]
    return f"note.{digest}"


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _now() -> str:
    return datetime.now(UTC).isoformat()
