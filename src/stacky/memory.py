from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


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
class Memory:
    id: str
    kind: str
    text: str
    importance: float
    source: str
    tags: tuple[str, ...]
    created_at: str
    updated_at: str
    score: float = 0.0


class HashEmbedder:
    """Small local embedding fallback for fresh memory search.

    This is intentionally simple and dependency-free. It gives Stacky a local
    semantic-ish index now, and can be replaced later by a real embedding model.
    """

    def __init__(self, dimensions: int = 64) -> None:
        self.dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in _tokens(text):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]

    def similarity(self, left: list[float], right: list[float]) -> float:
        return sum(a * b for a, b in zip(left, right, strict=False))


class MemoryStore:
    def __init__(self, path: Path, embedder: HashEmbedder | None = None) -> None:
        self.path = path
        self.embedder = embedder or HashEmbedder()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def remember(
        self,
        text: str,
        *,
        kind: str = "episode",
        importance: float = 0.5,
        source: str = "stacky",
        tags: tuple[str, ...] = (),
    ) -> Memory:
        now = _now()
        memory_id = str(uuid.uuid4())
        embedding = self.embedder.embed(text)
        with self._connect() as db:
            db.execute(
                """
                INSERT INTO memories (
                    id, kind, text, importance, source, tags, embedding, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    memory_id,
                    kind,
                    text.strip(),
                    float(importance),
                    source,
                    json.dumps(list(tags), ensure_ascii=False),
                    json.dumps(embedding),
                    now,
                    now,
                ),
            )
        return Memory(memory_id, kind, text.strip(), float(importance), source, tags, now, now)

    def recall(self, query: str, *, limit: int = 6, include_dialogue: bool = False) -> list[Memory]:
        query_embedding = self.embedder.embed(query)
        query_tokens = set(_tokens(query))
        memories = [
            memory
            for memory in self.all()
            if include_dialogue or "dialogue" not in memory.tags
        ]
        scored: list[Memory] = []
        for memory in memories:
            embedding = self._embedding_for(memory)
            vector_score = self.embedder.similarity(query_embedding, embedding)
            lexical_score = _lexical_overlap(query_tokens, set(_tokens(memory.text)))
            score = (vector_score * 0.7) + (lexical_score * 0.3) + (memory.importance * 0.05)
            if score > 0:
                scored.append(_replace_score(memory, score))
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:limit]

    def pinned(self, *, limit: int = 8) -> list[Memory]:
        """Return high-importance identity/preference facts that should travel with Stacky."""
        with self._connect() as db:
            rows = db.execute(
                """
                SELECT id, kind, text, importance, source, tags, created_at, updated_at
                FROM memories
                WHERE kind IN ('identity_fact', 'preference') OR importance >= 0.95
                ORDER BY importance DESC, updated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
            Memory(
                id=str(row["id"]),
                kind=str(row["kind"]),
                text=str(row["text"]),
                importance=float(row["importance"]),
                source=str(row["source"]),
                tags=tuple(json.loads(str(row["tags"] or "[]"))),
                created_at=str(row["created_at"]),
                updated_at=str(row["updated_at"]),
            )
            for row in rows
        ]

    def correct(self, memory_id: str, new_text: str, *, tags: tuple[str, ...] | None = None) -> Memory:
        now = _now()
        embedding = self.embedder.embed(new_text)
        with self._connect() as db:
            existing = db.execute("SELECT kind, importance, source, tags, created_at FROM memories WHERE id = ?", (memory_id,)).fetchone()
            if not existing:
                raise KeyError(f"Memory not found: {memory_id}")
            tag_json = json.dumps(list(tags), ensure_ascii=False) if tags is not None else existing["tags"]
            db.execute(
                """
                UPDATE memories
                SET text = ?, tags = ?, embedding = ?, updated_at = ?
                WHERE id = ?
                """,
                (new_text.strip(), tag_json, json.dumps(embedding), now, memory_id),
            )
        return Memory(
            id=memory_id,
            kind=str(existing["kind"]),
            text=new_text.strip(),
            importance=float(existing["importance"]),
            source=str(existing["source"]),
            tags=tuple(json.loads(tag_json)),
            created_at=str(existing["created_at"]),
            updated_at=now,
        )

    def forget(self, memory_id: str) -> bool:
        with self._connect() as db:
            cursor = db.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            return cursor.rowcount > 0

    def forget_by_tag(self, tag: str) -> int:
        deleted = 0
        with self._connect() as db:
            rows = db.execute("SELECT id, tags FROM memories").fetchall()
            for row in rows:
                try:
                    tags = tuple(json.loads(str(row["tags"] or "[]")))
                except json.JSONDecodeError:
                    tags = ()
                if tag in tags:
                    cursor = db.execute("DELETE FROM memories WHERE id = ?", (str(row["id"]),))
                    deleted += cursor.rowcount
        return deleted

    def all(self) -> list[Memory]:
        with self._connect() as db:
            rows = db.execute(
                """
                SELECT id, kind, text, importance, source, tags, created_at, updated_at
                FROM memories
                ORDER BY created_at ASC
                """
            ).fetchall()
        return [
            Memory(
                id=str(row["id"]),
                kind=str(row["kind"]),
                text=str(row["text"]),
                importance=float(row["importance"]),
                source=str(row["source"]),
                tags=tuple(json.loads(str(row["tags"] or "[]"))),
                created_at=str(row["created_at"]),
                updated_at=str(row["updated_at"]),
            )
            for row in rows
        ]

    def count(self) -> int:
        with self._connect() as db:
            return int(db.execute("SELECT COUNT(*) FROM memories").fetchone()[0])

    def _embedding_for(self, memory: Memory) -> list[float]:
        with self._connect() as db:
            row = db.execute("SELECT embedding FROM memories WHERE id = ?", (memory.id,)).fetchone()
        if not row:
            return []
        return [float(value) for value in json.loads(str(row["embedding"]))]

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self.path)
        db.row_factory = sqlite3.Row
        try:
            yield db
            db.commit()
        finally:
            db.close()

    def _init_db(self) -> None:
        with self._connect() as db:
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    text TEXT NOT NULL,
                    importance REAL NOT NULL DEFAULT 0.5,
                    source TEXT NOT NULL DEFAULT 'stacky',
                    tags TEXT NOT NULL DEFAULT '[]',
                    embedding TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            db.execute("CREATE INDEX IF NOT EXISTS idx_memories_kind ON memories(kind)")
            db.execute("CREATE INDEX IF NOT EXISTS idx_memories_created_at ON memories(created_at)")


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


def _replace_score(memory: Memory, score: float) -> Memory:
    return Memory(
        id=memory.id,
        kind=memory.kind,
        text=memory.text,
        importance=memory.importance,
        source=memory.source,
        tags=memory.tags,
        created_at=memory.created_at,
        updated_at=memory.updated_at,
        score=score,
    )


def _now() -> str:
    return datetime.now(UTC).isoformat()
