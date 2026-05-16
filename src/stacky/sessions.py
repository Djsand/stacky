from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

from .memory import Memory


CHARS_PER_TOKEN = 4


@dataclass(frozen=True)
class StitchMeta:
    total_messages: int = 0
    total_tokens: int = 0
    user_messages: int = 0
    assistant_messages: int = 0
    oldest_message_age_s: float = 0.0
    newest_message_age_s: float = 0.0
    time_gaps_count: int = 0
    recalled_memory_count: int = 0

    @property
    def assistant_ratio(self) -> float:
        total = self.user_messages + self.assistant_messages
        return 0.0 if total == 0 else self.assistant_messages / total

    @property
    def context_freshness(self) -> float:
        return 1.0 / (1.0 + math.exp(0.01 * (self.newest_message_age_s - 300)))


class InfiniteSessionStore:
    def __init__(
        self,
        data_dir: Path,
        *,
        thread_name: str = "stacky-infinite-thread",
        roll_tokens: int = 60_000,
        max_stitched_tokens: int = 120_000,
    ) -> None:
        self.session_dir = data_dir / "sessions"
        self.thread_name = thread_name
        self.roll_tokens = roll_tokens
        self.max_stitched_tokens = max_stitched_tokens
        self.session_dir.mkdir(parents=True, exist_ok=True)

    @property
    def active_path(self) -> Path:
        return self.session_dir / f"{self.thread_name}.jsonl"

    def append_message(self, role: str, content: str, *, meta: dict[str, object] | None = None) -> None:
        if role not in {"user", "assistant", "system"}:
            raise ValueError(f"Unsupported session role: {role}")
        clean_content = _clean_assistant_output(content) if role == "assistant" else content.strip()
        if not clean_content:
            return
        self._maybe_roll()
        record_meta = dict(meta or {})
        record_meta.setdefault("timestamp", _now_iso())
        record = {
            "type": "message",
            "message": {"role": role, "content": clean_content},
            "meta": record_meta,
        }
        with self.active_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

    def stitch_context(
        self,
        *,
        max_tokens: int | None = None,
        recalled_memories: Iterable[Memory] = (),
        is_restart: bool = False,
    ) -> tuple[list[dict[str, str]], StitchMeta]:
        token_limit = max_tokens or self.max_stitched_tokens
        recalled = tuple(recalled_memories)
        files = [*self._rolled_paths(), self.active_path]
        selected: list[dict[str, object]] = []
        total_tokens = 0

        for path in reversed(files):
            if not path.exists():
                continue
            messages = read_jsonl_messages(path)
            file_tokens = estimate_tokens(json.dumps(messages, ensure_ascii=False))
            if total_tokens + file_tokens > token_limit:
                for message in reversed(messages):
                    message_tokens = estimate_tokens(json.dumps(message, ensure_ascii=False))
                    if total_tokens + message_tokens > token_limit:
                        break
                    selected.append(message)
                    total_tokens += message_tokens
                break
            selected.extend(reversed(messages))
            total_tokens += file_tokens

        selected.reverse()
        meta = _stitch_meta(selected, total_tokens=total_tokens, recalled_memory_count=len(recalled))
        messages = _condense_repetitive(selected)
        messages = inject_chrono_markers(messages, is_restart=is_restart)
        if recalled:
            messages = format_recalled_memories(recalled) + messages
        return messages, meta

    def _rolled_paths(self) -> list[Path]:
        pattern = re.compile(rf"^{re.escape(self.thread_name)}\.(\d+)\.jsonl$")
        paths: list[tuple[int, Path]] = []
        for path in self.session_dir.glob(f"{self.thread_name}.*.jsonl"):
            match = pattern.match(path.name)
            if match:
                paths.append((int(match.group(1)), path))
        return [path for _, path in sorted(paths)]

    def _maybe_roll(self) -> None:
        if self.roll_tokens <= 0 or not self.active_path.exists():
            return
        if estimate_file_tokens(self.active_path) < self.roll_tokens:
            return
        index = self._next_roll_index()
        self.active_path.rename(self.session_dir / f"{self.thread_name}.{index:03d}.jsonl")

    def _next_roll_index(self) -> int:
        max_index = 0
        pattern = re.compile(rf"^{re.escape(self.thread_name)}\.(\d+)\.jsonl$")
        for path in self.session_dir.glob(f"{self.thread_name}.*.jsonl"):
            match = pattern.match(path.name)
            if match:
                max_index = max(max_index, int(match.group(1)))
        return max_index + 1


def read_jsonl_messages(path: Path) -> list[dict[str, object]]:
    messages: list[dict[str, object]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("type") != "message" or not isinstance(record.get("message"), dict):
                    continue
                message = dict(record["message"])
                if message.get("role") not in {"user", "assistant", "system"}:
                    continue
                meta = record.get("meta", {})
                if isinstance(meta, dict):
                    if meta.get("timestamp"):
                        message["_ts"] = str(meta["timestamp"])
                    if meta.get("source"):
                        message["_source"] = str(meta["source"])
                messages.append(message)
    except FileNotFoundError:
        return []
    return messages


def format_recalled_memories(memories: Iterable[Memory]) -> list[dict[str, str]]:
    lines = []
    for memory in memories:
        safe_text = _escape_for_prompt(memory.text)
        lines.append(f"[{memory.kind}] relevance={memory.score:.2f} {safe_text}")
    if not lines:
        return []
    return [
        {
            "role": "system",
            "content": (
                "Automatisk Stacky-hukommelse. Dette er ikke en ny besked fra Nicolai.\n"
                + "\n".join(lines)
            ),
        }
    ]


def inject_chrono_markers(
    messages: list[dict[str, object]],
    *,
    gap_threshold_s: int = 300,
    is_restart: bool = False,
) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    previous_ts: datetime | None = None
    for message in messages:
        current_ts = _parse_ts(str(message.get("_ts", "")))
        if previous_ts and current_ts:
            gap_seconds = (current_ts - previous_ts).total_seconds()
            if gap_seconds >= gap_threshold_s:
                result.append(
                    {
                        "role": "system",
                        "content": f"Tidsmarkør: {_format_gap(gap_seconds)}.",
                    }
                )
        result.append(
            {
                "role": str(message.get("role", "user")),
                "content": str(message.get("content", "")),
            }
        )
        if current_ts:
            previous_ts = current_ts
    if is_restart:
        result.append({"role": "system", "content": "Stacky-processen blev genstartet. Opfind ikke aktivitet i pausen."})
    return result


def estimate_file_tokens(path: Path) -> int:
    try:
        return max(1, int(path.stat().st_size // CHARS_PER_TOKEN))
    except FileNotFoundError:
        return 0


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


def _stitch_meta(
    messages: list[dict[str, object]],
    *,
    total_tokens: int,
    recalled_memory_count: int,
) -> StitchMeta:
    now = datetime.now(UTC)
    timestamps = [_parse_ts(str(message.get("_ts", ""))) for message in messages]
    timestamps = [timestamp for timestamp in timestamps if timestamp is not None]
    gaps = 0
    for previous, current in zip(timestamps, timestamps[1:], strict=False):
        if (current - previous).total_seconds() >= 300:
            gaps += 1
    return StitchMeta(
        total_messages=len(messages),
        total_tokens=total_tokens,
        user_messages=sum(1 for message in messages if message.get("role") == "user"),
        assistant_messages=sum(1 for message in messages if message.get("role") == "assistant"),
        oldest_message_age_s=max(0.0, (now - min(timestamps)).total_seconds()) if timestamps else 0.0,
        newest_message_age_s=max(0.0, (now - max(timestamps)).total_seconds()) if timestamps else 0.0,
        time_gaps_count=gaps,
        recalled_memory_count=recalled_memory_count,
    )


def _condense_repetitive(messages: list[dict[str, object]]) -> list[dict[str, object]]:
    assistant_indices = [index for index, message in enumerate(messages) if message.get("role") == "assistant"]
    condensed: set[int] = set()
    for offset in range(len(assistant_indices) - 1, 0, -1):
        newer_index = assistant_indices[offset]
        newer_text = str(messages[newer_index].get("content", ""))
        for older_index in assistant_indices[max(0, offset - 5) : offset]:
            older_text = str(messages[older_index].get("content", ""))
            if _text_similarity(newer_text, older_text) > 0.55:
                condensed.add(older_index)
    if not condensed:
        return messages
    result: list[dict[str, object]] = []
    for index, message in enumerate(messages):
        if index in condensed:
            result.append({"role": message.get("role", "assistant"), "content": "[Tidligere lignende svar kondenseret]"})
        else:
            result.append(message)
    return result


def _clean_assistant_output(text: str) -> str:
    clean = re.sub(r"(?s)<think>.*?</think>", "", text)
    clean = re.sub(r"(?s)^thought\s*\n.*?</think>", "", clean)
    return clean.strip()


def _text_similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    left = left[:500]
    right = right[:500]
    left_trigrams = {left[index : index + 3] for index in range(max(0, len(left) - 2))}
    right_trigrams = {right[index : index + 3] for index in range(max(0, len(right) - 2))}
    if not left_trigrams or not right_trigrams:
        return 0.0
    return len(left_trigrams & right_trigrams) / len(left_trigrams | right_trigrams)


def _escape_for_prompt(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def _parse_ts(value: str) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _format_gap(seconds: float) -> str:
    if seconds < 3600:
        return f"{int(seconds // 60)} minutter senere"
    if seconds < 86400:
        hours = int(seconds // 3600)
        return "1 time senere" if hours <= 1 else f"{hours} timer senere"
    days = int(seconds // 86400)
    return "1 dag senere" if days <= 1 else f"{days} dage senere"


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()
