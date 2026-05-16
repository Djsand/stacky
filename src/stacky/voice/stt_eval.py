from __future__ import annotations

import glob
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_DANISH_STT_PHRASES: tuple[str, ...] = (
    "Hej Stacky.",
    "Hvad laver du lige nu?",
    "Kan du høre mig tydeligt?",
    "Skru lidt op for lyden.",
    "Kig lidt til højre.",
    "Gem den her position som center.",
    "Jeg hedder Nicolai.",
    "Det her skal være en naturlig dansk samtale.",
    "Start et kodningsprojekt i Sandcode.",
    "Opsummer kun det vigtigste.",
    "Vent lige mens jeg tænker.",
    "Det var ikke det jeg sagde.",
    "Husk at du ikke skal gemme dårlige transskripter.",
    "Min computer står ved skrivebordet.",
    "Tænd lyset i stuen når Home Assistant virker.",
    "Jeg vil have lav latency og stabil forståelse.",
)


@dataclass(frozen=True)
class STTDatasetItem:
    audio_path: Path
    expected_text: str | None = None
    item_id: str = ""


def load_capture_phrases(
    *,
    phrase_args: Iterable[str] = (),
    phrases_file: Path | None = None,
    limit: int = 0,
) -> list[str]:
    phrases: list[str] = []
    for phrase in phrase_args:
        clean = phrase.strip()
        if clean:
            phrases.append(clean)
    if phrases_file is not None:
        for line in phrases_file.read_text(encoding="utf-8").splitlines():
            clean = line.strip()
            if clean and not clean.startswith("#"):
                phrases.append(clean)
    if not phrases:
        phrases = list(DEFAULT_DANISH_STT_PHRASES)
    if limit > 0:
        phrases = phrases[:limit]
    return phrases


def load_reference_file(path: Path) -> dict[str, str]:
    refs: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if "\t" in line:
            name, text = line.split("\t", 1)
        elif "|" in line:
            name, text = line.split("|", 1)
        else:
            continue
        refs[Path(name.strip()).name.lower()] = text.strip()
    return refs


def load_dataset_manifest(path: Path) -> list[STTDatasetItem]:
    if path.suffix.lower() == ".jsonl":
        return _load_jsonl_manifest(path)
    return _load_text_manifest(path)


def resolve_audio_inputs(patterns: list[str], *, default_pattern: str, limit: int) -> list[Path]:
    requested = patterns or [default_pattern]
    paths: list[Path] = []
    for item in requested:
        path = Path(item)
        if path.is_dir():
            paths.extend(sorted(path.glob("*.wav")))
            continue
        matches = [Path(match) for match in glob.glob(item)]
        if matches:
            paths.extend(matches)
            continue
        if path.exists():
            paths.append(path)

    unique: dict[str, Path] = {}
    for path in paths:
        if path.suffix.lower() == ".wav":
            unique[str(path.resolve()).lower()] = path.resolve()
    ordered = sorted(unique.values(), key=lambda item: item.stat().st_mtime, reverse=True)
    if limit > 0:
        ordered = ordered[:limit]
    return ordered


def apply_references(items: list[STTDatasetItem], refs: dict[str, str]) -> list[STTDatasetItem]:
    if not refs:
        return items
    result: list[STTDatasetItem] = []
    for item in items:
        expected = refs.get(item.audio_path.name.lower(), item.expected_text)
        result.append(STTDatasetItem(item.audio_path, expected, item.item_id))
    return result


def write_dataset_record(
    manifest_path: Path,
    *,
    audio_path: Path,
    expected_text: str,
    item_id: str,
    sample_rate: int,
    channels: int,
    duration_seconds: float,
    rms: int,
    peak: int,
    quality: dict[str, object],
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        audio_value = str(audio_path.resolve().relative_to(manifest_path.parent.resolve()))
    except ValueError:
        audio_value = str(audio_path.resolve())
    record = {
        "id": item_id,
        "audio": audio_value,
        "expected": expected_text,
        "sampleRate": sample_rate,
        "channels": channels,
        "durationSeconds": round(duration_seconds, 3),
        "rms": rms,
        "peak": peak,
        "quality": quality,
    }
    with manifest_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def word_error_rate(reference: str, hypothesis: str) -> float:
    return _token_error_rate(_score_words(reference), _score_words(hypothesis))


def char_error_rate(reference: str, hypothesis: str) -> float:
    return _token_error_rate(list(_score_chars(reference)), list(_score_chars(hypothesis)))


def _load_jsonl_manifest(path: Path) -> list[STTDatasetItem]:
    items: list[STTDatasetItem] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        record = json.loads(line)
        audio_value = str(record.get("audio") or record.get("path") or record.get("wav") or "").strip()
        if not audio_value:
            continue
        audio_path = _resolve_relative(path.parent, audio_value)
        expected = record.get("expected", record.get("text", record.get("reference")))
        items.append(
            STTDatasetItem(
                audio_path=audio_path,
                expected_text=None if expected is None else str(expected).strip(),
                item_id=str(record.get("id") or audio_path.stem),
            )
        )
    return items


def _load_text_manifest(path: Path) -> list[STTDatasetItem]:
    items: list[STTDatasetItem] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if "\t" in line:
            audio_value, expected = line.split("\t", 1)
        elif "|" in line:
            audio_value, expected = line.split("|", 1)
        else:
            audio_value, expected = line, ""
        audio_path = _resolve_relative(path.parent, audio_value.strip())
        items.append(STTDatasetItem(audio_path=audio_path, expected_text=expected.strip(), item_id=audio_path.stem))
    return items


def _resolve_relative(base_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _score_words(text: str) -> list[str]:
    lowered = text.lower()
    normalized = re.sub(r"[^0-9a-zæøå]+", " ", lowered)
    return [word for word in normalized.split() if word]


def _score_chars(text: str) -> str:
    lowered = text.lower()
    return re.sub(r"[^0-9a-zæøå]+", "", lowered)


def _token_error_rate(reference: list[str], hypothesis: list[str]) -> float:
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return _edit_distance(reference, hypothesis) / len(reference)


def _edit_distance(left: list[str], right: list[str]) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_item in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_item in enumerate(right, start=1):
            cost = 0 if left_item == right_item else 1
            current.append(
                min(
                    previous[right_index] + 1,
                    current[right_index - 1] + 1,
                    previous[right_index - 1] + cost,
                )
            )
        previous = current
    return previous[-1]
