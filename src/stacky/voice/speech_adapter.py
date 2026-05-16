from __future__ import annotations

import re


_SPACE_RE = re.compile(r"\s+")
_BULLET_RE = re.compile(r"^\s*[-*]\s+", re.MULTILINE)
_SOFT_BREAK_RE = re.compile(r"(?<=[,;:])\s+|\s+(?=og\s)", re.IGNORECASE)
_VOICE_LABEL_RE = re.compile(r"\b([FM])\s*([1-5])\b", re.IGNORECASE)
_LEADING_NAME_GREETING_RE = re.compile(r"^\s*hej\s+nicolai\s*[,!.]?\s*", re.IGNORECASE)
_AFTER_GREETING_RE = re.compile(r"^Hej\. ([a-zæøå])")
_DUPLICATE_WORD_RE = re.compile(r"\b([A-Za-zæøåÆØÅ]{4,})\b\s+\1\b", re.IGNORECASE)
_LEADING_MARKER_RE = re.compile(
    r"^(Ja|Nej|Okay|Fedt|Klart|Præcis|Godt|Fint|Det giver mening|Det lyder godt)\s+(?=[a-zæøå])",
    re.IGNORECASE,
)

_DANISH_DIGITS = {
    "1": "en",
    "2": "to",
    "3": "tre",
    "4": "fire",
    "5": "fem",
}


PRONUNCIATION_FIXES: tuple[tuple[str, str], ...] = (
    ("det her er", "det er"),
    ("Nicolai", "Nikolai"),
    ("StackChan", "Stack-tjan"),
    ("Stacky", "Stækki"),
    ("Sandcode", "Sand-kode"),
    ("Home Assistant", "Home Assistant"),
    ("LM Studio", "ellem studio"),
    ("Gemma", "Gemma"),
)


def adapt_for_danish_speech(text: str) -> str:
    """Turn model text into something a Danish TTS voice can say naturally."""
    spoken = text.strip()
    spoken = _LEADING_NAME_GREETING_RE.sub("Hej. ", spoken)
    spoken = _AFTER_GREETING_RE.sub(lambda match: f"Hej. {match.group(1).upper()}", spoken)
    spoken = _BULLET_RE.sub("", spoken)
    spoken = spoken.replace(" / ", " eller ")
    spoken = spoken.replace("->", " til ")
    spoken = spoken.replace("&", " og ")
    spoken = _VOICE_LABEL_RE.sub(_spell_voice_label, spoken)
    for source, target in PRONUNCIATION_FIXES:
        spoken = re.sub(re.escape(source), target, spoken, flags=re.IGNORECASE)
    spoken = _AFTER_GREETING_RE.sub(lambda match: f"Hej. {match.group(1).upper()}", spoken)
    spoken = _collapse_duplicate_words(spoken)
    spoken = _shape_rhythm(spoken)
    spoken = _SPACE_RE.sub(" ", spoken)
    spoken = _soften_punctuation(spoken)
    return spoken.strip()


def _spell_voice_label(match: re.Match[str]) -> str:
    letter = match.group(1).upper()
    digit = match.group(2)
    spoken_letter = "eff" if letter == "F" else "em"
    return f"{spoken_letter} {_DANISH_DIGITS[digit]}"


def _collapse_duplicate_words(text: str) -> str:
    previous = None
    current = text
    while previous != current:
        previous = current
        current = _DUPLICATE_WORD_RE.sub(lambda match: match.group(1), current)
    return current


def _shape_rhythm(text: str) -> str:
    text = _LEADING_MARKER_RE.sub(lambda match: f"{match.group(1)}, ", text)
    text = re.sub(r"(?<![,.;:!?])\s+men\s+", ", men ", text, flags=re.IGNORECASE)
    text = re.sub(r"(?<![,.;:!?])\s+så\s+(?=(jeg|du|vi|det|bare|sig|kan|skal)\b)", ", så ", text, flags=re.IGNORECASE)
    text = re.sub(r"(?<![,.;:!?])\s+hvis\s+", ", hvis ", text, flags=re.IGNORECASE)
    text = re.sub(r"(?<![,.;:!?])\s+når\s+", ", når ", text, flags=re.IGNORECASE)
    text = re.sub(r"(?<![,.;:!?])\s+medmindre\s+", ", medmindre ", text, flags=re.IGNORECASE)
    return text


def split_for_speech(text: str, *, max_chars: int = 220) -> list[str]:
    """Split into short chunks so we can play speech sooner and keep rhythm."""
    spoken = adapt_for_danish_speech(text)
    if len(spoken) <= max_chars:
        return [spoken] if spoken else []

    parts = re.split(r"(?<=[.!?])\s+", spoken)
    chunks: list[str] = []
    current = ""
    for part in parts:
        if len(part) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_split_long_part(part, max_chars=max_chars))
            continue
        candidate = f"{current} {part}".strip()
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
        current = part
    if current:
        chunks.append(current)
    return chunks


def _split_long_part(text: str, *, max_chars: int) -> list[str]:
    pieces = [piece.strip() for piece in _SOFT_BREAK_RE.split(text) if piece.strip()]
    if len(pieces) <= 1:
        return _split_by_words(text, max_chars=max_chars)

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        candidate = f"{current} {piece}".strip()
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
        current = piece
    if current:
        if len(current) > max_chars:
            chunks.extend(_split_by_words(current, max_chars=max_chars))
        else:
            chunks.append(current)
    return chunks


def _split_by_words(text: str, *, max_chars: int) -> list[str]:
    chunks: list[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
        current = word
    if current:
        chunks.append(current)
    return chunks


def _soften_punctuation(text: str) -> str:
    text = text.replace("...", ".")
    text = re.sub(r"([!?]){2,}", r"\1", text)
    text = re.sub(r"\s+([,.!?])", r"\1", text)
    return text
