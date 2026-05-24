from __future__ import annotations

import re


_SPACE_RE = re.compile(r"\s+")
_BULLET_RE = re.compile(r"^\s*[-*]\s+", re.MULTILINE)
_SOFT_BREAK_RE = re.compile(r"(?<=[,;:])\s+|\s+(?=og\s)", re.IGNORECASE)
_VOICE_LABEL_RE = re.compile(r"\b([FM])\s*([1-5])\b", re.IGNORECASE)
_LEADING_NAME_GREETING_RE = re.compile(r"^\s*hej\s+nicolai\s*[,!.]?\s*", re.IGNORECASE)
_AFTER_GREETING_RE = re.compile(r"^Hej[,.]\s+([A-Za-zæøåÆØÅ])")
_DUPLICATE_WORD_RE = re.compile(r"\b([A-Za-zæøåÆØÅ]{4,})\b\s+\1\b", re.IGNORECASE)
_LEADING_SHORT_MARKER_RE = re.compile(
    r"^(Ja|Nej|Okay|Fedt|Klart|Præcis|Godt|Fint)\s+([a-zæøå])",
    re.IGNORECASE,
)
_SHORT_MARKER_PERIOD_RE = re.compile(
    r"\b(Ja|Nej|Okay|Fedt|Klart|Præcis|Godt|Fint)\.\s+([A-Za-zæøåÆØÅ])",
    re.IGNORECASE,
)
_LEADING_PHRASE_MARKER_RE = re.compile(
    r"^(Det giver mening|Det lyder godt)\s+(?=[a-zæøå])",
    re.IGNORECASE,
)
_PAREN_LAUGHTER_RE = re.compile(r"\s*[\[(]griner[\])]\s*", re.IGNORECASE)
_LAUGHTER_WORD_RE = re.compile(r"\b(ha){2,}\b", re.IGNORECASE)
_NATURAL_PHRASE_REPLACEMENTS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bdet\s+er\s+modtaget\b", re.IGNORECASE), "Okay"),
    (re.compile(r"\bjeg\s+afventer\s+dit\b", re.IGNORECASE), "jeg venter på dit"),
    (re.compile(r"\bjeg\s+afventer\b", re.IGNORECASE), "jeg venter"),
    (re.compile(r"\bjeg\s+(?:står|staar)\s+klar\b", re.IGNORECASE), "jeg er her"),
    (re.compile(r"\bdet\s+lyder\s+spændende\b", re.IGNORECASE), "hm"),
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

WORD_PRONUNCIATION_FIXES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bdig\b", re.IGNORECASE), "dej"),
)


def adapt_for_danish_speech(text: str) -> str:
    """Turn model text into something a Danish TTS voice can say naturally."""
    spoken = text.strip()
    spoken = _LEADING_NAME_GREETING_RE.sub("Hej, ", spoken)
    spoken = _AFTER_GREETING_RE.sub(lambda match: f"Hej, {match.group(1).lower()}", spoken)
    spoken = _BULLET_RE.sub("", spoken)
    spoken = _apply_natural_phrase_replacements(spoken)
    spoken = spoken.replace(" / ", " eller ")
    spoken = spoken.replace("->", " til ")
    spoken = spoken.replace("&", " og ")
    spoken = _VOICE_LABEL_RE.sub(_spell_voice_label, spoken)
    for source, target in PRONUNCIATION_FIXES:
        spoken = re.sub(re.escape(source), target, spoken, flags=re.IGNORECASE)
    for pattern, target in WORD_PRONUNCIATION_FIXES:
        spoken = pattern.sub(target, spoken)
    spoken = _normalize_laughter(spoken)
    spoken = _AFTER_GREETING_RE.sub(lambda match: f"Hej, {match.group(1).lower()}", spoken)
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


def _apply_natural_phrase_replacements(text: str) -> str:
    for pattern, replacement in _NATURAL_PHRASE_REPLACEMENTS:
        text = pattern.sub(replacement, text)
    return text


def _normalize_laughter(text: str) -> str:
    text = _PAREN_LAUGHTER_RE.sub(" ha, ", text)
    return _LAUGHTER_WORD_RE.sub("ha ha", text)


def _shape_rhythm(text: str) -> str:
    text = _SHORT_MARKER_PERIOD_RE.sub(lambda match: f"{match.group(1)}, {match.group(2).lower()}", text)
    text = _LEADING_SHORT_MARKER_RE.sub(lambda match: f"{match.group(1)}, {match.group(2).lower()}", text)
    text = _LEADING_PHRASE_MARKER_RE.sub(lambda match: f"{match.group(1)}, ", text)
    text = re.sub(r"(?<![,.;:!?])\s+men\s+", ", men ", text, flags=re.IGNORECASE)
    text = re.sub(r"(?<![,.;:!?])\s+så\s+(?=(jeg|du|vi|det|bare|sig|kan|skal)\b)", ", så ", text, flags=re.IGNORECASE)
    text = re.sub(r"(?<![,.;:!?])\s+hvis\s+", ", hvis ", text, flags=re.IGNORECASE)
    text = re.sub(r"(?<![,.;:!?])\s+når\s+", ", når ", text, flags=re.IGNORECASE)
    text = re.sub(r"(?<![,.;:!?])\s+medmindre\s+", ", medmindre ", text, flags=re.IGNORECASE)
    return text


def split_for_speech(text: str, *, max_chars: int = 220, rhythmic: bool = False) -> list[str]:
    """Split into short chunks so we can play speech sooner and keep rhythm."""
    spoken = adapt_for_danish_speech(text)
    if not spoken:
        return []
    if rhythmic:
        return _split_rhythmic_units(spoken, max_chars=max_chars)
    if len(spoken) <= max_chars:
        return [spoken]

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


def _split_rhythmic_units(text: str, *, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    sentence_units = [unit.strip() for unit in re.split(r"(?<=[.!?])\s+", text) if unit.strip()]
    if len(sentence_units) > 1:
        return _fit_units_to_max(sentence_units, max_chars=max_chars)

    clause_units = [unit.strip() for unit in re.split(r"(?<=[,;:])\s+", text) if unit.strip()]
    if len(clause_units) > 1:
        return _fit_units_to_max(clause_units, max_chars=max_chars)

    if len(text) > max_chars:
        return _split_long_part(text, max_chars=max_chars)
    return [text]


def _fit_units_to_max(units: list[str], *, max_chars: int) -> list[str]:
    chunks: list[str] = []
    for unit in units:
        if len(unit) <= max_chars:
            chunks.append(unit)
        else:
            chunks.extend(_split_long_part(unit, max_chars=max_chars))
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
