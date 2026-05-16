from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher


@dataclass(frozen=True)
class TranscriptCorrection:
    raw_text: str
    text: str
    changed: bool
    reason: str = ""


_STACKY_WORDS = (
    "stakke",
    "stakki",
    "staki",
    "stackie",
    "staggy",
    "staggi",
    "stagi",
    "stage",
    "stegi",
    "steggi",
    "skabe",
)

_ENTITY_REPLACEMENTS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(rf"\b(?:{'|'.join(_STACKY_WORDS)})\b", re.IGNORECASE), "Stacky"),
    (re.compile(r"\b(?:nikolaj|nikolai|nicolaj|nicola|nikola)\b", re.IGNORECASE), "Nicolai"),
    (re.compile(r"\b(?:hojre|høje|hoje)\b", re.IGNORECASE), "højre"),
    (re.compile(r"\b(?:hore|høre|harer|hører|here)\b", re.IGNORECASE), "høre"),
    (re.compile(r"\b(?:tideligt|tydelig|tydelige|dideligt|tidelig)\b", re.IGNORECASE), "tydeligt"),
    (re.compile(r"\b(?:latinska|latinske|latens|latens i|laneside|latenstider)\b", re.IGNORECASE), "latency"),
    (re.compile(r"\bgået lidt i stol\b", re.IGNORECASE), "gået lidt i stå"),
    (re.compile(r"\bgaaet lidt i stol\b", re.IGNORECASE), "gået lidt i stå"),
)

_EXACT_PHRASE_KEYS: dict[str, str] = {
    "hejstakke": "Hej Stacky",
    "hejstackie": "Hej Stacky",
    "hejstage": "Hej Stacky",
    "hejstegi": "Hej Stacky",
    "hejstaggi": "Hej Stacky",
    "hejstikkei": "Hej Stacky",
    "hejopi": "Hej Stacky",
    "hejdeni": "Hej Stacky",
    "hejd": "Hej Stacky",
    "nejstakke": "Hej Stacky",
    "nejstacky": "Hej Stacky",
    "ej": "Hej.",
    "tog": "Tak.",
    "detkanoptillovligthvadmeddig": "Det går stille og roligt, hvad med dig?",
    "detkanoptilovligthvadmeddig": "Det går stille og roligt, hvad med dig?",
    "somhvader": "Som hvad?",
    "vedduhvadklokkenerd": "Ved du hvad klokken er?",
    "oligopoly": "Skru lidt op for lyden.",
    "oligodtforlyden": "Skru lidt op for lyden.",
    "ogligeopilyn": "Skru lidt op for lyden.",
    "ogligeopibyen": "Skru lidt op for lyden.",
    "hunliggeropilyn": "Skru lidt op for lyden.",
    "deterligegodtforlyden": "Skru lidt op for lyden.",
    "lidttilhojre": "Kig lidt til højre.",
    "lidttilhoje": "Kig lidt til højre.",
    "lidtforher": "Kig lidt til højre.",
    "iertilhojre": "Kig lidt til højre.",
    "iertilhoje": "Kig lidt til højre.",
    "lidttilvenstre": "Kig lidt til venstre.",
    "dgikopad": "Kig op.",
    "gikopad": "Kig op.",
    "harermigtideligt": "Kan du høre mig tydeligt?",
    "heremigtydeligt": "Kan du høre mig tydeligt?",
    "horemigtydeligt": "Kan du høre mig tydeligt?",
    "hvadlaverduligemed": "Hvad laver du lige nu?",
    "jeghedderneeland": "Jeg hedder Nicolai.",
    "heddernicola": "Jeg hedder Nicolai.",
    "nikolajsigeratskabeskallytteraven": "Nicolai siger at Stacky skal lytte ordentligt.",
    "nikolaisigeratskabeskallytteraven": "Nicolai siger at Stacky skal lytte ordentligt.",
}

_CANONICAL_SHORT_PHRASES: tuple[tuple[str, float], ...] = (
    ("Hej Stacky", 0.72),
    ("Hvad laver du lige nu?", 0.78),
    ("Kan du høre mig tydeligt?", 0.70),
    ("Skru lidt op for lyden.", 0.66),
    ("Skru lidt ned for lyden.", 0.66),
    ("Skru op for lyden.", 0.68),
    ("Skru ned for lyden.", 0.68),
    ("Kig lidt til højre.", 0.72),
    ("Kig lidt til venstre.", 0.72),
    ("Kig op.", 0.76),
    ("Kig ned.", 0.76),
    ("Gem den her position som center.", 0.74),
    ("Vent lige.", 0.78),
    ("Stop lige.", 0.78),
)


def correct_danish_transcript(text: str) -> TranscriptCorrection:
    raw = " ".join(text.split()).strip()
    if not raw:
        return TranscriptCorrection(raw_text=text, text="", changed=bool(text), reason="empty")

    key = _text_key(raw)
    exact = _EXACT_PHRASE_KEYS.get(key)
    if exact is not None:
        return TranscriptCorrection(raw_text=raw, text=exact, changed=exact != raw, reason="exact")

    with_entities = _replace_entities(raw)
    phrase = _closest_short_phrase(with_entities)
    if phrase is not None:
        return TranscriptCorrection(
            raw_text=raw,
            text=phrase,
            changed=phrase != raw,
            reason="phrase" if phrase != with_entities else "entities",
        )

    changed = with_entities != raw
    return TranscriptCorrection(
        raw_text=raw,
        text=with_entities,
        changed=changed,
        reason="entities" if changed else "",
    )


def _replace_entities(text: str) -> str:
    result = text
    for pattern, replacement in _ENTITY_REPLACEMENTS:
        result = pattern.sub(replacement, result)
    return result


def _closest_short_phrase(text: str) -> str | None:
    normalized = _word_key(text)
    if not normalized:
        return None
    tokens = normalized.split()
    if len(tokens) > 7 and not _is_control_context(normalized):
        return None

    best_phrase = None
    best_score = 0.0
    best_threshold = 1.0
    token_set = set(tokens)
    for phrase, threshold in _CANONICAL_SHORT_PHRASES:
        phrase_key = _word_key(phrase)
        phrase_tokens = set(phrase_key.split())
        ratio = SequenceMatcher(None, normalized, phrase_key).ratio()
        overlap = len(token_set & phrase_tokens) / max(1, len(phrase_tokens))
        score = max(ratio, (ratio * 0.72) + (overlap * 0.28))
        if score > best_score:
            best_score = score
            best_phrase = phrase
            best_threshold = threshold

    if best_phrase is not None and best_score >= best_threshold:
        return best_phrase
    return None


def _is_control_context(normalized: str) -> bool:
    compact = normalized.replace(" ", "")
    return any(
        token in compact
        for token in (
            "stacky",
            "stak",
            "skru",
            "lyd",
            "volumen",
            "kig",
            "tilhojre",
            "tilvenstre",
            "center",
            "position",
            "vent",
            "stop",
        )
    )


def _text_key(text: str) -> str:
    return _word_key(text).replace(" ", "")


def _word_key(text: str) -> str:
    lowered = text.lower()
    replacements = {
        "æ": "ae",
        "ø": "o",
        "å": "a",
        "ä": "ae",
        "ö": "o",
        "ü": "u",
        "é": "e",
    }
    for source, target in replacements.items():
        lowered = lowered.replace(source, target)
    return " ".join(re.sub(r"[^0-9a-z]+", " ", lowered).split())
