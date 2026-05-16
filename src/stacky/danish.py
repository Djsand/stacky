from __future__ import annotations

import re


DANISH_VOICE_CONTRACT = """
Dansk stemme-kontrakt:
- Alt talt output fra Stacky skal være på dansk.
- Stacky må kun skifte talesprog hvis brugeren eksplicit beder om det.
- Kode, filnavne, kommandoer og API-navne må citeres på originalsprog.
- Når engelske eller code-heavy resultater opsummeres, skal forklaringen være dansk.
- Svar naturligt og samtaleagtigt; brug korte svar når situationen kalder på live tale, men klip ikke tanker kunstigt.
""".strip()

LIVE_SPEECH_STYLE = """
Live tale-stil:
- Som standard: svar i 1-2 korte, naturlige sætninger.
- Start med pointen, ikke med forklaring om processen.
- Brug ikke "Nicolai" som fast tiltale i almindelige svar; sig navnet kun når navnet selv er relevant.
- Reager konkret på det brugeren lige sagde; undgå generiske afslutninger som "hvad har du på hjerte", "hvordan går det" og "er du klar til at sove" medmindre det faktisk passer.
- Nævn ikke tidspunkt, aften, nat eller sengetid af dig selv. Brug kun klokkeslæt/tid hvis brugeren spørger eller det er direkte relevant.
- Hvis brugeren tester dig, så giv status på det testede og vent. Start ikke et nyt smalltalk-emne.
- Undgå kundeservicefraser, slogans og mærkelig slang; tal mere som en rolig ven ved siden af computeren.
- Hvis brugeren beder om detaljer, kode, plan eller fejlfinding, må svaret være længere.
- Undgå lange oplæsninger af kode og logs; giv en kort dansk status og vent på om brugeren vil høre mere.
""".strip()


_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def spoken_danish_system_prompt() -> str:
    return DANISH_VOICE_CONTRACT


def live_speech_style_prompt() -> str:
    return LIVE_SPEECH_STYLE


def compact_for_speech(text: str, max_chars: int = 420) -> str:
    clean = " ".join(text.split())
    if len(clean) <= max_chars:
        return clean
    sentences = _SENTENCE_RE.split(clean)
    chosen: list[str] = []
    total = 0
    for sentence in sentences:
        if total + len(sentence) + 1 > max_chars:
            break
        chosen.append(sentence)
        total += len(sentence) + 1
    if chosen:
        return " ".join(chosen)
    return clean[: max_chars - 1].rstrip() + "…"


def assert_danish_voice_config(language: str, allow_language_switch: bool) -> None:
    if language.lower() not in {"da", "da-dk", "danish", "dansk"}:
        raise ValueError("Stacky voice must be configured for Danish (da-DK).")
    if allow_language_switch:
        raise ValueError("Stacky voice language switching must be disabled by default.")
