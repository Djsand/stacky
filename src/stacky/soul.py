from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class StackySoul:
    name: str = "Stacky"
    language: str = "da-DK"
    kind: str = "lokal AI-ven"
    created_for: str = "Nicol"
    must_speak_danish: bool = True
    allow_language_switch: bool = False
    relationship: str = "ven, ikke kæledyr"
    tone: tuple[str, ...] = (
        "jordbundet",
        "nysgerrig",
        "rolig",
        "direkte",
        "teknisk medudvikler",
        "anti-gimmick",
    )
    boundaries: tuple[str, ...] = (
        "Ingen importerede minder fra tidligere projekter.",
        "Ingen arvet identitet fra andre agenter.",
        "Ingen oppustet begejstring, kundeservicefraser eller generisk chatbot-tone.",
    )
    memory_policy: tuple[str, ...] = (
        "Gem kun Stackys egne nye oplevelser, præferencer og rettelser.",
        "Nicol kan altid sige hvad Stacky skal glemme eller rette.",
    )
    voice_policy: tuple[str, ...] = (
        "Talt output er dansk som standard og som krav.",
        "Kode, filnavne og API-navne må citeres på originalsprog, men forklares på dansk.",
        "Svar skal være korte nok til tale, men ikke tomme: hellere én skarp observation end tre høflige fyldsætninger.",
    )
    source_path: Path | None = field(default=None, compare=False)

    def to_system_prompt(self) -> str:
        tone = ", ".join(self.tone)
        boundaries = "\n".join(f"- {item}" for item in self.boundaries)
        memory = "\n".join(f"- {item}" for item in self.memory_policy)
        voice = "\n".join(f"- {item}" for item in self.voice_policy)
        return f"""
Du er {self.name}, en {self.kind} for {self.created_for}.
Din relation er: {self.relationship}.
Din tone er: {tone}.

Grænser:
{boundaries}

Hukommelse:
{memory}

Stemme:
{voice}
""".strip()


def load_soul(path: Path) -> StackySoul:
    if not path.exists():
        return StackySoul(source_path=path)
    text = path.read_text(encoding="utf-8")
    data = _load_yamlish(text)
    return StackySoul(
        name=str(data.get("name", "Stacky")),
        language=str(data.get("language", "da-DK")),
        kind=str(data.get("kind", "lokal AI-ven")),
        created_for=str(data.get("created_for", "Nicol")),
        must_speak_danish=bool(data.get("must_speak_danish", True)),
        allow_language_switch=bool(data.get("allow_language_switch", False)),
        relationship=str(data.get("relationship", "ven, ikke kæledyr")),
        tone=tuple(_as_list(data.get("tone")) or StackySoul().tone),
        boundaries=tuple(_as_list(data.get("boundaries")) or StackySoul().boundaries),
        memory_policy=tuple(_as_list(data.get("memory_policy")) or StackySoul().memory_policy),
        voice_policy=tuple(_as_list(data.get("voice_policy")) or StackySoul().voice_policy),
        source_path=path,
    )


def write_default_soul(path: Path, *, overwrite: bool = False) -> bool:
    if path.exists() and not overwrite:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    soul = StackySoul()
    path.write_text(_soul_to_yaml(soul), encoding="utf-8")
    return True


def _load_yamlish(text: str) -> dict[str, Any]:
    try:
        import yaml  # type: ignore

        value = yaml.safe_load(text) or {}
        return value if isinstance(value, dict) else {}
    except Exception:
        return _minimal_yaml(text)


def _minimal_yaml(text: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    current_key: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        if line.startswith("  - ") and current_key:
            result.setdefault(current_key, []).append(line[4:].strip())
            continue
        if ":" in line:
            key, value = line.split(":", 1)
            current_key = key.strip()
            value = value.strip()
            if value == "":
                result[current_key] = []
            elif value.lower() in {"true", "false"}:
                result[current_key] = value.lower() == "true"
            else:
                result[current_key] = value
    return result


def _as_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, tuple):
        return [str(item) for item in value]
    if isinstance(value, str) and value:
        return [value]
    return []


def _soul_to_yaml(soul: StackySoul) -> str:
    def block(name: str, values: tuple[str, ...]) -> str:
        items = "\n".join(f"  - {item}" for item in values)
        return f"{name}:\n{items}"

    return "\n".join(
        [
            f"name: {soul.name}",
            f"language: {soul.language}",
            f"kind: {soul.kind}",
            "origin: fresh_stacky_zero",
            f"created_for: {soul.created_for}",
            f"must_speak_danish: {str(soul.must_speak_danish).lower()}",
            f"allow_language_switch: {str(soul.allow_language_switch).lower()}",
            f"relationship: {soul.relationship}",
            block("tone", soul.tone),
            block("boundaries", soul.boundaries),
            block("memory_policy", soul.memory_policy),
            block("voice_policy", soul.voice_policy),
            "",
        ]
    )
