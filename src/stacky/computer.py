from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


DEFAULT_CONTEXT_PATHS = ("src", "tests", "configs", "docs")


@dataclass(frozen=True)
class CommandResult:
    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


@dataclass(frozen=True)
class LocalComputerAction:
    kind: str
    query: str = ""
    target_path: Path | None = None
    content: str = ""


@dataclass(frozen=True)
class LocalComputerActionResult:
    ok: bool
    spoken: str
    detail: str = ""


class LocalComputerContext:
    """Read-only local computer context for Stacky's workspace.

    This is intentionally not a free shell. It only runs a fixed set of
    read-only commands and returns compact context that the brain can use.
    Sandcode can be layered on top later for real coding sessions.
    """

    def __init__(
        self,
        root: Path,
        *,
        max_chars: int = 4000,
        timeout_seconds: float = 4.0,
        context_paths: tuple[str, ...] = DEFAULT_CONTEXT_PATHS,
    ) -> None:
        self.root = root.resolve()
        self.max_chars = max(800, max_chars)
        self.timeout_seconds = max(1.0, timeout_seconds)
        self.context_paths = context_paths

    def context_for(self, text: str) -> str:
        if not wants_computer_context(text):
            return ""
        parts = [
            "Computer-kontekst (lokal read-only):",
            f"- workspace: {self.root}",
        ]

        branch = self._run(("git", "rev-parse", "--abbrev-ref", "HEAD"))
        if branch.returncode == 0 and branch.stdout.strip():
            parts.append(f"- git branch: {branch.stdout.strip()}")

        status = self._run(("git", "status", "--short"))
        if status.returncode == 0:
            clean_status = _clip(status.stdout.strip() or "clean", 1200)
            parts.append("- git status --short:\n" + _indent(clean_status))

        query = extract_computer_search_query(text)
        if query:
            search = self._search(query)
            parts.append(f"- lokal kode/fil-soegning efter {query!r}:\n" + _indent(search))
        else:
            files = self._list_project_files()
            parts.append("- relevante projektfiler:\n" + _indent(files))

        parts.append(
            "Regel: Dette er read-only kontekst. Stacky har ikke fri terminal-adgang her "
            "og maa ikke paastaa at have koert andre kommandoer end dem, der staar i konteksten."
        )
        return _clip("\n".join(parts), self.max_chars)

    def _run(self, command: tuple[str, ...]) -> CommandResult:
        try:
            completed = subprocess.run(
                command,
                cwd=self.root,
                text=True,
                capture_output=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return CommandResult(command, 1, "", str(exc))
        return CommandResult(command, completed.returncode, completed.stdout, completed.stderr)

    def _search(self, query: str) -> str:
        rg = shutil.which("rg")
        roots = [path for path in self.context_paths if (self.root / path).exists()]
        if not roots:
            roots = ["."]
        if rg:
            result = self._run((rg, "-n", "-S", "-m", "25", "--", query, *roots))
            if result.returncode == 0 and result.stdout.strip():
                return _clip(result.stdout.strip().replace("\\", "/"), 2600)
            if result.returncode == 1:
                return "Ingen lokale matches."
            output = result.stderr.strip() or result.stdout.strip() or "rg fejlede uden output."
            return _clip(output, 1600)
        return "ripgrep (rg) er ikke paa PATH, saa lokal soegning er ikke tilgaengelig."

    def _list_project_files(self) -> str:
        rg = shutil.which("rg")
        roots = [path for path in self.context_paths if (self.root / path).exists()]
        if rg and roots:
            result = self._run((rg, "--files", *roots))
            if result.returncode == 0 and result.stdout.strip():
                lines = [line.replace("\\", "/") for line in result.stdout.splitlines()]
                suffix = f"\n... ({len(lines)} filer i alt)" if len(lines) > 80 else ""
                return "\n".join(lines[:80]) + suffix
        files: list[str] = []
        for root in roots:
            for path in (self.root / root).rglob("*"):
                if path.is_file():
                    files.append(path.relative_to(self.root).as_posix())
                    if len(files) >= 80:
                        break
            if len(files) >= 80:
                break
        return "\n".join(files) if files else "Ingen projektfiler fundet i standard-kontekststier."


class LocalComputerActions:
    """Small explicit local actions for voice control.

    This is deliberately narrower than a shell: it only runs predictable local
    operations that are easy to summarize out loud.
    """

    def __init__(self, root: Path, *, timeout_seconds: float = 4.0, desktop: Path | None = None) -> None:
        self.root = root.resolve()
        self.timeout_seconds = max(1.0, timeout_seconds)
        self.desktop = (desktop or _default_desktop()).resolve()

    def run(self, action: LocalComputerAction) -> LocalComputerActionResult:
        if action.kind == "git_status":
            return self._git_status()
        if action.kind == "list_workspace":
            return self._list_workspace()
        if action.kind == "search":
            return self._search(action.query)
        if action.kind == "create_text_file" and action.target_path is not None:
            return self._create_text_file(action.target_path, action.content)
        return LocalComputerActionResult(False, "Den computerhandling kender jeg ikke endnu.", action.kind)

    def _git_status(self) -> LocalComputerActionResult:
        result = _run_command(("git", "status", "--short"), cwd=self.root, timeout_seconds=self.timeout_seconds)
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip()
            return LocalComputerActionResult(False, "Jeg kunne ikke læse git status.", detail)
        lines = [line for line in result.stdout.splitlines() if line.strip()]
        if not lines:
            return LocalComputerActionResult(True, "Git status er ren.", "git status --short: clean")
        preview = ", ".join(lines[:3])
        suffix = "" if len(lines) <= 3 else f" plus {len(lines) - 3} mere"
        return LocalComputerActionResult(True, f"Git status viser {len(lines)} ændringer: {preview}{suffix}.", result.stdout)

    def _list_workspace(self) -> LocalComputerActionResult:
        try:
            entries = sorted(self.root.iterdir(), key=lambda path: (path.is_file(), path.name.lower()))
        except OSError as exc:
            return LocalComputerActionResult(False, "Jeg kunne ikke læse workspace-mappen.", str(exc))
        names = [entry.name + ("/" if entry.is_dir() else "") for entry in entries[:12]]
        if not names:
            return LocalComputerActionResult(True, "Workspace-mappen er tom.", str(self.root))
        return LocalComputerActionResult(
            True,
            "Jeg kan se: " + ", ".join(names) + ("." if len(entries) <= 12 else f", og {len(entries) - 12} mere."),
            "\n".join(names),
        )

    def _search(self, query: str) -> LocalComputerActionResult:
        query = _cleanup_query(query)
        if not query:
            return LocalComputerActionResult(False, "Jeg mangler et søgeord til rg.", "")
        rg = shutil.which("rg")
        if not rg:
            return LocalComputerActionResult(False, "rg er ikke på PATH, så jeg kan ikke søge lokalt endnu.", "")
        roots = [path for path in DEFAULT_CONTEXT_PATHS if (self.root / path).exists()] or ["."]
        result = _run_command((rg, "-n", "-S", "-m", "8", "--", query, *roots), cwd=self.root, timeout_seconds=self.timeout_seconds)
        if result.returncode == 1:
            return LocalComputerActionResult(True, f"Jeg fandt ingen lokale matches for {query}.", "")
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip()
            return LocalComputerActionResult(False, "Den lokale søgning fejlede.", detail)
        lines = [line.replace("\\", "/") for line in result.stdout.splitlines() if line.strip()]
        first = lines[0] if lines else ""
        more = "" if len(lines) <= 1 else f" og {len(lines) - 1} mere"
        return LocalComputerActionResult(True, f"Jeg fandt {len(lines)} match{more}. Første er {first}.", "\n".join(lines))

    def _create_text_file(self, target_path: Path, content: str) -> LocalComputerActionResult:
        try:
            target = self._safe_unique_path(target_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        except OSError as exc:
            return LocalComputerActionResult(False, "Jeg kunne ikke skrive tekstfilen.", str(exc))
        location = "skrivebordet" if _is_relative_to(target, self.desktop) else "workspace"
        return LocalComputerActionResult(True, f"Jeg skrev tekstfilen {target.name} på {location}.", str(target))

    def _safe_unique_path(self, target_path: Path) -> Path:
        target = target_path.resolve()
        allowed_roots = (self.root, self.desktop)
        if not any(_is_relative_to(target, allowed_root) for allowed_root in allowed_roots):
            target = self.root / target.name
        suffix = target.suffix or ".txt"
        stem = target.stem or "stacky-note"
        candidate = target.with_suffix(suffix)
        index = 2
        while candidate.exists():
            candidate = target.with_name(f"{stem}-{index}{suffix}")
            index += 1
        return candidate


def parse_local_computer_action(text: str, *, root: Path, desktop: Path | None = None) -> LocalComputerAction | None:
    normalized = _normalize(text)
    if not normalized:
        return None
    if _looks_like_web_search_request(normalized):
        return None
    if _wants_create_text_file(normalized):
        target_root = (desktop or _default_desktop()) if "skrivebord" in normalized else root
        filename = _text_file_name_from(normalized)
        return LocalComputerAction(
            kind="create_text_file",
            target_path=target_root / filename,
            content=_text_file_content_from(normalized),
        )
    if "git status" in normalized or ("git" in normalized and "status" in normalized):
        return LocalComputerAction(kind="git_status")
    if _wants_list_workspace_action(normalized):
        return LocalComputerAction(kind="list_workspace")
    if re.search(r"\b(rg|grep|ripgrep|soeg|sog|find)\b", normalized):
        query = extract_computer_search_query(text)
        if query:
            return LocalComputerAction(kind="search", query=query)
    return None


_COMPUTER_TRIGGERS = (
    "terminal",
    "kommando",
    "shell",
    "bash",
    "powershell",
    "dir",
    "ls",
    "grep",
    "rg",
    "ripgrep",
    "repo",
    "repository",
    "projekt",
    "kode",
    "koden",
    "fil",
    "filer",
    "git status",
    "din egen kode",
    "egen kode",
    "stacky kode",
    "computer",
    "skrivebord",
)


def wants_computer_context(text: str) -> bool:
    lowered = _normalize(text)
    if _looks_like_web_search_request(lowered):
        return False
    return any(trigger in lowered for trigger in _COMPUTER_TRIGGERS)


def _looks_like_web_search_request(normalized: str) -> bool:
    return any(
        token in normalized
        for token in (
            "paa nettet",
            "pa nettet",
            "online",
            "internet",
            "web search",
            "websearch",
            "google",
        )
    )


def extract_computer_search_query(text: str) -> str:
    lowered = _normalize(text)
    patterns = (
        r"(?:grep|rg|ripgrep|soeg|sog|find|finde|led)\s+(?:efter|paa|pa|i)?\s*['\"]?([^'\"?.!,]{3,80})",
        r"(?:hvor|findes|ligger)\s+([^?.!,]{3,80})",
    )
    for pattern in patterns:
        match = re.search(pattern, lowered)
        if match:
            query = _cleanup_query(match.group(1))
            if query:
                return query
    return ""


def _normalize(text: str) -> str:
    return (
        text.lower()
        .replace("\u00f8", "oe")
        .replace("\u00e5", "aa")
        .replace("\u00e6", "ae")
    )


def _cleanup_query(query: str) -> str:
    query = query.strip().strip("'\"`")
    query = re.sub(r"^(hvor|findes|ligger)\b", "", query).strip()
    query = re.sub(r"\b(i|paa|pa|koden|kode|repo|projektet|projekt|filer|fil)\b$", "", query).strip()
    query = re.sub(r"\b(i|paa|pa)\b$", "", query).strip()
    query = re.sub(r"\s+", " ", query)
    return query[:80]


def _indent(text: str) -> str:
    return "\n".join(f"  {line}" for line in text.splitlines())


def _clip(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 14].rstrip() + "\n... [afkortet]"


def _run_command(command: tuple[str, ...], *, cwd: Path, timeout_seconds: float) -> CommandResult:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return CommandResult(command, 1, "", str(exc))
    return CommandResult(command, completed.returncode, completed.stdout, completed.stderr)


def _default_desktop() -> Path:
    candidates: list[Path] = []
    for env_name in ("OneDriveConsumer", "OneDriveCommercial", "OneDrive"):
        value = os.environ.get(env_name)
        if value:
            candidates.append(Path(value) / "Desktop")
    userprofile = os.environ.get("USERPROFILE")
    if userprofile:
        candidates.append(Path(userprofile) / "Desktop")
    candidates.extend([Path.home() / "OneDrive" / "Desktop", Path.home() / "Desktop"])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return Path.home() / "Desktop"


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _wants_create_text_file(normalized: str) -> bool:
    has_write_verb = any(token in normalized for token in ("opret", "lav", "skriv", "gem"))
    has_file_word = any(token in normalized for token in ("tekstfil", "textfil", "fil"))
    if not (has_write_verb and has_file_word):
        return False
    return any(token in normalized for token in ("hilsen", "test", "tekst", "skrivebord", "workspace", "mappe"))


def _wants_list_workspace_action(normalized: str) -> bool:
    has_list_word = any(
        token in normalized
        for token in (
            "list mappe",
            "vis mappe",
            "vis workspace",
            "vis projekt",
            "hvad ligger",
            "hvad er der i mappen",
            "hvad er der her",
            "hvad kan du se i mappen",
        )
    )
    has_shell_list = bool(re.search(r"\b(?:koer|kor|kore|vis|lav|tag|start)\s+(?:en\s+)?(?:dir|ls)\b", normalized))
    has_bare_dir = normalized.strip() in {"dir", "ls", "koer dir", "kor dir", "vis dir", "vis ls"}
    return bool(
        has_bare_dir
        or has_shell_list
        or (
            has_list_word
            and any(token in normalized for token in ("workspace", "mappe", "projekt", "repo", "her"))
        )
    )


def _text_file_name_from(normalized: str) -> str:
    match = re.search(r"\b(?:hedder|kaldet|navn(?:et)?)\s+([0-9a-z._ -]{2,40})", normalized)
    if match:
        name = _safe_filename(match.group(1))
        if name:
            return name if "." in name else f"{name}.txt"
    if "hilsen" in normalized:
        return "hilsen.txt"
    if "test" in normalized:
        return "stacky-test.txt"
    return "stacky-note.txt"


def _text_file_content_from(normalized: str) -> str:
    if "hilsen" in normalized:
        return "Hej Nicolai.\n\nStacky kan skrive en lokal tekstfil nu.\n"
    if "test" in normalized:
        return "Stacky lokal computer-test.\n"
    return "Kort note fra Stacky.\n"


def _safe_filename(name: str) -> str:
    cleaned = re.sub(r"[^0-9a-zA-Z._ -]+", "", name).strip(" ._-")
    cleaned = re.sub(r"\s+", "-", cleaned)
    if not cleaned:
        return ""
    if cleaned.lower() in {"con", "prn", "aux", "nul"}:
        return ""
    return cleaned[:48]
