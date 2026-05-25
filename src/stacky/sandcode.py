from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Callable

from .config import SandcodeConfig
from .danish import compact_for_speech
from .llm import ChatClient, ChatMessage


class SandcodeError(RuntimeError):
    pass


@dataclass(frozen=True)
class SandcodeSession:
    session_id: str
    cwd: Path
    provider: str
    model: str
    permission_mode: str
    effort: str
    chat_only: bool = False


@dataclass(frozen=True)
class SandcodeAction:
    prompt: str
    cwd: Path | None = None
    chat_only: bool = False
    mode: str = "read_only"


DEFAULT_SANDCODE_AGENT_PROMPT = (
    "Lav en kort read-only status paa projektet. Find de vigtigste relevante filer "
    "og rapporter hvad der virker vigtigt lige nu. Aendr ikke filer."
)


class SandcodeMobileHostClient:
    def __init__(self, config: SandcodeConfig) -> None:
        self.config = config
        self._process: subprocess.Popen[str] | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self.config.host}:{self.config.port}"

    @property
    def ws_url(self) -> str:
        return f"ws://{self.config.host}:{self.config.port}/?token={self.config.token}"

    async def ensure_host(self) -> None:
        healthy, last_error = await self._health_result()
        if healthy:
            return
        if not self.config.host_script.exists():
            raise SandcodeError(f"Sandcode mobile host script not found: {self.config.host_script}")
        env = os.environ.copy()
        env["SANDCODE_MOBILE_TOKEN"] = self.config.token
        env["SANDCODE_MOBILE_PORT"] = str(self.config.port)
        env["SANDCODE_MOBILE_HOST"] = self.config.host
        env["SANDCODE_REPO_ROOT"] = str(self.config.repo_root)
        creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        node = shutil.which("node")
        if not node:
            raise SandcodeError("Node.js was not found on PATH; Sandcode mobile host cannot start.")
        self._process = subprocess.Popen(
            [node, str(self.config.host_script)],
            cwd=str(self.config.repo_root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            creationflags=creationflags,
        )
        loop = asyncio.get_running_loop()
        deadline = loop.time() + max(0.1, self.config.startup_timeout_seconds)
        while loop.time() < deadline:
            healthy, last_error = await self._health_result()
            if healthy:
                return
            if self._process.poll() is not None:
                detail = self._host_process_detail()
                raise SandcodeError(f"Sandcode mobile host exited before becoming healthy.{detail}") from last_error
            await asyncio.sleep(0.2)
        detail = f" Last health error: {last_error}" if last_error else ""
        raise SandcodeError(
            f"Sandcode mobile host did not become healthy at {self.base_url} "
            f"within {self.config.startup_timeout_seconds:.1f}s.{detail}"
        ) from last_error

    async def is_healthy(self) -> bool:
        healthy, _ = await self._health_result()
        return healthy

    async def _health_result(self) -> tuple[bool, SandcodeError | None]:
        try:
            await asyncio.to_thread(
                self._request_json,
                "GET",
                "/api/health",
                None,
                timeout_seconds=self.config.health_timeout_seconds,
            )
            return True, None
        except SandcodeError as exc:
            return False, exc

    async def start_session(self, cwd: Path, *, chat_only: bool = False) -> SandcodeSession:
        await self.ensure_host()
        payload = {
            "cwd": str(cwd),
            "provider": self.config.provider,
            "model": self.config.model,
            "permissionMode": self.config.permission_mode,
            "effort": self.config.effort,
            "chatOnly": chat_only,
        }
        data = await asyncio.to_thread(
            self._request_json,
            "POST",
            "/api/sessions",
            payload,
            timeout_seconds=self.config.request_timeout_seconds,
        )
        return SandcodeSession(
            session_id=str(data["sessionId"]),
            cwd=Path(str(data.get("cwd", cwd))),
            provider=self.config.provider,
            model=self.config.model,
            permission_mode=self.config.permission_mode,
            effort=self.config.effort,
            chat_only=chat_only,
        )

    async def send_user_message(self, session: SandcodeSession, text: str) -> None:
        await self._send_ws(
            {
                "type": "user_message",
                "sessionId": session.session_id,
                "cwd": str(session.cwd),
                "text": text,
                "provider": session.provider,
                "model": session.model,
                "permissionMode": session.permission_mode,
                "effort": session.effort,
                "chatOnly": session.chat_only,
            }
        )

    async def cancel(self, session_id: str) -> None:
        await self._send_ws({"type": "cancel", "sessionId": session_id})

    async def events(self) -> AsyncIterator[dict[str, Any]]:
        try:
            import websockets  # type: ignore
        except ImportError as exc:
            raise SandcodeError("Install websockets or stacky[voice] to listen to Sandcode events.") from exc

        async with websockets.connect(
            self.ws_url,
            proxy=None,
            open_timeout=self.config.websocket_open_timeout_seconds,
        ) as ws:
            async for raw in ws:
                event = json.loads(str(raw))
                if isinstance(event, dict):
                    yield event

    async def run_session(
        self,
        cwd: Path,
        prompt: str,
        on_event: Callable[[dict[str, Any]], None],
        *,
        chat_only: bool = False,
    ) -> SandcodeSession:
        session = await self.start_session(cwd, chat_only=chat_only)
        listener = asyncio.create_task(self._listen_until_idle(on_event, session.session_id))
        await self.send_user_message(session, prompt)
        await listener
        return session

    async def _listen_until_idle(self, on_event: Callable[[dict[str, Any]], None], session_id: str | None = None) -> None:
        session_ids = {session_id} if session_id else set()
        async for event in self.events():
            event_session_id = str(event.get("sessionId") or "")
            if event.get("type") == "session_rekey" and str(event.get("oldSessionId") or "") in session_ids:
                new_session_id = str(event.get("sessionId") or "")
                if new_session_id:
                    session_ids.add(new_session_id)
            if session_ids and event_session_id and event_session_id not in session_ids:
                continue
            on_event(event)
            if event.get("type") == "session_state" and event.get("state") == "idle":
                return

    async def _send_ws(self, message: dict[str, Any]) -> None:
        try:
            import websockets  # type: ignore
        except ImportError as exc:
            raise SandcodeError("Install websockets or stacky[voice] to send Sandcode websocket messages.") from exc

        await self.ensure_host()
        async with websockets.connect(
            self.ws_url,
            proxy=None,
            open_timeout=self.config.websocket_open_timeout_seconds,
        ) as ws:
            await ws.send(json.dumps(message, ensure_ascii=False))

    def _request_json(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None,
        *,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        url = self.base_url + path
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            method=method,
            headers={
                "Authorization": f"Bearer {self.config.token}",
                "Content-Type": "application/json",
            },
        )
        timeout = self.config.request_timeout_seconds if timeout_seconds is None else timeout_seconds
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            with opener.open(request, timeout=timeout) as response:
                text = response.read().decode("utf-8")
                return json.loads(text) if text else {}
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            if exc.code in {401, 403}:
                raise SandcodeError(
                    f"Sandcode rejected the token at {self.base_url}. "
                    f"Check [sandcode].token matches the mobile host token. HTTP {exc.code}: {detail}"
                ) from exc
            raise SandcodeError(f"Sandcode HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            reason = getattr(exc, "reason", exc)
            raise SandcodeError(f"Sandcode connection failed at {url}: {reason}") from exc
        except OSError as exc:
            raise SandcodeError(f"Sandcode connection failed at {url}: {exc}") from exc

    def _host_process_detail(self) -> str:
        if self._process is None:
            return ""
        try:
            stdout, stderr = self._process.communicate(timeout=0.2)
        except (OSError, subprocess.TimeoutExpired):
            return ""
        parts = []
        if stdout.strip():
            parts.append(f"stdout: {_compact_log(stdout)}")
        if stderr.strip():
            parts.append(f"stderr: {_compact_log(stderr)}")
        return " " + " ".join(parts) if parts else ""


def _compact_log(text: str, *, max_chars: int = 360) -> str:
    clean = re.sub(r"\s+", " ", text).strip()
    if len(clean) <= max_chars:
        return clean
    return clean[: max_chars - 1].rstrip() + "..."


class SandcodeDanishSummarizer:
    def summarize_event(self, event: dict[str, Any]) -> str | None:
        event_type = event.get("type")
        if event_type == "assistant_message":
            if not event.get("done"):
                return None
            text = str(event.get("text") or "").strip()
            if not text:
                return "Agenten er færdig med sit svar."
            return "Agenten melder: " + compact_for_speech(text, max_chars=360)
        if event_type == "tool_call":
            tool = str(event.get("toolName") or event.get("displayName") or "et værktøj")
            description = str(event.get("description") or "").strip()
            return compact_for_speech(f"Agenten arbejder med {tool}. {description}", max_chars=220)
        if event_type == "tool_update" and event.get("status") in {"done", "failed"}:
            status = "færdig" if event.get("status") == "done" else "fejlede"
            body = str(event.get("body") or "")
            return compact_for_speech(f"Agentens værktøj er {status}. {body}", max_chars=220)
        if event_type == "permission_request":
            tool = str(event.get("displayName") or event.get("toolName") or "et værktøj")
            return compact_for_speech(f"Agenten beder om tilladelse til {tool}.", max_chars=220)
        if event_type == "turn_cancelled":
            return "Jeg har afbrudt agent-sessionen."
        if event_type == "error":
            return compact_for_speech(f"Agenten meldte fejl: {event.get('message')}", max_chars=260)
        return None

    def summarize_heartbeat(self, *, elapsed_seconds: float, last_update: str = "") -> str:
        elapsed = _format_elapsed(elapsed_seconds)
        clean_update = compact_for_speech(last_update.strip(), max_chars=120) if last_update.strip() else ""
        if clean_update:
            return compact_for_speech(
                f"Agenten arbejder stadig efter {elapsed}. Sidste livstegn: {clean_update}",
                max_chars=220,
            )
        return f"Agenten arbejder stadig efter {elapsed}; jeg har ikke fået nye detaljer endnu."


def _format_elapsed(seconds: float) -> str:
    total = max(0, int(seconds))
    minutes, secs = divmod(total, 60)
    if minutes <= 0:
        return f"{secs} sekunder"
    if secs <= 0:
        return f"{minutes} min"
    return f"{minutes} min {secs} sek"


def parse_sandcode_action(text: str) -> SandcodeAction | None:
    """Parse explicit Danish Sandcode requests.

    The request must still be explicit. Stacky should not start a coding agent
    from vague speech or noisy STT fragments.
    """

    normalized = _normalize_for_intent(text)
    if not _looks_like_sandcode_request(normalized):
        return None
    if _wants_cancel_sandcode(normalized):
        return SandcodeAction(prompt="__cancel__", mode="cancel")
    if _looks_like_sandcode_test_request(normalized):
        return SandcodeAction(prompt=DEFAULT_SANDCODE_AGENT_PROMPT)

    chat_only = any(phrase in normalized for phrase in ("chat only", "kun chat", "uden tools", "uden vaerktoejer"))
    prompt = _extract_sandcode_prompt(text)
    if not prompt:
        if _looks_like_agent_activation_without_task(normalized):
            prompt = DEFAULT_SANDCODE_AGENT_PROMPT
        else:
            return None
    elif _is_activation_only_prompt(prompt):
        prompt = DEFAULT_SANDCODE_AGENT_PROMPT
    return SandcodeAction(prompt=prompt, chat_only=chat_only, mode=_infer_sandcode_mode(prompt))


async def classify_sandcode_action(
    text: str,
    brain: ChatClient | None,
    *,
    recent_context: str = "",
) -> SandcodeAction | None:
    """Route natural agent requests before the free-form brain replies.

    The regex parser is a safety net for clear commands. The LLM router is the
    primary path for Danish phrasing like "agenten du kan saette i gang" where
    Stacky should choose a capability instead of pretending in plain text.
    """

    action = parse_sandcode_action(text)
    if action is not None:
        return action
    if _looks_like_agent_followup_activation(text, recent_context):
        return SandcodeAction(prompt=DEFAULT_SANDCODE_AGENT_PROMPT)
    if brain is None or not _looks_like_possible_agent_need(text):
        return None

    context_note = f"\n\nSeneste live-kontekst:\n{recent_context[:900]}" if recent_context.strip() else ""
    prompt = (
        "Du er Stackys lokale ability-router, ikke samtalehjernen. "
        "Afgor om Nicolais danske besked beder Stacky om at bruge Sandcode/Codex-agenten "
        "som en rigtig runtime-handling foer svaret. "
        "Svar KUN JSON: {\"sandcode_action\":\"start|cancel|none\", \"prompt\":\"kort opgave\", \"chat_only\":false}. "
        "Brug start naar han beder agenten om at kigge, scanne, rette, arbejde eller blive sat i gang. "
        "Hvis han bare vil have agenten sat i gang uden konkret opgave, brug en read-only projektstatus som prompt. "
        "Brug none for almindelig snak om agenter, identitet, fejlbeskrivelser som 'agent skills halter', "
        "eller spoergsmaal om hvad agenten er.\n\n"
        f"Besked: {text[:500]}{context_note}"
    )
    try:
        result = await brain.chat(
            [ChatMessage("system", prompt), ChatMessage("user", text)],
            temperature=0.0,
            max_tokens=160,
        )
    except Exception as exc:
        print(f"[sandcode] ability-router failed: {exc}", flush=True)
        return None
    return _parse_sandcode_intent(result)


def _extract_sandcode_prompt(text: str) -> str:
    clean = text.strip()
    if not clean:
        return ""
    clean = _canonicalize_sandcode_aliases(clean)
    patterns = (
        r"(?i)^\s*(?:brug|start|k[øo]r|bed|s[æa]t|lad|send)\s+sandcode\s+(?:til\s+at|om\s+at|på|paa|med)?\s*(.+)$",
        r"(?i)^\s*sandcode\s+(?:skal|må|maa|kan|til\s+at|om\s+at)?\s*(.+)$",
    )
    for pattern in patterns:
        match = re.match(pattern, clean)
        if match:
            return _cleanup_prompt(match.group(1))
    lowered = _normalize_for_intent(clean)
    index = lowered.find("sandcode")
    if index < 0:
        return ""
    return _cleanup_prompt(clean[index + len("sandcode") :])


def _cleanup_prompt(prompt: str) -> str:
    prompt = re.sub(r"(?i)\b(kun chat|chat only|uden tools|uden v[æa]rkt[øo]jer)\b", "", prompt)
    prompt = re.sub(r"(?i)^\s*(?:til\s+at|om\s+at|med\s+at|skal|kan|m[åa])\s+", "", prompt)
    prompt = re.sub(r"\s+", " ", prompt).strip(" .,:;-")
    return prompt


def _parse_sandcode_intent(raw: str) -> SandcodeAction | None:
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        data = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None
    action = str(data.get("sandcode_action") or data.get("action") or "").strip().lower()
    if action in {"cancel", "stop", "afbryd", "annuller"}:
        return SandcodeAction(prompt="__cancel__", mode="cancel")
    if action != "start":
        return None
    prompt = _cleanup_prompt(str(data.get("prompt") or ""))
    if not prompt or _is_activation_only_prompt(prompt):
        prompt = DEFAULT_SANDCODE_AGENT_PROMPT
    mode = _normalize_sandcode_mode(str(data.get("mode") or "")) or _infer_sandcode_mode(prompt)
    return SandcodeAction(prompt=prompt, chat_only=bool(data.get("chat_only")), mode=mode)


def _normalize_sandcode_mode(mode: str) -> str:
    clean = _normalize_for_intent(mode).strip().replace("-", "_")
    if clean in {"readonly", "read_only", "read only", "status", "scan", "inspect"}:
        return "read_only"
    if clean in {"work", "write", "edit", "change", "fix", "implement", "aendre", "endre"}:
        return "work"
    if clean in {"cancel", "stop", "afbryd", "annuller"}:
        return "cancel"
    return ""


def _infer_sandcode_mode(prompt: str) -> str:
    normalized = _normalize_for_intent(prompt)
    if any(
        token in normalized
        for token in ("read-only", "read only", "readonly", "status", "scan", "kig", "laes", "las", "undersoeg")
    ):
        return "read_only"
    if any(
        token in normalized
        for token in (
            "ret",
            "rette",
            "fix",
            "fikse",
            "byg",
            "bygge",
            "implement",
            "lav",
            "opret",
            "skriv",
            "aendr",
            "aendre",
            "endre",
        )
    ):
        return "work"
    return "read_only"


def _is_activation_only_prompt(prompt: str) -> bool:
    folded = " ".join(_normalize_for_intent(prompt).split()).strip(" .,:;-")
    if not folded:
        return True
    folded = re.sub(r"\b(?:du kan|kan du|lige|bare|nu|tak|please|venligst|for mig|saa|sa)\b", " ", folded)
    folded = " ".join(folded.split())
    return folded in {
        "den",
        "det",
        "i gang",
        "igang",
        "start",
        "starte",
        "start den",
        "starte den",
        "saet i gang",
        "saet igang",
        "saette i gang",
        "saette igang",
        "sat i gang",
        "sat igang",
        "koer",
        "kor",
        "koere",
        "kore",
        "koer den",
        "kor den",
    }


def _looks_like_sandcode_request(normalized: str) -> bool:
    if "sandcode" in normalized:
        command = r"(?:brug|start|koer|kor|bed|saet|send|lad)"
        if _wants_cancel_sandcode(normalized):
            return True
        if _looks_like_sandcode_test_request(normalized):
            return True
        if re.search(rf"\b{command}\s+sandcode\b", normalized):
            return True
        return bool(re.search(r"\bsandcode\s+(?:skal|maa|ma|kan|til\s+at|om\s+at|start|starte|i\s*gang|igang)\b", normalized))
    agent_alias = r"(?:kodeagenten?|kodningsagenten?|codex(?:\s*agent)?|agenten?|agent)"
    if re.search(rf"\b(?:stop|afbryd|annuller)\s+(?:den\s+)?{agent_alias}\b", normalized):
        return True
    command = r"(?:brug|start|koer|kor|bed|saet|send|lad)"
    if re.search(rf"\b{command}\s+(?:den\s+)?{agent_alias}\b", normalized):
        return True
    return bool(re.search(rf"\b{agent_alias}\s+(?:skal|maa|ma|kan|til\s+at|om\s+at)\b", normalized))


def _looks_like_sandcode_test_request(normalized: str) -> bool:
    if "sandcode" not in normalized:
        return False
    return bool(
        re.search(r"\b(?:test|tester|teste|proev|proever|prov|prover)\b.*\bsandcode\b", normalized)
        or re.search(r"\bsandcode\b.*\b(?:test|tester|teste|virker|igennem|gennem)\b", normalized)
        or re.search(r"\bse\s+om\s+sandcode\b", normalized)
    )


def _looks_like_agent_followup_activation(text: str, recent_context: str) -> bool:
    if not recent_context.strip():
        return False
    context_words = set(_fold_agent_words(recent_context))
    if not (context_words & {"sandcode", "agent", "agenten", "kodeagent", "kodeagenten", "codex"}):
        return False
    normalized = _normalize_for_intent(text)
    return bool(
        re.search(r"\b(?:start|starte|koer|kor|saet|send|aktiver|aktivere)\s+(?:den|det)\b", normalized)
        or re.search(r"\b(?:proever|prøver|forsoger|forsøger)\s+at\s+(?:start|starte|aktivere)\s+(?:den|det)\b", normalized)
        or re.search(r"\b(?:den|det)\s+(?:skal\s+)?(?:start|starte|i\s*gang|igang|koere|kore)\b", normalized)
    )


def _looks_like_agent_activation_without_task(normalized: str) -> bool:
    agent_alias = r"(?:sandcode|kodeagenten?|kodningsagenten?|codex(?:\s*agent)?|agenten?|agent)"
    command = r"(?:brug|start|koer|kor|bed|saet|send|lad)"
    return bool(
        re.search(rf"\b{command}\s+(?:den\s+)?{agent_alias}\s*$", normalized)
        or re.search(rf"\b{agent_alias}\s+(?:start|starte|i\s*gang|igang|koer|kor)\s*$", normalized)
    )


def _looks_like_possible_agent_need(text: str) -> bool:
    words = set(_fold_agent_words(text))
    if not words:
        return False
    agent_words = {
        "agent",
        "agenten",
        "kodeagent",
        "kodeagenten",
        "kodningsagent",
        "kodningsagenten",
        "codex",
        "sandcode",
    }
    action_words = {
        "brug",
        "bruge",
        "start",
        "starte",
        "saet",
        "saette",
        "sat",
        "gang",
        "igang",
        "koer",
        "kor",
        "koere",
        "kore",
        "kig",
        "kigge",
        "scan",
        "scanne",
        "laes",
        "las",
        "laese",
        "lase",
        "ret",
        "rette",
        "fix",
        "fikse",
        "arbejd",
        "arbejde",
    }
    return bool(words & agent_words and words & action_words)


def _wants_cancel_sandcode(normalized: str) -> bool:
    if any(phrase in normalized for phrase in ("stop sandcode", "afbryd sandcode", "annuller sandcode")):
        return True
    agent_alias = r"(?:kodeagenten?|kodningsagenten?|codex(?:\s*agent)?|agenten?|agent)"
    return bool(re.search(rf"\b(?:stop|afbryd|annuller)\s+(?:den\s+)?{agent_alias}\b", normalized))


def _canonicalize_sandcode_aliases(text: str) -> str:
    clean = re.sub(r"(?i)\bsand\s+(?:code|kode)\b", "sandcode", text)
    return re.sub(
        r"(?i)\b(?:kodeagent(?:en)?|kodningsagent(?:en)?|codex(?:\s*agent)?|agenten|agent)\b",
        "sandcode",
        clean,
    )


def _normalize_for_intent(text: str) -> str:
    normalized = (
        text.lower()
        .replace("\u00f8", "oe")
        .replace("\u00e5", "aa")
        .replace("\u00e6", "ae")
    )
    normalized = normalized.replace("sand code", "sandcode").replace("sand kode", "sandcode")
    normalized = re.sub(r"\bsancodi(?=[a-z])", "sandcode ", normalized)
    normalized = re.sub(r"\bsan(?:d)?cod(?:e|i)?\b", "sandcode", normalized)
    return normalized


def _fold_agent_words(text: str) -> list[str]:
    folded = _normalize_for_intent(text)
    return re.sub(r"[^0-9a-z]+", " ", folded).split()
