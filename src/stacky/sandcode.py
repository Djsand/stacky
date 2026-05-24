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
        if await self.is_healthy():
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
        for _ in range(30):
            if await self.is_healthy():
                return
            await asyncio.sleep(0.2)
        raise SandcodeError("Sandcode mobile host did not become healthy.")

    async def is_healthy(self) -> bool:
        try:
            await asyncio.to_thread(self._request_json, "GET", "/api/health", None)
            return True
        except SandcodeError:
            return False

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
        data = await asyncio.to_thread(self._request_json, "POST", "/api/sessions", payload)
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

        async with websockets.connect(self.ws_url) as ws:
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
        async with websockets.connect(self.ws_url) as ws:
            await ws.send(json.dumps(message, ensure_ascii=False))

    def _request_json(self, method: str, path: str, payload: dict[str, Any] | None) -> dict[str, Any]:
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
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                text = response.read().decode("utf-8")
                return json.loads(text) if text else {}
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise SandcodeError(f"Sandcode HTTP {exc.code}: {detail}") from exc
        except OSError as exc:
            raise SandcodeError(f"Sandcode connection failed: {exc}") from exc


class SandcodeDanishSummarizer:
    def summarize_event(self, event: dict[str, Any]) -> str | None:
        event_type = event.get("type")
        if event_type == "assistant_message":
            if not event.get("done"):
                return None
            text = str(event.get("text") or "").strip()
            if not text:
                return "Sandcode er færdig med sit svar."
            return "Sandcode siger: " + compact_for_speech(text, max_chars=360)
        if event_type == "tool_call":
            tool = str(event.get("toolName") or event.get("displayName") or "et værktøj")
            description = str(event.get("description") or "").strip()
            return compact_for_speech(f"Sandcode bruger {tool}. {description}", max_chars=220)
        if event_type == "tool_update" and event.get("status") in {"done", "failed"}:
            status = "færdig" if event.get("status") == "done" else "fejlede"
            body = str(event.get("body") or "")
            return compact_for_speech(f"Sandcode-værktøjet er {status}. {body}", max_chars=220)
        if event_type == "permission_request":
            tool = str(event.get("displayName") or event.get("toolName") or "et værktøj")
            return compact_for_speech(f"Sandcode beder om tilladelse til {tool}.", max_chars=220)
        if event_type == "turn_cancelled":
            return "Jeg har afbrudt Sandcode-sessionen."
        if event_type == "error":
            return compact_for_speech(f"Sandcode meldte fejl: {event.get('message')}", max_chars=260)
        return None


def parse_sandcode_action(text: str) -> SandcodeAction | None:
    """Parse explicit Danish Sandcode requests.

    The word "sandcode" is required on purpose. Stacky should not start a
    coding agent from vague speech or noisy STT fragments.
    """

    normalized = _normalize_for_intent(text)
    if "sandcode" not in normalized:
        return None
    if any(phrase in normalized for phrase in ("stop sandcode", "afbryd sandcode", "annuller sandcode")):
        return SandcodeAction(prompt="__cancel__")

    chat_only = any(phrase in normalized for phrase in ("chat only", "kun chat", "uden tools", "uden vaerktoejer"))
    prompt = _extract_sandcode_prompt(text)
    if not prompt:
        return None
    return SandcodeAction(prompt=prompt, chat_only=chat_only)


def _extract_sandcode_prompt(text: str) -> str:
    clean = text.strip()
    if not clean:
        return ""
    clean = re.sub(r"(?i)\bsand\s+(?:code|kode)\b", "sandcode", clean)
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
    prompt = re.sub(r"\s+", " ", prompt).strip(" .,:;-")
    return prompt


def _normalize_for_intent(text: str) -> str:
    normalized = (
        text.lower()
        .replace("\u00f8", "oe")
        .replace("\u00e5", "aa")
        .replace("\u00e6", "ae")
    )
    return normalized.replace("sand code", "sandcode").replace("sand kode", "sandcode")
