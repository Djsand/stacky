from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeAction:
    kind: str
    status: str
    summary: str
    detail: str = ""
    error: str = ""
    session_id: str = ""
    verified_at: float = 0.0
    can_speak_about: tuple[str, ...] = ()


class RuntimeState:
    """Short-lived truth layer for actions Stacky's runtime actually performed."""

    def __init__(
        self,
        *,
        max_events: int = 8,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._clock = clock
        self._events: deque[RuntimeAction] = deque(maxlen=max(1, max_events))
        self._agent_status = "idle"
        self._last_action: RuntimeAction | None = None
        self._last_error = ""
        self._last_verified_at = 0.0
        self._can_speak_about: tuple[str, ...] = ()

    @property
    def agent_status(self) -> str:
        return self._agent_status

    @property
    def last_action(self) -> RuntimeAction | None:
        return self._last_action

    @property
    def last_error(self) -> str:
        return self._last_error

    @property
    def last_verified_at(self) -> float:
        return self._last_verified_at

    @property
    def can_speak_about(self) -> tuple[str, ...]:
        return self._can_speak_about

    def record_action(
        self,
        *,
        kind: str,
        status: str,
        summary: str,
        detail: str = "",
        error: str = "",
        session_id: str = "",
        can_speak_about: tuple[str, ...] = ("runtime_action",),
    ) -> RuntimeAction:
        action = RuntimeAction(
            kind=kind.strip() or "runtime_action",
            status=status.strip() or "done",
            summary=_one_line(summary),
            detail=_one_line(detail),
            error=_one_line(error),
            session_id=_one_line(session_id),
            verified_at=self._clock(),
            can_speak_about=tuple(dict.fromkeys(can_speak_about)),
        )
        self._events.append(action)
        self._last_action = action
        self._last_error = action.error
        self._last_verified_at = action.verified_at
        self._can_speak_about = action.can_speak_about
        if action.kind == "sandcode_agent":
            self._agent_status = action.status
        return action

    def mark_sandcode_starting(self, prompt: str) -> RuntimeAction:
        return self.record_action(
            kind="sandcode_agent",
            status="starting",
            summary=f"Sandcode-agent starter: {_one_line(prompt)}",
            detail=prompt,
            can_speak_about=("sandcode_agent", "runtime_action"),
        )

    def mark_sandcode_running(self, prompt: str, *, note: str = "") -> RuntimeAction:
        suffix = f" - {_one_line(note)}" if note.strip() else ""
        return self.record_action(
            kind="sandcode_agent",
            status="running",
            summary=f"Sandcode-agent koerer: {_one_line(prompt)}{suffix}",
            detail=prompt,
            can_speak_about=("sandcode_agent", "runtime_action"),
        )

    def mark_sandcode_done(self, prompt: str, *, session_id: str = "") -> RuntimeAction:
        return self.record_action(
            kind="sandcode_agent",
            status="done",
            summary=f"Sandcode-agent faerdig: {_one_line(prompt)}",
            detail=prompt,
            session_id=session_id,
            can_speak_about=("sandcode_agent", "runtime_action"),
        )

    def mark_sandcode_failed(self, prompt: str, error: str) -> RuntimeAction:
        return self.record_action(
            kind="sandcode_agent",
            status="failed",
            summary=f"Sandcode-agent fejlede: {_one_line(prompt)}",
            detail=prompt,
            error=error,
            can_speak_about=("sandcode_agent", "runtime_action"),
        )

    def mark_sandcode_cancelled(self, prompt: str, *, session_id: str = "", error: str = "") -> RuntimeAction:
        return self.record_action(
            kind="sandcode_agent",
            status="cancelled",
            summary=f"Sandcode-agent stoppet: {_one_line(prompt)}",
            detail=prompt,
            error=error,
            session_id=session_id,
            can_speak_about=("sandcode_agent", "runtime_action"),
        )

    def context_for_prompt(self, *, now: float | None = None, max_events: int = 5) -> str:
        current = self._clock() if now is None else now
        action = self._last_action
        age = "never" if action is None else f"{max(0, int(current - action.verified_at))} sek siden"
        last_action = "none" if action is None else action.summary
        last_error = self._last_error or "none"
        can_speak_about = ", ".join(self._can_speak_about) if self._can_speak_about else "none"
        lines = [
            "Runtime-sandhedslag (kortlivet, verificeret af Stackys runtime):",
            f"- agent_status: {self._agent_status}",
            f"- last_action: {last_action}",
            f"- last_error: {last_error}",
            f"- last_verified_at: {age}",
            f"- can_speak_about: {can_speak_about}",
            "- action_ledger:",
        ]
        events = list(self._events)[-max(1, max_events) :]
        if not events:
            lines.append("  - none")
        else:
            for event in events:
                line = f"  - {event.kind}/{event.status}: {event.summary}"
                if event.session_id:
                    line += f" (session {event.session_id})"
                if event.error:
                    line += f" (error {event.error})"
                lines.append(line)
        lines.append(
            "Regel: Stacky maa kun paastaa at Sandcode, web eller computerhandlinger faktisk er koert, "
            "naar dette lag eller frisk action-kontekst siger det. Hvis laget siger idle/none, saa sig det kort."
        )
        return "\n".join(lines)

    def status_reply(self, question: str = "", *, now: float | None = None, stale_after_seconds: float = 55.0) -> str:
        current = self._clock() if now is None else now
        action = self._last_action
        if action is None:
            return "Der kører ikke nogen runtime-handling lige nu."

        age_seconds = max(0, current - action.verified_at)
        age = _format_age(age_seconds)
        wants_wait_reason = _looks_like_wait_question(question)
        wants_hang_check = _looks_like_hang_question(question)

        if action.kind == "sandcode_agent":
            return _sandcode_status_reply(
                action,
                age=age,
                stale=age_seconds >= stale_after_seconds,
                wants_wait_reason=wants_wait_reason,
                wants_hang_check=wants_hang_check,
            )
        if action.kind == "web_search":
            if action.status == "failed":
                return f"Websearch fejlede sidst, {age}: {action.error or action.summary}"
            return f"Sidste websearch er kørt, {age}: {action.summary}"
        if action.kind.startswith("computer:"):
            if action.status == "failed":
                return f"Sidste computerhandling fejlede, {age}: {action.error or action.summary}"
            return f"Sidste computerhandling er færdig, {age}: {action.summary}"
        if action.status == "failed":
            return f"Sidste runtime-handling fejlede, {age}: {action.error or action.summary}"
        return f"Sidste runtime-handling er {action.status}, {age}: {action.summary}"


def _one_line(value: str) -> str:
    return " ".join(str(value).split()).strip()


def _sandcode_status_reply(
    action: RuntimeAction,
    *,
    age: str,
    stale: bool,
    wants_wait_reason: bool,
    wants_hang_check: bool,
) -> str:
    summary = action.summary
    if action.status == "starting":
        if wants_wait_reason:
            return f"Den venter på at Sandcode-sessionen kommer i gang. Sidste sikre status var {age}: {summary}"
        return f"Agenten er ved at starte. Sidste sikre status var {age}: {summary}"
    if action.status == "running":
        if stale:
            return (
                f"Jeg har ikke fået nyt livstegn fra agenten i {age}. "
                f"Det kan være et hæng, eller Sandcode der tygger langsomt: {summary}"
            )
        if wants_wait_reason:
            return f"Den venter på næste livstegn fra Sandcode. Sidste sikre status var {age}: {summary}"
        if wants_hang_check:
            return f"Den ser ikke hængt ud ud fra runtime. Sidste livstegn var {age}: {summary}"
        return f"Agenten kører. Sidste livstegn var {age}: {summary}"
    if action.status == "done":
        session = f" Sessionen hedder {action.session_id}." if action.session_id else ""
        return f"Agenten er færdig.{session} Sidste sikre status var {age}: {summary}"
    if action.status == "failed":
        error = action.error or "ukendt fejl"
        return f"Agenten fejlede, {age}: {error}"
    if action.status == "cancelled":
        error = f" {action.error}" if action.error else ""
        return f"Agenten er stoppet, {age}.{error}"
    return f"Agentstatus er {action.status}, {age}: {summary}"


def _format_age(seconds: float) -> str:
    whole = max(0, int(seconds))
    if whole < 2:
        return "lige nu"
    if whole < 60:
        return f"for {whole} sek siden"
    minutes = whole // 60
    if minutes == 1:
        return "for 1 min siden"
    return f"for {minutes} min siden"


def _looks_like_wait_question(text: str) -> bool:
    key = _fold_for_status(text)
    return "venter" in key or "venterden" in key or "hvadvent" in key


def _looks_like_hang_question(text: str) -> bool:
    key = _fold_for_status(text)
    return "haenger" in key or "hanger" in key or "hang" in key


def _fold_for_status(text: str) -> str:
    lowered = text.lower()
    replacements = {
        "æ": "ae",
        "ø": "o",
        "å": "a",
        "ä": "ae",
        "ö": "o",
        "ü": "u",
    }
    for source, target in replacements.items():
        lowered = lowered.replace(source, target)
    return "".join(ch for ch in lowered if ch.isalnum())
