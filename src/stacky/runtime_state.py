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


def _one_line(value: str) -> str:
    return " ".join(str(value).split()).strip()
