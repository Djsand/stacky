from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol

from ..brain import StackyBrain
from ..body.controller import BodyPresence
from ..danish import assert_danish_voice_config


@dataclass(frozen=True)
class DanishVoiceSettings:
    language: str = "da-DK"
    allow_language_switch: bool = False
    barge_in: bool = True

    def validate(self) -> None:
        assert_danish_voice_config(self.language, self.allow_language_switch)


class SpeechOutput(Protocol):
    async def speak(self, text: str) -> None:
        ...

    async def stop(self) -> None:
        ...


class ConsoleSpeechOutput:
    async def speak(self, text: str) -> None:
        print(f"Stacky: {text}")

    async def stop(self) -> None:
        print("Stacky: stopper.")


class LocalTextVoiceRuntime:
    """Text-mode voice harness for testing the Danish conversation loop."""

    def __init__(
        self,
        brain: StackyBrain,
        output: SpeechOutput | None = None,
        presence: BodyPresence | None = None,
        web_context_provider: Callable[[str], Awaitable[str]] | None = None,
        computer_context_provider: Callable[[str], Awaitable[str]] | None = None,
    ) -> None:
        self.brain = brain
        self.output = output or ConsoleSpeechOutput()
        self.presence = presence or BodyPresence(None)
        self.web_context_provider = web_context_provider
        self.computer_context_provider = computer_context_provider
        self.settings = DanishVoiceSettings()
        self.settings.validate()

    async def run_once(self, user_text: str) -> str:
        if user_text.strip().lower() in {"stop", "afbryd", "hold stop"}:
            self.presence.set("neutral")
            await self.output.stop()
            return "stopped"
        self.presence.set("listening")
        await asyncio.sleep(0.1)
        self.presence.set("thinking")
        web_context = await self.web_context_provider(user_text) if self.web_context_provider is not None else ""
        computer_context = (
            await self.computer_context_provider(user_text) if self.computer_context_provider is not None else ""
        )
        reply = await self.brain.respond(user_text, web_context=web_context, computer_context=computer_context)
        self.presence.set("happy")
        await self.output.speak(reply.spoken_text or reply.text)
        return reply.text

    async def interactive(self) -> None:
        print("Stacky tekst-voice. Skriv dansk. Skriv 'exit' for at stoppe.")
        try:
            while True:
                self.presence.set("listening")
                try:
                    user_text = await asyncio.to_thread(input, "Nicol: ")
                except EOFError:
                    return
                if user_text.strip().lower() in {"exit", "quit"}:
                    self.presence.set("neutral")
                    return
                await self.run_once(user_text)
        finally:
            await self.output.stop()
            self.presence.set("neutral")
