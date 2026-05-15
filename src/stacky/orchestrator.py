from __future__ import annotations

from pathlib import Path

from .brain import StackyBrain
from .sandcode import SandcodeDanishSummarizer, SandcodeMobileHostClient


class StackyOrchestrator:
    def __init__(self, brain: StackyBrain, sandcode: SandcodeMobileHostClient) -> None:
        self.brain = brain
        self.sandcode = sandcode
        self.sandcode_summarizer = SandcodeDanishSummarizer()

    async def chat(self, text: str) -> str:
        reply = await self.brain.respond(text)
        return reply.text

    async def start_coding_project(self, cwd: Path, prompt: str) -> list[str]:
        spoken_updates: list[str] = []

        def on_event(event: dict[str, object]) -> None:
            spoken = self.sandcode_summarizer.summarize_event(event)
            if spoken:
                spoken_updates.append(spoken)

        await self.sandcode.run_session(cwd, prompt, on_event)
        return spoken_updates
