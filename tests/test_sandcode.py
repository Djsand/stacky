from __future__ import annotations

import asyncio
import unittest
from pathlib import Path

from stacky.config import SandcodeConfig
from stacky.sandcode import SandcodeDanishSummarizer, SandcodeMobileHostClient


class FakeSandcodeClient(SandcodeMobileHostClient):
    def __init__(self) -> None:
        super().__init__(
            SandcodeConfig(
                repo_root=Path("C:/Users/nicol/SANDCODE"),
                host_script=Path("C:/Users/nicol/SANDCODE/ios/host/sandcode-mobile-host.mjs"),
                token="test-token",
            )
        )
        self.payloads: list[dict[str, object]] = []

    async def ensure_host(self) -> None:
        return None

    def _request_json(self, method: str, path: str, payload: dict[str, object] | None) -> dict[str, object]:
        self.payloads.append({"method": method, "path": path, "payload": payload or {}})
        return {"sessionId": "mobile-test", "cwd": "C:/project"}


class SandcodeTest(unittest.IsolatedAsyncioTestCase):
    async def test_start_session_uses_autonomous_agent(self) -> None:
        client = FakeSandcodeClient()
        session = await client.start_session(Path("C:/project"))

        self.assertEqual(session.permission_mode, "autonomousAgent")
        payload = client.payloads[0]["payload"]
        self.assertIsInstance(payload, dict)
        self.assertEqual(payload["permissionMode"], "autonomousAgent")
        self.assertFalse(payload["chatOnly"])

    async def test_summarizer_speaks_danish_status(self) -> None:
        summarizer = SandcodeDanishSummarizer()
        spoken = summarizer.summarize_event(
            {
                "type": "tool_call",
                "toolName": "Read",
                "description": "C:/project/src/app.py",
            }
        )
        self.assertIsNotNone(spoken)
        self.assertIn("Sandcode bruger", spoken)


if __name__ == "__main__":
    unittest.main()
