from __future__ import annotations

import asyncio
import unittest
from pathlib import Path
from unittest.mock import patch

from stacky.config import SandcodeConfig
from stacky.sandcode import SandcodeDanishSummarizer, SandcodeMobileHostClient, parse_sandcode_action


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
        self.ws_messages: list[dict[str, object]] = []

    async def ensure_host(self) -> None:
        return None

    def _request_json(
        self,
        method: str,
        path: str,
        payload: dict[str, object] | None,
        *,
        timeout_seconds: float | None = None,
    ) -> dict[str, object]:
        self.payloads.append(
            {
                "method": method,
                "path": path,
                "payload": payload or {},
                "timeout_seconds": timeout_seconds,
            }
        )
        return {"sessionId": "mobile-test", "cwd": "C:/project"}

    async def _send_ws(self, message: dict[str, object]) -> None:
        self.ws_messages.append(message)


class SandcodeTest(unittest.IsolatedAsyncioTestCase):
    async def test_start_session_uses_autonomous_agent(self) -> None:
        client = FakeSandcodeClient()
        session = await client.start_session(Path("C:/project"))

        self.assertEqual(session.permission_mode, "autonomousAgent")
        payload = client.payloads[0]["payload"]
        self.assertIsInstance(payload, dict)
        self.assertEqual(payload["permissionMode"], "autonomousAgent")
        self.assertFalse(payload["chatOnly"])
        self.assertEqual(client.payloads[0]["timeout_seconds"], client.config.request_timeout_seconds)

    async def test_is_healthy_uses_short_health_timeout(self) -> None:
        client = FakeSandcodeClient()
        healthy = await client.is_healthy()

        self.assertTrue(healthy)
        self.assertEqual(client.payloads[0]["path"], "/api/health")
        self.assertEqual(client.payloads[0]["timeout_seconds"], client.config.health_timeout_seconds)

    def test_request_json_bypasses_proxy_environment(self) -> None:
        class FakeResponse:
            def __enter__(self) -> "FakeResponse":
                return self

            def __exit__(self, *args: object) -> None:
                return None

            def read(self) -> bytes:
                return b'{"status":"ok"}'

        class FakeOpener:
            def __init__(self) -> None:
                self.timeout_seconds: float | None = None

            def open(self, request: object, *, timeout: float) -> FakeResponse:
                self.timeout_seconds = timeout
                return FakeResponse()

        opener = FakeOpener()
        client = SandcodeMobileHostClient(SandcodeConfig(token="sandcode-local"))

        with (
            patch("stacky.sandcode.urllib.request.ProxyHandler") as proxy_handler,
            patch("stacky.sandcode.urllib.request.build_opener", return_value=opener) as build_opener,
        ):
            proxy_handler.return_value = "empty-proxy-handler"
            data = client._request_json("GET", "/api/health", None, timeout_seconds=0.7)

        self.assertEqual(data["status"], "ok")
        proxy_handler.assert_called_once_with({})
        build_opener.assert_called_once_with("empty-proxy-handler")
        self.assertEqual(opener.timeout_seconds, 0.7)

    async def test_chat_only_is_sent_to_session_and_message(self) -> None:
        client = FakeSandcodeClient()
        session = await client.start_session(Path("C:/project"), chat_only=True)
        await client.send_user_message(session, "forklar uden tools")

        payload = client.payloads[0]["payload"]
        self.assertIsInstance(payload, dict)
        self.assertTrue(payload["chatOnly"])
        self.assertTrue(client.ws_messages[0]["chatOnly"])

    def test_parse_sandcode_action_requires_explicit_sandcode(self) -> None:
        self.assertIsNone(parse_sandcode_action("lav en fil på skrivebordet"))
        action = parse_sandcode_action("brug sand code til at rette testen")
        self.assertIsNotNone(action)
        assert action is not None
        self.assertEqual(action.prompt, "rette testen")

    def test_parse_sandcode_action_accepts_agent_aliases_without_vague_trigger(self) -> None:
        self.assertIsNone(parse_sandcode_action("agent skills halter stadig"))

        action = parse_sandcode_action("brug agenten til at fikse web search")
        self.assertIsNotNone(action)
        assert action is not None
        self.assertEqual(action.prompt, "fikse web search")

        codex_action = parse_sandcode_action("codex agent skal rette testen")
        self.assertIsNotNone(codex_action)
        assert codex_action is not None
        self.assertEqual(codex_action.prompt, "rette testen")

    def test_parse_sandcode_action_accepts_agent_cancel_alias(self) -> None:
        action = parse_sandcode_action("stop agenten")

        self.assertIsNotNone(action)
        assert action is not None
        self.assertEqual(action.prompt, "__cancel__")

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
