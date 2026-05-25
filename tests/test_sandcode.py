from __future__ import annotations

import asyncio
import unittest
from pathlib import Path
from unittest.mock import patch

from stacky.cli import _run_sandcode_with_updates
from stacky.config import SandcodeConfig
from stacky.llm import ChatMessage
from stacky.sandcode import (
    DEFAULT_SANDCODE_AGENT_PROMPT,
    SandcodeDanishSummarizer,
    SandcodeMobileHostClient,
    SandcodeSession,
    classify_sandcode_action,
    parse_sandcode_action,
)


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


class SlowFakeSandcodeClient(FakeSandcodeClient):
    async def run_session(self, cwd: Path, prompt: str, on_event, *, chat_only: bool = False) -> SandcodeSession:  # type: ignore[no-untyped-def]
        await asyncio.sleep(0.035)
        return SandcodeSession(
            session_id="slow-test",
            cwd=cwd,
            provider=self.config.provider,
            model=self.config.model,
            permission_mode=self.config.permission_mode,
            effort=self.config.effort,
            chat_only=chat_only,
        )


class FakeIntentBrain:
    def __init__(self, response: str) -> None:
        self.response = response
        self.messages: list[list[ChatMessage]] = []

    async def chat(self, messages: list[ChatMessage], *, temperature: float = 0.4, max_tokens: int | None = None) -> str:
        self.messages.append(messages)
        return self.response


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
        self.assertEqual(action.mode, "work")

    def test_parse_sandcode_action_accepts_agent_aliases_without_vague_trigger(self) -> None:
        self.assertIsNone(parse_sandcode_action("agent skills halter stadig"))
        self.assertIsNone(parse_sandcode_action("hvad med sandcode som ide"))

        action = parse_sandcode_action("brug agenten til at fikse web search")
        self.assertIsNotNone(action)
        assert action is not None
        self.assertEqual(action.prompt, "fikse web search")

        codex_action = parse_sandcode_action("codex agent skal rette testen")
        self.assertIsNotNone(codex_action)
        assert codex_action is not None
        self.assertEqual(codex_action.prompt, "rette testen")

    def test_parse_sandcode_action_defaults_empty_start_to_read_only_status(self) -> None:
        for text in ("start agenten", "saet agenten i gang", "sandcode start"):
            with self.subTest(text=text):
                action = parse_sandcode_action(text)
                self.assertIsNotNone(action)
                assert action is not None
                self.assertEqual(action.prompt, DEFAULT_SANDCODE_AGENT_PROMPT)
                self.assertEqual(action.mode, "read_only")

    def test_parse_sandcode_action_cleans_agent_task_prefix(self) -> None:
        action = parse_sandcode_action("kan du faa agenten til at kigge projektet igennem")

        self.assertIsNotNone(action)
        assert action is not None
        self.assertEqual(action.prompt, "kigge projektet igennem")

    def test_parse_sandcode_action_handles_stt_sancodi_mishearing(self) -> None:
        action = parse_sandcode_action("jeg bare at se om sancodigennem den virker")

        self.assertIsNotNone(action)
        assert action is not None
        self.assertEqual(action.prompt, DEFAULT_SANDCODE_AGENT_PROMPT)

    def test_parse_sandcode_action_accepts_agent_cancel_alias(self) -> None:
        action = parse_sandcode_action("stop agenten")

        self.assertIsNotNone(action)
        assert action is not None
        self.assertEqual(action.prompt, "__cancel__")

    async def test_agentic_router_routes_natural_agent_start(self) -> None:
        brain = FakeIntentBrain('{"sandcode_action":"start","prompt":"kig projektet igennem read-only","chat_only":false}')

        action = await classify_sandcode_action("nej jeg mener agenten du kan saet igang", brain)

        self.assertIsNotNone(action)
        assert action is not None
        self.assertEqual(action.prompt, "kig projektet igennem read-only")
        self.assertTrue(brain.messages)

    async def test_agentic_router_ignores_plain_agent_talk(self) -> None:
        brain = FakeIntentBrain('{"sandcode_action":"start","prompt":"forkert","chat_only":false}')

        action = await classify_sandcode_action("hvordan ser du ud med agenten", brain)

        self.assertIsNone(action)
        self.assertFalse(brain.messages)

    async def test_agentic_router_defaults_empty_prompt_to_read_only_status(self) -> None:
        brain = FakeIntentBrain('{"sandcode_action":"start","prompt":"","chat_only":false}')

        action = await classify_sandcode_action("nej jeg mener agenten du kan saet igang", brain)

        self.assertIsNotNone(action)
        assert action is not None
        self.assertEqual(action.prompt, DEFAULT_SANDCODE_AGENT_PROMPT)
        self.assertEqual(action.mode, "read_only")

    async def test_agentic_router_accepts_work_mode_from_brain(self) -> None:
        brain = FakeIntentBrain('{"sandcode_action":"start","prompt":"ret testen","mode":"work","chat_only":false}')

        action = await classify_sandcode_action("nej jeg mener agenten du kan saet igang", brain)

        self.assertIsNotNone(action)
        assert action is not None
        self.assertEqual(action.prompt, "ret testen")
        self.assertEqual(action.mode, "work")

    async def test_agentic_router_uses_recent_agent_context_for_start_den_followup(self) -> None:
        brain = FakeIntentBrain('{"sandcode_action":"none","prompt":"","chat_only":false}')

        action = await classify_sandcode_action(
            "som proever at starte den",
            brain,
            recent_context="Stacky: Jeg startede ikke Sandcode-agenten der.",
        )

        self.assertIsNotNone(action)
        assert action is not None
        self.assertEqual(action.prompt, DEFAULT_SANDCODE_AGENT_PROMPT)
        self.assertFalse(brain.messages)

    async def test_agentic_router_does_not_route_plain_followup_from_agent_context(self) -> None:
        brain = FakeIntentBrain('{"sandcode_action":"start","prompt":"forkert","chat_only":false}')

        action = await classify_sandcode_action(
            "bestemmer du",
            brain,
            recent_context="Stacky: Jeg startede ikke Sandcode-agenten der.",
        )

        self.assertIsNone(action)
        self.assertFalse(brain.messages)

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
        self.assertIn("Agenten arbejder", spoken)
        self.assertNotIn("Sandcode bruger", spoken)

    def test_summarizer_builds_heartbeat_status(self) -> None:
        spoken = SandcodeDanishSummarizer().summarize_heartbeat(
            elapsed_seconds=75,
            last_update="Agenten arbejder med Read.",
        )

        self.assertIn("Agenten arbejder stadig", spoken)
        self.assertIn("1 min", spoken)
        self.assertIn("Sidste livstegn", spoken)

    async def test_run_sandcode_with_updates_emits_heartbeat_when_silent(self) -> None:
        updates: list[str] = []

        async def on_update(update: str) -> None:
            updates.append(update)

        session = await _run_sandcode_with_updates(
            SlowFakeSandcodeClient(),
            Path("C:/project"),
            "ret testen",
            on_update=on_update,
            heartbeat_seconds=0.01,
        )

        self.assertEqual(session.session_id, "slow-test")
        self.assertTrue(any("Agenten arbejder stadig" in update for update in updates))


if __name__ == "__main__":
    unittest.main()
