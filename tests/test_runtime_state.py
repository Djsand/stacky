from __future__ import annotations

import unittest

from stacky.runtime_state import RuntimeState


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


class RuntimeStateTest(unittest.TestCase):
    def test_idle_context_is_explicit(self) -> None:
        state = RuntimeState(clock=FakeClock())

        context = state.context_for_prompt()

        self.assertIn("agent_status: idle", context)
        self.assertIn("last_action: none", context)
        self.assertIn("last_error: none", context)
        self.assertIn("can_speak_about: none", context)

    def test_sandcode_starting_is_speakable_truth(self) -> None:
        clock = FakeClock()
        state = RuntimeState(clock=clock)

        state.mark_sandcode_starting("scan projektet")
        context = state.context_for_prompt()

        self.assertEqual(state.agent_status, "starting")
        self.assertIn("agent_status: starting", context)
        self.assertIn("Sandcode-agent starter: scan projektet", context)
        self.assertIn("can_speak_about: sandcode_agent, runtime_action", context)

    def test_sandcode_done_records_session(self) -> None:
        clock = FakeClock()
        state = RuntimeState(clock=clock)

        state.mark_sandcode_done("scan projektet", session_id="abc123")
        context = state.context_for_prompt()

        self.assertEqual(state.agent_status, "done")
        self.assertIn("agent_status: done", context)
        self.assertIn("Sandcode-agent faerdig: scan projektet", context)
        self.assertIn("session abc123", context)

    def test_sandcode_failed_records_error(self) -> None:
        clock = FakeClock()
        state = RuntimeState(clock=clock)

        state.mark_sandcode_failed("scan projektet", "timeout")
        context = state.context_for_prompt()

        self.assertEqual(state.agent_status, "failed")
        self.assertEqual(state.last_error, "timeout")
        self.assertIn("agent_status: failed", context)
        self.assertIn("last_error: timeout", context)
        self.assertIn("Sandcode-agent fejlede: scan projektet", context)


if __name__ == "__main__":
    unittest.main()
