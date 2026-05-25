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

    def test_status_reply_reports_running_agent_without_guessing(self) -> None:
        clock = FakeClock()
        state = RuntimeState(clock=clock)
        state.mark_sandcode_running("scan projektet", note="Agenten arbejder med Read.")
        clock.now += 12

        reply = state.status_reply("kører den")

        self.assertIn("Agenten kører", reply)
        self.assertIn("Sidste livstegn", reply)
        self.assertIn("Agenten arbejder med Read", reply)

    def test_status_reply_flags_stale_agent_as_possible_hang(self) -> None:
        clock = FakeClock()
        state = RuntimeState(clock=clock)
        state.mark_sandcode_running("scan projektet", note="Agenten arbejder med Read.")
        clock.now += 70

        reply = state.status_reply("hænger den stadig")

        self.assertIn("ikke fået nyt livstegn", reply)
        self.assertIn("Det kan være et hæng", reply)

    def test_status_reply_explains_wait_reason(self) -> None:
        clock = FakeClock()
        state = RuntimeState(clock=clock)
        state.mark_sandcode_starting("scan projektet")

        reply = state.status_reply("hvad er det den venter på")

        self.assertIn("venter på at Sandcode-sessionen kommer i gang", reply)

    def test_status_reply_reports_done_agent_session(self) -> None:
        clock = FakeClock()
        state = RuntimeState(clock=clock)
        state.mark_sandcode_done("scan projektet", session_id="abc123")

        reply = state.status_reply("kører den")

        self.assertIn("Agenten er færdig", reply)
        self.assertIn("abc123", reply)


    def test_status_reply_reports_cancelled_agent(self) -> None:
        clock = FakeClock()
        state = RuntimeState(clock=clock)
        state.mark_sandcode_cancelled("scan projektet", session_id="abc123")

        reply = state.status_reply("agent status")

        self.assertEqual(state.agent_status, "cancelled")
        self.assertIn("Agenten er stoppet", reply)


if __name__ == "__main__":
    unittest.main()
