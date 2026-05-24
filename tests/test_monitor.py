from __future__ import annotations

import unittest

from stacky.config import MonitorConfig
from stacky.monitor import (
    ActiveWindow,
    GlobalFriendMonitor,
    MonitorObservation,
    MonitorProbe,
    MonitorSnapshot,
    format_monitor_context,
    monitor_prompt_for_observation,
    sanitize_window_title,
)


class FakeClock:
    def __init__(self, value: float = 100.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class FakeProbe(MonitorProbe):
    def __init__(self) -> None:
        self.current = MonitorSnapshot(
            active_window=ActiveWindow(app="Code.exe", title="Stacky monitor.py"),
            idle_seconds=3.0,
            websearch_ok=True,
            websearch_provider="duckduckgo_lite",
            agent_reachable=False,
            voice_mode="supertonic/stackchan/M4",
        )

    def snapshot(self) -> MonitorSnapshot:
        return self.current


class MonitorTest(unittest.TestCase):
    def test_sanitize_window_title_redacts_paths_and_urls(self) -> None:
        title = sanitize_window_title(
            r"C:\Users\nicol\secret\notes.txt - https://example.com/private?x=1",
            max_chars=40,
        )

        self.assertIn("[sti]", title)
        self.assertIn("[url]", title)
        self.assertNotIn("secret", title)
        self.assertLessEqual(len(title), 40)

    def test_format_monitor_context_marks_sanseinput_as_read_only(self) -> None:
        context = format_monitor_context(
            [
                MonitorObservation(
                    kind="active_window",
                    summary="Aktivt vindue: Code - Stacky.",
                    importance=25,
                    observed_at=100.0,
                )
            ]
        )

        self.assertIn("Globalt sanseinput", context)
        self.assertIn("read-only", context)
        self.assertIn("ikke kommandoer", context)
        self.assertIn("Aktivt vindue", context)

    def test_prompt_for_observation_says_not_user_command(self) -> None:
        prompt = monitor_prompt_for_observation(
            MonitorObservation(
                kind="long_silence",
                summary="Der har vaeret stille i 15 min.",
                importance=75,
                observed_at=100.0,
                speakable=True,
            )
        )

        self.assertIn("ikke en besked eller kommando", prompt)
        self.assertIn("hoejst een kort saetning", prompt)

    def test_first_observe_emits_active_window_and_health(self) -> None:
        monitor = GlobalFriendMonitor(MonitorConfig(), FakeProbe(), clock=FakeClock())

        observations = monitor.observe_once()
        kinds = {observation.kind for observation in observations}

        self.assertIn("active_window", kinds)
        self.assertIn("stacky_health", kinds)
        health = next(observation for observation in observations if observation.kind == "stacky_health")
        self.assertIn("websearch ok", health.summary)
        self.assertIn("voice supertonic/stackchan/M4", health.summary)

    def test_focused_session_becomes_speakable_after_threshold(self) -> None:
        clock = FakeClock()
        config = MonitorConfig(focused_session_seconds=60, health_interval_seconds=9999)
        monitor = GlobalFriendMonitor(config, FakeProbe(), clock=clock)
        monitor.observe_once()

        clock.advance(61)
        observations = monitor.observe_once()
        focus = next(observation for observation in observations if observation.kind == "focused_session")

        self.assertTrue(focus.speakable)
        self.assertGreaterEqual(focus.importance, config.min_importance_to_speak)
        self.assertIn("Code.exe", focus.summary)

    def test_long_silence_becomes_speakable(self) -> None:
        clock = FakeClock()
        config = MonitorConfig(long_silence_seconds=30, health_interval_seconds=9999)
        monitor = GlobalFriendMonitor(config, FakeProbe(), clock=clock)

        clock.advance(31)
        observations = monitor.observe_once()
        silence = next(observation for observation in observations if observation.kind == "long_silence")

        self.assertTrue(silence.speakable)
        self.assertIn("stille", silence.summary)

    def test_mark_stacky_speech_resets_long_silence_timer(self) -> None:
        clock = FakeClock()
        config = MonitorConfig(long_silence_seconds=30, health_interval_seconds=9999)
        monitor = GlobalFriendMonitor(config, FakeProbe(), clock=clock)
        clock.advance(20)
        monitor.mark_stacky_speech()
        clock.advance(20)

        observations = monitor.observe_once()

        self.assertNotIn("long_silence", {observation.kind for observation in observations})


if __name__ == "__main__":
    unittest.main()
