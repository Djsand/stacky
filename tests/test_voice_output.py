from __future__ import annotations

import asyncio
import tempfile
import unittest
import wave
from pathlib import Path

from stacky.voice.output import (
    PiperSpeechOutput,
    StackChanSpeechOutput,
    boost_pcm16_for_stackchan,
    join_pcm16_chunks,
    split_pcm16_segments,
)


class FakeTTS:
    def load(self) -> None:
        pass

    def synthesize_to_file(self, text: str, output_path: Path, **kwargs: object) -> Path:
        output_path.write_bytes(b"RIFF")
        return output_path


class FakeController:
    def __init__(self) -> None:
        self.audio_calls: list[bytes] = []
        self.hold_states: list[bool] = []

    def stop_audio(self) -> bool:
        return True

    def hold_audio(self, hold: bool) -> bool:
        self.hold_states.append(hold)
        return True

    def speak_audio_chunks(self, pcm: bytes, **kwargs: object) -> bool:
        self.audio_calls.append(pcm)
        return True


class FakeWavTTS:
    def load(self) -> None:
        pass

    def synthesize_to_file(self, text: str, output_path: Path, **kwargs: object) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sample_rate = 24_000
        samples = max(80, len(text) * 16)
        with wave.open(str(output_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(b"".join(int(1200).to_bytes(2, "little", signed=True) for _ in range(samples)))
        return output_path


class VoiceOutputTest(unittest.IsolatedAsyncioTestCase):
    async def test_speak_returns_while_playback_task_runs(self) -> None:
        output = PiperSpeechOutput(FakeTTS())
        started = asyncio.Event()
        released = asyncio.Event()

        async def fake_speak_chunks(text: str, utterance_id: int) -> None:
            started.set()
            await released.wait()

        output._speak_chunks = fake_speak_chunks  # type: ignore[method-assign]

        await output.speak("Hej Nicolai.")
        await asyncio.wait_for(started.wait(), timeout=1)

        self.assertIsNotNone(output._task)
        released.set()
        await output.wait()
        self.assertIsNone(output._task)

    async def test_stop_cancels_playback_task(self) -> None:
        output = PiperSpeechOutput(FakeTTS())
        started = asyncio.Event()
        released = asyncio.Event()

        async def fake_speak_chunks(text: str, utterance_id: int) -> None:
            started.set()
            await released.wait()

        output._speak_chunks = fake_speak_chunks  # type: ignore[method-assign]

        await output.speak("Hej Nicolai.")
        await asyncio.wait_for(started.wait(), timeout=1)

        await output.stop()
        self.assertIsNone(output._task)
        released.set()

    def test_boost_pcm16_for_stackchan_raises_soft_speech_level(self) -> None:
        pcm = b"".join(int(1000).to_bytes(2, "little", signed=True) for _ in range(20))

        boosted = boost_pcm16_for_stackchan(pcm, target_active_rms=4000, max_gain=4.0)
        first = int.from_bytes(boosted[:2], "little", signed=True)

        self.assertEqual(first, 4000)

    def test_boost_pcm16_for_stackchan_clips_safely(self) -> None:
        pcm = int(12000).to_bytes(2, "little", signed=True)

        boosted = boost_pcm16_for_stackchan(pcm, target_active_rms=60000, max_gain=4.0)
        first = int.from_bytes(boosted[:2], "little", signed=True)

        self.assertEqual(first, 32767)

    def test_stackchan_output_uses_bounded_text_chunks(self) -> None:
        output = StackChanSpeechOutput(FakeTTS(), FakeController())  # type: ignore[arg-type]

        self.assertLessEqual(output.chunk_chars, 300)
        self.assertGreaterEqual(output.max_stackchan_pcm_bytes, 1_000_000)

    async def test_stackchan_output_combines_tts_chunks_before_sending(self) -> None:
        controller = FakeController()
        with tempfile.TemporaryDirectory() as temp_dir:
            output = StackChanSpeechOutput(
                FakeWavTTS(),
                controller,  # type: ignore[arg-type]
                output_dir=Path(temp_dir),
                chunk_chars=35,
                max_stackchan_pcm_bytes=200_000,
            )

            await output.speak("Nicolai, dette er en længere test. Den skal stadig sendes samlet.")
            await output.wait()

        self.assertEqual(len(controller.audio_calls), 1)
        self.assertEqual(controller.hold_states, [True, False])

    def test_join_pcm16_chunks_adds_short_aligned_gap(self) -> None:
        first = int(100).to_bytes(2, "little", signed=True)
        second = int(200).to_bytes(2, "little", signed=True)

        joined = join_pcm16_chunks([first, second], sample_rate=100, channels=1, gap_seconds=0.02)

        self.assertEqual(joined, first + b"\x00\x00\x00\x00" + second)

    def test_split_pcm16_segments_keeps_frame_alignment(self) -> None:
        pcm = b"".join(int(index).to_bytes(2, "little", signed=True) for index in range(10))

        segments = split_pcm16_segments(pcm, sample_rate=16000, channels=1, max_bytes=8)

        self.assertEqual([len(segment) for segment in segments], [8, 8, 4])
        self.assertEqual(b"".join(segments), pcm)


if __name__ == "__main__":
    unittest.main()
