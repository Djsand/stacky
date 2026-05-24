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
    create_stackchan_supertonic_output,
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
        self.volume_levels: list[int] = []
        self.interrupts = 0

    def stop_audio(self) -> bool:
        return True

    def interrupt_audio(self) -> bool:
        self.interrupts += 1
        return True

    def hold_audio(self, hold: bool) -> bool:
        self.hold_states.append(hold)
        return True

    def speak_audio_chunks(self, pcm: bytes, **kwargs: object) -> bool:
        self.audio_calls.append(pcm)
        return True

    def set_volume(self, level: int) -> bool:
        self.volume_levels.append(level)
        return True


class FakeWavTTS:
    def __init__(self, sample_rate: int = 24_000) -> None:
        self.sample_rate = sample_rate

    def load(self) -> None:
        pass

    def synthesize_to_file(self, text: str, output_path: Path, **kwargs: object) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        samples = max(80, len(text) * 16)
        with wave.open(str(output_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.sample_rate)
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

    async def test_piper_output_normalizes_question_mark_before_logging_and_speech(self) -> None:
        output = PiperSpeechOutput(FakeTTS())
        seen: list[str] = []

        async def fake_speak_chunks(text: str, utterance_id: int) -> None:
            seen.append(text)

        output._speak_chunks = fake_speak_chunks  # type: ignore[method-assign]

        await output.speak("Har du haft en god dag?")
        await output.wait()

        self.assertEqual(seen, ["Har du haft en god dag spørgsmål"])

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
        self.assertEqual(output.rhythm_gap_seconds, 0.04)
        self.assertFalse(output.rhythmic_chunks)
        self.assertEqual(output.target_active_rms, 9000)
        self.assertEqual(output.max_gain, 4.0)
        self.assertEqual(output.volume_level, 80)

    def test_stackchan_supertonic_output_uses_rhythmic_chunks(self) -> None:
        output = create_stackchan_supertonic_output(FakeController())  # type: ignore[arg-type]

        self.assertEqual(output.chunk_chars, 200)
        self.assertLessEqual(output.rhythm_gap_seconds, 0.04)
        self.assertTrue(output.rhythmic_chunks)

    def test_stackchan_output_can_change_volume(self) -> None:
        controller = FakeController()
        output = StackChanSpeechOutput(FakeTTS(), controller)  # type: ignore[arg-type]

        self.assertTrue(output.set_volume(35))

        self.assertEqual(output.volume_level, 35)
        self.assertEqual(controller.volume_levels, [35])
        self.assertEqual(output.target_active_rms, 4950)
        self.assertAlmostEqual(output.max_gain, 3.4)

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

    async def test_stackchan_output_normalizes_question_mark_before_speech(self) -> None:
        controller = FakeController()
        output = StackChanSpeechOutput(FakeTTS(), controller)  # type: ignore[arg-type]
        seen: list[str] = []

        async def fake_speak_chunks(text: str, utterance_id: int) -> None:
            seen.append(text)

        output._speak_chunks = fake_speak_chunks  # type: ignore[method-assign]

        await output.speak("Hvad tænker du?")
        await output.wait()

        self.assertEqual(seen, ["Hvad tænker du spørgsmål"])

    async def test_stackchan_wait_absorbs_playback_task_error(self) -> None:
        output = StackChanSpeechOutput(FakeTTS(), FakeController())  # type: ignore[arg-type]

        async def fail_speak_chunks(text: str, utterance_id: int) -> None:
            raise RuntimeError("conversion failed")

        output._speak_chunks = fail_speak_chunks  # type: ignore[method-assign]

        await output.speak("Hej")
        await output.wait()

        self.assertIsNone(output._task)

    async def test_stackchan_stop_absorbs_failed_previous_playback_task(self) -> None:
        output = StackChanSpeechOutput(FakeTTS(), FakeController())  # type: ignore[arg-type]

        async def fail_speak_chunks(text: str, utterance_id: int) -> None:
            raise RuntimeError("conversion failed")

        async def ok_speak_chunks(text: str, utterance_id: int) -> None:
            return None

        output._speak_chunks = fail_speak_chunks  # type: ignore[method-assign]
        await output.speak("Hej")
        await asyncio.sleep(0)
        output._speak_chunks = ok_speak_chunks  # type: ignore[method-assign]

        await output.speak("Igen")
        await output.wait()

        self.assertIsNone(output._task)

    def test_stackchan_wav_conversion_resamples_without_ffmpeg_for_pcm16(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "speech.wav"
            FakeWavTTS(sample_rate=44_100).synthesize_to_file("hej", path)
            output = StackChanSpeechOutput(FakeWavTTS(), FakeController())  # type: ignore[arg-type]

            converted = output._prepare_stackchan_wav(path)

            self.assertEqual(converted, path.with_suffix(".stackchan.wav"))
            self.assertGreater(converted.stat().st_size, 44)
            with wave.open(str(converted), "rb") as wav_file:
                self.assertEqual(wav_file.getnchannels(), 1)
                self.assertEqual(wav_file.getsampwidth(), 2)
                self.assertEqual(wav_file.getframerate(), 24_000)

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
