from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from stacky.voice.stt import write_pcm_wav
from stacky.voice.stt_eval import (
    STTDatasetItem,
    apply_references,
    char_error_rate,
    load_capture_phrases,
    load_dataset_manifest,
    resolve_audio_inputs,
    word_error_rate,
    write_dataset_record,
)


class STTEvalTest(unittest.TestCase):
    def test_load_capture_phrases_uses_arguments_file_and_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            phrases = Path(tmp) / "phrases.txt"
            phrases.write_text("# skip\nanden sætning\n", encoding="utf-8")

            result = load_capture_phrases(
                phrase_args=["første sætning"],
                phrases_file=phrases,
                limit=1,
            )

        self.assertEqual(result, ["første sætning"])

    def test_write_and_load_jsonl_manifest_resolves_relative_audio(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = write_pcm_wav(root / "clip.wav", b"\x00\x00" * 160, sample_rate=16000)
            manifest = root / "manifest.jsonl"

            write_dataset_record(
                manifest,
                audio_path=audio,
                expected_text="hej stacky",
                item_id="clip-1",
                sample_rate=16000,
                channels=1,
                duration_seconds=0.01,
                rms=0,
                peak=0,
                quality={"speechLike": True},
                speech_style="mumble",
            )

            items = load_dataset_manifest(manifest)

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].audio_path.name, "clip.wav")
        self.assertEqual(items[0].expected_text, "hej stacky")
        self.assertEqual(items[0].item_id, "clip-1")
        self.assertEqual(items[0].speech_style, "mumble")

    def test_resolve_audio_inputs_sorts_recent_first_and_limits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = write_pcm_wav(root / "old.wav", b"\x00\x00" * 160, sample_rate=16000)
            new = write_pcm_wav(root / "new.wav", b"\x01\x00" * 160, sample_rate=16000)
            os.utime(old, (1000, 1000))
            os.utime(new, (2000, 2000))

            paths = resolve_audio_inputs([str(root)], default_pattern="", limit=1)

        self.assertEqual(paths, [new.resolve()])

    def test_resolve_audio_inputs_limit_zero_returns_all(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = write_pcm_wav(root / "first.wav", b"\x00\x00" * 160, sample_rate=16000)
            second = write_pcm_wav(root / "second.wav", b"\x01\x00" * 160, sample_rate=16000)

            paths = resolve_audio_inputs([str(root)], default_pattern="", limit=0)

        self.assertEqual(set(paths), {first.resolve(), second.resolve()})

    def test_apply_references_accepts_empty_reference_for_noise_clip(self) -> None:
        item = STTDatasetItem(Path("noise.wav"), expected_text=None)

        result = apply_references([item], {"noise.wav": ""})

        self.assertEqual(result[0].expected_text, "")

    def test_error_rates(self) -> None:
        self.assertEqual(word_error_rate("hej med dig", "hej med dig"), 0.0)
        self.assertAlmostEqual(word_error_rate("hej med dig", "hej dig"), 1 / 3)
        self.assertGreater(char_error_rate("stacky", "stakki"), 0.0)


if __name__ == "__main__":
    unittest.main()
