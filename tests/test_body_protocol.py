from __future__ import annotations

import unittest

from stacky.body.protocol import (
    BodyEvent,
    audio_chunk,
    audio_end,
    audio_start,
    body_status,
    decode_pcm_payload,
    display_brightness,
    expression,
    gesture,
    hold_audio,
    look_at,
    mobility_intent,
    mic_input_gain,
    motion_config,
    speak_audio,
    speaker_volume,
    speaker_tone,
    stop_audio,
    vision_capture,
)


class BodyProtocolTest(unittest.TestCase):
    def test_expression_command_encodes(self) -> None:
        raw = expression("listening").to_json()
        self.assertIn("body.set_expression", raw)
        self.assertIn("listening", raw)

    def test_look_at_command_clamps_normalized_coordinates(self) -> None:
        command = look_at(2.0, -2.0, speed=1200)
        raw = command.to_json()

        self.assertIn("body.look_at", raw)
        self.assertEqual(command.payload["x"], 1.0)
        self.assertEqual(command.payload["y"], -1.0)
        self.assertEqual(command.payload["speed"], 1000)

    def test_gesture_command_encodes(self) -> None:
        command = gesture("nod", intensity=2.0, speed=-1)
        raw = command.to_json()

        self.assertIn("body.gesture", raw)
        self.assertEqual(command.payload["intensity"], 1.0)
        self.assertEqual(command.payload["speed"], 0)

    def test_motion_config_command_clamps(self) -> None:
        command = motion_config(
            center_yaw=2000,
            center_pitch=-100,
            yaw_range=5000,
            look_up_range=900,
            look_down_range=-10,
        )
        raw = command.to_json()

        self.assertIn("body.motion_config", raw)
        self.assertEqual(command.payload["centerYaw"], 1280)
        self.assertEqual(command.payload["centerPitch"], 30)
        self.assertEqual(command.payload["yawRange"], 1280)
        self.assertEqual(command.payload["lookUpRange"], 870)
        self.assertEqual(command.payload["lookDownRange"], 0)

    def test_audio_command_encodes_pcm(self) -> None:
        raw = speak_audio(b"\x00\x01").to_json()
        self.assertIn("audio.out", raw)
        self.assertIn("AAE=", raw)

    def test_stop_audio_command_encodes(self) -> None:
        raw = stop_audio().to_json()
        self.assertIn("audio.stop", raw)

    def test_hold_audio_command_encodes(self) -> None:
        raw = hold_audio(active=True).to_json()
        self.assertIn("audio.hold", raw)
        self.assertIn('"active":true', raw)

    def test_speaker_tone_command_encodes(self) -> None:
        raw = speaker_tone(frequency=660, duration_ms=120).to_json()
        self.assertIn("audio.tone", raw)
        self.assertIn('"frequency":660', raw)

    def test_speaker_volume_command_encodes(self) -> None:
        raw = speaker_volume(140).to_json()
        self.assertIn("audio.volume", raw)
        self.assertIn('"level":100', raw)

    def test_mic_input_gain_command_encodes(self) -> None:
        raw = mic_input_gain(140).to_json()
        self.assertIn("audio.input_gain", raw)
        self.assertIn('"level":100', raw)

    def test_display_brightness_command_encodes(self) -> None:
        command = display_brightness(0, permanent=False)
        raw = command.to_json()

        self.assertIn("display.brightness", raw)
        self.assertEqual(command.payload["level"], 1)
        self.assertFalse(command.payload["permanent"])

    def test_body_status_command_encodes(self) -> None:
        raw = body_status().to_json()

        self.assertIn("body.status", raw)

    def test_vision_capture_command_encodes(self) -> None:
        command = vision_capture(width=20, height=900, format="jpeg")
        raw = command.to_json()

        self.assertIn("vision.capture", raw)
        self.assertEqual(command.payload["width"], 64)
        self.assertEqual(command.payload["height"], 720)
        self.assertEqual(command.payload["format"], "jpeg")

    def test_audio_chunk_protocol_encodes(self) -> None:
        start = audio_start(sample_rate=44100, total_bytes=2048).to_json()
        self.assertIn("audio.start", start)
        self.assertIn('"totalBytes":2048', start)
        self.assertIn("audio.chunk", audio_chunk(b"\x00\x01", seq=3).to_json())
        self.assertIn("audio.end", audio_end().to_json())

    def test_audio_payload_decodes_pcm(self) -> None:
        command = speak_audio(b"\x00\x01", sample_rate=16000)

        pcm, sample_rate, channels = decode_pcm_payload(command.payload)

        self.assertEqual(pcm, b"\x00\x01")
        self.assertEqual(sample_rate, 16000)
        self.assertEqual(channels, 1)

    def test_audio_payload_decodes_raw_pcm(self) -> None:
        pcm, sample_rate, channels = decode_pcm_payload(
            {
                "encoding": "pcm16le",
                "sampleRate": 16000,
                "channels": 1,
                "pcm": b"\x02\x03",
            }
        )

        self.assertEqual(pcm, b"\x02\x03")
        self.assertEqual(sample_rate, 16000)
        self.assertEqual(channels, 1)

    def test_mobility_is_disabled_by_default(self) -> None:
        raw = mobility_intent("forward", speed=0.2).to_json()
        self.assertIn('"enabled":false', raw)

    def test_event_decodes(self) -> None:
        event = BodyEvent.from_json('{"type":"touch","payload":{"zone":"screen"},"ts":1.0}')
        self.assertEqual(event.type, "touch")
        self.assertEqual(event.payload["zone"], "screen")

    def test_audio_play_done_event_decodes(self) -> None:
        event = BodyEvent.from_json('{"type":"audio.play_done","payload":{"reason":"finished"},"ts":1.0}')
        self.assertEqual(event.type, "audio.play_done")
        self.assertEqual(event.payload["reason"], "finished")


if __name__ == "__main__":
    unittest.main()
