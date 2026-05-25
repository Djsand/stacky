from __future__ import annotations

import json
import threading
import unittest

from stacky.body.controller import BodyPresence, StackChanBodyController


class FakeController:
    def __init__(self) -> None:
        self.expressions: list[str] = []
        self.looks: list[tuple[float, float, int]] = []
        self.gestures: list[str] = []

    def set_expression(self, name: str) -> bool:
        self.expressions.append(name)
        return True

    def look_at(self, x: float, y: float, *, speed: int = 500) -> bool:
        self.looks.append((x, y, speed))
        return True

    def gesture(self, name: str, *, intensity: float = 1.0, speed: int = 500) -> bool:
        self.gestures.append(name)
        return True


class BodyPresenceTest(unittest.TestCase):
    def test_presence_sets_expression_when_controller_exists(self) -> None:
        fake = FakeController()
        presence = BodyPresence(fake)  # type: ignore[arg-type]

        presence.set("thinking")

        self.assertEqual(fake.expressions, ["thinking"])

    def test_presence_noops_without_controller(self) -> None:
        presence = BodyPresence(None)
        presence.set("thinking")


class BodyControllerRawAudioTest(unittest.TestCase):
    def test_controller_sends_look_at_command(self) -> None:
        sent = []
        controller = StackChanBodyController()

        def send(command) -> bool:
            sent.append(command)
            return True

        controller.send = send  # type: ignore[method-assign]

        self.assertTrue(controller.look_at(0.25, -0.5, speed=600))

        self.assertEqual(sent[0].type, "body.look_at")
        self.assertEqual(sent[0].payload["x"], 0.25)
        self.assertEqual(sent[0].payload["y"], -0.5)
        self.assertEqual(sent[0].payload["speed"], 600)

    def test_controller_sends_gesture_command(self) -> None:
        sent = []
        controller = StackChanBodyController()

        def send(command) -> bool:
            sent.append(command)
            return True

        controller.send = send  # type: ignore[method-assign]

        self.assertTrue(controller.gesture("nod"))

        self.assertEqual(sent[0].type, "body.gesture")
        self.assertEqual(sent[0].payload["name"], "nod")

    def test_controller_sends_gesture_with_face_lock_base(self) -> None:
        sent = []
        controller = StackChanBodyController()

        def send(command) -> bool:
            sent.append(command)
            return True

        controller.send = send  # type: ignore[method-assign]

        self.assertTrue(controller.gesture("nod", base_x=0.31, base_y=-0.19))

        self.assertEqual(sent[0].payload["baseX"], 0.31)
        self.assertEqual(sent[0].payload["baseY"], -0.19)

    def test_controller_sends_leds_command(self) -> None:
        sent = []
        controller = StackChanBodyController()

        def send(command) -> bool:
            sent.append(command)
            return True

        controller.send = send  # type: ignore[method-assign]

        self.assertTrue(controller.set_leds(r=12, g=34, b=56, brightness=0.4))

        self.assertEqual(sent[0].type, "body.leds")
        self.assertEqual(sent[0].payload["r"], 12)
        self.assertEqual(sent[0].payload["brightness"], 0.4)

    def test_controller_sends_motion_config_command(self) -> None:
        sent = []
        controller = StackChanBodyController()

        def send(command) -> bool:
            sent.append(command)
            return True

        controller.send = send  # type: ignore[method-assign]

        self.assertTrue(controller.configure_motion(center_yaw=120, center_pitch=250))

        self.assertEqual(sent[0].type, "body.motion_config")
        self.assertEqual(sent[0].payload["centerYaw"], 120)
        self.assertEqual(sent[0].payload["centerPitch"], 250)

    def test_controller_sends_mic_gain_command(self) -> None:
        sent = []
        controller = StackChanBodyController()

        def send(command) -> bool:
            sent.append(command)
            return True

        controller.send = send  # type: ignore[method-assign]

        self.assertTrue(controller.set_mic_gain(75))

        self.assertEqual(sent[0].type, "audio.input_gain")
        self.assertEqual(sent[0].payload["level"], 75)

    def test_controller_sends_display_brightness_command(self) -> None:
        sent = []
        controller = StackChanBodyController()

        def send(command) -> bool:
            sent.append(command)
            return True

        controller.send = send  # type: ignore[method-assign]

        self.assertTrue(controller.set_display_brightness(35))

        self.assertEqual(sent[0].type, "display.brightness")
        self.assertEqual(sent[0].payload["level"], 35)
        self.assertTrue(sent[0].payload["permanent"])

    def test_controller_sends_status_request_command(self) -> None:
        sent = []
        controller = StackChanBodyController()

        def send(command) -> bool:
            sent.append(command)
            return True

        controller.send = send  # type: ignore[method-assign]

        self.assertTrue(controller.request_status())

        self.assertEqual(sent[0].type, "body.status")

    def test_controller_sends_i2c_scan_request_command(self) -> None:
        sent = []
        controller = StackChanBodyController()

        def send(command) -> bool:
            sent.append(command)
            return True

        controller.send = send  # type: ignore[method-assign]

        self.assertTrue(controller.request_i2c_scan())

        self.assertEqual(sent[0].type, "body.i2c_scan")

    def test_controller_interrupt_audio_sends_stop_and_releases_waiters(self) -> None:
        sent = []
        controller = StackChanBodyController()

        def send(command) -> bool:
            sent.append(command)
            return True

        controller.send = send  # type: ignore[method-assign]
        before = controller._audio_done_generation  # type: ignore[attr-defined]

        self.assertTrue(controller.interrupt_audio())

        self.assertEqual(sent[0].type, "audio.stop")
        self.assertGreater(controller._audio_done_generation, before)  # type: ignore[attr-defined]

    def test_raw_audio_header_and_body_are_sent_atomically(self) -> None:
        class BlockingSocket:
            def __init__(self) -> None:
                self.sent: list[bytes] = []
                self.first_send_started = threading.Event()
                self.release_first_send = threading.Event()
                self.lock = threading.Lock()

            def sendall(self, data: bytes) -> None:
                with self.lock:
                    self.sent.append(data)
                    is_first = len(self.sent) == 1
                if is_first:
                    self.first_send_started.set()
                    self.release_first_send.wait(timeout=1.0)

        fake_socket = BlockingSocket()
        controller = StackChanBodyController()
        controller._client = fake_socket  # type: ignore[attr-defined]

        audio_thread = threading.Thread(
            target=lambda: controller._send_audio_raw_chunk(b"\x01\x02\x03\x04", seq=7),  # type: ignore[attr-defined]
        )
        audio_thread.start()
        self.assertTrue(fake_socket.first_send_started.wait(timeout=1.0))

        body_thread = threading.Thread(target=lambda: controller.set_leds(r=1, g=2, b=3))
        body_thread.start()
        fake_socket.release_first_send.set()
        audio_thread.join(timeout=1.0)
        body_thread.join(timeout=1.0)

        self.assertFalse(audio_thread.is_alive())
        self.assertFalse(body_thread.is_alive())
        self.assertIn(b'"type":"audio.raw"', fake_socket.sent[0])
        self.assertEqual(fake_socket.sent[1], b"\x01\x02\x03\x04")
        self.assertIn(b'"type":"body.leds"', fake_socket.sent[2])

    def test_controller_sends_vision_capture_command(self) -> None:
        sent = []
        controller = StackChanBodyController()

        def send(command) -> bool:
            sent.append(command)
            return True

        controller.send = send  # type: ignore[method-assign]

        self.assertTrue(controller.capture_vision_frame(width=320, height=240, quality=35))

        self.assertEqual(sent[0].type, "vision.capture")
        self.assertEqual(sent[0].payload["width"], 320)
        self.assertEqual(sent[0].payload["height"], 240)
        self.assertEqual(sent[0].payload["quality"], 35)
        self.assertEqual(sent[0].payload["aeLevel"], 2)

    def test_processes_raw_audio_in_header_and_binary_body(self) -> None:
        events = []
        controller = StackChanBodyController(on_event=events.append)
        header = json.dumps(
            {
                "type": "audio.in",
                "payload": {
                    "encoding": "pcm16le",
                    "transport": "raw",
                    "sampleRate": 16000,
                    "channels": 1,
                    "seq": 7,
                    "bytes": 4,
                },
                "ts": 1.5,
            },
            separators=(",", ":"),
        ).encode("utf-8")

        controller._buffer = header + b"\n\x01\x02\x03\x04"  # type: ignore[attr-defined]
        controller._process_buffered_events()  # type: ignore[attr-defined]

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].type, "audio.in")
        self.assertEqual(events[0].payload["pcm"], b"\x01\x02\x03\x04")
        self.assertEqual(events[0].payload["sampleRate"], 16000)

    def test_waits_for_full_raw_audio_body(self) -> None:
        events = []
        controller = StackChanBodyController(on_event=events.append)
        controller._buffer = (  # type: ignore[attr-defined]
            b'{"type":"audio.in","payload":{"encoding":"pcm16le","transport":"raw","bytes":4},"ts":1}\n'
            b"\x01\x02"
        )

        controller._process_buffered_events()  # type: ignore[attr-defined]
        self.assertEqual(events, [])

        controller._buffer += b"\x03\x04"  # type: ignore[attr-defined]
        controller._process_buffered_events()  # type: ignore[attr-defined]
        self.assertEqual(events[0].payload["pcm"], b"\x01\x02\x03\x04")


if __name__ == "__main__":
    unittest.main()
