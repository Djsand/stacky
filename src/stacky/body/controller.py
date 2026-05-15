from __future__ import annotations

import socket
import threading
import time
import json
from collections.abc import Callable

from .protocol import (
    BodyCommand,
    BodyEvent,
    audio_chunk,
    audio_end,
    audio_start,
    expression,
    hold_audio,
    speak_audio,
    speaker_tone,
    stop_audio,
)


EventHandler = Callable[[BodyEvent], None]


class StackChanBodyController:
    """Small threaded TCP controller for the current StackChan firmware."""

    def __init__(
        self,
        *,
        host: str = "0.0.0.0",
        port: int = 8765,
        on_event: EventHandler | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.on_event = on_event
        self._stop = threading.Event()
        self._connected = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._server: socket.socket | None = None
        self._client: socket.socket | None = None
        self._client_address: tuple[str, int] | None = None
        self._buffer = b""
        self._pending_audio_in: dict[str, object] | None = None
        self._audio_ack_condition = threading.Condition()
        self._audio_acks: dict[int, bool] = {}
        self._audio_done_condition = threading.Condition()
        self._audio_done_generation = 0

    @property
    def connected(self) -> bool:
        return self._connected.is_set()

    @property
    def client_address(self) -> tuple[str, int] | None:
        return self._client_address

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="stacky-body-controller", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            client = self._client
            server = self._server
            self._client = None
            self._server = None
        for sock in (client, server):
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass
        if self._thread:
            self._thread.join(timeout=2.0)

    def wait_connected(self, timeout: float = 8.0) -> bool:
        return self._connected.wait(timeout)

    def set_expression(self, name: str, *, intensity: float = 1.0) -> bool:
        return self.send(expression(name, intensity=intensity))

    def speak_audio(self, pcm: bytes, *, sample_rate: int, channels: int = 1) -> bool:
        return self.send(speak_audio(pcm, sample_rate=sample_rate, channels=channels))

    def speak_audio_chunks(
        self,
        pcm: bytes,
        *,
        sample_rate: int,
        channels: int = 1,
        chunk_bytes: int = 2048,
        chunk_delay_seconds: float | None = None,
        wait_for_ack: bool = True,
        ack_timeout_seconds: float = 2.0,
        wait_for_playback_done: bool = True,
        playback_timeout_seconds: float | None = None,
        binary_chunks: bool = True,
    ) -> bool:
        with self._audio_ack_condition:
            self._audio_acks.clear()
        with self._audio_done_condition:
            target_done_generation = self._audio_done_generation + 1
        if not self.send(audio_start(sample_rate=sample_rate, channels=channels, total_bytes=len(pcm))):
            return False
        ok = True
        seq = 0
        frame_bytes = max(2, channels * 2)
        chunk_bytes = max(frame_bytes, (chunk_bytes // frame_bytes) * frame_bytes)
        bytes_per_second = max(frame_bytes, sample_rate * frame_bytes)
        if chunk_delay_seconds is None and not wait_for_ack:
            chunk_duration = chunk_bytes / bytes_per_second
            chunk_delay_seconds = min(0.08, max(0.012, chunk_duration * 0.8))
        elif chunk_delay_seconds is None:
            chunk_delay_seconds = 0.0
        for offset in range(0, len(pcm), chunk_bytes):
            chunk = pcm[offset : offset + chunk_bytes]
            sent = self._send_audio_raw_chunk(chunk, seq=seq) if binary_chunks else self.send(audio_chunk(chunk, seq=seq))
            ok = sent and ok
            if wait_for_ack and not self._wait_audio_ack(seq, timeout=ack_timeout_seconds):
                ok = False
                break
            seq += 1
            if chunk_delay_seconds > 0 and offset + chunk_bytes < len(pcm):
                time.sleep(chunk_delay_seconds)
        ok = self.send(audio_end()) and ok
        if ok and wait_for_playback_done:
            if playback_timeout_seconds is None:
                frame_bytes = max(2, channels * 2)
                duration = len(pcm) / max(frame_bytes, sample_rate * frame_bytes)
                playback_timeout_seconds = max(3.0, duration + 4.0)
            ok = self._wait_audio_done(target_done_generation, timeout=playback_timeout_seconds) and ok
        return ok

    def stop_audio(self) -> bool:
        return self.send(stop_audio())

    def hold_audio(self, active: bool) -> bool:
        return self.send(hold_audio(active=active))

    def speaker_tone(self, *, frequency: int = 880, duration_ms: int = 180) -> bool:
        return self.send(speaker_tone(frequency=frequency, duration_ms=duration_ms))

    def send(self, command: BodyCommand) -> bool:
        payload = (command.to_json() + "\n").encode("utf-8")
        with self._lock:
            client = self._client
        if client is None:
            return False
        try:
            client.sendall(payload)
            return True
        except OSError:
            self._drop_client()
            return False

    def _send_audio_raw_chunk(self, chunk: bytes, *, seq: int) -> bool:
        header = json.dumps(
            {
                "type": "audio.raw",
                "payload": {
                    "encoding": "pcm16le",
                    "seq": seq,
                    "bytes": len(chunk),
                },
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8") + b"\n"
        with self._lock:
            client = self._client
        if client is None:
            return False
        try:
            client.sendall(header)
            client.sendall(chunk)
            return True
        except OSError:
            self._drop_client()
            return False

    def _run(self) -> None:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self.host, self.port))
        server.listen(1)
        server.settimeout(0.5)
        with self._lock:
            self._server = server
        try:
            while not self._stop.is_set():
                if not self.connected:
                    self._accept_client(server)
                    continue
                self._receive_events()
        finally:
            self._drop_client()
            try:
                server.close()
            except OSError:
                pass

    def _accept_client(self, server: socket.socket) -> None:
        try:
            client, address = server.accept()
        except socket.timeout:
            return
        except OSError:
            return
        client.settimeout(0.5)
        with self._lock:
            old = self._client
            self._client = client
            self._client_address = address
            self._connected.set()
        if old is not None:
            try:
                old.close()
            except OSError:
                pass

    def _receive_events(self) -> None:
        with self._lock:
            client = self._client
        if client is None:
            self._connected.clear()
            return
        try:
            raw = client.recv(4096)
        except socket.timeout:
            return
        except OSError:
            self._drop_client()
            return
        if not raw:
            self._drop_client()
            return
        self._buffer += raw
        self._process_buffered_events()

    def _process_buffered_events(self) -> None:
        while True:
            if self._pending_audio_in is not None:
                needed = int(self._pending_audio_in.get("bytes", 0))
                if len(self._buffer) < needed:
                    return
                pcm = self._buffer[:needed]
                self._buffer = self._buffer[needed:]
                payload = dict(self._pending_audio_in)
                ts = float(payload.pop("_ts", time.time()))
                payload["pcm"] = pcm
                self._pending_audio_in = None
                self._dispatch_event(BodyEvent("audio.in", payload, ts))
                continue
            if b"\n" not in self._buffer:
                return
            line_bytes, self._buffer = self._buffer.split(b"\n", 1)
            line = line_bytes.decode("utf-8", errors="replace")
            if not line.strip():
                continue
            try:
                event = BodyEvent.from_json(line)
            except ValueError:
                continue
            if self._is_raw_audio_in_header(event):
                self._pending_audio_in = dict(event.payload)
                self._pending_audio_in["_ts"] = event.ts
                continue
            self._dispatch_event(event)

    def _dispatch_event(self, event: BodyEvent) -> None:
        if event.type == "audio.chunk_ack":
            self._record_audio_ack(event)
        if event.type == "audio.play_done":
            self._record_audio_done()
        if self.on_event:
            self.on_event(event)

    def _is_raw_audio_in_header(self, event: BodyEvent) -> bool:
        if event.type != "audio.in":
            return False
        if "data" in event.payload or "pcm" in event.payload:
            return False
        if event.payload.get("transport") != "raw":
            return False
        try:
            byte_count = int(event.payload.get("bytes", 0))
        except (TypeError, ValueError):
            return False
        return byte_count > 0

    def _record_audio_ack(self, event: BodyEvent) -> None:
        try:
            seq = int(event.payload.get("seq", -1))
        except (TypeError, ValueError):
            return
        if seq < 0:
            return
        queued = bool(event.payload.get("queued", False))
        with self._audio_ack_condition:
            self._audio_acks[seq] = queued
            self._audio_ack_condition.notify_all()

    def _wait_audio_ack(self, seq: int, *, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        with self._audio_ack_condition:
            while not self._stop.is_set() and self.connected:
                if seq in self._audio_acks:
                    return self._audio_acks.pop(seq)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._audio_ack_condition.wait(timeout=remaining)
        return False

    def _record_audio_done(self) -> None:
        with self._audio_done_condition:
            self._audio_done_generation += 1
            self._audio_done_condition.notify_all()

    def _wait_audio_done(self, target_generation: int, *, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        with self._audio_done_condition:
            while not self._stop.is_set() and self.connected:
                if self._audio_done_generation >= target_generation:
                    return True
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._audio_done_condition.wait(timeout=remaining)
        return False

    def _drop_client(self) -> None:
        with self._lock:
            client = self._client
            self._client = None
            self._client_address = None
            self._buffer = b""
            self._pending_audio_in = None
            self._connected.clear()
        with self._audio_ack_condition:
            self._audio_ack_condition.notify_all()
        with self._audio_done_condition:
            self._audio_done_condition.notify_all()
        if client is not None:
            try:
                client.close()
            except OSError:
                pass


class BodyPresence:
    def __init__(self, controller: StackChanBodyController | None) -> None:
        self.controller = controller

    def set(self, expression_name: str) -> None:
        if self.controller:
            self.controller.set_expression(expression_name)

    def pause_then(self, expression_name: str, delay: float = 0.35) -> None:
        if not self.controller:
            return
        time.sleep(delay)
        self.controller.set_expression(expression_name)
