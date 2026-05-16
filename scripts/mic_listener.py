#!/usr/bin/env python3
"""Minimal Stacky mic-debug listener.

Accepts a StackChan TCP connection on port 8765, captures audio.in raw PCM
frames to a WAV file, and prints per-frame RMS/peak so mic quality can be
evaluated independently of the full STT stack.

Requires only the Python stdlib. Run with:

    python3 scripts/mic_listener.py
    python3 scripts/mic_listener.py --output captured.wav --port 8765
"""

from __future__ import annotations

import argparse
import json
import socket
import statistics
import struct
import time
import wave


def compute_rms_peak(pcm_bytes: bytes) -> tuple[int, int]:
    sample_count = len(pcm_bytes) // 2
    if sample_count == 0:
        return 0, 0
    samples = struct.unpack(f"<{sample_count}h", pcm_bytes)
    peak = max(abs(s) for s in samples)
    rms = (sum(s * s for s in samples) / sample_count) ** 0.5
    return int(rms), peak


def read_line(sock_file) -> str | None:
    line = sock_file.readline()
    if not line:
        return None
    return line.decode("utf-8", errors="replace").strip()


def read_exact(sock_file, n: int) -> bytes | None:
    chunks: list[bytes] = []
    remaining = n
    while remaining > 0:
        chunk = sock_file.read(remaining)
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="captured.wav")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--bind", default="0.0.0.0")
    parser.add_argument("--seconds", type=float, default=0.0, help="Stop after this many seconds; 0 runs until interrupted.")
    args = parser.parse_args()

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.bind, args.port))
    server.listen(1)
    print(f"[listen] {args.bind}:{args.port} - waiting for StackChan...")

    conn, addr = server.accept()
    print(f"[connect] from {addr[0]}:{addr[1]}")
    sock_file = conn.makefile("rb", buffering=0)

    sample_rate = 16000
    pcm_chunks: list[bytes] = []
    rms_values: list[int] = []
    peak_values: list[int] = []
    frame_count = 0
    started_at = time.time()
    deadline = started_at + args.seconds if args.seconds > 0 else None

    try:
        while True:
            if deadline is not None and time.time() >= deadline:
                print("[stop] time limit reached")
                break
            line = read_line(sock_file)
            if line is None:
                print("[disconnect] peer closed connection")
                break
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                print(f"[skip] non-JSON line: {line[:80]!r}")
                continue

            mtype = msg.get("type")
            payload = msg.get("payload", {})

            if mtype == "audio.in" and payload.get("transport") == "raw":
                n = int(payload.get("bytes", 0))
                sample_rate = int(payload.get("sampleRate", sample_rate))
                pcm = read_exact(sock_file, n)
                if pcm is None:
                    print("[disconnect] peer closed mid-frame")
                    break
                rms, peak = compute_rms_peak(pcm)
                rms_values.append(rms)
                peak_values.append(peak)
                pcm_chunks.append(pcm)
                frame_count += 1
                if frame_count % 25 == 0:
                    captured_sec = (frame_count * (n // 2)) / sample_rate
                    avg_rms = statistics.mean(rms_values[-25:])
                    max_peak = max(peak_values[-25:])
                    clipping = "CLIP" if max_peak >= 32700 else "ok"
                    print(
                        f"[mic] frame={frame_count:5d} t={captured_sec:5.2f}s "
                        f"rms_avg={avg_rms:6.0f} peak={max_peak:5d} {clipping}"
                    )
            elif mtype == "status":
                fw = payload.get("firmware", "?")
                expr = payload.get("expression", "?")
                print(f"[status] firmware={fw} expression={expr}")
            elif mtype == "touch":
                print(f"[touch] {payload}")
            else:
                print(f"[recv] {mtype}: {payload}")

    except KeyboardInterrupt:
        print("\n[stop] Ctrl+C")
    finally:
        conn.close()
        server.close()

    if not pcm_chunks:
        print("[wav] no audio captured")
        return

    total_pcm = b"".join(pcm_chunks)
    with wave.open(args.output, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(total_pcm)
    duration = len(total_pcm) / 2 / sample_rate
    elapsed = time.time() - started_at
    print()
    print(f"[wav]      wrote {args.output}  {duration:.2f}s @ {sample_rate} Hz")
    print(f"[summary]  frames={frame_count} wall={elapsed:.1f}s")
    print(f"[summary]  rms  avg={statistics.mean(rms_values):.0f} "
          f"min={min(rms_values)} max={max(rms_values)}")
    print(f"[summary]  peak max={max(peak_values)} "
          f"clipped_frames={sum(1 for p in peak_values if p >= 32700)}")


if __name__ == "__main__":
    main()
