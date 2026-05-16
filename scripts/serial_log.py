#!/usr/bin/env python3
"""Capture StackChan/CoreS3 serial output to a log file."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import serial


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="COM3")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--seconds", type=float, default=120.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + args.seconds

    with serial.Serial(args.port, args.baud, timeout=0.2, dsrdtr=False, rtscts=False) as ser:
        ser.dtr = False
        ser.rts = False
        with output.open("wb") as log_file:
            print(f"[serial] logging {args.port} @ {args.baud} to {output}")
            while time.monotonic() < deadline:
                chunk = ser.read(4096)
                if chunk:
                    log_file.write(chunk)
                    log_file.flush()
                    print(chunk.decode("utf-8", errors="replace"), end="", flush=True)


if __name__ == "__main__":
    main()
