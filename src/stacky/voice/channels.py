from __future__ import annotations


def select_pcm16_channel(pcm: bytes, *, channels: int, selection: str) -> tuple[bytes, int]:
    """Return PCM16 audio for one selected channel, a mono mix, or the original stream."""
    selection = str(selection).strip().lower()
    if channels <= 1 or selection == "all":
        return pcm, max(1, channels)
    if selection == "mix":
        return _mix_pcm16_channels(pcm, channels=channels), 1
    if selection in {"auto", "best"}:
        channel_index = _loudest_pcm16_channel(pcm, channels=channels)
        return _extract_pcm16_channel(pcm, channels=channels, channel_index=channel_index), 1
    try:
        channel_index = int(selection)
    except ValueError as exc:
        raise ValueError(f"Invalid mic channel selection: {selection}") from exc
    if channel_index < 0 or channel_index >= channels:
        raise ValueError(f"Mic channel {channel_index} is unavailable; firmware sent {channels} channel(s).")
    return _extract_pcm16_channel(pcm, channels=channels, channel_index=channel_index), 1


def apply_pcm16_gain(pcm: bytes, *, gain: float) -> bytes:
    if gain <= 1.0 or len(pcm) < 2:
        return pcm
    out = bytearray()
    for index in range(0, len(pcm) - 1, 2):
        sample = int.from_bytes(pcm[index : index + 2], "little", signed=True)
        amplified = int(round(sample * gain))
        amplified = max(-32768, min(32767, amplified))
        out.extend(amplified.to_bytes(2, "little", signed=True))
    if len(pcm) % 2:
        out.extend(pcm[-1:])
    return bytes(out)


def _extract_pcm16_channel(pcm: bytes, *, channels: int, channel_index: int) -> bytes:
    frame_bytes = channels * 2
    offset = channel_index * 2
    out = bytearray()
    for index in range(0, len(pcm) - frame_bytes + 1, frame_bytes):
        out.extend(pcm[index + offset : index + offset + 2])
    return bytes(out)


def _mix_pcm16_channels(pcm: bytes, *, channels: int) -> bytes:
    frame_bytes = channels * 2
    out = bytearray()
    for index in range(0, len(pcm) - frame_bytes + 1, frame_bytes):
        total = 0
        for channel in range(channels):
            sample_index = index + channel * 2
            total += int.from_bytes(pcm[sample_index : sample_index + 2], "little", signed=True)
        mixed = int(total / channels)
        mixed = max(-32768, min(32767, mixed))
        out.extend(mixed.to_bytes(2, "little", signed=True))
    return bytes(out)


def _loudest_pcm16_channel(pcm: bytes, *, channels: int) -> int:
    best_channel = 0
    best_energy = -1
    for channel in range(channels):
        energy = 0
        count = 0
        offset = channel * 2
        frame_bytes = channels * 2
        for index in range(0, len(pcm) - frame_bytes + 1, frame_bytes):
            sample = int.from_bytes(pcm[index + offset : index + offset + 2], "little", signed=True)
            energy += sample * sample
            count += 1
        if count and energy > best_energy:
            best_energy = energy
            best_channel = channel
    return best_channel
