from __future__ import annotations


def select_pcm16_channel(pcm: bytes, *, channels: int, selection: str) -> tuple[bytes, int]:
    """Return PCM16 audio for one selected channel, a mono mix, or the original stream."""
    selection = str(selection).strip().lower()
    if channels <= 1 or selection == "all":
        return pcm, max(1, channels)
    if selection == "mix":
        return _mix_pcm16_channels(pcm, channels=channels), 1
    try:
        channel_index = int(selection)
    except ValueError as exc:
        raise ValueError(f"Invalid mic channel selection: {selection}") from exc
    if channel_index < 0 or channel_index >= channels:
        raise ValueError(f"Mic channel {channel_index} is unavailable; firmware sent {channels} channel(s).")
    return _extract_pcm16_channel(pcm, channels=channels, channel_index=channel_index), 1


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
