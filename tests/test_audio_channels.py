from __future__ import annotations

import unittest

from stacky.voice.channels import select_pcm16_channel


def pcm_sample(value: int) -> bytes:
    return int(value).to_bytes(2, "little", signed=True)


class AudioChannelTest(unittest.TestCase):
    def test_selects_single_channel_from_stereo_pcm16(self) -> None:
        pcm = pcm_sample(100) + pcm_sample(1000) + pcm_sample(200) + pcm_sample(2000)

        selected, channels = select_pcm16_channel(pcm, channels=2, selection="1")

        self.assertEqual(channels, 1)
        self.assertEqual(selected, pcm_sample(1000) + pcm_sample(2000))

    def test_mixes_stereo_pcm16_to_mono(self) -> None:
        pcm = pcm_sample(100) + pcm_sample(300) + pcm_sample(-200) + pcm_sample(100)

        selected, channels = select_pcm16_channel(pcm, channels=2, selection="mix")

        self.assertEqual(channels, 1)
        self.assertEqual(selected, pcm_sample(200) + pcm_sample(-50))

    def test_all_keeps_original_stream(self) -> None:
        pcm = pcm_sample(100) + pcm_sample(300)

        selected, channels = select_pcm16_channel(pcm, channels=2, selection="all")

        self.assertEqual(channels, 2)
        self.assertEqual(selected, pcm)


if __name__ == "__main__":
    unittest.main()
