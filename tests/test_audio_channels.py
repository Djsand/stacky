from __future__ import annotations

import unittest

from stacky.voice.channels import Pcm16ChannelSelector, apply_pcm16_gain, select_pcm16_channel


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

    def test_auto_selects_loudest_channel(self) -> None:
        pcm = pcm_sample(10) + pcm_sample(300) + pcm_sample(-20) + pcm_sample(-400)

        selected, channels = select_pcm16_channel(pcm, channels=2, selection="auto")

        self.assertEqual(channels, 1)
        self.assertEqual(selected, pcm_sample(300) + pcm_sample(-400))

    def test_all_keeps_original_stream(self) -> None:
        pcm = pcm_sample(100) + pcm_sample(300)

        selected, channels = select_pcm16_channel(pcm, channels=2, selection="all")

        self.assertEqual(channels, 2)
        self.assertEqual(selected, pcm)

    def test_applies_pcm16_gain_without_clipping_when_near_limit(self) -> None:
        pcm = pcm_sample(1000) + pcm_sample(-20000) + pcm_sample(20000)

        amplified = apply_pcm16_gain(pcm, gain=2.0)

        self.assertEqual(amplified, pcm_sample(1500) + pcm_sample(-30000) + pcm_sample(30000))

    def test_stateful_auto_channel_does_not_switch_on_one_loud_chunk(self) -> None:
        selector = Pcm16ChannelSelector("auto")
        left_loud = pcm_sample(800) + pcm_sample(100) + pcm_sample(700) + pcm_sample(120)
        right_loud = pcm_sample(100) + pcm_sample(1200) + pcm_sample(120) + pcm_sample(1000)

        first, _ = selector.select(left_loud, channels=2)
        second, _ = selector.select(right_loud, channels=2)

        self.assertEqual(first, pcm_sample(800) + pcm_sample(700))
        self.assertEqual(second, pcm_sample(100) + pcm_sample(120))

    def test_stateful_auto_channel_switches_after_sustained_advantage(self) -> None:
        selector = Pcm16ChannelSelector("auto")
        left_loud = pcm_sample(800) + pcm_sample(100) + pcm_sample(700) + pcm_sample(120)
        right_loud = pcm_sample(100) + pcm_sample(2200) + pcm_sample(120) + pcm_sample(2000)

        selector.select(left_loud, channels=2)
        selected = b""
        for _ in range(18):
            selected, _ = selector.select(right_loud, channels=2)

        self.assertEqual(selected, pcm_sample(2200) + pcm_sample(2000))


if __name__ == "__main__":
    unittest.main()
