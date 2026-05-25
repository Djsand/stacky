from __future__ import annotations

import argparse
import asyncio
import json
import re
import shutil
import socket
import subprocess
import sys
import time
import unicodedata
import wave
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

try:
    from slugify import slugify
except ModuleNotFoundError:

    def slugify(value: object, *, max_length: int = 0) -> str:
        normalized = unicodedata.normalize("NFKD", str(value))
        ascii_text = normalized.encode("ascii", "ignore").decode("ascii").lower()
        slug = re.sub(r"[^a-z0-9]+", "-", ascii_text).strip("-")
        slug = re.sub(r"-{2,}", "-", slug) or "stacky"
        return slug[:max_length].strip("-") if max_length > 0 else slug

from .brain import StackyBrain
from .body.calibration import BodyCalibration, load_body_calibration, save_body_calibration
from .body.controller import BodyPresence, StackChanBodyController
from .body.director import BodyDirector
from .body.protocol import decode_pcm_payload, decode_vision_frame_payload, expression
from .computer import LocalComputerActions, LocalComputerContext, parse_local_computer_action
from .config import DEFAULT_CONFIG_PATH, ROOT, MonitorConfig, load_config
from .evolution import StackyEvolutionEngine
from .llm import create_chat_client
from .llm import ChatImageAttachment
from .memory import MemoryStore
from .memory_map import MemoryMapStore
from .monitor import (
    DefaultMonitorProbe,
    GlobalFriendMonitor,
    MonitorObservation,
    format_monitor_context,
    monitor_prompt_for_observation,
)
from .personality import StackySelfModel
from .runtime_state import RuntimeState
from .sandcode import (
    SandcodeDanishSummarizer,
    SandcodeError,
    SandcodeMobileHostClient,
    SandcodeSession,
    classify_sandcode_action,
)
from .sessions import InfiniteSessionStore
from .soul import load_soul, write_default_soul
from .voice.output import (
    create_fast_piper_output,
    create_stackchan_piper_output,
    create_stackchan_supertonic_output,
    create_supertonic_output,
)
from .voice.channels import Pcm16ChannelSelector, apply_pcm16_gain
from .voice.supertonic_tts import SupertonicVoice, supertonic_voice_preset
from .voice.runtime import LocalTextVoiceRuntime
from .voice.stt import STTResult, create_danish_stt, resolve_stt_model_name, wav_audio_stats, write_pcm_wav
from .voice.stt_eval import (
    STTDatasetItem,
    apply_references,
    char_error_rate,
    load_capture_phrases,
    load_dataset_manifest,
    load_reference_file,
    resolve_audio_inputs,
    word_error_rate,
    write_dataset_record,
)
from .voice.transcript_correction import correct_danish_transcript
from .voice.turn_detection import EnergyTurnDetector, TurnSignalQuality, analyze_turn_signal, pcm16_rms
from .vision import VisionSnapshot, VisionState, create_face_detector
from .websearch import (
    DuckDuckGoLiteSearch,
    WebSearchClient,
    WebSearchError,
    classify_web_search_intent,
    extract_web_search_query,
    format_web_search_context,
    wants_web_search,
)
from .voice.piper_tts import FastPiperTTS, ensure_danish_piper_voice, pitch_shift_wav
from .voice.roest_tts import RoestTTS, roest_voice
from .voice.speech_adapter import adapt_for_danish_speech


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="stacky")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    sub = parser.add_subparsers(dest="command", required=False)
    sub.add_parser("init", help="Create fresh Stacky local state.")
    sub.add_parser("self-status", help="Show Stacky's local personality/self-model state.")
    web_search = sub.add_parser("web-search", help="Run Stacky's configured web search once.")
    web_search.add_argument("query", nargs="+", help="Search query.")
    computer_context = sub.add_parser("computer-context", help="Show Stacky's local read-only computer context once.")
    computer_context.add_argument("query", nargs="*", help="Optional Danish request/search text.")
    chat = sub.add_parser("chat", help="Run a Danish text-mode voice loop.")
    chat.add_argument("--speak", action="store_true", help="Speak replies through local low-latency Piper TTS.")
    live = sub.add_parser("live-text", help="Run Danish text chat while driving StackChan's face.")
    live.add_argument("--body-timeout", type=float, default=8.0, help="Seconds to wait for StackChan to connect.")
    live.add_argument("--speak", action="store_true", help="Speak replies through local low-latency Piper TTS.")
    handsfree = sub.add_parser("handsfree", help="Run hands-free Danish voice directly through StackChan.")
    handsfree.add_argument("--body-timeout", type=float, default=12.0, help="Seconds to wait for StackChan to connect.")
    handsfree.add_argument(
        "--stt-engine",
        choices=("wav2vec2", "whisper", "qwen3"),
        default="wav2vec2",
        help="Local Danish STT backend. wav2vec2 is the low-latency default.",
    )
    handsfree.add_argument(
        "--stt-model",
        default="",
        help="Model name. wav2vec2 default is CoRal-project/roest-v3-wav2vec2-315m; whisper default is small.",
    )
    handsfree.add_argument("--vad-threshold", type=int, default=280, help="PCM RMS threshold for speech start.")
    handsfree.add_argument("--start-speech-ms", type=int, default=120, help="Continuous speech needed before a turn starts.")
    handsfree.add_argument("--end-silence-ms", type=int, default=850, help="Silence duration that ends a voice turn.")
    handsfree.add_argument("--min-speech-ms", type=int, default=220, help="Minimum voiced audio before accepting a turn.")
    handsfree.add_argument(
        "--mic-channel",
        choices=("auto", "best", "0", "1", "mix", "all"),
        default="0",
        help="StackChan input channel to use after firmware capture. CoreS3 official firmware sends mic on channel 0 and reference/noise on channel 1; auto/best is diagnostics only.",
    )
    handsfree.add_argument("--listen-only", action="store_true", help="Only print StackChan STT results; do not call the brain or TTS.")
    handsfree.add_argument("--debug-audio", action="store_true", help="Print live StackChan mic RMS/peak while listening.")
    handsfree.add_argument(
        "--speaker",
        choices=("stackchan", "pc"),
        default="stackchan",
        help="Where Stacky speaks. StackChan is the hands-free body speaker.",
    )
    handsfree.add_argument(
        "--tts-engine",
        choices=("piper", "supertonic"),
        default="supertonic",
        help="Local TTS engine. Supertonic is the livelier default; Piper is the realtime-stable fallback.",
    )
    handsfree.add_argument(
        "--supertonic-profile",
        choices=("stacky", "calm", "clear", "quick", "alive"),
        default="alive",
        help="Supertonic tuning profile for Stacky's Danish voice.",
    )
    handsfree.add_argument("--supertonic-voice", default="", help="Override Supertonic voice style: F1-F5 or M1-M5.")
    handsfree.add_argument("--supertonic-speed", type=float, default=None, help="Override Supertonic speed multiplier.")
    handsfree.add_argument("--supertonic-steps", type=int, default=None, help="Override Supertonic quality steps; higher is clearer but slower.")
    handsfree.add_argument("--supertonic-silence", type=float, default=None, help="Override silence between Supertonic internal chunks.")
    handsfree.add_argument("--stackchan-target-rms", type=int, default=9000, help="Target active PCM RMS for StackChan speaker loudness.")
    handsfree.add_argument("--stackchan-max-gain", type=float, default=4.0, help="Maximum StackChan speaker PCM gain before clipping.")
    handsfree.add_argument("--stackchan-volume", type=int, default=80, help="Initial StackChan codec volume, 0-100.")
    handsfree.add_argument("--stackchan-mic-gain", type=int, default=85, help="Initial StackChan codec mic gain, 0-100.")
    handsfree.add_argument("--mic-preamp", type=float, default=2.0, help="Digital PCM gain before VAD/STT. Limited to avoid PCM clipping; use 1.0 to disable.")
    handsfree.add_argument("--reply-chars", type=int, default=260, help="Default spoken reply character budget for low-latency live chat.")
    handsfree.add_argument("--detail-reply-chars", type=int, default=650, help="Spoken reply character budget when the user asks for details.")
    handsfree.add_argument(
        "--vision",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Capture StackChan camera frames during handsfree and provide them to the brain.",
    )
    handsfree.add_argument(
        "--vision-image",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Attach the latest 320x240 JPEG snapshot to Gemini/openai-compatible brain prompts.",
    )
    handsfree.add_argument(
        "--face-tracking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use local face tracking to gently orient StackChan's head while listening.",
    )
    handsfree.add_argument("--vision-interval", type=float, default=1.0, help="Seconds between idle camera captures.")
    handsfree.add_argument("--vision-prompt-timeout", type=float, default=0.8, help="Seconds to wait for a fresh prompt snapshot.")
    handsfree.add_argument(
        "--websearch",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable explicit web search requests such as 'søg på nettet efter ...'.",
    )
    handsfree.add_argument(
        "--computer",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable local read-only computer/code context for explicit terminal, grep, repo, or code requests.",
    )
    handsfree.add_argument(
        "--monitor",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable Stacky's sparse global friend monitor sanseinput.",
    )
    handsfree.add_argument(
        "--voice-trust",
        choices=("trusted", "session-only", "off"),
        default="session-only",
        help="How accepted StackChan voice turns are logged. trusted writes session + safe memories; session-only logs context; off keeps the old untrusted mode.",
    )
    stt_capture = sub.add_parser("stt-capture", help="Record labelled StackChan mic clips for Danish STT evaluation.")
    stt_capture.add_argument("--body-timeout", type=float, default=18.0, help="Seconds to wait for StackChan to connect.")
    stt_capture.add_argument("--phrase", action="append", default=[], help="Expected Danish phrase to record. Can be repeated.")
    stt_capture.add_argument("--phrases-file", default="", help="UTF-8 text file with one expected phrase per line.")
    stt_capture.add_argument("--noise-count", type=int, default=0, help="Also capture N non-speech/noise clips with empty expected text.")
    stt_capture.add_argument(
        "--speech-style",
        action="append",
        choices=("normal", "fast", "mumble", "quiet"),
        default=[],
        help="Capture style. Repeat it to build robustness, e.g. normal + fast + mumble.",
    )
    stt_capture.add_argument("--limit", type=int, default=0, help="Limit phrase count. 0 records all selected/default phrases.")
    stt_capture.add_argument("--output-dir", default=str(ROOT / "artifacts" / "stt_dataset" / "stackchan"), help="Directory for captured WAV clips.")
    stt_capture.add_argument("--manifest", default="", help="JSONL manifest path. Defaults to <output-dir>/manifest.jsonl.")
    stt_capture.add_argument("--vad-threshold", type=int, default=280, help="PCM RMS threshold for speech start.")
    stt_capture.add_argument("--start-speech-ms", type=int, default=120, help="Continuous speech needed before a turn starts.")
    stt_capture.add_argument("--end-silence-ms", type=int, default=850, help="Silence duration that ends a voice turn.")
    stt_capture.add_argument("--min-speech-ms", type=int, default=220, help="Minimum voiced audio before accepting a turn.")
    stt_capture.add_argument(
        "--mic-channel",
        choices=("auto", "best", "0", "1", "mix", "all"),
        default="0",
        help="StackChan input channel to record. Channel 0 is the CoreS3 mic; use 1/mix/auto only for diagnostics.",
    )
    stt_capture.add_argument("--debug-audio", action="store_true", help="Print accepted/rejected signal quality while capturing.")
    stt_capture.add_argument("--stackchan-mic-gain", type=int, default=85, help="Initial StackChan codec mic gain, 0-100.")
    stt_capture.add_argument("--mic-preamp", type=float, default=2.0, help="Digital PCM gain before VAD/STT. Limited to avoid PCM clipping; use 1.0 to disable.")
    stt_bench = sub.add_parser("stt-bench", help="Benchmark local Danish STT models on saved StackChan WAV turns.")
    stt_bench.add_argument("--audio", action="append", default=[], help="WAV file, directory, or glob. Defaults to artifacts/handsfree_turns/*.wav.")
    stt_bench.add_argument("--dataset", default="", help="JSONL/TSV manifest from stt-capture. Provides expected text for scoring.")
    stt_bench.add_argument("--engine", action="append", default=[], help="Model spec: roest, roest-v2, roest-v2-1b, roest-v2-2b, qwen3, saga, milo, or engine:model.")
    stt_bench.add_argument("--limit", type=int, default=0, help="Maximum number of WAV files to test. 0 tests all.")
    stt_bench.add_argument("--include-heavy", action="store_true", help="Also test heavier Qwen3-ASR candidates.")
    stt_bench.add_argument("--refs", default="", help="Optional tab-separated references file: wav_filename<TAB>expected text.")
    stt_bench.add_argument("--report", default="", help="Optional JSONL report output path.")
    stt_bench.add_argument(
        "--correct-transcripts",
        action="store_true",
        help="Score Stacky live post-correction instead of raw ASR output.",
    )
    stt_bench.add_argument(
        "--live-gate",
        action="store_true",
        help="Use manifest signal-quality gate before STT, matching handsfree behavior for rejected noise.",
    )
    voice_lab = sub.add_parser("voice-lab", help="Generate local Danish TTS samples.")
    voice_lab.add_argument("--play", action="store_true", help="Play generated samples with ffplay.")
    voice_lab.add_argument(
        "--engine",
        choices=("piper", "roest", "supertonic"),
        default="piper",
        help="TTS engine to audition. Piper is fast; Supertonic is natural/local; Roest is heavier.",
    )
    voice_lab.add_argument(
        "--style",
        choices=("neutral", "female"),
        default="neutral",
        help="Piper-only audition style. 'female' pitch-shifts the Danish Piper voice.",
    )
    voice_lab.add_argument(
        "--speaker",
        choices=("nic", "mic"),
        default="nic",
        help="Roest speaker prompt. 'nic' is the brighter default audition.",
    )
    voice_lab.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit sample count; 0 uses the engine default.",
    )
    sub.add_parser("sandcode-health", help="Start/check Sandcode mobile host.")
    sandcode_run = sub.add_parser("sandcode-run", help="Start a Sandcode session for an explicit coding task.")
    sandcode_run.add_argument("--cwd", default="", help="Project directory. Defaults to Stacky's configured workspace.")
    sandcode_run.add_argument("--chat-only", action="store_true", help="Ask Sandcode without allowing tool use.")
    sandcode_run.add_argument("prompt", nargs="+", help="Task prompt for Sandcode.")
    speaker_test = sub.add_parser("speaker-test", help="Play a short Danish test phrase through StackChan.")
    speaker_test.add_argument("--body-timeout", type=float, default=12.0, help="Seconds to wait for StackChan to connect.")
    speaker_test.add_argument(
        "--tts-engine",
        choices=("piper", "supertonic"),
        default="supertonic",
        help="Local TTS engine to test on StackChan speaker.",
    )
    speaker_test.add_argument("--text", default="Hej Nicolai, jeg er her.", help="Danish phrase to speak.")
    speaker_test.add_argument(
        "--supertonic-profile",
        choices=("stacky", "calm", "clear", "quick", "alive"),
        default="alive",
        help="Supertonic tuning profile for Stacky's Danish voice.",
    )
    speaker_test.add_argument("--supertonic-voice", default="", help="Override Supertonic voice style: F1-F5 or M1-M5.")
    speaker_test.add_argument("--supertonic-speed", type=float, default=None, help="Override Supertonic speed multiplier.")
    speaker_test.add_argument("--supertonic-steps", type=int, default=None, help="Override Supertonic quality steps; higher is clearer but slower.")
    speaker_test.add_argument("--supertonic-silence", type=float, default=None, help="Override silence between Supertonic internal chunks.")
    speaker_test.add_argument("--stackchan-target-rms", type=int, default=9000, help="Target active PCM RMS for StackChan speaker loudness.")
    speaker_test.add_argument("--stackchan-max-gain", type=float, default=4.0, help="Maximum StackChan speaker PCM gain before clipping.")
    speaker_test.add_argument("--stackchan-volume", type=int, default=80, help="Initial StackChan codec volume, 0-100.")
    speaker_tone = sub.add_parser("speaker-tone", help="Play a tiny firmware tone on StackChan without TTS streaming.")
    speaker_tone.add_argument("--body-timeout", type=float, default=12.0, help="Seconds to wait for StackChan to connect.")
    speaker_tone.add_argument("--frequency", type=int, default=880, help="Tone frequency in Hz.")
    speaker_tone.add_argument("--duration-ms", type=int, default=180, help="Tone duration in milliseconds.")
    motion_test = sub.add_parser("motion-test", help="Move StackChan head servos through the Stacky bridge.")
    motion_test.add_argument("--body-timeout", type=float, default=12.0, help="Seconds to wait for StackChan to connect.")
    motion_test.add_argument(
        "--gesture",
        choices=("demo", "center", "look_left", "look_right", "look_up", "look_down", "nod", "shake"),
        default="demo",
        help="Gesture to run. demo runs a short safe sequence.",
    )
    motion_test.add_argument("--speed", type=int, default=550, help="Servo speed from 0 to 1000.")
    camera_test = sub.add_parser("camera-test", help="Capture one StackChan camera JPEG through the Stacky bridge.")
    camera_test.add_argument("--body-timeout", type=float, default=12.0, help="Seconds to wait for StackChan to connect.")
    camera_test.add_argument("--frame-timeout", type=float, default=8.0, help="Seconds to wait for a camera frame.")
    camera_test.add_argument("--width", type=int, default=320, help="Requested frame width metadata.")
    camera_test.add_argument("--height", type=int, default=240, help="Requested frame height metadata.")
    camera_test.add_argument("--quality", type=int, default=50, help="JPEG quality from 5 to 80. Lower is smaller/faster.")
    camera_test.add_argument("--discard-frames", type=int, default=4, help="Frames to discard before saving a capture.")
    camera_test.add_argument("--settle-ms", type=int, default=30, help="Delay between discarded camera frames.")
    camera_test.add_argument("--ae-level", type=int, default=2, help="GC0308 auto-exposure target level from -2 to 2.")
    camera_test.add_argument("--sensor-gain", type=int, default=30, help="Optional manual GC0308 gain from 0 to 30.")
    camera_test.add_argument("--sensor-exposure", type=int, default=1200, help="Optional manual GC0308 exposure from 0 to 1200.")
    camera_test.add_argument(
        "--enhance",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save an additional conservative brightened JPEG for LLM vision input.",
    )
    camera_test.add_argument(
        "--enhance-target",
        type=float,
        default=110.0,
        help="Target mean luminance for camera-test auto-enhance.",
    )
    camera_test.add_argument("--count", type=int, default=1, help="Number of frames to capture.")
    camera_test.add_argument("--delay-ms", type=int, default=250, help="Delay between captures when --count is above 1.")
    camera_test.add_argument(
        "--output",
        default=str(ROOT / "artifacts" / "vision" / "stackchan-latest.jpg"),
        help="Output JPEG path.",
    )
    sensor_test = sub.add_parser("sensor-test", help="Read StackChan status and I2C sensor scan through the bridge.")
    sensor_test.add_argument("--body-timeout", type=float, default=12.0, help="Seconds to wait for StackChan to connect.")
    sensor_test.add_argument("--event-timeout", type=float, default=8.0, help="Seconds to wait for i2c.scan.")
    body = sub.add_parser("body-server", help="Run the StackChan body server.")
    body.add_argument("--duration", type=float, default=0.0, help="Stop after N seconds; 0 means run forever.")
    args = parser.parse_args(argv)

    if args.command == "init":
        return _init(args.config)
    if args.command == "self-status":
        return _self_status(args.config)
    if args.command == "web-search":
        return _run_async(_web_search_once(args.config, " ".join(args.query)))
    if args.command == "computer-context":
        query = " ".join(args.query) if args.query else "terminal status kode"
        return _run_async(_computer_context_once(args.config, query))
    if args.command == "sandcode-health":
        return _run_async(_sandcode_health(args.config))
    if args.command == "sandcode-run":
        return _run_async(
            _sandcode_run_once(
                args.config,
                " ".join(args.prompt),
                cwd_arg=args.cwd,
                chat_only=args.chat_only,
            )
        )
    if args.command == "body-server":
        return _body_server(args.config, duration=args.duration)
    if args.command == "sensor-test":
        return _run_async(
            _sensor_test(args.config, body_timeout=args.body_timeout, event_timeout=args.event_timeout)
        )
    if args.command == "speaker-test":
        return _run_async(
            _speaker_test(
                args.config,
                body_timeout=args.body_timeout,
                tts_engine=args.tts_engine,
                text=args.text,
                supertonic_profile=args.supertonic_profile,
                supertonic_voice=args.supertonic_voice,
                supertonic_speed=args.supertonic_speed,
                supertonic_steps=args.supertonic_steps,
                supertonic_silence=args.supertonic_silence,
                stackchan_target_rms=args.stackchan_target_rms,
                stackchan_max_gain=args.stackchan_max_gain,
                stackchan_volume=args.stackchan_volume,
            )
        )
    if args.command == "speaker-tone":
        return _run_async(
            _speaker_tone(
                args.config,
                body_timeout=args.body_timeout,
                frequency=args.frequency,
                duration_ms=args.duration_ms,
            )
        )
    if args.command == "motion-test":
        return _run_async(
            _motion_test(
                args.config,
                body_timeout=args.body_timeout,
                gesture_name=args.gesture,
                speed=args.speed,
            )
        )
    if args.command == "camera-test":
        return _run_async(
            _camera_test(
                args.config,
                body_timeout=args.body_timeout,
                frame_timeout=args.frame_timeout,
                width=args.width,
                height=args.height,
                quality=args.quality,
                discard_frames=args.discard_frames,
                settle_ms=args.settle_ms,
                ae_level=args.ae_level,
                sensor_gain=args.sensor_gain,
                sensor_exposure=args.sensor_exposure,
                enhance=args.enhance,
                enhance_target=args.enhance_target,
                count=args.count,
                delay_ms=args.delay_ms,
                output=args.output,
            )
        )
    if args.command == "live-text":
        return _run_async(_live_text(args.config, body_timeout=args.body_timeout, speak=args.speak))
    if args.command == "handsfree":
        return _run_async(
            _handsfree(
                args.config,
                body_timeout=args.body_timeout,
                stt_engine=args.stt_engine,
                stt_model=args.stt_model,
                vad_threshold=args.vad_threshold,
                start_speech_ms=args.start_speech_ms,
                end_silence_ms=args.end_silence_ms,
                min_speech_ms=args.min_speech_ms,
                mic_channel=args.mic_channel,
                mic_preamp=args.mic_preamp,
                speaker=args.speaker,
                tts_engine=args.tts_engine,
                supertonic_profile=args.supertonic_profile,
                supertonic_voice=args.supertonic_voice,
                supertonic_speed=args.supertonic_speed,
                supertonic_steps=args.supertonic_steps,
                supertonic_silence=args.supertonic_silence,
                stackchan_target_rms=args.stackchan_target_rms,
                stackchan_max_gain=args.stackchan_max_gain,
                stackchan_volume=args.stackchan_volume,
                stackchan_mic_gain=args.stackchan_mic_gain,
                reply_chars=args.reply_chars,
                detail_reply_chars=args.detail_reply_chars,
                vision=args.vision,
                vision_image=args.vision_image,
                face_tracking=args.face_tracking,
                vision_interval=args.vision_interval,
                vision_prompt_timeout=args.vision_prompt_timeout,
                websearch=args.websearch,
                computer=args.computer,
                monitor=args.monitor,
                voice_trust=args.voice_trust,
                listen_only=args.listen_only,
                debug_audio=args.debug_audio,
            )
        )
    if args.command == "stt-capture":
        return _run_async(
            _stt_capture(
                args.config,
                body_timeout=args.body_timeout,
                phrase_args=args.phrase,
                phrases_file=args.phrases_file,
                noise_count=args.noise_count,
                speech_styles=args.speech_style,
                limit=args.limit,
                output_dir=args.output_dir,
                manifest=args.manifest,
                vad_threshold=args.vad_threshold,
                start_speech_ms=args.start_speech_ms,
                end_silence_ms=args.end_silence_ms,
                min_speech_ms=args.min_speech_ms,
                mic_channel=args.mic_channel,
                mic_preamp=args.mic_preamp,
                debug_audio=args.debug_audio,
                stackchan_mic_gain=args.stackchan_mic_gain,
            )
        )
    if args.command == "stt-bench":
        return _run_async(
            _stt_bench(
                audio_patterns=args.audio,
                dataset_path=args.dataset,
                engine_specs=args.engine,
                limit=args.limit,
                include_heavy=args.include_heavy,
                refs_path=args.refs,
                report_path=args.report,
                correct_transcripts=args.correct_transcripts,
                live_gate=args.live_gate,
            )
        )
    if args.command == "voice-lab":
        return _voice_lab(play=args.play, engine=args.engine, style=args.style, speaker=args.speaker, limit=args.limit)
    return _run_async(_chat(args.config, speak=getattr(args, "speak", False)))


def _run_async(coro) -> int:
    try:
        return asyncio.run(coro)
    except KeyboardInterrupt:
        return 0


def _safe_console_text(text: str) -> str:
    encoding = sys.stdout.encoding or "utf-8"
    return text.encode(encoding, errors="replace").decode(encoding, errors="replace")


_LOW_LATENCY_STT_SPECS = (("wav2vec2", "roest-v3"), ("wav2vec2", "roest-v2"))
_HEAVY_STT_SPECS = (
    ("wav2vec2", "roest-v2-1b"),
    ("wav2vec2", "roest-v2-2b"),
    ("qwen3", "qwen3-0.6b"),
    ("qwen3", "saga"),
    ("qwen3", "milo"),
)
_STT_SPEC_ALIASES = {
    "roest": ("wav2vec2", "roest"),
    "coral": ("wav2vec2", "roest"),
    "coral-v3": ("wav2vec2", "roest"),
    "roest-v3": ("wav2vec2", "roest-v3"),
    "roest-v3-315m": ("wav2vec2", "roest-v3-315m"),
    "coral-v2": ("wav2vec2", "roest-v2"),
    "roest-v2": ("wav2vec2", "roest-v2"),
    "roest-v2-315m": ("wav2vec2", "roest-v2-315m"),
    "roest-v2-1b": ("wav2vec2", "roest-v2-1b"),
    "roest-v2-2b": ("wav2vec2", "roest-v2-2b"),
    "roest-accurate": ("wav2vec2", "roest-accurate"),
    "ftspeech": ("wav2vec2", "ftspeech"),
    "qwen3": ("qwen3", "qwen3-0.6b"),
    "qwen3-0.6b": ("qwen3", "qwen3-0.6b"),
    "saga": ("qwen3", "saga"),
    "milo": ("qwen3", "milo"),
}


@dataclass(frozen=True)
class _VoiceMemoryPolicy:
    persist_session: bool
    allow_memory_writes: bool
    remember_recent: bool
    session_source: str


def _voice_memory_policy(mode: str) -> _VoiceMemoryPolicy:
    if mode == "off":
        return _VoiceMemoryPolicy(
            persist_session=False,
            allow_memory_writes=False,
            remember_recent=False,
            session_source="stackchan-voice-untrusted",
        )
    if mode == "session-only":
        return _VoiceMemoryPolicy(
            persist_session=True,
            allow_memory_writes=False,
            remember_recent=True,
            session_source="stackchan-voice-session",
        )
    return _VoiceMemoryPolicy(
        persist_session=True,
        allow_memory_writes=True,
        remember_recent=True,
        session_source="stackchan-voice",
    )


@dataclass
class _SttBenchStats:
    count: int = 0
    scored: int = 0
    total_audio: float = 0.0
    total_infer: float = 0.0
    wer_sum: float = 0.0
    cer_sum: float = 0.0

    def add(self, *, duration: float, infer_seconds: float, wer: float | None, cer: float | None) -> None:
        self.count += 1
        self.total_audio += duration
        self.total_infer += infer_seconds
        if wer is not None:
            self.scored += 1
            self.wer_sum += wer
        if cer is not None:
            self.cer_sum += cer

    def summary(self, label: str) -> str:
        rtf = self.total_infer / max(self.total_audio, 0.001)
        line = (
            f"  {label}: count={self.count} scored={self.scored} "
            f"audio={self.total_audio:.2f}s infer={self.total_infer:.2f}s rtf={rtf:.2f}"
        )
        if self.scored:
            line += f" mean_wer={self.wer_sum / self.scored:.1%} mean_cer={self.cer_sum / self.scored:.1%}"
        return line


async def _stt_bench(
    *,
    audio_patterns: list[str],
    dataset_path: str,
    engine_specs: list[str],
    limit: int,
    include_heavy: bool,
    refs_path: str,
    report_path: str,
    correct_transcripts: bool,
    live_gate: bool,
) -> int:
    items = _resolve_stt_bench_items(audio_patterns, dataset_path=dataset_path, refs_path=refs_path, limit=limit)
    if not items:
        print("No WAV files found. Run stt-capture or pass --audio path\\to\\turn.wav.", flush=True)
        return 1

    specs = _resolve_stt_bench_specs(engine_specs, include_heavy=include_heavy)
    report_file = Path(report_path) if report_path else None
    if report_file:
        report_file.parent.mkdir(parents=True, exist_ok=True)
        report_file.write_text("", encoding="utf-8")
    scored_count = sum(1 for item in items if item.expected_text is not None)
    print(f"Benchmarking {len(items)} StackChan WAV file(s), scored={scored_count}.", flush=True)
    for item in items:
        style = f" [{item.speech_style}]" if item.speech_style else ""
        expected = "" if item.expected_text is None else f" :: ref={item.expected_text!r}"
        print(f"- {item.audio_path}{style}{expected}", flush=True)

    for engine, requested_model in specs:
        model_name = resolve_stt_model_name(engine, requested_model)
        stt = create_danish_stt(engine, model_name)
        print(f"\n[{engine}] {model_name}", flush=True)
        started = time.perf_counter()
        try:
            await stt.preload()
        except Exception as exc:
            print(f"  SKIP load failed: {type(exc).__name__}: {exc}", flush=True)
            continue
        load_seconds = time.perf_counter() - started
        print(f"  load={load_seconds:.2f}s", flush=True)

        total_stats = _SttBenchStats()
        style_stats: dict[str, _SttBenchStats] = {}
        for item in items:
            path = item.audio_path
            gated_quality = _wav_signal_quality(path) if live_gate else None
            if live_gate and gated_quality is not None and not gated_quality.speech_like:
                result = STTResult(
                    text="",
                    audio=wav_audio_stats(path),
                    avg_logprob=-10.0,
                    no_speech_prob=1.0,
                    compression_ratio=0.0,
                )
                infer_seconds = 0.0
                duration = max(result.audio.duration_seconds, 0.001)
                expected = item.expected_text
                wer = word_error_rate(expected, "") if expected is not None else None
                cer = char_error_rate(expected, "") if expected is not None else None
                score = "" if wer is None or cer is None else f" wer={wer:.1%} cer={cer:.1%}"
                total_stats.add(duration=duration, infer_seconds=infer_seconds, wer=wer, cer=cer)
                style_label = item.speech_style or "unlabeled"
                style_stats.setdefault(style_label, _SttBenchStats()).add(
                    duration=duration,
                    infer_seconds=infer_seconds,
                    wer=wer,
                    cer=cer,
                )
                print(
                    f"  {path.name}: GATE non-speech dur={duration:.2f}s reason={gated_quality.reason!r}{score}",
                    flush=True,
                )
                if report_file is not None:
                    _append_stt_report(
                        report_file,
                        engine=engine,
                        model=model_name,
                        item=item,
                        result=result,
                        infer_seconds=infer_seconds,
                        wer=wer,
                        cer=cer,
                        live_gate_rejected=True,
                    )
                continue
            started = time.perf_counter()
            try:
                result = await stt.transcribe_wav_result(path)
            except Exception as exc:
                print(f"  {path.name}: ERROR {type(exc).__name__}: {exc}", flush=True)
                continue
            infer_seconds = time.perf_counter() - started
            duration = max(result.audio.duration_seconds, 0.001)
            rtf = infer_seconds / duration
            hypothesis = result.text
            correction_reason = ""
            if correct_transcripts:
                correction = correct_danish_transcript(result.text)
                hypothesis = correction.text
                correction_reason = correction.reason if correction.changed else ""
            expected = item.expected_text
            score = ""
            wer = None
            cer = None
            if expected is not None:
                wer = word_error_rate(expected, hypothesis)
                cer = char_error_rate(expected, hypothesis)
                score = f" wer={wer:.1%} cer={cer:.1%}"
            total_stats.add(duration=duration, infer_seconds=infer_seconds, wer=wer, cer=cer)
            style_label = item.speech_style or "unlabeled"
            style_stats.setdefault(style_label, _SttBenchStats()).add(
                duration=duration,
                infer_seconds=infer_seconds,
                wer=wer,
                cer=cer,
            )
            correction_note = f" -> {hypothesis}" if correct_transcripts and hypothesis != result.text else ""
            print(
                f"  {path.name}: dur={duration:.2f}s infer={infer_seconds:.2f}s "
                f"rtf={rtf:.2f} logprob={result.avg_logprob:.2f}{score} :: {result.text}{correction_note}",
                flush=True,
            )
            if report_file is not None:
                _append_stt_report(
                    report_file,
                    engine=engine,
                    model=model_name,
                    item=item,
                    result=result,
                    hypothesis=hypothesis,
                    correction_reason=correction_reason,
                    infer_seconds=infer_seconds,
                    wer=wer,
                    cer=cer,
                )

        if total_stats.count:
            print(total_stats.summary("total"), flush=True)
            if len(style_stats) > 1:
                for style_label in sorted(style_stats):
                    print(style_stats[style_label].summary(f"style[{style_label}]"), flush=True)
    if report_file is not None:
        print(f"\nReport: {report_file}", flush=True)
    return 0


async def _stt_capture(
    config_path: str,
    *,
    body_timeout: float,
    phrase_args: list[str],
    phrases_file: str,
    noise_count: int,
    speech_styles: list[str],
    limit: int,
    output_dir: str,
    manifest: str,
    vad_threshold: int,
    start_speech_ms: int,
    end_silence_ms: int,
    min_speech_ms: int,
    mic_channel: str,
    mic_preamp: float,
    debug_audio: bool,
    stackchan_mic_gain: int,
) -> int:
    config = load_config(config_path)
    phrases = load_capture_phrases(
        phrase_args=phrase_args,
        phrases_file=Path(phrases_file) if phrases_file else None,
        limit=limit,
    )
    output_path = Path(output_dir)
    manifest_path = Path(manifest) if manifest else output_path / "manifest.jsonl"
    output_path.mkdir(parents=True, exist_ok=True)

    loop = asyncio.get_running_loop()
    audio_queue: asyncio.Queue[tuple[bytes, int, int]] = asyncio.Queue(maxsize=80)
    accepting_audio = False
    audio_meter = {"last_at": 0.0, "max_rms": 0, "max_peak": 0, "chunks": 0}
    channel_selector = Pcm16ChannelSelector(mic_channel)

    def on_event(event) -> None:
        if event.type == "status":
            print(f"[StackChan] status: {event.payload}", flush=True)
            return
        if event.type == "touch":
            print(f"[StackChan] touch: {event.payload}", flush=True)
            return
        if event.type != "audio.in":
            return
        try:
            pcm, sample_rate, channels = decode_pcm_payload(event.payload)
        except ValueError as exc:
            print(f"[StackChan] bad audio.in: {exc}", flush=True)
            return
        try:
            pcm, channels = channel_selector.select(pcm, channels=channels)
        except ValueError as exc:
            print(f"[StackChan] bad mic channel: {exc}", flush=True)
            return
        pcm = apply_pcm16_gain(pcm, gain=mic_preamp)
        if debug_audio:
            rms = pcm16_rms(pcm)
            peak = _pcm16_peak(pcm)
            audio_meter["max_rms"] = max(int(audio_meter["max_rms"]), rms)
            audio_meter["max_peak"] = max(int(audio_meter["max_peak"]), peak)
            audio_meter["chunks"] = int(audio_meter["chunks"]) + 1
            now = time.monotonic()
            if now - float(audio_meter["last_at"]) >= 1.0:
                print(
                    "[mic] "
                    f"rms={audio_meter['max_rms']} peak={audio_meter['max_peak']} "
                    f"chunks={audio_meter['chunks']} accepting={accepting_audio}",
                    flush=True,
                )
                audio_meter.update({"last_at": now, "max_rms": 0, "max_peak": 0, "chunks": 0})
        if not accepting_audio:
            return

        def enqueue() -> None:
            if audio_queue.full():
                try:
                    audio_queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            audio_queue.put_nowait((pcm, sample_rate, channels))

        loop.call_soon_threadsafe(enqueue)

    controller = StackChanBodyController(port=config.stackchan.port, on_event=on_event)
    controller.start()
    print(f"Stacky STT capture server listening on 0.0.0.0:{config.stackchan.port}", flush=True)
    if not controller.wait_connected(body_timeout):
        print("StackChan did not connect yet. Start/flashing firmware and run this again.", flush=True)
        controller.stop()
        return 1

    address = controller.client_address
    where = f"{address[0]}:{address[1]}" if address else "StackChan"
    print(f"StackChan connected from {where}", flush=True)
    print(f"Using StackChan mic channel: {mic_channel}", flush=True)
    print(f"Setting StackChan mic gain: {stackchan_mic_gain}", flush=True)
    print(f"Applying StackChan mic preamp: {mic_preamp:.2f}x", flush=True)
    controller.set_mic_gain(stackchan_mic_gain)
    selected_styles = _resolve_capture_speech_styles(speech_styles)
    capture_items = [
        (_capture_prompt_for_style(phrase, style), phrase, False, style)
        for style in selected_styles
        for phrase in phrases
    ]
    for index in range(max(0, noise_count)):
        label = f"noise-{index + 1:02d}"
        capture_items.append((label, "", True, "noise"))
    print(f"Capturing {len(capture_items)} clip(s). Manifest: {manifest_path}", flush=True)
    controller.set_expression("listening")

    detector = EnergyTurnDetector(
        threshold=vad_threshold,
        start_speech_ms=start_speech_ms,
        min_speech_ms=min_speech_ms,
        end_silence_ms=end_silence_ms,
    )
    try:
        for index, (prompt_text, expected_text, allow_rejected, speech_style) in enumerate(capture_items, start=1):
            slug_source = f"{speech_style}-{expected_text}" if expected_text else f"{prompt_text}-non-speech"
            item_id = f"{index:03d}-{slugify(slug_source, max_length=52)}"
            wav_path = output_path / f"{item_id}.wav"
            print("", flush=True)
            if expected_text:
                print(f"[{index}/{len(capture_items)}] Sig præcist: {prompt_text}", flush=True)
            else:
                print(f"[{index}/{len(capture_items)}] Lav IKKE tale. Brug fx tastatur/klik/støj: {prompt_text}", flush=True)
            accepting_audio = True
            detector.reset()
            while True:
                pcm, sample_rate, channels = await audio_queue.get()
                turn = detector.push(pcm, sample_rate=sample_rate, channels=channels)
                if turn is None:
                    continue
                accepting_audio = False
                _drain_queue(audio_queue)
                detector.reset()
                quality = analyze_turn_signal(turn.pcm, sample_rate=turn.sample_rate, channels=turn.channels)
                if debug_audio:
                    print(f"[audio] {_format_signal_quality(quality)}", flush=True)
                if not quality.speech_like and not allow_rejected:
                    print(f"[capture] rejected ({quality.reason}); prøv sætningen igen.", flush=True)
                    accepting_audio = True
                    continue
                write_pcm_wav(wav_path, turn.pcm, sample_rate=turn.sample_rate, channels=turn.channels)
                write_dataset_record(
                    manifest_path,
                    audio_path=wav_path,
                    expected_text=expected_text,
                    item_id=item_id,
                    sample_rate=turn.sample_rate,
                    channels=turn.channels,
                    duration_seconds=quality.duration_seconds,
                    rms=quality.median_rms,
                    peak=quality.peak,
                    quality=_quality_record(quality),
                    speech_style=speech_style,
                )
                print(f"[capture] saved {wav_path.name} dur={quality.duration_seconds:.2f}s peak={quality.peak}", flush=True)
                break
        controller.set_expression("happy")
        print("", flush=True)
        print(f"Done. Dataset manifest: {manifest_path}", flush=True)
        return 0
    except (KeyboardInterrupt, asyncio.CancelledError):
        return 0
    finally:
        accepting_audio = False
        controller.set_expression("neutral")
        controller.stop()


def _resolve_capture_speech_styles(speech_styles: list[str]) -> list[str]:
    styles = speech_styles or ["normal"]
    result: list[str] = []
    seen: set[str] = set()
    for style in styles:
        value = style.strip().lower()
        if value and value not in seen:
            result.append(value)
            seen.add(value)
    return result or ["normal"]


def _capture_prompt_for_style(phrase: str, style: str) -> str:
    if style == "fast":
        return f"Sig hurtigt, men naturligt: {phrase}"
    if style == "mumble":
        return f"Muml lidt, men sig stadig sætningen: {phrase}"
    if style == "quiet":
        return f"Sig lavt, som i normal hverdagstale: {phrase}"
    return phrase


def _resolve_stt_bench_specs(engine_specs: list[str], *, include_heavy: bool) -> list[tuple[str, str]]:
    raw_specs = engine_specs or [f"{engine}:{model}" for engine, model in _LOW_LATENCY_STT_SPECS]
    if include_heavy:
        raw_specs = [*raw_specs, *(f"{engine}:{model}" for engine, model in _HEAVY_STT_SPECS)]

    specs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for raw in raw_specs:
        spec = _parse_stt_bench_spec(raw)
        if spec not in seen:
            specs.append(spec)
            seen.add(spec)
    return specs


def _parse_stt_bench_spec(raw: str) -> tuple[str, str]:
    value = raw.strip()
    if not value:
        raise ValueError("Empty STT bench spec.")
    alias = _STT_SPEC_ALIASES.get(value.lower())
    if alias is not None:
        return alias
    if ":" in value:
        engine, model = value.split(":", 1)
        engine = engine.strip().lower()
        model = model.strip()
        if engine not in {"wav2vec2", "whisper", "qwen3"} or not model:
            raise ValueError(f"Invalid STT bench spec: {raw}")
        return engine, model
    return "wav2vec2", value


def _resolve_stt_bench_items(
    audio_patterns: list[str],
    *,
    dataset_path: str,
    refs_path: str,
    limit: int,
) -> list[STTDatasetItem]:
    if dataset_path:
        items = load_dataset_manifest(Path(dataset_path))
        if limit > 0:
            items = items[:limit]
    else:
        paths = resolve_audio_inputs(
            audio_patterns,
            default_pattern=str(ROOT / "artifacts" / "handsfree_turns" / "*.wav"),
            limit=limit,
        )
        items = [STTDatasetItem(path) for path in paths]
    refs = load_reference_file(Path(refs_path)) if refs_path else {}
    return apply_references(items, refs)


def _append_stt_report(
    report_path: Path,
    *,
    engine: str,
    model: str,
    item: STTDatasetItem,
    result: STTResult,
    infer_seconds: float,
    wer: float | None,
    cer: float | None,
    hypothesis: str | None = None,
    correction_reason: str = "",
    live_gate_rejected: bool = False,
) -> None:
    duration = max(result.audio.duration_seconds, 0.001)
    final_hypothesis = result.text if hypothesis is None else hypothesis
    record = {
        "engine": engine,
        "model": model,
        "id": item.item_id or item.audio_path.stem,
        "audio": str(item.audio_path),
        "expected": item.expected_text,
        "hypothesis": final_hypothesis,
        "durationSeconds": round(result.audio.duration_seconds, 4),
        "inferSeconds": round(infer_seconds, 4),
        "rtf": round(infer_seconds / duration, 4),
        "wer": None if wer is None else round(wer, 6),
        "cer": None if cer is None else round(cer, 6),
        "avgLogprob": round(result.avg_logprob, 4),
        "noSpeechProb": round(result.no_speech_prob, 4),
        "rms": result.audio.rms,
        "peak": result.audio.peak,
    }
    if final_hypothesis != result.text:
        record["rawHypothesis"] = result.text
        record["correctionReason"] = correction_reason
    if live_gate_rejected:
        record["liveGateRejected"] = True
    if item.speech_style:
        record["speechStyle"] = item.speech_style
    with report_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def _wav_signal_quality(path: Path) -> TurnSignalQuality | None:
    try:
        with wave.open(str(path), "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            pcm = wav_file.readframes(wav_file.getnframes())
    except (OSError, wave.Error):
        return None
    return analyze_turn_signal(pcm, sample_rate=sample_rate, channels=channels)


def _quality_record(quality: TurnSignalQuality) -> dict[str, object]:
    return {
        "speechLike": quality.speech_like,
        "reason": quality.reason,
        "medianRms": quality.median_rms,
        "p80Rms": quality.p80_rms,
        "p95Rms": quality.p95_rms,
        "activeRatio": round(quality.active_ratio, 4),
        "activeMs": quality.active_ms,
        "maxActiveRunMs": quality.max_active_run_ms,
        "crestFactor": round(quality.crest_factor, 4),
        "zeroCrossingRate": round(quality.zero_crossing_rate, 6),
        "activeThreshold": quality.active_threshold,
    }


def _word_error_rate(reference: str, hypothesis: str) -> float:
    return word_error_rate(reference, hypothesis)


def _supertonic_voice(
    *,
    profile: str,
    voice_name: str,
    speed: float | None,
    steps: int | None,
    silence: float | None,
) -> SupertonicVoice:
    return supertonic_voice_preset(
        profile,
        voice_name=voice_name or None,
        speed=speed,
        total_steps=steps,
        silence_duration=silence,
    )


def _init(config_path: str) -> int:
    config = load_config(config_path)
    config.data_dir.mkdir(parents=True, exist_ok=True)
    wrote_soul = write_default_soul(config.soul_path, overwrite=False)
    if not Path(config_path).exists():
        example = ROOT / "configs" / "stacky.example.toml"
        if example.exists():
            Path(config_path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(example, config_path)
    memory = MemoryStore(config.memory_path)
    print(f"Stacky data: {config.data_dir}")
    print(f"Soul: {'created' if wrote_soul else 'already exists'} at {config.soul_path}")
    print(f"Fresh memory DB: {config.memory_path} ({memory.count()} memories)")
    return 0


def _create_brain(config) -> StackyBrain:
    soul = load_soul(config.soul_path)
    memory = MemoryStore(config.memory_path)
    self_model = StackySelfModel(config.data_dir)
    evolution = StackyEvolutionEngine(config.data_dir)
    memory_map = MemoryMapStore(config.data_dir / "memory_map.json")
    return StackyBrain(
        soul,
        memory,
        create_chat_client(config.lmstudio),
        InfiniteSessionStore(config.data_dir),
        self_model,
        evolution,
        memory_map,
    )


def _create_web_search_client(config, *, enabled_override: bool | None = None) -> WebSearchClient | None:
    enabled = config.websearch.enabled if enabled_override is None else enabled_override
    if not enabled:
        return None
    provider = config.websearch.provider.strip().lower()
    if provider in {"duckduckgo_lite", "duckduckgo-lite", "ddg_lite", "ddg"}:
        return DuckDuckGoLiteSearch(
            timeout_seconds=config.websearch.timeout_seconds,
            allow_insecure_tls_fallback=config.websearch.allow_insecure_tls_fallback,
        )
    raise ValueError(f"Ukendt websearch-provider: {config.websearch.provider}")


def _create_web_context_provider(config, *, enabled_override: bool | None = None, intent_brain=None):
    client = _create_web_search_client(config, enabled_override=enabled_override)
    max_results = config.websearch.max_results

    async def provider(text: str) -> str:
        return await _web_context_for_text(text, client, max_results=max_results, intent_brain=intent_brain)

    return provider


async def _web_context_for_text(
    text: str,
    client: WebSearchClient | None,
    *,
    max_results: int,
    intent_brain=None,
) -> str:
    if client is None:
        return ""
    intent = await classify_web_search_intent(text, intent_brain)
    if not intent.wants_search:
        return ""
    query = intent.query or extract_web_search_query(text)
    if not query:
        return format_web_search_context("", (), error="tom søgeforespørgsel")
    try:
        results = await asyncio.to_thread(lambda: client.search(query, max_results=max_results))
    except WebSearchError as exc:
        print(f"[web] search failed query={query!r}: {exc}", flush=True)
        return format_web_search_context(query, (), error=str(exc))
    print(f"[web] search query={query!r} results={len(results)}", flush=True)
    return format_web_search_context(query, results)


async def _web_search_once(config_path: str, query: str) -> int:
    config = load_config(config_path)
    client = _create_web_search_client(config, enabled_override=True)
    context = await _web_context_for_text(
        f"søg på nettet efter {query}",
        client,
        max_results=config.websearch.max_results,
    )
    print(_safe_console_text(context))
    return 0


def _create_computer_context_provider(config, *, enabled_override: bool | None = None):
    enabled = config.computer.enabled if enabled_override is None else enabled_override
    if not enabled:
        return None
    context = LocalComputerContext(
        config.computer.workspace_root,
        max_chars=config.computer.max_context_chars,
        timeout_seconds=config.computer.timeout_seconds,
    )

    async def provider(text: str) -> str:
        return await asyncio.to_thread(context.context_for, text)

    return provider


async def _computer_context_once(config_path: str, query: str) -> int:
    config = load_config(config_path)
    provider = _create_computer_context_provider(config, enabled_override=True)
    if provider is None:
        print("Computer-kontekst er slået fra.")
        return 1
    context = await provider(query)
    print(_safe_console_text(context or "Ingen computer-kontekst for denne forespørgsel."))
    return 0


def _self_status(config_path: str) -> int:
    config = load_config(config_path)
    self_model = StackySelfModel(config.data_dir)
    evolution = StackyEvolutionEngine(config.data_dir)
    memory_map = MemoryMapStore(config.data_dir / "memory_map.json")
    summary = self_model.summary()
    evolution_summary = evolution.summary()
    memory_map_summary = memory_map.summary()
    print(f"Stacky self-model: {summary['path']}")
    print(f"Stacky memory-map: {memory_map_summary['path']} ({memory_map_summary['count']} entries)")
    print(f"Trusted turns: {summary['trusted_turns']}")
    print(f"Untrusted voice turns: {summary['untrusted_turns']}")
    print(f"Tid: {summary['temporal']['wall_clock']} ({summary['temporal']['continuity']})")
    print(f"Nicolai-model: {summary['social']['mood']} / {summary['social']['phase']}")
    print("Style notes:")
    for note in summary["style_notes"] or ["Ingen endnu."]:
        print(f"- {note}")
    print("Convictions:")
    for conviction in summary["convictions"] or ["Ingen endnu."]:
        print(f"- {conviction}")
    print(f"Stacky evolution: {evolution_summary['path']}")
    print(f"Assistant turns measured: {evolution_summary['assistant_turns']}")
    print(f"Evolution observation: {evolution_summary['recent_summary']}")
    print("Tunings:")
    for key, value in evolution_summary["tuning"].items():
        print(f"- {key}: {value:.2f}")
    print("Evolution reflections:")
    for reflection in evolution_summary["reflections"] or ["Ingen endnu."]:
        print(f"- {reflection}")
    return 0


def _voice_lab_roest(phrases: list[str], *, play: bool, speaker: str) -> int:
    voice = roest_voice(speaker)
    tts = RoestTTS(voice)
    out_dir = ROOT / "artifacts" / f"voice_lab_roest_{speaker}"
    print(f"Using local Danish Roest voice: {voice.model_dir}", flush=True)
    print(f"Speaker prompt: {speaker} ({voice.prompt_path})", flush=True)
    print(f"Device: {tts.device}", flush=True)
    if tts.device == "cpu":
        print("Note: Roest is natural but slow on CPU; Piper remains the realtime fallback.", flush=True)
    for index, phrase in enumerate(phrases, start=1):
        adapted = adapt_for_danish_speech(phrase)
        filename = f"{index:02d}-{slugify(phrase, max_length=42)}.wav"
        output = out_dir / filename
        started = time.perf_counter()
        output = tts.synthesize_to_file(adapted, output)
        elapsed = time.perf_counter() - started
        print(f"{output} ({elapsed:.1f}s) :: {adapted}", flush=True)
        if play:
            subprocess.run(
                ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(output)],
                check=False,
            )
    return 0


def _voice_lab_supertonic(phrases: list[str], *, play: bool) -> int:
    from .voice.supertonic_tts import SupertonicTTS

    tts = SupertonicTTS()
    out_dir = ROOT / "artifacts" / "voice_lab_supertonic"
    print(f"Using local Danish Supertonic 3 voice: {tts.voice.voice_name}", flush=True)
    started = time.perf_counter()
    tts.load()
    print(f"Voice ready ({time.perf_counter() - started:.1f}s).", flush=True)
    for index, phrase in enumerate(phrases, start=1):
        adapted = adapt_for_danish_speech(phrase)
        filename = f"{index:02d}-{slugify(phrase, max_length=42)}.wav"
        output = out_dir / filename
        started = time.perf_counter()
        output = tts.synthesize_to_file(adapted, output)
        elapsed = time.perf_counter() - started
        print(f"{output} ({elapsed:.2f}s) :: {adapted}", flush=True)
        if play:
            subprocess.run(
                ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(output)],
                check=False,
            )
    return 0


async def _chat(config_path: str, *, speak: bool = False) -> int:
    config = load_config(config_path)
    brain = _create_brain(config)
    web_context_provider = _create_web_context_provider(config, intent_brain=brain.lmstudio)
    computer_context_provider = _create_computer_context_provider(config)
    output = await _speech_output(speak)
    await LocalTextVoiceRuntime(
        brain,
        output=output,
        web_context_provider=web_context_provider,
        computer_context_provider=computer_context_provider,
    ).interactive()
    return 0


async def _live_text(config_path: str, *, body_timeout: float, speak: bool = False) -> int:
    config = load_config(config_path)
    brain = _create_brain(config)
    web_context_provider = _create_web_context_provider(config, intent_brain=brain.lmstudio)
    computer_context_provider = _create_computer_context_provider(config)

    def on_event(event) -> None:
        if event.type in {"status", "touch"}:
            print(f"[StackChan] {event.type}: {event.payload}", flush=True)

    controller = StackChanBodyController(port=config.stackchan.port, on_event=on_event)
    controller.start()
    print(f"Stacky body server listening on 0.0.0.0:{config.stackchan.port}", flush=True)
    if controller.wait_connected(body_timeout):
        address = controller.client_address
        where = f"{address[0]}:{address[1]}" if address else "StackChan"
        print(f"StackChan connected from {where}", flush=True)
        controller.set_expression("happy")
    else:
        print("No StackChan connection yet; continuing text chat without body.", flush=True)
    try:
        output = await _speech_output(speak)
        await LocalTextVoiceRuntime(
            brain,
            output=output,
            presence=BodyPresence(controller),
            web_context_provider=web_context_provider,
            computer_context_provider=computer_context_provider,
        ).interactive()
    finally:
        controller.stop()
    return 0


async def _speaker_test(
    config_path: str,
    *,
    body_timeout: float,
    tts_engine: str,
    text: str,
    supertonic_profile: str,
    supertonic_voice: str,
    supertonic_speed: float | None,
    supertonic_steps: int | None,
    supertonic_silence: float | None,
    stackchan_target_rms: int,
    stackchan_max_gain: float,
    stackchan_volume: int,
) -> int:
    config = load_config(config_path)
    controller = StackChanBodyController(port=config.stackchan.port)
    controller.start()
    print(f"Stacky speaker-test server listening on 0.0.0.0:{config.stackchan.port}", flush=True)
    if not controller.wait_connected(body_timeout):
        print("StackChan did not connect yet.", flush=True)
        controller.stop()
        return 1
    output = create_stackchan_piper_output(
        controller,
        target_active_rms=stackchan_target_rms,
        max_gain=stackchan_max_gain,
        volume_level=stackchan_volume,
    )
    if tts_engine == "supertonic":
        output = create_stackchan_supertonic_output(
            controller,
            _supertonic_voice(
                profile=supertonic_profile,
                voice_name=supertonic_voice,
                speed=supertonic_speed,
                steps=supertonic_steps,
                silence=supertonic_silence,
            ),
            target_active_rms=stackchan_target_rms,
            max_gain=stackchan_max_gain,
            volume_level=stackchan_volume,
        )
    try:
        controller.set_expression("happy")
        print(f"Loading local Danish {tts_engine} voice...", flush=True)
        started = time.perf_counter()
        await output.preload()
        print(f"Voice ready ({time.perf_counter() - started:.1f}s). Speaking on StackChan.", flush=True)
        await output.speak(text)
        await output.wait()
        await asyncio.sleep(0.2)
        controller.set_expression("listening")
        return 0
    finally:
        await output.stop()
        controller.stop()


async def _speaker_tone(config_path: str, *, body_timeout: float, frequency: int, duration_ms: int) -> int:
    config = load_config(config_path)
    controller = StackChanBodyController(port=config.stackchan.port)
    controller.start()
    print(f"Stacky speaker-tone server listening on 0.0.0.0:{config.stackchan.port}", flush=True)
    if not controller.wait_connected(body_timeout):
        print("StackChan did not connect yet.", flush=True)
        controller.stop()
        return 1
    try:
        print(f"Playing firmware tone on StackChan: {frequency} Hz for {duration_ms} ms.", flush=True)
        ok = controller.speaker_tone(frequency=frequency, duration_ms=duration_ms)
        await asyncio.sleep(max(0.4, duration_ms / 1000 + 0.2))
        return 0 if ok else 1
    finally:
        controller.stop()


async def _motion_test(config_path: str, *, body_timeout: float, gesture_name: str, speed: int) -> int:
    config = load_config(config_path)
    controller = StackChanBodyController(port=config.stackchan.port)
    controller.start()
    print(f"Stacky motion-test server listening on 0.0.0.0:{config.stackchan.port}", flush=True)
    if not controller.wait_connected(body_timeout):
        print("StackChan did not connect yet.", flush=True)
        controller.stop()
        return 1
    try:
        calibration = load_body_calibration(config.data_dir)
        BodyDirector(controller, calibration).apply_calibration()
        actor = BodyDirector(controller, calibration)
        controller.set_expression("happy")
        print(f"Motion: {gesture_name}", flush=True)
        _run_motion_gesture(actor, gesture_name, speed=speed)
        controller.set_expression("listening")
        return 0
    finally:
        controller.stop()


async def _camera_test(
    config_path: str,
    *,
    body_timeout: float,
    frame_timeout: float,
    width: int,
    height: int,
    quality: int,
    discard_frames: int,
    settle_ms: int,
    ae_level: int,
    sensor_gain: int | None,
    sensor_exposure: int | None,
    enhance: bool,
    enhance_target: float,
    count: int,
    delay_ms: int,
    output: str,
) -> int:
    config = load_config(config_path)
    loop = asyncio.get_running_loop()
    frame_future: asyncio.Future[dict[str, object]] | None = None

    def on_event(event) -> None:
        nonlocal frame_future
        if event.type == "status":
            print(f"[StackChan] status: {event.payload}", flush=True)
            return
        if event.type == "vision.frame" and frame_future is not None and not frame_future.done():
            loop.call_soon_threadsafe(frame_future.set_result, dict(event.payload))

    controller = StackChanBodyController(port=config.stackchan.port, on_event=on_event)
    controller.start()
    print(f"Stacky camera-test server listening on 0.0.0.0:{config.stackchan.port}", flush=True)
    try:
        if not controller.wait_connected(body_timeout):
            print("StackChan did not connect yet.", flush=True)
            return 1

        address = controller.client_address
        where = f"{address[0]}:{address[1]}" if address else "StackChan"
        print(f"StackChan connected from {where}", flush=True)
        controller.set_expression("thinking")
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        capture_count = max(1, int(count))

        for index in range(capture_count):
            frame_future = loop.create_future()
            print(
                f"Requesting camera frame {index + 1}/{capture_count} "
                f"{width}x{height} quality={quality} discard={discard_frames} settle={settle_ms}ms "
                f"ae={ae_level} gain={sensor_gain if sensor_gain is not None else 'auto'} "
                f"exposure={sensor_exposure if sensor_exposure is not None else 'auto'}.",
                flush=True,
            )
            if not controller.capture_vision_frame(
                width=width,
                height=height,
                quality=quality,
                discard_frames=discard_frames,
                settle_ms=settle_ms,
                ae_level=ae_level,
                sensor_gain=sensor_gain,
                sensor_exposure=sensor_exposure,
            ):
                print("Failed to send vision.capture.", flush=True)
                return 1

            try:
                payload = await asyncio.wait_for(frame_future, timeout=frame_timeout)
            except TimeoutError:
                print("Timed out waiting for vision.frame.", flush=True)
                return 1

            if not bool(payload.get("available", False)):
                print(f"Camera unavailable: {payload.get('reason', 'unknown')}", flush=True)
                return 1

            try:
                jpeg = decode_vision_frame_payload(payload)
            except ValueError as exc:
                print(f"Invalid vision frame: {exc}", flush=True)
                return 1

            frame_path = output_path
            if capture_count > 1:
                frame_path = output_path.with_name(f"{output_path.stem}-{index + 1:02d}{output_path.suffix}")
            metadata = {key: value for key, value in payload.items() if key != "data"}
            llm_path: Path | None = None
            if enhance:
                llm_path = frame_path.with_name(f"{frame_path.stem}-llm{frame_path.suffix}")
                llm_jpeg = _enhance_camera_jpeg(jpeg, target_mean=enhance_target)
                llm_path.write_bytes(llm_jpeg)
                metadata["llmEnhanced"] = True
                metadata["llmPath"] = str(llm_path)
                metadata["enhanceTargetMean"] = enhance_target
            else:
                metadata["llmEnhanced"] = False
            frame_path.write_bytes(jpeg)
            frame_path.with_suffix(".json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            llm_note = f", llm={llm_path} stats={_image_stats(llm_path)}" if llm_path is not None else ""
            print(
                f"Saved camera frame: {frame_path} "
                f"({payload.get('width')}x{payload.get('height')}, jpeg={len(jpeg)} bytes, "
                f"source={_fourcc(payload.get('sourceFormat'))}/{payload.get('sourceBytes')} bytes, "
                f"stats={_image_stats(frame_path)}{llm_note}).",
                flush=True,
            )
            if index + 1 < capture_count and delay_ms > 0:
                await asyncio.sleep(delay_ms / 1000)

        controller.set_expression("happy")
        return 0
    finally:
        controller.stop()


async def _sensor_test(config_path: str, *, body_timeout: float, event_timeout: float) -> int:
    config = load_config(config_path)
    loop = asyncio.get_running_loop()
    scan_future: asyncio.Future[dict[str, object]] | None = None

    def on_event(event) -> None:
        nonlocal scan_future
        if event.type == "status":
            print(f"[StackChan] status: {event.payload}", flush=True)
            return
        if event.type == "i2c.scan" and scan_future is not None and not scan_future.done():
            loop.call_soon_threadsafe(scan_future.set_result, dict(event.payload))
            return
        print(f"[StackChan] event {event.type}: {event.payload}", flush=True)

    controller = StackChanBodyController(port=config.stackchan.port, on_event=on_event)
    controller.start()
    print(f"Stacky sensor-test server listening on 0.0.0.0:{config.stackchan.port}", flush=True)
    try:
        if not controller.wait_connected(body_timeout):
            print("StackChan did not connect yet.", flush=True)
            return 1
        address = controller.client_address
        where = f"{address[0]}:{address[1]}" if address else "StackChan"
        print(f"StackChan connected from {where}", flush=True)
        controller.request_status()
        scan_future = loop.create_future()
        print("Requesting I2C scan.", flush=True)
        if not controller.request_i2c_scan():
            print("Failed to send body.i2c_scan.", flush=True)
            return 1
        try:
            payload = await asyncio.wait_for(scan_future, timeout=event_timeout)
        except TimeoutError:
            print("Timed out waiting for i2c.scan.", flush=True)
            return 1
        devices = payload.get("devices", [])
        if isinstance(devices, list):
            labels = []
            for device in devices:
                if isinstance(device, dict):
                    name = str(device.get("name") or "unknown")
                    labels.append(f"{device.get('hex', '?')}:{name}")
            print(f"I2C devices: {', '.join(labels) if labels else 'none'}", flush=True)
        proximity = bool(payload.get("proximityAvailable", False))
        if proximity:
            print("Proximity sensor candidate detected on I2C.", flush=True)
        else:
            print("No physical I2C proximity sensor detected; use camera-derived proximity or add external hardware.", flush=True)
        return 0
    finally:
        controller.stop()


def _fourcc(value: object) -> str:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return str(value)
    chars = "".join(chr((number >> shift) & 0xFF) for shift in (0, 8, 16, 24))
    printable = "".join(char if 32 <= ord(char) <= 126 else "." for char in chars)
    return f"{printable}/0x{number:08x}"


def _enhance_camera_jpeg(jpeg: bytes, *, target_mean: float = 96.0) -> bytes:
    try:
        from PIL import Image, ImageEnhance, ImageStat

        with Image.open(BytesIO(jpeg)) as image:
            rgb = image.convert("RGB")
        stat = ImageStat.Stat(rgb)
        luma = 0.2126 * stat.mean[0] + 0.7152 * stat.mean[1] + 0.0722 * stat.mean[2]
        if luma <= 1:
            return jpeg
        factor = min(5.0, max(1.0, float(target_mean) / luma))
        enhanced = ImageEnhance.Brightness(rgb).enhance(factor)
        enhanced = ImageEnhance.Contrast(enhanced).enhance(1.05)
        output = BytesIO()
        enhanced.save(output, format="JPEG", quality=85, optimize=True)
        return output.getvalue()
    except Exception:
        return jpeg


def _image_stats(path: Path) -> str:
    try:
        from PIL import Image, ImageStat

        with Image.open(path) as image:
            stat = ImageStat.Stat(image.convert("RGB"))
        mean = ",".join(f"{value:.1f}" for value in stat.mean)
        extrema = ",".join(f"{low}-{high}" for low, high in stat.extrema)
        return f"mean={mean} range={extrema}"
    except Exception:
        return "n/a"


async def _handsfree(
    config_path: str,
    *,
    body_timeout: float,
    stt_engine: str,
    stt_model: str,
    vad_threshold: int,
    start_speech_ms: int,
    end_silence_ms: int,
    min_speech_ms: int,
    speaker: str,
    tts_engine: str,
    supertonic_profile: str,
    supertonic_voice: str,
    supertonic_speed: float | None,
    supertonic_steps: int | None,
    supertonic_silence: float | None,
    stackchan_target_rms: int,
    stackchan_max_gain: float,
    stackchan_volume: int,
    stackchan_mic_gain: int,
    reply_chars: int,
    detail_reply_chars: int,
    vision: bool,
    vision_image: bool,
    face_tracking: bool,
    vision_interval: float,
    vision_prompt_timeout: float,
    websearch: bool | None,
    computer: bool | None,
    monitor: bool | None,
    voice_trust: str,
    mic_channel: str,
    mic_preamp: float,
    listen_only: bool,
    debug_audio: bool,
) -> int:
    config = load_config(config_path)
    brain = None
    if not listen_only:
        brain = _create_brain(config)
    voice_policy = _voice_memory_policy(voice_trust)
    web_context_provider = (
        _create_web_context_provider(config, enabled_override=websearch, intent_brain=brain.lmstudio if brain else None)
        if not listen_only
        else None
    )
    computer_enabled = config.computer.enabled if computer is None else computer
    computer_context_provider = (
        _create_computer_context_provider(config, enabled_override=computer) if not listen_only else None
    )
    computer_actions = (
        LocalComputerActions(config.computer.workspace_root, timeout_seconds=config.computer.timeout_seconds)
        if not listen_only and computer_enabled
        else None
    )
    sandcode_client = SandcodeMobileHostClient(config.sandcode) if not listen_only else None
    runtime_state = RuntimeState()
    monitor_enabled = (config.monitor.enabled if monitor is None else monitor) and not listen_only
    monitor_queue: asyncio.Queue[MonitorObservation] | None = (
        asyncio.Queue(maxsize=8) if monitor_enabled else None
    )
    recent_monitor_observations: deque[MonitorObservation] = deque(maxlen=max(1, config.monitor.max_context_observations))
    friend_monitor: GlobalFriendMonitor | None = None
    last_user_voice_at = time.monotonic()
    last_stacky_speech_at = last_user_voice_at
    last_monitor_comment_at = 0.0
    if monitor_enabled:
        voice_bits = [tts_engine, speaker]
        if tts_engine == "supertonic":
            voice_bits.append(supertonic_voice or supertonic_profile)
        friend_monitor = GlobalFriendMonitor(
            config.monitor,
            DefaultMonitorProbe(
                monitor_config=config.monitor,
                websearch_config=config.websearch,
                sandcode_config=config.sandcode,
                voice_mode="/".join(voice_bits),
            ),
        )

    loop = asyncio.get_running_loop()
    audio_queue: asyncio.Queue[tuple[bytes, int, int]] = asyncio.Queue(maxsize=80)
    vision_payload_queue: asyncio.Queue[dict[str, object]] = asyncio.Queue(maxsize=4)
    vision_snapshot_queue: asyncio.Queue[VisionSnapshot] = asyncio.Queue(maxsize=4)
    vision_state = VisionState(create_face_detector(auto_download_yunet=True)) if vision else None
    accepting_audio = False
    body_state_name = "neutral"
    detector: EnergyTurnDetector | None = None
    motion_audio_guard_until = 0.0
    audio_meter = {"last_at": 0.0, "max_rms": 0, "max_peak": 0, "chunks": 0}
    channel_selector = Pcm16ChannelSelector(mic_channel)
    body_status: dict[str, object] = {}
    body_calibration = load_body_calibration(config.data_dir)
    body_director: BodyDirector | None = None
    display_brightness_level = 80
    last_brightness_direction = 0

    def guard_audio_after_body_motion() -> None:
        nonlocal motion_audio_guard_until
        motion_audio_guard_until = max(motion_audio_guard_until, time.monotonic() + 0.45)

    def guard_audio_after_tracking_motion() -> None:
        nonlocal motion_audio_guard_until
        motion_audio_guard_until = max(motion_audio_guard_until, time.monotonic() + 0.45)

    def on_event(event) -> None:
        nonlocal display_brightness_level
        if event.type == "status":
            body_status.clear()
            body_status.update(event.payload)
            try:
                display_brightness_level = _clamp_percent(
                    int(event.payload.get("displayBrightness", display_brightness_level)),
                    minimum=1,
                )
            except (TypeError, ValueError):
                pass
            print(f"[StackChan] status: {event.payload}", flush=True)
            return
        if event.type == "touch":
            print(f"[StackChan] touch: {event.payload}", flush=True)
            payload = dict(event.payload)

            def react_touch() -> None:
                if body_director is not None:
                    asyncio.create_task(
                        _body_event_reaction(
                            body_director,
                            "touch",
                            payload,
                            on_motion=guard_audio_after_body_motion,
                        )
                    )

            loop.call_soon_threadsafe(react_touch)
            return
        if event.type == "proximity":
            if debug_audio or listen_only:
                print(f"[StackChan] proximity: {event.payload}", flush=True)
            payload = dict(event.payload)

            def react_proximity() -> None:
                if body_director is not None:
                    asyncio.create_task(
                        _body_event_reaction(
                            body_director,
                            "proximity",
                            payload,
                            on_motion=guard_audio_after_body_motion,
                        )
                    )

            loop.call_soon_threadsafe(react_proximity)
            return
        if event.type == "vision.frame":
            if vision_state is None:
                return

            def enqueue_vision() -> None:
                if vision_payload_queue.full():
                    try:
                        vision_payload_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                vision_payload_queue.put_nowait(dict(event.payload))

            loop.call_soon_threadsafe(enqueue_vision)
            return
        if event.type != "audio.in":
            if debug_audio or listen_only:
                print(f"[debug] on_event non-audio.in type={event.type!r}", flush=True)
            return
        on_event._audio_in_count = getattr(on_event, "_audio_in_count", 0) + 1
        if (debug_audio or listen_only) and (on_event._audio_in_count <= 3 or on_event._audio_in_count % 200 == 0):
            print(f"[debug] audio.in #{on_event._audio_in_count} accepting={accepting_audio}", flush=True)
        try:
            pcm, sample_rate, channels = decode_pcm_payload(event.payload)
        except ValueError as exc:
            print(f"[StackChan] bad audio.in: {exc}", flush=True)
            return
        try:
            pcm, channels = channel_selector.select(pcm, channels=channels)
        except ValueError as exc:
            print(f"[StackChan] bad mic channel: {exc}", flush=True)
            return
        pcm = apply_pcm16_gain(pcm, gain=mic_preamp)
        if debug_audio or listen_only:
            rms = pcm16_rms(pcm)
            peak = _pcm16_peak(pcm)
            audio_meter["max_rms"] = max(int(audio_meter["max_rms"]), rms)
            audio_meter["max_peak"] = max(int(audio_meter["max_peak"]), peak)
            audio_meter["chunks"] = int(audio_meter["chunks"]) + 1
            now = time.monotonic()
            if now - float(audio_meter["last_at"]) >= 1.0:
                print(
                    "[mic] "
                    f"rms={audio_meter['max_rms']} peak={audio_meter['max_peak']} "
                    f"chunks={audio_meter['chunks']} accepting={accepting_audio}",
                    flush=True,
                )
                audio_meter.update({"last_at": now, "max_rms": 0, "max_peak": 0, "chunks": 0})
        if not accepting_audio:
            return

        def enqueue() -> None:
            if audio_queue.full():
                try:
                    audio_queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            audio_queue.put_nowait((pcm, sample_rate, channels))

        loop.call_soon_threadsafe(enqueue)

    controller = StackChanBodyController(port=config.stackchan.port, on_event=on_event)
    controller.start()
    print(f"Stacky hands-free server listening on 0.0.0.0:{config.stackchan.port}", flush=True)
    if not controller.wait_connected(body_timeout):
        print("StackChan did not connect yet. Start/flashing firmware and run this again.", flush=True)
        controller.stop()
        return 1

    address = controller.client_address
    where = f"{address[0]}:{address[1]}" if address else "StackChan"
    print(f"StackChan connected from {where}", flush=True)
    print(f"Using StackChan mic channel: {mic_channel}", flush=True)
    print(f"Setting StackChan mic gain: {stackchan_mic_gain}", flush=True)
    print(f"Applying StackChan mic preamp: {mic_preamp:.2f}x", flush=True)
    if not listen_only:
        print(f"Voice memory mode: {voice_trust}", flush=True)
    controller.set_mic_gain(stackchan_mic_gain)
    body_director = BodyDirector(controller, body_calibration)
    body_director.apply_calibration()
    if brain is not None:
        body_director.set_presence_mode(brain.presence_mode())
        body_director.set_stacky_mood(brain.stacky_mood_name())

    def sync_body_personality() -> None:
        if brain is None or body_director is None:
            return
        body_director.set_presence_mode(brain.presence_mode())
        body_director.set_stacky_mood(brain.stacky_mood_name())

    def set_body_state(name: str) -> bool:
        nonlocal body_state_name
        body_state_name = name
        return body_director.set_state(name) if body_director is not None else controller.set_expression(name)

    def should_track_face() -> bool:
        return bool(
            face_tracking
            and accepting_audio
            and body_state_name == "listening"
            and detector is not None
            and not detector.active
            and time.monotonic() >= motion_audio_guard_until
        )

    vision_processor_task = None
    vision_capture_task = None
    body_presence_task = None
    monitor_task = None
    if vision_state is not None:
        print(f"Vision mode ready ({vision_state.detector_status}).", flush=True)
        vision_processor_task = asyncio.create_task(
            _vision_processor_loop(
                vision_state,
                vision_payload_queue,
                vision_snapshot_queue,
                body_director,
                should_track_face=should_track_face,
                on_tracking_motion=guard_audio_after_tracking_motion,
                debug=debug_audio,
            )
        )
        vision_capture_task = asyncio.create_task(
            _vision_capture_loop(
                controller,
                interval_seconds=vision_interval,
                should_capture=lambda: _should_capture_vision_runtime(
                    controller_connected=controller.connected,
                    accepting_audio=accepting_audio,
                ),
            )
        )
    if friend_monitor is not None and monitor_queue is not None:
        print("Global friend monitor ready (read-only sparse sanseinput).", flush=True)
        monitor_task = asyncio.create_task(friend_monitor.run(monitor_queue))

    def record_local_turn(user_text: str, assistant_text: str) -> None:
        if brain is None:
            return
        brain.record_observed_turn(
            user_text,
            assistant_text,
            persist_session=voice_policy.persist_session,
            allow_memory_writes=voice_policy.allow_memory_writes,
            remember_recent=voice_policy.remember_recent,
            session_source=voice_policy.session_source,
        )
        sync_body_personality()

    set_body_state("thinking")

    output = None
    if listen_only:
        print("Listen-only mode: Stacky prints what it heard but does not answer.", flush=True)
    else:
        if tts_engine == "supertonic":
            voice = _supertonic_voice(
                profile=supertonic_profile,
                voice_name=supertonic_voice,
                speed=supertonic_speed,
                steps=supertonic_steps,
                silence=supertonic_silence,
            )
            output = (
                create_stackchan_supertonic_output(
                    controller,
                    voice,
                    target_active_rms=stackchan_target_rms,
                    max_gain=stackchan_max_gain,
                    volume_level=stackchan_volume,
                )
                if speaker == "stackchan"
                else create_supertonic_output(voice)
            )
            print("Loading local Danish Supertonic voice...", flush=True)
        else:
            output = (
                create_stackchan_piper_output(
                    controller,
                    target_active_rms=stackchan_target_rms,
                    max_gain=stackchan_max_gain,
                    volume_level=stackchan_volume,
                )
                if speaker == "stackchan"
                else create_fast_piper_output()
            )
            print("Loading local Danish Piper voice...", flush=True)
        started = time.perf_counter()
        try:
            await output.preload()
        except Exception as exc:
            if tts_engine != "supertonic":
                raise
            print(f"Supertonic kunne ikke starte ({exc}). Falder tilbage til Piper.", flush=True)
            output = (
                create_stackchan_piper_output(
                    controller,
                    target_active_rms=stackchan_target_rms,
                    max_gain=stackchan_max_gain,
                    volume_level=stackchan_volume,
                )
                if speaker == "stackchan"
                else create_fast_piper_output()
            )
            started = time.perf_counter()
            await output.preload()
            print(f"Piper fallback ready ({time.perf_counter() - started:.1f}s).", flush=True)
        else:
            print(f"Voice ready ({time.perf_counter() - started:.1f}s).", flush=True)

    stt = create_danish_stt(stt_engine, stt_model or None)
    stt_name = getattr(stt, "model_id", getattr(stt, "model_size", stt_model or "default"))
    print(f"Loading local Danish STT model ({stt_engine}: {stt_name})...", flush=True)
    started = time.perf_counter()
    await stt.preload()
    print(f"STT ready ({time.perf_counter() - started:.1f}s). Speak to StackChan now.", flush=True)

    async def speak_reply(spoken: str) -> None:
        nonlocal last_stacky_speech_at
        if output is None:
            return
        speaking_animation_task = (
            asyncio.create_task(
                _speaking_body_loop(
                    body_director,
                    spoken,
                    on_motion=guard_audio_after_body_motion,
                )
            )
            if body_director is not None
            else None
        )
        try:
            await output.speak(spoken)
            await output.wait()
        finally:
            if speaking_animation_task is not None:
                speaking_animation_task.cancel()
                try:
                    await speaking_animation_task
                except asyncio.CancelledError:
                    pass
        last_stacky_speech_at = time.monotonic()
        if friend_monitor is not None:
            friend_monitor.mark_stacky_speech(last_stacky_speech_at)

    detector = EnergyTurnDetector(
        threshold=vad_threshold,
        start_speech_ms=start_speech_ms,
        min_speech_ms=min_speech_ms,
        end_silence_ms=end_silence_ms,
    )
    turn_index = 0
    last_transcript = ""
    last_transcript_at = 0.0
    accepting_audio = True
    set_body_state("listening")
    if body_director is not None:
        body_presence_task = asyncio.create_task(
            _body_presence_loop(
                body_director,
                get_state=lambda: body_state_name,
                should_tick=lambda: accepting_audio and time.monotonic() >= motion_audio_guard_until,
                on_motion=guard_audio_after_body_motion,
            )
        )
    try:
        while True:
            observation = _pop_monitor_observation(monitor_queue)
            if observation is not None:
                recent_monitor_observations.append(observation)
                print(f"[monitor] {observation.kind}: {observation.summary}", flush=True)
                if brain is not None:
                    stored_sense = brain.observe_monitor_observation(observation)
                    if stored_sense:
                        print(f"[monitor] sanse-dagbog: {stored_sense[0]}", flush=True)
                    sync_body_personality()
                now = time.monotonic()
                presence_mode = brain.presence_mode() if brain is not None else "stille_ven"
                if (
                    brain is not None
                    and output is not None
                    and _should_comment_on_monitor_observation(
                        observation,
                        config.monitor,
                        presence_mode=presence_mode,
                        accepting_audio=accepting_audio,
                        body_state_name=body_state_name,
                        now=now,
                        last_user_voice_at=last_user_voice_at,
                        last_stacky_speech_at=last_stacky_speech_at,
                        last_monitor_comment_at=last_monitor_comment_at,
                    )
                ):
                    accepting_audio = False
                    _drain_queue(audio_queue)
                    detector.reset()
                    set_body_state("thinking")
                    monitor_context = format_monitor_context([observation], max_items=1)
                    reply = await brain.respond(
                        monitor_prompt_for_observation(
                            observation,
                            presence_mode=presence_mode,
                            stacky_mood=brain.stacky_mood_name(),
                        ),
                        max_spoken_chars=config.monitor.max_spoken_chars,
                        detail_spoken_chars=config.monitor.max_spoken_chars,
                        persist_session=False,
                        allow_memory_writes=False,
                        remember_recent=False,
                        session_source="stacky-monitor",
                        observe_turn=False,
                        monitor_context=monitor_context,
                        runtime_context=runtime_state.context_for_prompt(),
                    )
                    set_body_state("speaking")
                    if body_director is not None:
                        await asyncio.to_thread(body_director.reply_started, reply.spoken_text or reply.text)
                    await speak_reply(reply.spoken_text or reply.text)
                    last_monitor_comment_at = last_stacky_speech_at
                    _drain_queue(audio_queue)
                    detector.reset()
                    await asyncio.sleep(0.25)
                    set_body_state("listening")
                    accepting_audio = True
                continue
            pcm, sample_rate, channels = await audio_queue.get()
            if time.monotonic() < motion_audio_guard_until:
                detector.reset()
                continue
            turn = detector.push(pcm, sample_rate=sample_rate, channels=channels)
            if turn is None:
                continue
            accepting_audio = False
            _drain_queue(audio_queue)
            detector.reset()
            turn_index += 1
            pipeline_started = time.perf_counter()
            wav_path = write_pcm_wav(
                ROOT / "artifacts" / "handsfree_turns" / f"turn-{turn_index:04d}.wav",
                turn.pcm,
                sample_rate=turn.sample_rate,
                channels=turn.channels,
            )
            signal_quality = analyze_turn_signal(turn.pcm, sample_rate=turn.sample_rate, channels=turn.channels)
            print(f"[audio] {_format_signal_quality(signal_quality)}", flush=True)
            if not signal_quality.speech_like:
                print(f"[Stacky] ignorerer audio ({signal_quality.reason})", flush=True)
                set_body_state("listening")
                accepting_audio = True
                continue
            set_body_state("thinking")
            stt_started = time.perf_counter()
            stt_result = await stt.transcribe_wav_result(wav_path)
            stt_seconds = time.perf_counter() - stt_started
            raw_text = _clean_transcript(stt_result.text)
            correction = correct_danish_transcript(raw_text)
            text = _clean_transcript(correction.text)
            print(f"[STT] {_format_stt_result(stt_result, text)}", flush=True)
            if correction.changed:
                print(
                    f"[STT] corrected raw={correction.raw_text!r} -> {text!r} ({correction.reason})",
                    flush=True,
                )
            accepted, reason = _accept_stt_result(
                stt_result,
                text,
                signal_quality=signal_quality,
                trusted_transcript=correction.reason in {"exact", "phrase"},
            )
            if not accepted:
                print(f"[Stacky] ignorerer STT ({reason}): {text}", flush=True)
                set_body_state("listening")
                accepting_audio = True
                continue
            transcript_key = _transcript_key(text)
            now = time.monotonic()
            if transcript_key and transcript_key == last_transcript and now - last_transcript_at < 6.0:
                print(f"[Stacky] ignorerer gentaget STT: {text}", flush=True)
                set_body_state("listening")
                accepting_audio = True
                continue
            last_transcript = transcript_key
            last_transcript_at = now
            last_user_voice_at = time.monotonic()
            if friend_monitor is not None:
                friend_monitor.mark_user_turn(last_user_voice_at)
            print(f"Nicolai: {text}", flush=True)
            if listen_only:
                set_body_state("listening")
                accepting_audio = True
                continue
            if brain is None or output is None:
                set_body_state("listening")
                accepting_audio = True
                continue
            presence_command = _parse_presence_mode_command(text)
            if presence_command is not None:
                reply_started = time.perf_counter()
                mode = brain.set_presence_mode(presence_command.mode, source="stackchan-voice-command")
                sync_body_personality()
                print(f"[Stacky] presence_mode={mode}", flush=True)
                set_body_state("happy")
                speak_started = time.perf_counter()
                record_local_turn(text, presence_command.spoken)
                await speak_reply(presence_command.spoken)
                speak_seconds = time.perf_counter() - speak_started
                print(
                    f"[timing] stt={stt_seconds:.2f}s presence={time.perf_counter() - reply_started:.2f}s "
                    f"tts_send={speak_seconds:.2f}s total={time.perf_counter() - pipeline_started:.2f}s",
                    flush=True,
                )
                _drain_queue(audio_queue)
                detector.reset()
                await asyncio.sleep(0.25)
                set_body_state("listening")
                accepting_audio = True
                continue
            memory_write = _parse_memory_map_write_command(text)
            if memory_write is not None:
                reply_started = time.perf_counter()
                spoken_reply = brain.remember_memory_map(memory_write, source="stackchan-voice-command")
                record_local_turn(text, spoken_reply)
                set_body_state("happy")
                speak_started = time.perf_counter()
                await speak_reply(spoken_reply)
                speak_seconds = time.perf_counter() - speak_started
                print(
                    f"[timing] stt={stt_seconds:.2f}s memory_map={time.perf_counter() - reply_started:.2f}s "
                    f"tts_send={speak_seconds:.2f}s total={time.perf_counter() - pipeline_started:.2f}s",
                    flush=True,
                )
                _drain_queue(audio_queue)
                detector.reset()
                await asyncio.sleep(0.25)
                set_body_state("listening")
                accepting_audio = True
                continue
            if _wants_capability_report(text):
                reply_started = time.perf_counter()
                spoken_reply = brain.capability_reply()
                record_local_turn(text, spoken_reply)
                set_body_state("speaking")
                speak_started = time.perf_counter()
                await speak_reply(spoken_reply)
                speak_seconds = time.perf_counter() - speak_started
                print(
                    f"[timing] stt={stt_seconds:.2f}s capability={time.perf_counter() - reply_started:.2f}s "
                    f"tts_send={speak_seconds:.2f}s total={time.perf_counter() - pipeline_started:.2f}s",
                    flush=True,
                )
                _drain_queue(audio_queue)
                detector.reset()
                await asyncio.sleep(0.25)
                set_body_state("listening")
                accepting_audio = True
                continue
            if _wants_memory_map_recall(text):
                reply_started = time.perf_counter()
                spoken_reply = brain.memory_map_reply(text)
                record_local_turn(text, spoken_reply)
                set_body_state("speaking")
                speak_started = time.perf_counter()
                await speak_reply(spoken_reply)
                speak_seconds = time.perf_counter() - speak_started
                print(
                    f"[timing] stt={stt_seconds:.2f}s memory_map={time.perf_counter() - reply_started:.2f}s "
                    f"tts_send={speak_seconds:.2f}s total={time.perf_counter() - pipeline_started:.2f}s",
                    flush=True,
                )
                _drain_queue(audio_queue)
                detector.reset()
                await asyncio.sleep(0.25)
                set_body_state("listening")
                accepting_audio = True
                continue
            if _wants_sense_diary_recall(text):
                reply_started = time.perf_counter()
                set_body_state("thinking")
                spoken_reply = brain.sense_diary_reply()
                record_local_turn(text, spoken_reply)
                set_body_state("speaking")
                speak_started = time.perf_counter()
                await speak_reply(spoken_reply)
                speak_seconds = time.perf_counter() - speak_started
                print(
                    f"[timing] stt={stt_seconds:.2f}s sense_diary={time.perf_counter() - reply_started:.2f}s "
                    f"tts_send={speak_seconds:.2f}s total={time.perf_counter() - pipeline_started:.2f}s",
                    flush=True,
                )
                _drain_queue(audio_queue)
                detector.reset()
                await asyncio.sleep(0.25)
                set_body_state("listening")
                accepting_audio = True
                continue
            if _wants_stacky_state_report(text):
                reply_started = time.perf_counter()
                spoken_reply = brain.stacky_state_reply()
                record_local_turn(text, spoken_reply)
                set_body_state("speaking")
                speak_started = time.perf_counter()
                await speak_reply(spoken_reply)
                speak_seconds = time.perf_counter() - speak_started
                print(
                    f"[timing] stt={stt_seconds:.2f}s stacky_state={time.perf_counter() - reply_started:.2f}s "
                    f"tts_send={speak_seconds:.2f}s total={time.perf_counter() - pipeline_started:.2f}s",
                    flush=True,
                )
                _drain_queue(audio_queue)
                detector.reset()
                await asyncio.sleep(0.25)
                set_body_state("listening")
                accepting_audio = True
                continue
            if _wants_runtime_status_reply(text):
                reply_started = time.perf_counter()
                spoken_reply = runtime_state.status_reply(text)
                record_local_turn(text, spoken_reply)
                set_body_state("speaking")
                speak_started = time.perf_counter()
                await speak_reply(spoken_reply)
                speak_seconds = time.perf_counter() - speak_started
                print(
                    f"[timing] stt={stt_seconds:.2f}s runtime_status={time.perf_counter() - reply_started:.2f}s "
                    f"tts_send={speak_seconds:.2f}s total={time.perf_counter() - pipeline_started:.2f}s",
                    flush=True,
                )
                _drain_queue(audio_queue)
                detector.reset()
                await asyncio.sleep(0.25)
                set_body_state("listening")
                accepting_audio = True
                continue
            if _parse_battery_status_command(text):
                reply_started = time.perf_counter()
                ok = controller.request_status()
                await asyncio.sleep(0.12)
                reply_seconds = time.perf_counter() - reply_started
                print(f"[Stacky] battery_status ok={ok} status={_battery_status_debug(body_status)}", flush=True)
                set_body_state("happy")
                speak_started = time.perf_counter()
                spoken_reply = _format_battery_status_reply(body_status) if ok else "Jeg kunne ikke hente batteristatus lige nu."
                record_local_turn(text, spoken_reply)
                await speak_reply(spoken_reply)
                speak_seconds = time.perf_counter() - speak_started
                print(
                    f"[timing] stt={stt_seconds:.2f}s command={reply_seconds:.2f}s "
                    f"tts_send={speak_seconds:.2f}s total={time.perf_counter() - pipeline_started:.2f}s",
                    flush=True,
                )
                _drain_queue(audio_queue)
                detector.reset()
                await asyncio.sleep(0.25)
                set_body_state("listening")
                accepting_audio = True
                continue
            volume_command = _parse_volume_command(text, current_level=getattr(output, "volume_level", stackchan_volume))
            if volume_command is not None and hasattr(output, "set_volume"):
                volume_level, spoken = volume_command
                reply_started = time.perf_counter()
                ok = output.set_volume(volume_level)
                reply_seconds = time.perf_counter() - reply_started
                print(f"[Stacky] volumen={volume_level} ok={ok}", flush=True)
                set_body_state("happy")
                speak_started = time.perf_counter()
                spoken_reply = spoken if ok else "Jeg kunne ikke ændre min volumen lige nu."
                record_local_turn(text, spoken_reply)
                await speak_reply(spoken_reply)
                speak_seconds = time.perf_counter() - speak_started
                print(
                    f"[timing] stt={stt_seconds:.2f}s command={reply_seconds:.2f}s "
                    f"tts_send={speak_seconds:.2f}s total={time.perf_counter() - pipeline_started:.2f}s",
                    flush=True,
                )
                _drain_queue(audio_queue)
                detector.reset()
                await asyncio.sleep(0.25)
                set_body_state("listening")
                accepting_audio = True
                continue
            brightness_command = _parse_display_brightness_command(
                text,
                current_level=display_brightness_level,
                previous_direction=last_brightness_direction,
            )
            if brightness_command is not None:
                reply_started = time.perf_counter()
                display_brightness_level = brightness_command.level
                last_brightness_direction = brightness_command.direction or last_brightness_direction
                ok = controller.set_display_brightness(display_brightness_level)
                reply_seconds = time.perf_counter() - reply_started
                print(f"[Stacky] brightness={display_brightness_level} ok={ok}", flush=True)
                set_body_state("happy")
                speak_started = time.perf_counter()
                spoken_reply = brightness_command.spoken if ok else "Jeg kunne ikke ændre skærmens lysstyrke lige nu."
                record_local_turn(text, spoken_reply)
                await speak_reply(spoken_reply)
                speak_seconds = time.perf_counter() - speak_started
                print(
                    f"[timing] stt={stt_seconds:.2f}s command={reply_seconds:.2f}s "
                    f"tts_send={speak_seconds:.2f}s total={time.perf_counter() - pipeline_started:.2f}s",
                    flush=True,
                )
                _drain_queue(audio_queue)
                detector.reset()
                await asyncio.sleep(0.25)
                set_body_state("listening")
                accepting_audio = True
                continue
            motion_command = _parse_motion_command(text)
            if motion_command is not None:
                speak_started = time.perf_counter()
                controller.set_expression("happy")
                spoken_reply = motion_command.spoken
                record_local_turn(text, spoken_reply)
                await speak_reply(spoken_reply)
                speak_seconds = time.perf_counter() - speak_started
                reply_started = time.perf_counter()
                ok = _run_motion_gesture(body_director or controller, motion_command.gesture)
                reply_seconds = time.perf_counter() - reply_started
                print(f"[Stacky] motion={motion_command.gesture} ok={ok}", flush=True)
                print(
                    f"[timing] stt={stt_seconds:.2f}s command={reply_seconds:.2f}s "
                    f"tts_send={speak_seconds:.2f}s total={time.perf_counter() - pipeline_started:.2f}s",
                    flush=True,
                )
                _drain_queue(audio_queue)
                detector.reset()
                await asyncio.sleep(0.25)
                set_body_state("listening")
                accepting_audio = True
                continue
            calibration_command = _parse_calibration_command(text)
            if calibration_command is not None:
                reply_started = time.perf_counter()
                if calibration_command.save_current:
                    body_calibration = BodyCalibration(
                        center_yaw=int(body_status.get("yaw", body_calibration.center_yaw)),
                        center_pitch=int(body_status.get("pitch", body_calibration.center_pitch)),
                        yaw_range=body_calibration.yaw_range,
                        look_up_range=body_calibration.look_up_range,
                        look_down_range=body_calibration.look_down_range,
                    ).clamp()
                else:
                    body_calibration = body_calibration.nudge(
                        yaw_delta=calibration_command.yaw_delta,
                        pitch_delta=calibration_command.pitch_delta,
                    )
                save_body_calibration(config.data_dir, body_calibration)
                if body_director is not None:
                    ok = body_director.update_calibration(body_calibration)
                else:
                    ok = controller.configure_motion(
                        center_yaw=body_calibration.center_yaw,
                        center_pitch=body_calibration.center_pitch,
                        yaw_range=body_calibration.yaw_range,
                        look_up_range=body_calibration.look_up_range,
                        look_down_range=body_calibration.look_down_range,
                    )
                ok = _run_motion_gesture(body_director or controller, "center", speed=260) and ok
                reply_seconds = time.perf_counter() - reply_started
                print(
                    f"[Stacky] calibration center_yaw={body_calibration.center_yaw} "
                    f"center_pitch={body_calibration.center_pitch} ok={ok}",
                    flush=True,
                )
                set_body_state("happy")
                speak_started = time.perf_counter()
                spoken_reply = calibration_command.spoken if ok else "Jeg kunne ikke gemme hovedkalibreringen lige nu."
                record_local_turn(text, spoken_reply)
                await speak_reply(spoken_reply)
                speak_seconds = time.perf_counter() - speak_started
                print(
                    f"[timing] stt={stt_seconds:.2f}s command={reply_seconds:.2f}s "
                    f"tts_send={speak_seconds:.2f}s total={time.perf_counter() - pipeline_started:.2f}s",
                    flush=True,
                )
                _drain_queue(audio_queue)
                detector.reset()
                await asyncio.sleep(0.25)
                set_body_state("listening")
                accepting_audio = True
                continue
            local_reply = _parse_local_realtime_reply(text)
            if local_reply is not None:
                set_body_state("happy")
                speak_started = time.perf_counter()
                record_local_turn(text, local_reply)
                await speak_reply(local_reply)
                speak_seconds = time.perf_counter() - speak_started
                print(
                    f"[timing] stt={stt_seconds:.2f}s local=0.00s "
                    f"tts_send={speak_seconds:.2f}s total={time.perf_counter() - pipeline_started:.2f}s",
                    flush=True,
                )
                _drain_queue(audio_queue)
                detector.reset()
                await asyncio.sleep(0.25)
                set_body_state("listening")
                accepting_audio = True
                continue
            sandcode_action = await classify_sandcode_action(
                text,
                brain.lmstudio,
                recent_context=brain.recent_context_text(),
            )
            if sandcode_action is not None:
                reply_started = time.perf_counter()
                set_body_state("thinking")
                cwd = _resolve_sandcode_cwd(config, "")
                if sandcode_action.prompt == "__cancel__":
                    runtime_state.record_action(
                        kind="sandcode_agent",
                        status="failed",
                        summary="Sandcode-agent cancel er ikke understoettet fra voice endnu.",
                        error="voice cancel unsupported",
                        can_speak_about=("sandcode_agent", "runtime_action"),
                    )
                    spoken_reply = "Jeg kan ikke afbryde en kørende Sandcode-session fra voice endnu."
                    record_local_turn(text, spoken_reply)
                    await speak_reply(spoken_reply)
                    print(
                        f"[timing] stt={stt_seconds:.2f}s sandcode=0.00s "
                        f"tts_send={time.perf_counter() - reply_started:.2f}s "
                        f"total={time.perf_counter() - pipeline_started:.2f}s",
                        flush=True,
                    )
                    _drain_queue(audio_queue)
                    detector.reset()
                    await asyncio.sleep(0.25)
                    set_body_state("listening")
                    accepting_audio = True
                    continue
                runtime_state.mark_sandcode_starting(sandcode_action.prompt)
                lead_reply = _sandcode_lead_reply(
                    sandcode_action.prompt,
                    presence_mode=brain.presence_mode() if brain is not None else "stille_ven",
                )
                record_local_turn(text, f"{lead_reply} Opgave: {sandcode_action.prompt}")
                set_body_state("speaking")
                await speak_reply(lead_reply)
                set_body_state("thinking")
                await asyncio.sleep(0.08)

                spoken_updates = 0

                async def speak_sandcode_update(update: str) -> None:
                    nonlocal spoken_updates
                    print(f"[Sandcode] {update}", flush=True)
                    runtime_state.mark_sandcode_running(sandcode_action.prompt, note=update)
                    if not _should_speak_sandcode_update(update, spoken_updates=spoken_updates):
                        return
                    spoken_updates += 1
                    set_body_state("speaking")
                    await speak_reply(update)
                    set_body_state("thinking")

                try:
                    session = await _run_sandcode_with_updates(
                        sandcode_client,
                        cwd,
                        sandcode_action.prompt,
                        on_update=speak_sandcode_update,
                        chat_only=sandcode_action.chat_only,
                    )
                    runtime_state.mark_sandcode_done(sandcode_action.prompt, session_id=session.session_id)
                    brain.remember_memory_map(
                        f"Seneste Sandcode-agentkørsel: {sandcode_action.prompt}. Session: {session.session_id}.",
                        source="sandcode-agent",
                    )
                    if spoken_updates == 0:
                        set_body_state("speaking")
                        await speak_reply(f"Agenten er færdig. Sessionen hedder {session.session_id}.")
                except SandcodeError as exc:
                    runtime_state.mark_sandcode_failed(sandcode_action.prompt, str(exc))
                    error_reply = f"Agenten kunne ikke starte: {exc}"
                    print(f"[Sandcode] {error_reply}", flush=True)
                    brain.remember_memory_map(
                        f"Sandcode-agenten kunne ikke starte for opgaven: {sandcode_action.prompt}. Fejl: {exc}.",
                        source="sandcode-agent",
                    )
                    set_body_state("speaking")
                    await speak_reply(error_reply)
                set_body_state("happy")
                print(
                    f"[timing] stt={stt_seconds:.2f}s sandcode={time.perf_counter() - reply_started:.2f}s "
                    f"total={time.perf_counter() - pipeline_started:.2f}s",
                    flush=True,
                )
                _drain_queue(audio_queue)
                detector.reset()
                await asyncio.sleep(0.25)
                set_body_state("listening")
                accepting_audio = True
                continue
            early_web_context = ""
            if web_context_provider is not None:
                early_web_context = await web_context_provider(text)
                if early_web_context:
                    web_first_line = early_web_context.splitlines()[0]
                    web_failed = "fejlede" in web_first_line.lower()
                    runtime_state.record_action(
                        kind="web_search",
                        status="failed" if web_failed else "done",
                        summary=web_first_line[:240],
                        error=web_first_line[:240] if web_failed else "",
                        can_speak_about=("web_search", "runtime_action"),
                    )
            computer_action = None
            if not early_web_context:
                computer_action = (
                    parse_local_computer_action(text, root=config.computer.workspace_root)
                    if computer_actions is not None
                    else None
                )
            if computer_action is not None:
                reply_started = time.perf_counter()
                set_body_state("thinking")
                result = await asyncio.to_thread(computer_actions.run, computer_action)
                print(f"[computer] {computer_action.kind} ok={result.ok} detail={result.detail}", flush=True)
                runtime_state.record_action(
                    kind=f"computer:{computer_action.kind}",
                    status="done" if result.ok else "failed",
                    summary=result.spoken,
                    detail=result.detail,
                    error="" if result.ok else result.detail,
                    can_speak_about=("computer_action", "runtime_action"),
                )
                spoken_reply = result.spoken
                record_local_turn(text, spoken_reply)
                set_body_state("happy" if result.ok else "thinking")
                speak_started = time.perf_counter()
                await speak_reply(spoken_reply)
                speak_seconds = time.perf_counter() - speak_started
                print(
                    f"[timing] stt={stt_seconds:.2f}s computer={time.perf_counter() - reply_started:.2f}s "
                    f"tts_send={speak_seconds:.2f}s total={time.perf_counter() - pipeline_started:.2f}s",
                    flush=True,
                )
                _drain_queue(audio_queue)
                detector.reset()
                await asyncio.sleep(0.25)
                set_body_state("listening")
                accepting_audio = True
                continue
            brain_started = time.perf_counter()
            visual_context = ""
            prompt_image = None
            web_context = early_web_context
            computer_context = ""
            if computer_context_provider is not None and not web_context:
                computer_context = await computer_context_provider(text)
            use_visual_context = _wants_visual_context(text)
            if vision_state is not None and use_visual_context:
                await _capture_prompt_vision(
                    controller,
                    vision_snapshot_queue,
                    timeout_seconds=vision_prompt_timeout,
                )
                visual_context = vision_state.prompt_context(max_age_seconds=10.0)
                if vision_image:
                    image_base64 = vision_state.image_base64(max_age_seconds=10.0)
                    if image_base64 is not None:
                        prompt_image = ChatImageAttachment("image/jpeg", image_base64)
            monitor_context = format_monitor_context(
                recent_monitor_observations,
                max_items=config.monitor.max_context_observations,
            )
            reply = await brain.respond(
                text,
                max_spoken_chars=reply_chars,
                detail_spoken_chars=detail_reply_chars,
                persist_session=voice_policy.persist_session,
                allow_memory_writes=voice_policy.allow_memory_writes,
                remember_recent=voice_policy.remember_recent,
                session_source=voice_policy.session_source,
                visual_context=visual_context,
                vision_image=prompt_image,
                web_context=web_context,
                computer_context=computer_context,
                monitor_context=monitor_context,
                runtime_context=runtime_state.context_for_prompt(),
            )
            sync_body_personality()
            brain_seconds = time.perf_counter() - brain_started
            set_body_state("speaking")
            speak_started = time.perf_counter()
            reply_text = reply.spoken_text or reply.text
            if body_director is not None:
                await asyncio.to_thread(body_director.reply_started, reply_text)
            await speak_reply(reply_text)
            speak_seconds = time.perf_counter() - speak_started
            print(
                f"[timing] stt={stt_seconds:.2f}s brain={brain_seconds:.2f}s "
                f"tts_send={speak_seconds:.2f}s total={time.perf_counter() - pipeline_started:.2f}s",
                flush=True,
            )
            _drain_queue(audio_queue)
            detector.reset()
            await asyncio.sleep(0.25)
            set_body_state("listening")
            accepting_audio = True
    except (KeyboardInterrupt, asyncio.CancelledError):
        return 0
    finally:
        for task in (vision_capture_task, vision_processor_task, body_presence_task, monitor_task):
            if task is not None:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        if output is not None:
            await output.stop()
        set_body_state("neutral")
        controller.stop()


def _pop_monitor_observation(
    monitor_queue: asyncio.Queue[MonitorObservation] | None,
) -> MonitorObservation | None:
    if monitor_queue is None:
        return None
    try:
        return monitor_queue.get_nowait()
    except asyncio.QueueEmpty:
        return None


def _should_comment_on_monitor_observation(
    observation: MonitorObservation,
    config: MonitorConfig,
    *,
    presence_mode: str = "stille_ven",
    accepting_audio: bool,
    body_state_name: str,
    now: float,
    last_user_voice_at: float,
    last_stacky_speech_at: float,
    last_monitor_comment_at: float,
) -> bool:
    if not config.enabled:
        return False
    mode = presence_mode.strip().lower()
    if mode == "ikke_forstyr":
        return False
    if not accepting_audio or body_state_name != "listening":
        return False
    min_importance = config.min_importance_to_speak
    recent_speech_grace = config.recent_speech_grace_seconds
    if mode == "vaagen_makker":
        min_importance = max(60, min_importance - 10)
        recent_speech_grace = min(recent_speech_grace, 75)
    elif mode == "agent_vagt" and observation.kind == "stacky_health":
        min_importance = min(min_importance, 65)
        recent_speech_grace = min(recent_speech_grace, 45)
    if not observation.speakable or observation.importance < min_importance:
        return False
    last_conversation_at = max(last_user_voice_at, last_stacky_speech_at)
    if now - last_conversation_at < recent_speech_grace:
        return False
    if last_monitor_comment_at > 0 and now - last_monitor_comment_at < config.speak_cooldown_seconds:
        return False
    return True


def _sandcode_lead_reply(prompt: str, *, presence_mode: str = "stille_ven") -> str:
    del prompt
    mode = presence_mode.strip().lower()
    if mode == "agent_vagt":
        return "Jeg sender agenten bag forhænget og holder øje."
    if mode == "moerk_humor_lavt_blus":
        return "Jeg slipper agenten løs i maskinrummet. Pænt, men med hjelm på."
    return "Jeg sender agenten afsted. Jeg bliver her."


def _should_speak_sandcode_update(update: str, *, spoken_updates: int) -> bool:
    clean = update.strip()
    if not clean:
        return False
    if clean.startswith(("Agenten melder", "Agenten meldte fejl", "Agenten er færdig", "Jeg har afbrudt")):
        return True
    if clean.startswith("Agenten arbejder stadig"):
        return spoken_updates < 10
    return spoken_updates < 5


def _drain_queue(queue: asyncio.Queue[tuple[bytes, int, int]]) -> None:
    while True:
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            return


async def _delayed_reply_motion(director: BodyDirector, text: str, *, delay_seconds: float = 0.18) -> None:
    await asyncio.sleep(delay_seconds)
    try:
        await asyncio.to_thread(director.reply_started, text)
    except Exception as exc:  # pragma: no cover - body motion must never break speech.
        print(f"[Stacky] body motion skipped: {exc}", flush=True)


async def _speaking_body_loop(
    director: BodyDirector,
    text: str,
    *,
    on_motion: Callable[[], None] | None = None,
    interval_seconds: float = 0.18,
) -> None:
    interval_seconds = max(0.12, float(interval_seconds))
    while True:
        await asyncio.sleep(interval_seconds)
        try:
            before = director.last_motion_at
            await asyncio.to_thread(director.speaking_tick, text)
            if director.last_motion_at > before and on_motion is not None:
                on_motion()
        except Exception as exc:  # pragma: no cover - speech animation must not break audio.
            print(f"[Stacky] speaking animation skipped: {exc}", flush=True)
            return


def _should_track_face_runtime(
    *,
    face_tracking: bool,
    body_state_name: str,
    detector_ready: bool,
    detector_active: bool,
) -> bool:
    return bool(
        face_tracking
        and detector_ready
        and body_state_name == "listening"
        and not detector_active
    )


def _should_capture_vision_runtime(
    *,
    controller_connected: bool,
    accepting_audio: bool,
) -> bool:
    return bool(controller_connected and accepting_audio)


async def _body_presence_loop(
    director: BodyDirector,
    *,
    get_state: Callable[[], str],
    should_tick: Callable[[], bool],
    on_motion: Callable[[], None] | None = None,
    interval_seconds: float = 0.35,
) -> None:
    interval_seconds = max(0.20, float(interval_seconds))
    while True:
        await asyncio.sleep(interval_seconds)
        if not should_tick():
            continue
        state = get_state()
        try:
            before = director.last_motion_at
            await asyncio.to_thread(director.presence_tick, state)
            if director.last_motion_at > before and on_motion is not None:
                on_motion()
        except Exception as exc:  # pragma: no cover - body presence must not break audio.
            print(f"[Stacky] body presence skipped: {exc}", flush=True)


async def _body_event_reaction(
    director: BodyDirector,
    event_type: str,
    payload: dict[str, object],
    *,
    on_motion: Callable[[], None] | None = None,
) -> None:
    try:
        before = director.last_motion_at
        if event_type == "touch":
            await asyncio.to_thread(director.handle_touch, payload)
        elif event_type == "proximity":
            await asyncio.to_thread(director.handle_proximity, payload)
        if director.last_motion_at > before and on_motion is not None:
            on_motion()
    except Exception as exc:  # pragma: no cover - physical reactions must not break audio.
        print(f"[Stacky] body event reaction skipped: {exc}", flush=True)


async def _vision_processor_loop(
    vision_state: VisionState,
    payload_queue: asyncio.Queue[dict[str, object]],
    snapshot_queue: asyncio.Queue[VisionSnapshot],
    body_director: BodyDirector | None,
    *,
    should_track_face: Callable[[], bool],
    on_tracking_motion: Callable[[], None] | None = None,
    debug: bool = False,
) -> None:
    while True:
        payload = await payload_queue.get()
        snapshot = await asyncio.to_thread(vision_state.observe_payload, payload)
        if snapshot_queue.full():
            try:
                snapshot_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        snapshot_queue.put_nowait(snapshot)
        face = snapshot.primary_face
        if debug:
            if snapshot.error:
                print(f"[vision] error={snapshot.error}", flush=True)
            elif face is None:
                print(f"[vision] faces=0 detector={snapshot.detector}", flush=True)
            else:
                print(
                    f"[vision] face x={face.x:.2f} y={face.y:.2f} "
                    f"area={face.area:.2f} detector={snapshot.detector}",
                    flush=True,
                )
        if body_director is not None and face is not None and should_track_face():
            try:
                last_motion_at = body_director.last_motion_at
                body_director.track_face(face.x, face.y, confidence=face.confidence, area=face.area)
                if body_director.last_motion_at > last_motion_at and on_tracking_motion is not None:
                    on_tracking_motion()
            except Exception as exc:  # pragma: no cover - vision must not break audio.
                if debug:
                    print(f"[vision] face tracking skipped: {exc}", flush=True)


async def _vision_capture_loop(
    controller: StackChanBodyController,
    *,
    interval_seconds: float,
    should_capture: Callable[[], bool],
) -> None:
    interval_seconds = max(0.6, float(interval_seconds))
    await asyncio.sleep(0.8)
    while True:
        if should_capture():
            controller.capture_vision_frame(quality=50, discard_frames=4, settle_ms=30)
        await asyncio.sleep(interval_seconds)


async def _capture_prompt_vision(
    controller: StackChanBodyController,
    snapshot_queue: asyncio.Queue[VisionSnapshot],
    *,
    timeout_seconds: float,
) -> VisionSnapshot | None:
    while True:
        try:
            snapshot_queue.get_nowait()
        except asyncio.QueueEmpty:
            break
    if not controller.capture_vision_frame(quality=50, discard_frames=4, settle_ms=30):
        return None
    try:
        return await asyncio.wait_for(snapshot_queue.get(), timeout=max(0.05, float(timeout_seconds)))
    except asyncio.TimeoutError:
        return None


def _clean_transcript(text: str) -> str:
    clean = " ".join(text.split()).strip()
    if not clean:
        return ""
    sentence_parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+", clean) if part.strip()]
    if len(sentence_parts) >= 2:
        first_key = _transcript_key(sentence_parts[0])
        if first_key and all(_transcript_key(part) == first_key for part in sentence_parts):
            return _normalize_danish_transcript(sentence_parts[0])

    words = clean.split()
    if len(words) >= 2 and len(words) % 2 == 0:
        midpoint = len(words) // 2
        first = " ".join(words[:midpoint])
        second = " ".join(words[midpoint:])
        if _transcript_key(first) == _transcript_key(second):
            return _normalize_danish_transcript(first)
    return _normalize_danish_transcript(clean)


def _normalize_danish_transcript(text: str) -> str:
    key = _transcript_key(text)
    if key in {"haj", "hai"}:
        return "Hej!"
    if key in {"hajstacky", "haistacky"}:
        return "Hej Stacky"
    return text


def _transcript_key(text: str) -> str:
    return re.sub(r"[^0-9a-zæøå]+", "", text.lower())


def _is_likely_hallucination(text: str) -> bool:
    key = _transcript_key(text)
    if not key:
        return True
    known_bad = (
        "detersåjegharværetpåatdetderdetjegharværetpå",
        "deterdetjegharværetpå",
        "detersåjegharværetpå",
        "deterdetderdet",
    )
    return any(bad in key for bad in known_bad)


def _accept_stt_result(
    result: STTResult,
    text: str | None = None,
    *,
    signal_quality: TurnSignalQuality | None = None,
    trusted_transcript: bool = False,
) -> tuple[bool, str]:
    transcript = _clean_transcript(text if text is not None else result.text)
    key = _transcript_key(transcript)
    if len(transcript.strip()) < 2 or not key:
        return False, "tom tekst"
    if _is_likely_hallucination(transcript):
        return False, "kendt hallucination"
    if signal_quality is not None and not signal_quality.speech_like:
        return False, signal_quality.reason
    if signal_quality is not None and _is_clipped_sparse_noise_turn(result, transcript, signal_quality):
        return False, "clippet støj uden sammenhængende tale"
    if signal_quality is not None and signal_quality.zero_crossing_rate >= 0.45 and result.avg_logprob < -0.75:
        return False, "støjfyldt højfrekvent transcript"
    if signal_quality is not None and _is_short_high_frequency_stt_fragment(transcript, signal_quality):
        return False, "kort højfrekvent STT-fragment"
    if signal_quality is not None and _is_repetitive_filler_noise_turn(result, transcript, signal_quality):
        return False, "repetitivt filler-støjfragment"
    if trusted_transcript:
        return True, "trusted transcript correction"
    if key in {"ja", "nej", "ok", "okay"} and result.avg_logprob < -0.8:
        if signal_quality is None or signal_quality.crest_factor >= 24.0 or signal_quality.active_ratio < 0.18:
            return False, "kort uklart svar"
    if key in {"deer", "deeri", "deter", "deteri", "jageri", "jegeri"}:
        if signal_quality is None or signal_quality.active_ratio < 0.22 or signal_quality.max_active_run_ms < 120:
            return False, "typisk STT-støjfragment"
    if key in {"deter", "deteri"} and result.avg_logprob < -0.65:
        return False, "ufærdigt STT-fragment"
    if signal_quality is not None and signal_quality.max_active_run_ms < 80 and signal_quality.active_ratio < 0.20:
        return False, "for lidt sammenhængende tale"
    if key in {"hej", "hejsa", "hejstacky", "stacky"}:
        if signal_quality is not None and _is_noisy_short_greeting(result, signal_quality):
            return False, "kort hilsen fra støj"
        return True, "kort hilsen"
    if signal_quality is not None and _is_too_thin_single_word_fragment(transcript, signal_quality):
        return False, "kort tyndt STT-fragment"
    if _is_short_uncertain_stt_fragment(transcript, result):
        return False, "kort usikkert STT-fragment"
    if signal_quality is not None and _is_semantically_thin_reference_fragment(transcript, result, signal_quality):
        return False, "for tyndt referencefragment"

    audio = result.audio
    if audio.duration_seconds < 0.45:
        return False, "for kort lyd"
    if (
        result.avg_logprob >= -0.7
        and result.no_speech_prob <= 0.25
        and audio.peak >= 1200
        and len(key) >= 4
    ):
        return True, "klar STT-confidence"
    if audio.rms < 180 and audio.peak < 1000:
        return False, "for lavt mic-niveau"
    if audio.rms < 240 and audio.peak < 1600 and audio.duration_seconds < 1.2:
        return False, "kort og svag lyd"
    if result.avg_logprob < -0.9 and audio.rms < 700:
        return False, "lav STT confidence"
    if result.no_speech_prob >= 0.82 and audio.rms < 900:
        return False, "Whisper tror der er stilhed"
    if result.avg_logprob < -1.8 and audio.rms < 1000:
        return False, "lav STT confidence"
    if result.compression_ratio > 2.6:
        return False, "gentagelses-artefakt"
    return True, "ok"


def _is_short_uncertain_stt_fragment(transcript: str, result: STTResult) -> bool:
    words = [word for word in transcript.split() if word]
    if len(words) > 3:
        return False
    key = _transcript_key(transcript)
    if key in {"deer", "deeri", "deter", "deteri", "jageri", "jegeri"}:
        return False
    if key in {
        "vent",
        "ventlige",
        "stop",
        "stoplige",
        "pause",
        "skruop",
        "skruned",
        "kigop",
        "kigned",
        "center",
        "ligeud",
    }:
        return False
    if any(token in key for token in ("volumen", "volume", "hojre", "venstre", "batteri", "status", "strom")):
        return False
    return result.avg_logprob < -0.55


def _is_too_thin_single_word_fragment(transcript: str, signal_quality: TurnSignalQuality) -> bool:
    words = [word.strip(".,!?").lower() for word in transcript.split() if word.strip(".,!?")]
    if len(words) != 1:
        return False
    key = _transcript_key(words[0])
    protected_keys = {
        "ja",
        "nej",
        "ok",
        "okay",
        "hej",
        "hejsa",
        "tak",
        "vent",
        "stop",
        "pause",
        "igen",
        "center",
        "op",
        "ned",
        "lys",
        "lyd",
        "nik",
    }
    if key in protected_keys:
        return False
    if any(token in key for token in ("volumen", "volume", "hojre", "venstre", "batteri", "status", "strom")):
        return False
    return len(key) < 4 and (
        signal_quality.active_ratio < 0.30
        or signal_quality.active_ms <= 380
        or signal_quality.max_active_run_ms <= 360
    )


def _is_semantically_thin_reference_fragment(
    transcript: str,
    result: STTResult,
    signal_quality: TurnSignalQuality,
) -> bool:
    words = [word.strip(".,!?").lower() for word in transcript.split() if word.strip(".,!?")]
    if not words or len(words) > 2:
        return False
    key = _transcript_key(transcript)
    protected_keys = {
        "ja",
        "nej",
        "ok",
        "okay",
        "hej",
        "hejsa",
        "hejstacky",
        "stacky",
        "vent",
        "ventlige",
        "stop",
        "stoplige",
        "pause",
        "holdpause",
        "skruop",
        "skruned",
        "kigop",
        "kigned",
        "center",
        "ligeud",
    }
    if key in protected_keys:
        return False
    if any(token in key for token in ("volumen", "volume", "hojre", "venstre", "batteri", "status", "strom")):
        return False
    thin_words = {"den", "det", "der", "her", "du", "dig", "jeg", "kan", "er", "nu"}
    if any(word not in thin_words for word in words):
        return False
    if len(words) == 1:
        return True
    return (
        result.avg_logprob <= -0.35
        or signal_quality.crest_factor >= 10.0
        or signal_quality.duration_seconds >= 2.0
    )


def _is_clipped_sparse_noise_turn(result: STTResult, transcript: str, signal_quality: TurnSignalQuality) -> bool:
    words = [word.strip(".,!?").lower() for word in transcript.split() if word.strip(".,!?")]
    if not words:
        return False
    filler_words = {"den", "det", "her", "du", "jeg", "kan", "for", "til", "ned", "op"}
    filler_count = sum(1 for word in words if word in filler_words)
    repeated_count = max(words.count(word) for word in set(words))
    filler_ratio = filler_count / len(words)
    sparse_runs = signal_quality.max_speech_band_run_ms <= 320 or signal_quality.max_active_run_ms <= 360

    if len(words) >= 5 and sparse_runs and signal_quality.crest_factor >= 18.0:
        if filler_ratio >= 0.75 or repeated_count >= 4:
            return True

    if result.avg_logprob > -0.65:
        return False
    if signal_quality.crest_factor < 35.0:
        return False
    if not sparse_runs:
        return False

    if len(words) < 4:
        return signal_quality.max_speech_band_run_ms <= 260
    return filler_ratio >= 0.65 or repeated_count >= 3


def _is_noisy_short_greeting(result: STTResult, signal_quality: TurnSignalQuality) -> bool:
    if result.avg_logprob >= -0.75:
        return False
    if signal_quality.crest_factor >= 14.0 and signal_quality.max_speech_band_run_ms <= 260:
        return True
    if signal_quality.active_ratio < 0.24 and signal_quality.max_active_run_ms <= 260:
        return True
    return result.audio.peak >= 12000 and signal_quality.max_speech_band_run_ms <= 220


def _is_repetitive_filler_noise_turn(result: STTResult, transcript: str, signal_quality: TurnSignalQuality) -> bool:
    words = [word.strip(".,!?").lower() for word in transcript.split() if word.strip(".,!?")]
    if len(words) < 3:
        return False
    filler_words = {
        "den",
        "det",
        "der",
        "her",
        "du",
        "dig",
        "jeg",
        "kan",
        "er",
        "for",
        "til",
        "ned",
        "op",
        "nu",
        "i",
        "på",
        "paa",
    }
    weak_connectors = {"og", "så", "saa", "ja", "nej", "ok", "okay", "ej", "lige", "lidt"}
    content_words = [word for word in words if word not in filler_words and word not in weak_connectors]
    if content_words:
        return False
    key = _transcript_key(transcript)
    if key in {"jajegden", "jajegdet", "jajegkan", "jajegkanden", "jegden", "jegdet"}:
        return (
            signal_quality.peak >= 12000
            or signal_quality.crest_factor >= 18.0
            or signal_quality.max_speech_band_run_ms <= 420
        )
    noise_word_ratio = sum(1 for word in words if word in filler_words or word in weak_connectors) / len(words)
    repeated_count = max(words.count(word) for word in set(words))
    if (
        noise_word_ratio >= 0.95
        and len(words) >= 3
        and result.avg_logprob <= -0.55
        and (
            result.audio.rms < 900
            or signal_quality.active_ratio < 0.45
            or signal_quality.duration_seconds >= 2.5
        )
    ):
        return True
    sparse_or_spiky = (
        signal_quality.peak >= 7000
        or signal_quality.crest_factor >= 8.0
        or signal_quality.max_speech_band_run_ms <= 520
    )
    if noise_word_ratio < 0.80 or not sparse_or_spiky:
        return False
    if result.avg_logprob <= -0.55:
        return True
    return repeated_count >= 2 and signal_quality.duration_seconds >= 2.0


def _is_short_high_frequency_stt_fragment(transcript: str, signal_quality: TurnSignalQuality) -> bool:
    words = [word for word in transcript.split() if word]
    if len(words) > 2:
        return False
    key = _transcript_key(transcript)
    if key in {"deer", "deeri", "deter", "deteri", "jageri", "jegeri"}:
        return False
    if key in {
        "vent",
        "ventlige",
        "stop",
        "stoplige",
        "pause",
        "skruop",
        "skruned",
        "kigop",
        "kigned",
        "center",
        "ligeud",
    }:
        return False
    if any(token in key for token in ("volumen", "volume", "hojre", "venstre", "batteri", "status", "strom")):
        return False
    if key in {"jegkan", "jegkanher"} and (
        signal_quality.crest_factor >= 12.0
        or signal_quality.active_ratio < 0.35
        or signal_quality.max_active_run_ms < 320
    ):
        return True
    if (
        signal_quality.crest_factor >= 16.0
        and signal_quality.active_ratio < 0.32
        and signal_quality.max_active_run_ms <= 280
        and signal_quality.max_speech_band_run_ms <= 280
    ):
        return True
    if signal_quality.zero_crossing_rate < 0.38:
        return False
    return (
        signal_quality.peak >= 30000
        or signal_quality.active_ratio < 0.35
        or signal_quality.max_speech_band_run_ms < 260
    )


def _parse_local_realtime_reply(text: str) -> str | None:
    key = _transcript_key(text)
    if key in {"vent", "ventlige", "stop", "stoplige", "pause", "holdpause"}:
        return "Jeg venter."
    return None


def _wants_runtime_status_reply(text: str) -> bool:
    key = _motion_text_key(text)
    if not key:
        return False
    if any(token in key for token in ("batteri", "strom", "volumen", "volume", "gitstatus", "presencestatus")):
        return False
    if key in {"status", "hvadstatus"}:
        return False

    agent_hint = any(
        token in key
        for token in (
            "agent",
            "sandcode",
            "sandkode",
            "sancode",
            "sancodi",
            "runtime",
            "computerhandling",
        )
    )
    pronoun_status_hint = any(
        token in key
        for token in (
            "koererden",
            "korerden",
            "virkerden",
            "haengerden",
            "hangerden",
            "venterden",
            "hvadventerden",
            "hvadvardetdenventer",
            "hvaderdetdenventer",
            "erdengang",
            "erdetigang",
            "gikdenigang",
            "startededen",
            "fikdenstartet",
        )
    )
    status_hint = any(
        token in key
        for token in (
            "status",
            "koerer",
            "korer",
            "virker",
            "haenger",
            "hanger",
            "venter",
            "livstegn",
            "faerdig",
            "fejlede",
            "igang",
            "startet",
            "hvadlaver",
        )
    )
    return status_hint and (agent_hint or pronoun_status_hint)


def _parse_presence_mode_command(text: str) -> PresenceModeCommand | None:
    key = _motion_text_key(text)
    if not key:
        return None
    if any(token in key for token in ("ikkeforstyr", "forstyrikke", "donotdisturb", "dontdisturb")):
        return PresenceModeCommand("ikke_forstyr", "Okay. Jeg går i ikke-forstyr og holder mig næsten helt stille.")
    if "agentvagt" in key or ("hold" in key and "agent" in key and ("oeje" in key or "oje" in key or "vagt" in key)):
        return PresenceModeCommand("agent_vagt", "Okay. Agent-vagt er på; jeg holder øje uden at råbe brandalarm for en hoste.")
    if any(
        token in key
        for token in (
            "vagenmakker",
            "vaagenmakker",
            "merevagen",
            "merevaagen",
            "vaarvagenmakker",
            "vaervagenmakker",
            "vaervaagenmakker",
        )
    ):
        return PresenceModeCommand("vaagen_makker", "Okay. Vågen makker: mere opmærksom, stadig uden kontorstolenergi.")
    if (
        ("morkhumor" in key or "moerkhumor" in key or "darkhumor" in key or "galgenhumor" in key)
        and ("lavtblus" in key or "lavblus" in key or "paalavtblus" in key or "palavtblus" in key)
    ):
        return PresenceModeCommand("moerk_humor_lavt_blus", "Okay. Mørk humor på lavt blus, ikke kirkegård med konfetti.")
    if any(token in key for token in ("stilleven", "sparsomven", "merestille", "vaerstille", "varstille")):
        return PresenceModeCommand("stille_ven", "Okay. Stille ven igen; jeg er her, bare uden trommesolo.")
    return None


def _parse_memory_map_write_command(text: str) -> str | None:
    clean = re.sub(r"\s+", " ", text).strip(" .,:;-")
    if not clean:
        return None
    lowered = clean.lower()
    if lowered.startswith(("husk mig", "mind mig")):
        return None
    patterns = (
        r"(?i)^\s*husk\s+(?:at\s+)?(.+)$",
        r"(?i)^\s*gem\s+(?:i\s+(?:din\s+)?(?:memory[- ]map|hukommelse|røde tråd|rode traad)\s+)?(?:at\s+)?(.+)$",
        r"(?i)^\s*skriv\s+(?:i\s+(?:din\s+)?(?:memory[- ]map|hukommelse|røde tråd|rode traad)\s+)(?:at\s+)?(.+)$",
    )
    for pattern in patterns:
        match = re.match(pattern, clean)
        if not match:
            continue
        note = re.sub(r"\s+", " ", match.group(1)).strip(" .,:;-")
        if len(note) >= 6:
            return note
    return None


def _wants_capability_report(text: str) -> bool:
    key = _motion_text_key(text)
    if not key:
        return False
    return any(
        token in key
        for token in (
            "hvadkandulave",
            "hvadkandugore",
            "hvadkandugøre",
            "hvilkefunktionerhardu",
            "kan dubrugeagenten",
            "kandubrugeagenten",
            "harduagentfunktionen",
            "hardusandcode",
            "kandustartesandcode",
            "kandustarteagenten",
        )
    )


def _wants_memory_map_recall(text: str) -> bool:
    key = _motion_text_key(text)
    if not key:
        return False
    return any(
        token in key
        for token in (
            "hvadhuskerdu",
            "hvadkanduhuske",
            "hvaderdenrodetrad",
            "hvaderdenrodetråd",
            "hvaderdinrodetrad",
            "hvaderdinrodetråd",
            "vismemorymap",
            "visdinmemorymap",
            "memorymap",
            "hukommelsesindex",
            "hukommelseindex",
        )
    )


def _wants_sense_diary_recall(text: str) -> bool:
    key = _motion_text_key(text)
    if not key:
        return False
    return any(
        token in key
        for token in (
            "hvadlagdumaerketil",
            "hvadlagtdumaerketil",
            "hvadlagdumerketil",
            "hvadlagtdumerketil",
            "hvadhardulagtmaerketil",
            "hvadhardulagtmerketil",
            "hvadhardulagtmarketil",
            "hvadbardulagtmaerketil",
            "hvadbardulagtmerketil",
            "sensedagbog",
            "sansedagbog",
            "hvadbardubemaerket",
            "hvadbardubemerket",
            "hvadhardubemaerket",
            "hvadhardubemerket",
        )
    )


def _wants_stacky_state_report(text: str) -> bool:
    key = _motion_text_key(text)
    if not key:
        return False
    return any(
        token in key
        for token in (
            "hvordanfoelesdet",
            "hvordanfolesdet",
            "hvordanhardudet",
            "hvilkenmode",
            "hvilkentilstand",
            "hvilkenpresence",
            "presence status",
            "presencestatus",
            "hvadmodeerd",
            "hvadmodeerdu",
        )
    )


def _wants_visual_context(text: str) -> bool:
    lowered = text.lower()
    key = _transcript_key(text)
    if not key:
        return False
    visual_phrases = (
        "kan du se",
        "hvad kan du se",
        "hvad ser du",
        "hvad kigger du på",
        "kig på",
        "se på",
        "ser det ud",
        "hvordan ser",
        "tag et billede",
        "tag et snapshot",
        "brug kamera",
        "tjek kamera",
        "visuelt",
        "i billedet",
        "på billedet",
    )
    if any(phrase in lowered for phrase in visual_phrases):
        return True
    visual_tokens = (
        "kamera",
        "billede",
        "snapshot",
        "foto",
        "ansigt",
        "face",
        "genkend",
        "objekt",
        "farve",
        "lys",
        "mørk",
        "moerk",
        "rød",
        "roed",
    )
    if any(token in lowered for token in visual_tokens):
        return True
    compact_visual = (
        "hvadkanduse",
        "hvadserdu",
        "kandusemig",
        "kamera",
        "billede",
        "snapshot",
        "ansigt",
    )
    return any(token in key for token in compact_visual)


def _parse_battery_status_command(text: str) -> bool:
    key = _motion_text_key(text)
    if not key:
        return False
    has_battery_context = any(token in key for token in ("batteri", "battery", "strom", "stroem", "oplad"))
    if not has_battery_context:
        return False
    return any(token in key for token in ("status", "niveau", "procent", "hvormeget", "hvorer", "vis", "fortael", "tjek"))


def _format_battery_status_reply(status: dict[str, object]) -> str:
    level = _coerce_int_status(status.get("batteryLevel"))
    charging = _coerce_bool_status(status.get("batteryCharging"))
    if level is None:
        return "Jeg har ikke fået batteridata fra firmware endnu."

    state = ""
    if charging is True:
        state = " og jeg oplader"
    elif charging is False:
        state = " og jeg kører på batteri"
    if level <= 20:
        state += ", så det er lavt"
    return f"Mit batteri er på {level} procent{state}."


def _battery_status_debug(status: dict[str, object]) -> str:
    level = status.get("batteryLevel", "?")
    charging = status.get("batteryCharging", "?")
    return f"level={level} charging={charging}"


def _coerce_int_status(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return _clamp_percent(int(value))
    except (TypeError, ValueError):
        return None


def _coerce_bool_status(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "ja"}:
            return True
        if lowered in {"false", "0", "no", "nej"}:
            return False
    return None


@dataclass(frozen=True)
class MotionCommand:
    gesture: str
    spoken: str


@dataclass(frozen=True)
class CalibrationCommand:
    yaw_delta: int = 0
    pitch_delta: int = 0
    save_current: bool = False
    spoken: str = "Okay, jeg justerer mit center."


@dataclass(frozen=True)
class BrightnessCommand:
    level: int
    direction: int = 0
    spoken: str = "Okay, jeg justerer skærmens lysstyrke."


@dataclass(frozen=True)
class PresenceModeCommand:
    mode: str
    spoken: str


def _run_motion_gesture(actor: BodyDirector | StackChanBodyController | None, gesture_name: str, *, speed: int = 550) -> bool:
    if actor is None:
        return False
    profiles = {
        "center": (0.12, min(speed, 200)),
        "look_left": (0.16, min(speed, 220)),
        "look_right": (0.16, min(speed, 220)),
        "look_up": (0.12, min(speed, 200)),
        "look_down": (0.12, min(speed, 200)),
        "nod": (0.12, min(speed, 210)),
        "shake": (0.10, min(speed, 190)),
    }
    if gesture_name == "demo":
        sequence = ["center", "look_left", "center", "look_right", "center", "look_up", "look_down", "nod", "shake", "center"]
    elif gesture_name == "dance":
        sequence = ["look_left", "center", "look_right", "center", "nod", "center"]
    else:
        sequence = [gesture_name]
    ok = True
    for index, name in enumerate(sequence):
        intensity, step_speed = profiles.get(name, (0.12, min(speed, 220)))
        ok = actor.gesture(name, intensity=intensity, speed=step_speed) and ok
        if index + 1 < len(sequence):
            time.sleep(
                0.36
                if gesture_name == "dance"
                else 0.34 if name not in {"nod", "shake"} else 0.62
            )
    return ok


def _parse_calibration_command(text: str) -> CalibrationCommand | None:
    lowered = text.lower()
    key = _motion_text_key(text)
    if any(token in key for token in ("volumen", "volume", "skruop", "skruned", "hojerevolumen", "laverevolumen")):
        return None
    if _has_brightness_context(key):
        return None
    if any(token in key for token in ("kig", "kigger", "kik", "se", "drej", "ryst", "ryste", "rest", "nik", "dans")):
        return None
    if any(token in key for token in ("gemcenter", "gemligeud", "gemdenherposition", "gemnuposition", "gemmitcenter")):
        return CalibrationCommand(save_current=True, spoken="Okay, jeg gemmer den her position som mit center.")
    if "kalibr" not in key and "center" not in key and "midt" not in key and "ligeud" not in key and "mere" not in key and "lidt" not in key:
        return None

    small = 30 if "lidt" in key else 50
    if any(token in key for token in ("merehojre", "lidttilhojre", "modhojre", "hojre")):
        return CalibrationCommand(yaw_delta=small, spoken="Okay, jeg flytter mit center lidt mod højre.")
    if any(token in key for token in ("merevenstre", "lidttilvenstre", "modvenstre", "venstre")):
        return CalibrationCommand(yaw_delta=-small, spoken="Okay, jeg flytter mit center lidt mod venstre.")
    if any(token in key for token in ("mereop", "lidtop", "hovedetop")):
        return CalibrationCommand(pitch_delta=small, spoken="Okay, jeg flytter mit center lidt op.")
    if any(token in key for token in ("merened", "lidtned", "nedad", "hovedetned")):
        return CalibrationCommand(pitch_delta=-small, spoken="Okay, jeg flytter mit center lidt ned.")
    if any(token in lowered for token in ("gem center", "gem ligeud", "gem position")):
        return CalibrationCommand(save_current=True, spoken="Okay, jeg gemmer den her position som mit center.")
    return None


def _parse_motion_command(text: str) -> MotionCommand | None:
    lowered = text.lower()
    key = _motion_text_key(text)
    words = _motion_words(text)
    if "skru" in lowered or "volumen" in lowered:
        return None
    if _has_brightness_context(key):
        return None
    if any(word in {"dans", "danse", "danser"} for word in words):
        return MotionCommand("dance", "Okay.")
    if any(token in key for token in ("provnoget", "provenbevaegelse", "bevaegdig", "bevaegelsekommando", "bevaegelseskommando")):
        return MotionCommand("demo", "Okay, jeg prøver en bevægelse.")
    if "nik" in lowered or "nod" in key:
        return MotionCommand("nod", "Okay.")
    if any(token in key for token in ("ryst", "ryste", "rest")) and "hoved" in key:
        return MotionCommand("shake", "Okay.")
    if any(token in key for token in ("ligeud", "midten", "center", "centrer", "nulstilhoved")):
        return MotionCommand("center", "Jeg kigger ligeud.")
    if any(token in key for token in ("tilvenstre", "modvenstre", "venstre")) and any(
        token in key for token in ("kig", "kik", "gik", "se", "drej")
    ):
        return MotionCommand("look_left", "Jeg kigger til venstre.")
    if any(token in key for token in ("tilhojre", "modhojre", "hojre")) and any(
        token in key for token in ("kig", "kik", "gik", "se", "drej")
    ):
        return MotionCommand("look_right", "Jeg kigger til højre.")
    has_look_verb = any(token in key for token in ("kig", "kigger", "kik", "gik", "se", "drej"))
    if any(token in key for token in ("kigop", "kikop", "gikop", "seop", "hovedetop")) or (has_look_verb and "opad" in key):
        return MotionCommand("look_up", "Jeg kigger op.")
    if any(token in key for token in ("kigned", "kikned", "gikned", "sened", "hovedetned")) or (has_look_verb and "nedad" in key):
        return MotionCommand("look_down", "Jeg kigger ned.")
    return None


def _motion_text_key(text: str) -> str:
    lowered = text.lower()
    replacements = {
        "æ": "ae",
        "ø": "o",
        "å": "a",
        "Ã¦": "ae",
        "Ã¸": "o",
        "Ã¥": "a",
        "ö": "o",
        "ä": "ae",
        "ü": "u",
    }
    for source, target in replacements.items():
        lowered = lowered.replace(source, target)
    return re.sub(r"[^0-9a-z]+", "", lowered)


def _motion_words(text: str) -> list[str]:
    lowered = text.lower()
    replacements = {
        "Ã¦": "ae",
        "Ã¸": "o",
        "Ã¥": "a",
        "ÃƒÂ¦": "ae",
        "ÃƒÂ¸": "o",
        "ÃƒÂ¥": "a",
        "Ã¶": "o",
        "Ã¤": "ae",
        "Ã¼": "u",
    }
    for source, target in replacements.items():
        lowered = lowered.replace(source, target)
    return [word for word in re.split(r"[^0-9a-z]+", lowered) if word]


def _has_brightness_context(key: str) -> bool:
    return any(
        token in key
        for token in (
            "lysstyrk",
            "lysestyrk",
            "skaermlys",
            "skarmlys",
            "skaerm",
            "skarm",
            "display",
            "backlight",
        )
    )


def _parse_display_brightness_command(
    text: str,
    *,
    current_level: int,
    previous_direction: int = 0,
) -> BrightnessCommand | None:
    lowered = text.lower()
    key = _motion_text_key(text)
    current_level = _clamp_percent(current_level, minimum=1)
    has_context = _has_brightness_context(key)

    explicit_level = re.search(r"\b(?:til|på|pa)\s+(\d{1,3})\b", lowered)
    if has_context and explicit_level:
        level = _clamp_percent(int(explicit_level.group(1)), minimum=1)
        return BrightnessCommand(level, spoken=f"Okay, skærmen er nu {level} procent.")

    if has_context:
        match = re.search(r"\b(\d{1,3})\s*(?:procent|%)?", lowered)
        if match and any(word in lowered for word in ("procent", "%", "lysstyr", "skærm", "skaerm", "display")):
            level = _clamp_percent(int(match.group(1)), minimum=1)
            return BrightnessCommand(level, spoken=f"Okay, skærmen er nu {level} procent.")

    down = (
        "ned" in key
        or "daemp" in key
        or "damp" in key
        or "morkere" in key
        or "moerkere" in key
        or "lavere" in key
        or "svagere" in key
    )
    up = (
        "op" in key
        or "lysere" in key
        or "hojere" in key
        or "merebrightness" in key
    )

    followup_more = previous_direction != 0 and any(token in key for token in ("lidtmere", "mer", "endnumere", "laenger", "langer"))
    if not has_context and not followup_more:
        return None

    if "heltned" in key or "slukskaerm" in key or "slukdisplay" in key:
        return BrightnessCommand(10, direction=-1, spoken="Okay, skærmen er nu 10 procent.")
    if "heltop" in key or "fuldlysstyrke" in key or "maxlysstyrke" in key or "makslysstyrke" in key:
        return BrightnessCommand(100, direction=1, spoken="Okay, skærmen er nu 100 procent.")

    direction = -1 if down else 1 if up else previous_direction if followup_more else 0
    if direction == 0:
        return None

    step = 25 if any(token in key for token in ("meget", "laengere", "længere", "langere", "langer")) else 15
    level = _clamp_percent(current_level + direction * step, minimum=1)
    return BrightnessCommand(level, direction=direction, spoken=f"Okay, skærmen er nu {level} procent.")


_VOLUME_WORDS = {
    "nul": 0,
    "ti": 10,
    "tyve": 20,
    "tredive": 30,
    "fyrre": 40,
    "halvtreds": 50,
    "halvtredsindstyve": 50,
    "tres": 60,
    "tresindstyve": 60,
    "halvfjerds": 70,
    "halvfjerdsindstyve": 70,
    "firs": 80,
    "firsindstyve": 80,
    "halvfems": 90,
    "halvfemsindstyve": 90,
    "hundrede": 100,
    "max": 100,
    "maks": 100,
    "maksimum": 100,
}


def _parse_volume_command(text: str, *, current_level: int) -> tuple[int, str] | None:
    lowered = text.lower()
    key = _transcript_key(lowered)
    compact = _motion_text_key(lowered)
    if not key:
        return None
    current_level = _clamp_volume_level(current_level)

    directional_level = re.search(r"\b(?:ned|op)\s+til\s+(\d{1,3})\b", lowered)
    if directional_level:
        level = _clamp_volume_level(int(directional_level.group(1)))
        return level, f"Okay, min volumen er nu {level} procent."

    explicit_level = re.search(
        r"\b(?:sæt|saet|set|juster|justerer|justere|skru|skrue|kronet|krone|skole)\b.*\btil\s+(\d{1,3})\b",
        lowered,
    )
    if explicit_level:
        level = _clamp_volume_level(int(explicit_level.group(1)))
        return level, f"Okay, min volumen er nu {level} procent."

    volume_context = any(
        word in lowered
        for word in (
            "volumen",
            "volume",
            "lyd",
            "højere",
            "hojere",
            "lavere",
            "dæmp",
            "daemp",
            "skru",
            "juster",
            "justerer",
            "justere",
        )
    ) or any(token in compact for token in ("lyd", "lydstyrk", "volumen", "volume"))
    if not volume_context:
        return None

    if any(phrase in lowered for phrase in ("sluk lyden", "mute", "helt stille")):
        return 0, "Okay, jeg skruer helt ned."
    if any(phrase in lowered for phrase in ("fuld volumen", "max volumen", "maks volumen", "skru helt op")):
        return 100, "Okay, jeg skruer helt op."

    match = re.search(r"\b(\d{1,3})\s*(?:procent|%)?", lowered)
    if match and any(word in lowered for word in ("volumen", "volume", "lyd", "procent", "%", "juster", "justerer", "justere")):
        level = _clamp_volume_level(int(match.group(1)))
        return level, f"Okay, min volumen er nu {level} procent."

    for word, level in _VOLUME_WORDS.items():
        if re.search(rf"\b{re.escape(word)}\b", lowered) and any(token in lowered for token in ("volumen", "volume", "lyd")):
            level = _clamp_volume_level(level)
            return level, f"Okay, min volumen er nu {level} procent."

    if (
        any(phrase in lowered for phrase in ("skru op", "skru lidt op", "skru en smule op", "højere", "hojere", "mere lyd", "for lav"))
        or re.search(r"\bskru(?:e)?\b.*\bop\b", lowered)
    ):
        step = 35 if any(word in lowered for word in ("meget", "langt", "længere", "laengere", "mere")) else 15
        level = _clamp_volume_level(current_level + step)
        return level, f"Okay, jeg skruer op til {level} procent."
    if (
        any(phrase in lowered for phrase in ("skru ned", "skru lidt ned", "skru en smule ned", "lavere", "dæmp", "daemp", "mindre lyd", "for høj", "for hoj"))
        or re.search(r"\bskru(?:e)?\b.*\bned\b", lowered)
        or ("lydstyrk" in compact and "ned" in compact)
        or ("skole" in compact and "lyd" in compact and "ned" in compact)
    ):
        step = 35 if any(word in lowered for word in ("meget", "langt", "længere", "laengere", "mindre")) else 15
        level = _clamp_volume_level(current_level - step)
        return level, f"Okay, jeg skruer ned til {level} procent."
    return None


def _clamp_volume_level(level: int) -> int:
    return max(0, min(100, int(level)))


def _clamp_percent(level: int, *, minimum: int = 0) -> int:
    return max(minimum, min(100, int(level)))


def _format_stt_result(result: STTResult, text: str) -> str:
    audio = result.audio
    return (
        f"text={text!r} dur={audio.duration_seconds:.2f}s rms={audio.rms} peak={audio.peak} "
        f"logprob={result.avg_logprob:.2f} no_speech={result.no_speech_prob:.2f} "
        f"compression={result.compression_ratio:.2f}"
    )


def _format_signal_quality(quality: TurnSignalQuality) -> str:
    return (
        f"speech_like={quality.speech_like} reason={quality.reason!r} "
        f"dur={quality.duration_seconds:.2f}s med={quality.median_rms} "
        f"p95={quality.p95_rms} peak={quality.peak} "
        f"active={quality.active_ratio:.2f}/{quality.active_ms}ms "
        f"run={quality.max_active_run_ms}ms band={quality.speech_band_ms}/{quality.max_speech_band_run_ms}ms "
        f"crest={quality.crest_factor:.1f} zcr={quality.zero_crossing_rate:.2f} "
        f"thr={quality.active_threshold}"
    )


def _pcm16_peak(pcm: bytes) -> int:
    peak = 0
    for index in range(0, len(pcm) - 1, 2):
        sample = int.from_bytes(pcm[index : index + 2], "little", signed=True)
        peak = max(peak, abs(sample))
    return peak


async def _speech_output(enabled: bool):
    if not enabled:
        return None
    output = create_fast_piper_output()
    print("Loading local Danish Piper voice...", flush=True)
    started = time.perf_counter()
    await output.preload()
    print(f"Local Danish voice ready ({time.perf_counter() - started:.1f}s). Type 'afbryd' to stop speech.", flush=True)
    return output


async def _sandcode_health(config_path: str) -> int:
    config = load_config(config_path)
    client = SandcodeMobileHostClient(config.sandcode)
    try:
        await client.ensure_host()
    except SandcodeError as exc:
        print(f"Sandcode health failed: {exc}", flush=True)
        return 1
    print(f"Sandcode mobile host is healthy at {client.base_url}")
    return 0


async def _sandcode_run_once(config_path: str, prompt: str, *, cwd_arg: str = "", chat_only: bool = False) -> int:
    config = load_config(config_path)
    cwd = _resolve_sandcode_cwd(config, cwd_arg)
    client = SandcodeMobileHostClient(config.sandcode)
    print(f"Starting Sandcode in {cwd}", flush=True)

    async def print_update(text: str) -> None:
        print(text, flush=True)

    try:
        session = await _run_sandcode_with_updates(
            client,
            cwd,
            prompt,
            on_update=print_update,
            chat_only=chat_only,
        )
    except SandcodeError as exc:
        print(f"Sandcode failed: {exc}", flush=True)
        return 1
    print(f"Sandcode session complete: {session.session_id}", flush=True)
    return 0


def _resolve_sandcode_cwd(config, cwd_arg: str = "") -> Path:
    cwd = Path(cwd_arg.strip()) if cwd_arg.strip() else config.computer.workspace_root
    if not cwd.is_absolute():
        cwd = ROOT / cwd
    return cwd.resolve()


async def _run_sandcode_with_updates(
    client: SandcodeMobileHostClient,
    cwd: Path,
    prompt: str,
    *,
    on_update: Callable[[str], Awaitable[None]],
    chat_only: bool = False,
    heartbeat_seconds: float = 25.0,
) -> SandcodeSession:
    summarizer = SandcodeDanishSummarizer()
    queue: asyncio.Queue[str | None] = asyncio.Queue()
    loop = asyncio.get_running_loop()
    started_at = loop.time()
    last_update_at = started_at
    last_update = ""

    def enqueue_update(spoken: str) -> None:
        nonlocal last_update_at, last_update
        last_update_at = loop.time()
        last_update = spoken
        loop.call_soon_threadsafe(queue.put_nowait, spoken)

    def on_event(event: dict[str, object]) -> None:
        spoken = summarizer.summarize_event(event)
        if spoken:
            enqueue_update(spoken)

    async def run() -> SandcodeSession:
        try:
            return await client.run_session(cwd, prompt, on_event, chat_only=chat_only)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    async def heartbeat() -> None:
        interval = max(0.01, float(heartbeat_seconds))
        while True:
            await asyncio.sleep(interval)
            if task.done():
                return
            if loop.time() - last_update_at >= interval:
                enqueue_update(
                    summarizer.summarize_heartbeat(
                        elapsed_seconds=loop.time() - started_at,
                        last_update=last_update,
                    )
                )

    task = asyncio.create_task(run())
    heartbeat_task = asyncio.create_task(heartbeat())
    try:
        while True:
            update = await queue.get()
            if update is None:
                break
            await on_update(update)
        return await task
    finally:
        heartbeat_task.cancel()
        try:
            await heartbeat_task
        except asyncio.CancelledError:
            pass


def _body_server(config_path: str, *, duration: float = 0.0) -> int:
    config = load_config(config_path)
    deadline = time.monotonic() + duration if duration > 0 else None
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", config.stackchan.port))
    server.listen(1)
    server.settimeout(0.5)
    print(f"Stacky body server listening on 0.0.0.0:{config.stackchan.port}", flush=True)
    print("Waiting for StackChan status/touch events. Press Ctrl+C to stop.", flush=True)
    client: socket.socket | None = None
    buffer = b""
    pending_raw_bytes = 0
    audio_in_count = 0
    try:
        while True:
            if deadline is not None and time.monotonic() >= deadline:
                print("Body server duration elapsed.", flush=True)
                return 0
            if client is None:
                try:
                    client, address = server.accept()
                except socket.timeout:
                    continue
                client.settimeout(0.5)
                print(f"StackChan connected from {address[0]}:{address[1]}", flush=True)
                buffer = b""
                pending_raw_bytes = 0
                audio_in_count = 0
                continue
            try:
                raw = client.recv(4096)
            except socket.timeout:
                continue
            if not raw:
                print("StackChan disconnected.", flush=True)
                client.close()
                client = None
                buffer = b""
                pending_raw_bytes = 0
                audio_in_count = 0
                continue
            buffer += raw
            while buffer:
                if pending_raw_bytes > 0:
                    consumed = min(pending_raw_bytes, len(buffer))
                    buffer = buffer[consumed:]
                    pending_raw_bytes -= consumed
                    if pending_raw_bytes > 0:
                        break
                    continue
                if b"\n" not in buffer:
                    break
                line_bytes, buffer = buffer.split(b"\n", 1)
                if not line_bytes.strip():
                    continue
                line = line_bytes.decode("utf-8", errors="replace")
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    print(f"StackChan raw event: {line}", flush=True)
                    continue
                event_type = str(event.get("type", ""))
                payload = event.get("payload", {})
                if not isinstance(payload, dict):
                    payload = {}
                if event_type == "audio.in" and payload.get("transport") == "raw":
                    pending_raw_bytes = max(0, int(payload.get("bytes", 0) or 0))
                    audio_in_count += 1
                    if audio_in_count <= 3 or audio_in_count % 100 == 0:
                        print(
                            "StackChan raw event: "
                            f"audio.in #{audio_in_count} raw {payload.get('sampleRate', '?')} Hz "
                            f"{payload.get('channels', '?')} ch {pending_raw_bytes} bytes",
                            flush=True,
                        )
                    continue
                print(f"StackChan raw event: {line}", flush=True)
                if event_type == "touch":
                    try:
                        client.sendall((expression("happy").to_json() + "\n").encode("utf-8"))
                    except OSError:
                        client.close()
                        client = None
                        buffer = b""
                        pending_raw_bytes = 0
                        audio_in_count = 0
                        break
    except KeyboardInterrupt:
        return 0
    finally:
        if client is not None:
            client.close()
        server.close()


def _voice_lab(
    *,
    play: bool = False,
    engine: str = "piper",
    style: str = "neutral",
    speaker: str = "nic",
    limit: int = 0,
) -> int:
    phrases = [
        "Hej Nicolai, jeg er her.",
        "Mm, jeg tænker lige.",
        "Skal jeg tænde lyset i stuen?",
        "Sandcode er færdig og har ændret tre filer.",
        "Rødgrød med fløde.",
        "Jeg kan godt høre dig, men jeg skal lige bruge et øjeblik.",
    ]
    if engine == "roest":
        sample_count = limit if limit > 0 else 3
        return _voice_lab_roest(phrases[:sample_count], play=play, speaker=speaker)
    if engine == "supertonic":
        sample_count = limit if limit > 0 else 3
        return _voice_lab_supertonic(phrases[:sample_count], play=play)

    sample_count = limit if limit > 0 else len(phrases)
    phrases = phrases[:sample_count]
    voice = ensure_danish_piper_voice()
    tts = FastPiperTTS(voice)
    out_dir = ROOT / "artifacts" / ("voice_lab_female" if style == "female" else "voice_lab")
    print(f"Using local Danish Piper voice: {voice.model_path}", flush=True)
    if style == "female":
        print("Style: female audition, generated by local pitch-shift. No API used.", flush=True)
    for index, phrase in enumerate(phrases, start=1):
        adapted = adapt_for_danish_speech(phrase)
        filename = f"{index:02d}-{slugify(phrase, max_length=42)}.wav"
        output = out_dir / filename
        if style == "female":
            raw = out_dir / f".raw-{filename}"
            tts.synthesize_to_file(adapted, raw, length_scale=1.0, sentence_silence=0.14)
            output = pitch_shift_wav(raw, output, factor=1.13)
            raw.unlink(missing_ok=True)
        else:
            output = tts.synthesize_to_file(adapted, output)
        print(f"{output} :: {adapted}", flush=True)
        if play:
            subprocess.run(
                ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(output)],
                check=False,
            )
    return 0
