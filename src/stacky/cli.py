from __future__ import annotations

import argparse
import asyncio
import glob
import re
import shutil
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from slugify import slugify

from .brain import StackyBrain
from .body.controller import BodyPresence, StackChanBodyController
from .body.protocol import decode_pcm_payload, expression
from .config import DEFAULT_CONFIG_PATH, ROOT, load_config
from .llm import create_chat_client
from .memory import MemoryStore
from .sandcode import SandcodeMobileHostClient
from .soul import load_soul, write_default_soul
from .voice.output import (
    create_fast_piper_output,
    create_stackchan_piper_output,
    create_stackchan_supertonic_output,
    create_supertonic_output,
)
from .voice.supertonic_tts import SupertonicVoice, supertonic_voice_preset
from .voice.runtime import LocalTextVoiceRuntime
from .voice.stt import STTResult, create_danish_stt, resolve_stt_model_name, write_pcm_wav
from .voice.turn_detection import EnergyTurnDetector, TurnSignalQuality, analyze_turn_signal, pcm16_rms
from .voice.piper_tts import FastPiperTTS, ensure_danish_piper_voice, pitch_shift_wav
from .voice.roest_tts import RoestTTS, roest_voice
from .voice.speech_adapter import adapt_for_danish_speech


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="stacky")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    sub = parser.add_subparsers(dest="command", required=False)
    sub.add_parser("init", help="Create fresh Stacky local state.")
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
    handsfree.add_argument("--vad-threshold", type=int, default=700, help="PCM RMS threshold for speech start.")
    handsfree.add_argument("--end-silence-ms", type=int, default=850, help="Silence duration that ends a voice turn.")
    handsfree.add_argument("--min-speech-ms", type=int, default=150, help="Minimum voiced audio before accepting a turn.")
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
        default="piper",
        help="Local TTS engine. Piper is realtime-stable; Supertonic is more natural and local.",
    )
    handsfree.add_argument(
        "--supertonic-profile",
        choices=("stacky", "calm", "clear", "quick"),
        default="quick",
        help="Supertonic tuning profile for Stacky's Danish voice.",
    )
    handsfree.add_argument("--supertonic-voice", default="", help="Override Supertonic voice style: F1-F5 or M1-M5.")
    handsfree.add_argument("--supertonic-speed", type=float, default=None, help="Override Supertonic speed multiplier.")
    handsfree.add_argument("--supertonic-steps", type=int, default=None, help="Override Supertonic quality steps; higher is clearer but slower.")
    handsfree.add_argument("--supertonic-silence", type=float, default=None, help="Override silence between Supertonic internal chunks.")
    handsfree.add_argument("--stackchan-target-rms", type=int, default=9000, help="Target active PCM RMS for StackChan speaker loudness.")
    handsfree.add_argument("--stackchan-max-gain", type=float, default=4.0, help="Maximum StackChan speaker PCM gain before clipping.")
    handsfree.add_argument("--stackchan-volume", type=int, default=80, help="Initial StackChan codec volume, 0-100.")
    handsfree.add_argument("--reply-chars", type=int, default=150, help="Default spoken reply character budget for low-latency live chat.")
    handsfree.add_argument("--detail-reply-chars", type=int, default=260, help="Spoken reply character budget when the user asks for details.")
    stt_bench = sub.add_parser("stt-bench", help="Benchmark local Danish STT models on saved StackChan WAV turns.")
    stt_bench.add_argument("--audio", action="append", default=[], help="WAV file, directory, or glob. Defaults to artifacts/handsfree_turns/*.wav.")
    stt_bench.add_argument("--engine", action="append", default=[], help="Model spec: roest, ftspeech, qwen3, saga, milo, or engine:model.")
    stt_bench.add_argument("--limit", type=int, default=8, help="Maximum number of WAV files to test.")
    stt_bench.add_argument("--include-heavy", action="store_true", help="Also test heavier Qwen3-ASR candidates.")
    stt_bench.add_argument("--refs", default="", help="Optional tab-separated references file: wav_filename<TAB>expected text.")
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
        choices=("stacky", "calm", "clear", "quick"),
        default="stacky",
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
    body = sub.add_parser("body-server", help="Run the StackChan body server.")
    body.add_argument("--duration", type=float, default=0.0, help="Stop after N seconds; 0 means run forever.")
    args = parser.parse_args(argv)

    if args.command == "init":
        return _init(args.config)
    if args.command == "sandcode-health":
        return _run_async(_sandcode_health(args.config))
    if args.command == "body-server":
        return _body_server(args.config, duration=args.duration)
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
                end_silence_ms=args.end_silence_ms,
                min_speech_ms=args.min_speech_ms,
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
                reply_chars=args.reply_chars,
                detail_reply_chars=args.detail_reply_chars,
                listen_only=args.listen_only,
                debug_audio=args.debug_audio,
            )
        )
    if args.command == "stt-bench":
        return _run_async(
            _stt_bench(
                audio_patterns=args.audio,
                engine_specs=args.engine,
                limit=args.limit,
                include_heavy=args.include_heavy,
                refs_path=args.refs,
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


_LOW_LATENCY_STT_SPECS = (("wav2vec2", "roest"), ("wav2vec2", "ftspeech"))
_HEAVY_STT_SPECS = (("qwen3", "qwen3-0.6b"), ("qwen3", "saga"), ("qwen3", "milo"))
_STT_SPEC_ALIASES = {
    "roest": ("wav2vec2", "roest"),
    "coral": ("wav2vec2", "roest"),
    "coral-v3": ("wav2vec2", "roest"),
    "ftspeech": ("wav2vec2", "ftspeech"),
    "qwen3": ("qwen3", "qwen3-0.6b"),
    "qwen3-0.6b": ("qwen3", "qwen3-0.6b"),
    "saga": ("qwen3", "saga"),
    "milo": ("qwen3", "milo"),
}


async def _stt_bench(
    *,
    audio_patterns: list[str],
    engine_specs: list[str],
    limit: int,
    include_heavy: bool,
    refs_path: str,
) -> int:
    paths = _resolve_stt_bench_audio(audio_patterns, limit=limit)
    if not paths:
        print("No WAV files found. Run handsfree once or pass --audio path\\to\\turn.wav.", flush=True)
        return 1

    specs = _resolve_stt_bench_specs(engine_specs, include_heavy=include_heavy)
    refs = _load_stt_refs(Path(refs_path)) if refs_path else {}
    print(f"Benchmarking {len(paths)} StackChan WAV file(s).", flush=True)
    for path in paths:
        print(f"- {path}", flush=True)

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

        total_audio = 0.0
        total_infer = 0.0
        wer_values: list[float] = []
        for path in paths:
            started = time.perf_counter()
            try:
                result = await stt.transcribe_wav_result(path)
            except Exception as exc:
                print(f"  {path.name}: ERROR {type(exc).__name__}: {exc}", flush=True)
                continue
            infer_seconds = time.perf_counter() - started
            duration = max(result.audio.duration_seconds, 0.001)
            total_audio += duration
            total_infer += infer_seconds
            rtf = infer_seconds / duration
            expected = refs.get(path.name.lower())
            score = ""
            if expected:
                wer = _word_error_rate(expected, result.text)
                wer_values.append(wer)
                score = f" wer={wer:.1%}"
            print(
                f"  {path.name}: dur={duration:.2f}s infer={infer_seconds:.2f}s "
                f"rtf={rtf:.2f} logprob={result.avg_logprob:.2f}{score} :: {result.text}",
                flush=True,
            )

        if total_audio > 0 and total_infer > 0:
            summary = f"  total_audio={total_audio:.2f}s total_infer={total_infer:.2f}s rtf={total_infer / total_audio:.2f}"
            if wer_values:
                summary += f" mean_wer={sum(wer_values) / len(wer_values):.1%}"
            print(summary, flush=True)
    return 0


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


def _resolve_stt_bench_audio(patterns: list[str], *, limit: int) -> list[Path]:
    requested = patterns or [str(ROOT / "artifacts" / "handsfree_turns" / "*.wav")]
    paths: list[Path] = []
    for item in requested:
        path = Path(item)
        if path.is_dir():
            paths.extend(sorted(path.glob("*.wav")))
            continue
        matches = [Path(match) for match in glob.glob(item)]
        if matches:
            paths.extend(matches)
            continue
        if path.exists():
            paths.append(path)

    unique: dict[str, Path] = {}
    for path in paths:
        if path.suffix.lower() == ".wav":
            unique[str(path.resolve()).lower()] = path.resolve()
    ordered = sorted(unique.values(), key=lambda item: item.stat().st_mtime, reverse=True)
    if limit > 0:
        ordered = ordered[:limit]
    return ordered


def _load_stt_refs(path: Path) -> dict[str, str]:
    refs: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if "\t" in line:
            name, text = line.split("\t", 1)
        elif "|" in line:
            name, text = line.split("|", 1)
        else:
            continue
        refs[Path(name.strip()).name.lower()] = text.strip()
    return refs


def _word_error_rate(reference: str, hypothesis: str) -> float:
    ref_words = _score_words(reference)
    hyp_words = _score_words(hypothesis)
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return _edit_distance(ref_words, hyp_words) / len(ref_words)


def _score_words(text: str) -> list[str]:
    lowered = text.lower()
    normalized = re.sub(r"[^0-9a-zæøå]+", " ", lowered)
    return [word for word in normalized.split() if word]


def _edit_distance(left: list[str], right: list[str]) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_item in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_item in enumerate(right, start=1):
            cost = 0 if left_item == right_item else 1
            current.append(
                min(
                    previous[right_index] + 1,
                    current[right_index - 1] + 1,
                    previous[right_index - 1] + cost,
                )
            )
        previous = current
    return previous[-1]


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
    soul = load_soul(config.soul_path)
    memory = MemoryStore(config.memory_path)
    brain = StackyBrain(soul, memory, create_chat_client(config.lmstudio))
    output = await _speech_output(speak)
    await LocalTextVoiceRuntime(brain, output=output).interactive()
    return 0


async def _live_text(config_path: str, *, body_timeout: float, speak: bool = False) -> int:
    config = load_config(config_path)
    soul = load_soul(config.soul_path)
    memory = MemoryStore(config.memory_path)
    brain = StackyBrain(soul, memory, create_chat_client(config.lmstudio))

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
        await LocalTextVoiceRuntime(brain, output=output, presence=BodyPresence(controller)).interactive()
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
        controller.set_expression("happy")
        if gesture_name == "demo":
            sequence = ["center", "look_left", "look_right", "look_up", "look_down", "nod", "shake", "center"]
        else:
            sequence = [gesture_name]
        for name in sequence:
            print(f"Motion: {name}", flush=True)
            controller.gesture(name, speed=speed)
            await asyncio.sleep(0.75 if name in {"nod", "shake"} else 0.45)
        controller.set_expression("listening")
        return 0
    finally:
        controller.stop()


async def _handsfree(
    config_path: str,
    *,
    body_timeout: float,
    stt_engine: str,
    stt_model: str,
    vad_threshold: int,
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
    reply_chars: int,
    detail_reply_chars: int,
    listen_only: bool,
    debug_audio: bool,
) -> int:
    config = load_config(config_path)
    brain = None
    if not listen_only:
        soul = load_soul(config.soul_path)
        memory = MemoryStore(config.memory_path)
        brain = StackyBrain(soul, memory, create_chat_client(config.lmstudio))

    loop = asyncio.get_running_loop()
    audio_queue: asyncio.Queue[tuple[bytes, int, int]] = asyncio.Queue(maxsize=80)
    accepting_audio = False
    audio_meter = {"last_at": 0.0, "max_rms": 0, "max_peak": 0, "chunks": 0}

    def on_event(event) -> None:
        if event.type == "status":
            print(f"[StackChan] status: {event.payload}", flush=True)
            return
        if event.type == "touch":
            print(f"[StackChan] touch: {event.payload}", flush=True)
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
    controller.set_expression("thinking")

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
        await output.preload()
        print(f"Voice ready ({time.perf_counter() - started:.1f}s).", flush=True)

    stt = create_danish_stt(stt_engine, stt_model or None)
    stt_name = getattr(stt, "model_id", getattr(stt, "model_size", stt_model or "default"))
    print(f"Loading local Danish STT model ({stt_engine}: {stt_name})...", flush=True)
    started = time.perf_counter()
    await stt.preload()
    print(f"STT ready ({time.perf_counter() - started:.1f}s). Speak to StackChan now.", flush=True)

    detector = EnergyTurnDetector(
        threshold=vad_threshold,
        min_speech_ms=min_speech_ms,
        end_silence_ms=end_silence_ms,
    )
    turn_index = 0
    last_transcript = ""
    last_transcript_at = 0.0
    accepting_audio = True
    controller.set_expression("listening")
    try:
        while True:
            pcm, sample_rate, channels = await audio_queue.get()
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
                controller.set_expression("listening")
                accepting_audio = True
                continue
            controller.set_expression("thinking")
            stt_started = time.perf_counter()
            stt_result = await stt.transcribe_wav_result(wav_path)
            stt_seconds = time.perf_counter() - stt_started
            text = _clean_transcript(stt_result.text)
            print(f"[STT] {_format_stt_result(stt_result, text)}", flush=True)
            accepted, reason = _accept_stt_result(stt_result, text, signal_quality=signal_quality)
            if not accepted:
                print(f"[Stacky] ignorerer STT ({reason}): {text}", flush=True)
                controller.set_expression("listening")
                accepting_audio = True
                continue
            transcript_key = _transcript_key(text)
            now = time.monotonic()
            if transcript_key and transcript_key == last_transcript and now - last_transcript_at < 6.0:
                print(f"[Stacky] ignorerer gentaget STT: {text}", flush=True)
                controller.set_expression("listening")
                accepting_audio = True
                continue
            last_transcript = transcript_key
            last_transcript_at = now
            print(f"Nicolai: {text}", flush=True)
            if listen_only:
                controller.set_expression("listening")
                accepting_audio = True
                continue
            if brain is None or output is None:
                controller.set_expression("listening")
                accepting_audio = True
                continue
            volume_command = _parse_volume_command(text, current_level=getattr(output, "volume_level", stackchan_volume))
            if volume_command is not None and hasattr(output, "set_volume"):
                volume_level, spoken = volume_command
                reply_started = time.perf_counter()
                ok = output.set_volume(volume_level)
                reply_seconds = time.perf_counter() - reply_started
                print(f"[Stacky] volumen={volume_level} ok={ok}", flush=True)
                controller.set_expression("happy")
                speak_started = time.perf_counter()
                await output.speak(spoken if ok else "Jeg kunne ikke ændre min volumen lige nu.")
                await output.wait()
                speak_seconds = time.perf_counter() - speak_started
                print(
                    f"[timing] stt={stt_seconds:.2f}s command={reply_seconds:.2f}s "
                    f"tts_send={speak_seconds:.2f}s total={time.perf_counter() - pipeline_started:.2f}s",
                    flush=True,
                )
                _drain_queue(audio_queue)
                detector.reset()
                await asyncio.sleep(0.25)
                controller.set_expression("listening")
                accepting_audio = True
                continue
            motion_command = _parse_motion_command(text)
            if motion_command is not None:
                reply_started = time.perf_counter()
                ok = _run_motion_gesture(controller, motion_command.gesture)
                reply_seconds = time.perf_counter() - reply_started
                print(f"[Stacky] motion={motion_command.gesture} ok={ok}", flush=True)
                controller.set_expression("happy")
                speak_started = time.perf_counter()
                await output.speak(motion_command.spoken if ok else "Jeg kunne ikke bevæge hovedet lige nu.")
                await output.wait()
                speak_seconds = time.perf_counter() - speak_started
                print(
                    f"[timing] stt={stt_seconds:.2f}s command={reply_seconds:.2f}s "
                    f"tts_send={speak_seconds:.2f}s total={time.perf_counter() - pipeline_started:.2f}s",
                    flush=True,
                )
                _drain_queue(audio_queue)
                detector.reset()
                await asyncio.sleep(0.25)
                controller.set_expression("listening")
                accepting_audio = True
                continue
            local_reply = _parse_local_realtime_reply(text)
            if local_reply is not None:
                controller.set_expression("happy")
                speak_started = time.perf_counter()
                await output.speak(local_reply)
                await output.wait()
                speak_seconds = time.perf_counter() - speak_started
                print(
                    f"[timing] stt={stt_seconds:.2f}s local=0.00s "
                    f"tts_send={speak_seconds:.2f}s total={time.perf_counter() - pipeline_started:.2f}s",
                    flush=True,
                )
                _drain_queue(audio_queue)
                detector.reset()
                await asyncio.sleep(0.25)
                controller.set_expression("listening")
                accepting_audio = True
                continue
            brain_started = time.perf_counter()
            reply = await brain.respond(
                text,
                max_spoken_chars=reply_chars,
                detail_spoken_chars=detail_reply_chars,
            )
            brain_seconds = time.perf_counter() - brain_started
            controller.set_expression("happy")
            speak_started = time.perf_counter()
            await output.speak(reply.spoken_text or reply.text)
            await output.wait()
            speak_seconds = time.perf_counter() - speak_started
            print(
                f"[timing] stt={stt_seconds:.2f}s brain={brain_seconds:.2f}s "
                f"tts_send={speak_seconds:.2f}s total={time.perf_counter() - pipeline_started:.2f}s",
                flush=True,
            )
            _drain_queue(audio_queue)
            detector.reset()
            await asyncio.sleep(0.25)
            controller.set_expression("listening")
            accepting_audio = True
    except (KeyboardInterrupt, asyncio.CancelledError):
        return 0
    finally:
        if output is not None:
            await output.stop()
        controller.set_expression("neutral")
        controller.stop()


def _drain_queue(queue: asyncio.Queue[tuple[bytes, int, int]]) -> None:
    while True:
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            return


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
) -> tuple[bool, str]:
    transcript = _clean_transcript(text if text is not None else result.text)
    key = _transcript_key(transcript)
    if len(transcript.strip()) < 2 or not key:
        return False, "tom tekst"
    if _is_likely_hallucination(transcript):
        return False, "kendt hallucination"
    if signal_quality is not None and not signal_quality.speech_like:
        return False, signal_quality.reason
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
        return True, "kort hilsen"

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


def _parse_local_realtime_reply(text: str) -> str | None:
    key = _transcript_key(text)
    if key in {"vent", "ventlige", "stop", "stoplige", "pause", "holdpause"}:
        return "Jeg venter."
    return None


@dataclass(frozen=True)
class MotionCommand:
    gesture: str
    spoken: str


def _run_motion_gesture(controller: StackChanBodyController, gesture_name: str, *, speed: int = 550) -> bool:
    sequence = (
        ["center", "look_left", "look_right", "look_up", "look_down", "nod", "shake", "center"]
        if gesture_name == "demo"
        else [gesture_name]
    )
    ok = True
    for index, name in enumerate(sequence):
        ok = controller.gesture(name, speed=speed) and ok
        if index + 1 < len(sequence):
            time.sleep(0.28 if name not in {"nod", "shake"} else 0.55)
    return ok


def _parse_motion_command(text: str) -> MotionCommand | None:
    lowered = text.lower()
    key = _motion_text_key(text)
    if "skru" in lowered or "volumen" in lowered:
        return None
    if any(token in key for token in ("provnoget", "provenbevaegelse", "bevaegdig", "bevaegelsekommando", "bevaegelseskommando")):
        return MotionCommand("demo", "Okay, jeg prøver en bevægelse.")
    if "nik" in lowered or "nod" in key:
        return MotionCommand("nod", "Okay.")
    if "ryst" in lowered and ("hoved" in lowered or "hovedet" in lowered):
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
    if any(token in key for token in ("kigop", "kikop", "gikop", "seop", "hovedetop")):
        return MotionCommand("look_up", "Jeg kigger op.")
    if any(token in key for token in ("kigned", "kikned", "gikned", "sened", "hovedetned")):
        return MotionCommand("look_down", "Jeg kigger ned.")
    return None


def _motion_text_key(text: str) -> str:
    lowered = text.lower()
    replacements = {
        "æ": "ae",
        "ø": "o",
        "å": "a",
        "ö": "o",
        "ä": "ae",
        "ü": "u",
    }
    for source, target in replacements.items():
        lowered = lowered.replace(source, target)
    return re.sub(r"[^0-9a-z]+", "", lowered)


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
    if not key:
        return None
    current_level = _clamp_volume_level(current_level)

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
        )
    )
    if not volume_context:
        return None

    if any(phrase in lowered for phrase in ("sluk lyden", "mute", "helt stille")):
        return 0, "Okay, jeg skruer helt ned."
    if any(phrase in lowered for phrase in ("fuld volumen", "max volumen", "maks volumen", "skru helt op")):
        return 100, "Okay, jeg skruer helt op."

    match = re.search(r"\b(\d{1,3})\s*(?:procent|%)?", lowered)
    if match and any(word in lowered for word in ("volumen", "volume", "lyd", "procent", "%")):
        level = _clamp_volume_level(int(match.group(1)))
        return level, f"Okay, min volumen er nu {level} procent."

    for word, level in _VOLUME_WORDS.items():
        if re.search(rf"\b{re.escape(word)}\b", lowered) and any(token in lowered for token in ("volumen", "volume", "lyd")):
            level = _clamp_volume_level(level)
            return level, f"Okay, min volumen er nu {level} procent."

    if any(phrase in lowered for phrase in ("skru op", "højere", "hojere", "mere lyd", "for lav")):
        level = _clamp_volume_level(current_level + 15)
        return level, f"Okay, jeg skruer op til {level} procent."
    if any(phrase in lowered for phrase in ("skru ned", "lavere", "dæmp", "daemp", "mindre lyd", "for høj", "for hoj")):
        level = _clamp_volume_level(current_level - 15)
        return level, f"Okay, jeg skruer ned til {level} procent."
    return None


def _clamp_volume_level(level: int) -> int:
    return max(0, min(100, int(level)))


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
        f"run={quality.max_active_run_ms}ms crest={quality.crest_factor:.1f} "
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
    await client.ensure_host()
    print(f"Sandcode mobile host is healthy at {client.base_url}")
    return 0


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
                continue
            try:
                raw = client.recv(4096)
            except socket.timeout:
                continue
            if not raw:
                print("StackChan disconnected.", flush=True)
                client.close()
                client = None
                continue
            for line in raw.decode("utf-8", errors="replace").splitlines():
                if not line.strip():
                    continue
                print(f"StackChan raw event: {line}", flush=True)
                try:
                    client.sendall((expression("happy").to_json() + "\n").encode("utf-8"))
                except OSError:
                    client.close()
                    client = None
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
