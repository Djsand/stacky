from __future__ import annotations

import argparse
import asyncio
import re
import shutil
import socket
import subprocess
import time
from pathlib import Path
from slugify import slugify

from .brain import StackyBrain
from .body.controller import BodyPresence, StackChanBodyController
from .body.protocol import decode_pcm_payload, expression
from .config import DEFAULT_CONFIG_PATH, ROOT, load_config
from .llm import LMStudioClient
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
from .voice.stt import STTResult, create_danish_stt, write_pcm_wav
from .voice.turn_detection import EnergyTurnDetector, pcm16_rms
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
        choices=("wav2vec2", "whisper"),
        default="wav2vec2",
        help="Local Danish STT backend. wav2vec2 is default to avoid Whisper hallucinations.",
    )
    handsfree.add_argument(
        "--stt-model",
        default="",
        help="Model name. wav2vec2 default is CoRal-project/roest-v3-wav2vec2-315m; whisper default is small.",
    )
    handsfree.add_argument("--vad-threshold", type=int, default=120, help="PCM RMS threshold for speech start.")
    handsfree.add_argument("--end-silence-ms", type=int, default=1000, help="Silence duration that ends a voice turn.")
    handsfree.add_argument("--min-speech-ms", type=int, default=250, help="Minimum voiced audio before accepting a turn.")
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
        default="stacky",
        help="Supertonic tuning profile for Stacky's Danish voice.",
    )
    handsfree.add_argument("--supertonic-voice", default="", help="Override Supertonic voice style: F1-F5 or M1-M5.")
    handsfree.add_argument("--supertonic-speed", type=float, default=None, help="Override Supertonic speed multiplier.")
    handsfree.add_argument("--supertonic-steps", type=int, default=None, help="Override Supertonic quality steps; higher is clearer but slower.")
    handsfree.add_argument("--supertonic-silence", type=float, default=None, help="Override silence between Supertonic internal chunks.")
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
    speaker_tone = sub.add_parser("speaker-tone", help="Play a tiny firmware tone on StackChan without TTS streaming.")
    speaker_tone.add_argument("--body-timeout", type=float, default=12.0, help="Seconds to wait for StackChan to connect.")
    speaker_tone.add_argument("--frequency", type=int, default=880, help="Tone frequency in Hz.")
    speaker_tone.add_argument("--duration-ms", type=int, default=180, help="Tone duration in milliseconds.")
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
                listen_only=args.listen_only,
                debug_audio=args.debug_audio,
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
    brain = StackyBrain(soul, memory, LMStudioClient(config.lmstudio))
    output = await _speech_output(speak)
    await LocalTextVoiceRuntime(brain, output=output).interactive()
    return 0


async def _live_text(config_path: str, *, body_timeout: float, speak: bool = False) -> int:
    config = load_config(config_path)
    soul = load_soul(config.soul_path)
    memory = MemoryStore(config.memory_path)
    brain = StackyBrain(soul, memory, LMStudioClient(config.lmstudio))

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
) -> int:
    config = load_config(config_path)
    controller = StackChanBodyController(port=config.stackchan.port)
    controller.start()
    print(f"Stacky speaker-test server listening on 0.0.0.0:{config.stackchan.port}", flush=True)
    if not controller.wait_connected(body_timeout):
        print("StackChan did not connect yet.", flush=True)
        controller.stop()
        return 1
    output = create_stackchan_piper_output(controller)
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
    listen_only: bool,
    debug_audio: bool,
) -> int:
    config = load_config(config_path)
    brain = None
    if not listen_only:
        soul = load_soul(config.soul_path)
        memory = MemoryStore(config.memory_path)
        brain = StackyBrain(soul, memory, LMStudioClient(config.lmstudio))

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
            return
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
            output = create_stackchan_supertonic_output(controller, voice) if speaker == "stackchan" else create_supertonic_output(voice)
            print("Loading local Danish Supertonic voice...", flush=True)
        else:
            output = create_stackchan_piper_output(controller) if speaker == "stackchan" else create_fast_piper_output()
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
            controller.set_expression("thinking")
            wav_path = write_pcm_wav(
                ROOT / "artifacts" / "handsfree_turns" / f"turn-{turn_index:04d}.wav",
                turn.pcm,
                sample_rate=turn.sample_rate,
                channels=turn.channels,
            )
            stt_result = await stt.transcribe_wav_result(wav_path)
            text = _clean_transcript(stt_result.text)
            print(f"[STT] {_format_stt_result(stt_result, text)}", flush=True)
            accepted, reason = _accept_stt_result(stt_result, text)
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
            reply = await brain.respond(text)
            controller.set_expression("happy")
            await output.speak(reply.spoken_text or reply.text)
            await output.wait()
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


def _accept_stt_result(result: STTResult, text: str | None = None) -> tuple[bool, str]:
    transcript = _clean_transcript(text if text is not None else result.text)
    key = _transcript_key(transcript)
    if len(transcript.strip()) < 2 or not key:
        return False, "tom tekst"
    if _is_likely_hallucination(transcript):
        return False, "kendt hallucination"
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


def _format_stt_result(result: STTResult, text: str) -> str:
    audio = result.audio
    return (
        f"text={text!r} dur={audio.duration_seconds:.2f}s rms={audio.rms} peak={audio.peak} "
        f"logprob={result.avg_logprob:.2f} no_speech={result.no_speech_prob:.2f} "
        f"compression={result.compression_ratio:.2f}"
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
