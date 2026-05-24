# Stacky

Stacky is a fresh local Danish AI friend for M5Stack StackChan. The PC runs the brain, memory, realtime voice orchestration, Home Assistant tools, and Sandcode control. StackChan is the body: microphone, speaker, face, touch, servos, LEDs, and later wheels.

Important boundary: Stacky does not import Moss identity, memories, sessions, or names. This repository starts from a clean soul file and a fresh local database.

## Quick Start

```powershell
cd C:\Users\nicol\stackchan
$env:PYTHONPATH = "src"
python -m stacky init
python -m unittest discover -s tests
python -m stacky chat
python -m stacky chat --speak
python -m stacky handsfree
```

Create `configs/stacky.toml` from `configs/stacky.example.toml` when you are ready to point Stacky at LM Studio, Home Assistant, StackChan, and Sandcode.

## Official StackChan Firmware Patch

The official M5Stack StackChan firmware lives in a git submodule at `vendor/m5stack-stackchan`. Stacky's firmware changes are stored in this repo as a reproducible patch:

```powershell
.\scripts\apply-official-firmware-patch.ps1
```

Use this after cloning on another PC. The script initializes the submodule and applies `patches/official-stackchan/0001-stacky-bridge.patch`, which adds the Stacky app, bridge, audio/body protocol, boot screen branding, and version string. If the submodule is already dirty and you want to recreate it from the official base:

```powershell
.\scripts\apply-official-firmware-patch.ps1 -ForceReset
```

## Voice Rule

Danish speech is a hard v1 requirement. Stacky may quote code, file names, API names, and Sandcode output in the original language, but spoken explanations and summaries stay Danish unless you explicitly ask otherwise.

## Major Pieces

- `src/stacky/brain.py`: Danish-first brain around LM Studio and local memory.
- `src/stacky/memory.py`: fresh SQLite memory store with a tiny local vector index.
- `src/stacky/sessions.py`: append-only infinite session thread plus stitcher for trusted context.
- `src/stacky/personality.py`: Stacky's fresh self-model: continuity, Nicolai-model, style notes, and convictions from trusted feedback.
- `src/stacky/sandcode.py`: client for `C:\Users\nicol\SANDCODE\ios\host\sandcode-mobile-host.mjs`.
- `src/stacky/body`: StackChan body/audio protocol and local hub.
- `src/stacky/voice`: Pipecat pipeline factory plus a text voice loop for local testing.
- `firmware/stacky_cores3`: custom CoreS3 firmware skeleton for StackChan body control.

## Local Danish Voice Lab

Stacky's live voice path is latency-first: local Piper TTS is the default realtime engine and runs in-process after the first model load. Røst/Chatterbox can be used in the lab to audition a more natural Danish voice, but it is not a live runtime engine until it can generate fast enough on the target hardware.

```powershell
.\.venv\Scripts\python.exe -m stacky voice-lab
.\.venv\Scripts\python.exe -m stacky voice-lab --play
.\.venv\Scripts\python.exe -m stacky voice-lab --engine supertonic --limit 1 --play
.\.venv\Scripts\python.exe -m stacky voice-lab --engine roest --speaker nic --limit 1 --play
```

Samples are written to `artifacts/voice_lab*`. The current realtime voice is `da_DK-talesyntese-medium`.

For the first live test with fast local speech:

```powershell
.\.venv\Scripts\python.exe -m stacky chat --speak
.\.venv\Scripts\python.exe -m stacky live-text --speak
.\.venv\Scripts\python.exe -m stacky handsfree
```

Type `afbryd` while Stacky is speaking to stop playback.

`handsfree` listens through StackChan's CoreS3 mic, runs local Danish STT on the PC, and speaks through StackChan by default after the chunked speaker firmware is flashed. Use `--speaker pc` as a fallback.

Whisper is no longer the default for hands-free mode because it can hallucinate short Danish turns. Stacky now defaults to a local Danish wav2vec2/CTC backend:

```powershell
.\.venv\Scripts\python.exe -m stacky handsfree --listen-only
.\.venv\Scripts\python.exe -m stacky handsfree --listen-only --debug-audio
.\.venv\Scripts\python.exe -m stacky handsfree
.\.venv\Scripts\python.exe -m stacky handsfree --speaker pc --tts-engine supertonic
.\.venv\Scripts\python.exe -m stacky handsfree --stt-engine whisper --stt-model small
```

The wav2vec2 default is `CoRal-project/roest-v3-wav2vec2-315m`. First startup is slow while the model and language model load; after that, short StackChan turns transcribe in a fraction of a second on the current PC. After the first download, Stacky loads Roest from the local Hugging Face cache first and only contacts Hugging Face again if the model is missing locally.

Accepted hands-free turns now default to trusted Stacky conversation: they persist to the infinite session, update recent context, and can write narrow safe memories/personality observations. Use `--voice-trust session-only` to keep session context without long-term writes, or `--voice-trust off` to return to the old untrusted mode while debugging STT.

Hands-free replies default to live speech that can still carry a thought: `--reply-chars 260` and `--detail-reply-chars 650`. Stacky should answer simple turns in 1-3 concise sentences, allow 2-5 sentences for complex discussion, avoid automatic follow-up questions, and avoid bringing up bedtime/time unless asked.

The Danish speech adapter also shapes TTS rhythm before synthesis: short markers such as "Okay" and "Fedt" get a sentence pause, and clauses before "men", "hvis", "når", and relevant "så" get clearer pauses. Hands-free mode now defaults to Supertonic's `alive` voice for a less monotone Stacky. Piper remains available as the fast fallback with `--tts-engine piper`. The live Supertonic `alive` profile is tuned for less rushed Danish speech: speed `1.08`, chunk length `140`, silence `0.07`.

The hands-free VAD is tuned for the official Stacky bridge: default `--vad-threshold 280`, `--start-speech-ms 120`, and `--min-speech-ms 220`. It rejects sparse clicks and high-frequency noise before STT; use `--debug-audio` to see `[audio] ... reason='højfrekvent støj'` / `klik/percussiv støj` lines.
The start detector also ignores high-frequency mic noise as a voice candidate, and post-STT gating rejects clipped sparse turns that hallucinate repeated filler words such as `den her den her`.

Firmware `official-0.1.11` streams the CoreS3 mic plus a reference/noise channel, accepts PC-controlled mic gain, reports `displayBrightness`, accepts `display.brightness`, and has a `vision.capture` placeholder that returns `camera_capture_not_implemented` until the real camera bridge is added. `handsfree` and `stt-capture` default to `--mic-channel 0`, which is the real mic path used by the official mic test. Use `--mic-channel 1`, `--mic-channel mix`, or `--mic-channel auto` only for diagnostics, and `--mic-channel all` to keep multichannel diagnostic WAVs. Default `--stackchan-mic-gain` is `85`, and default `--mic-preamp` is `2.0` with clipping protection.

Trusted hands-free and text/chat turns persist to `data/stacky/sessions/stacky-infinite-thread.jsonl`; rolled blocks become `stacky-infinite-thread.001.jsonl`, etc. Local commands such as volume, calibration, motion, and pause are logged without an extra LLM call so the session remains continuous.

Stacky's personality/self-development layer stores persistent style notes, convictions, interaction density, and Nicolai-context in `data/stacky/personality/`. Inspect it with:

```powershell
.\.venv\Scripts\python.exe -m stacky self-status
```

Web search is planned as an early Stacky feature, but it is not active in runtime yet. Until that provider/router exists, Stacky should not claim that it has searched the web.

## Danish STT Dataset Loop

Use StackChan itself to capture labelled Danish clips before changing STT models or filters:

```powershell
.\.venv\Scripts\python.exe -m stacky stt-capture --limit 12 --debug-audio
.\.venv\Scripts\python.exe -m stacky stt-capture --phrases-file .\artifacts\stt_phrases.txt --noise-count 3 --debug-audio
.\.venv\Scripts\python.exe -m stacky stt-capture --limit 16 --speech-style normal --speech-style fast --speech-style mumble --noise-count 5 --debug-audio
.\.venv\Scripts\python.exe -m stacky stt-capture --limit 10 --speech-style normal --noise-count 3 --mic-channel 1 --output-dir .\artifacts\stt_dataset\stackchan-ch1 --debug-audio
```

This writes WAV clips plus `artifacts/stt_dataset/stackchan/manifest.jsonl`. Speech clips have the expected Danish sentence; noise clips have an empty expected text so false positives score as errors.

Run local candidates against the captured dataset:

```powershell
.\.venv\Scripts\python.exe -m stacky stt-bench --dataset .\artifacts\stt_dataset\stackchan\manifest.jsonl --report .\artifacts\stt_dataset\stt-report.jsonl
.\.venv\Scripts\python.exe -m stacky stt-bench --dataset .\artifacts\stt_dataset\stackchan\manifest.jsonl --engine roest-v3 --engine roest-v2
.\.venv\Scripts\python.exe -m stacky stt-bench --dataset .\artifacts\stt_dataset\stackchan\manifest.jsonl --engine roest-v3 --engine roest-v2 --include-heavy
```

Benchmark output includes load time, inference time, realtime factor, WER, CER, per-style summaries, and a JSONL report when `--report` is set. `stt-bench` tests all dataset clips by default; pass `--limit N` only for quick smoke runs.

Stacky's STT benchmark defaults are Danish-specific: Røst v3 315M and Røst v2 315M. Heavy mode adds Røst v2 1B/2B before the non-Røst experimental candidates. Use the `fast`, `mumble`, and `quiet` capture styles to test whether a model handles everyday unclear Danish rather than only clean read-aloud clips.

To run Stacky's brain through Gemini instead of the local OpenAI-compatible endpoint for latency testing:

```powershell
$env:STACKY_BRAIN_PROVIDER = "gemini"
$env:GEMINI_MODEL = "gemini-3.1-flash-lite-preview"
$env:GEMINI_API_KEY = "<key>"
.\.venv\Scripts\python.exe -m stacky handsfree --speaker stackchan
.\.venv\Scripts\python.exe -m stacky handsfree --tts-engine piper --speaker stackchan
```

Head motion is available after flashing the official Stacky bridge firmware with motion support:

```powershell
.\.venv\Scripts\python.exe -m stacky motion-test --gesture demo
.\.venv\Scripts\python.exe -m stacky motion-test --gesture nod
```

Stacky can move locally from spoken commands such as `kig til højre`, `kig op`, `nik med hovedet`, `ryst på hovedet`, and `kan du danse`. It can also adjust its head center at runtime and persists it in `data/stacky/body_calibration.json`:

- `lidt mere til højre`
- `lidt mere til venstre`
- `lidt op`
- `lidt ned`
- `gem den her position som center`

Display brightness is a local body command after flashing `official-0.1.11`:

- `skru skærmens lysstyrke ned`
- `sæt skærmen til 35`
- `lidt mere`

Camera snapshot capture is available after flashing `official-0.1.13`:

```powershell
.\.venv\Scripts\python.exe -m stacky camera-test --body-timeout 45
```

The captured JPEG is written to `artifacts\vision\stackchan-latest.jpg` by default.

## Safety Defaults

- Home Assistant actions are suggest-first unless explicitly allowed by the caller.
- Sandcode coding sessions default to `permissionMode: "autonomousAgent"` because that is the chosen mode for this project.
- Wheel commands are present in the protocol but disabled by default until the physical wheel build is calibrated.
