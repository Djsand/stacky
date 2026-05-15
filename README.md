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

## Voice Rule

Danish speech is a hard v1 requirement. Stacky may quote code, file names, API names, and Sandcode output in the original language, but spoken explanations and summaries stay Danish unless you explicitly ask otherwise.

## Major Pieces

- `src/stacky/brain.py`: Danish-first brain around LM Studio and local memory.
- `src/stacky/memory.py`: fresh SQLite memory store with a tiny local vector index.
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
.\.venv\Scripts\python.exe -m stacky handsfree --tts-engine supertonic
.\.venv\Scripts\python.exe -m stacky handsfree --speaker pc --tts-engine supertonic
.\.venv\Scripts\python.exe -m stacky handsfree --stt-engine whisper --stt-model small
```

The wav2vec2 default is `CoRal-project/roest-v3-wav2vec2-315m`. First startup is slow while the model and language model load; after that, short StackChan turns transcribe in a fraction of a second on the current PC.

## Safety Defaults

- Home Assistant actions are suggest-first unless explicitly allowed by the caller.
- Sandcode coding sessions default to `permissionMode: "autonomousAgent"` because that is the chosen mode for this project.
- Wheel commands are present in the protocol but disabled by default until the physical wheel build is calibrated.
