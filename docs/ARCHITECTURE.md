# Stacky Architecture

Stacky is split into PC brain and StackChan body.

## PC Brain

- `StackyBrain` builds a Danish system prompt from `data/stacky/soul.yaml`.
- `MemoryStore` owns a fresh SQLite DB at `data/stacky/memory.sqlite`.
- `LMStudioClient` calls LM Studio through its OpenAI-compatible `/v1/chat/completions` endpoint.
- `LocalTextVoiceRuntime` is the current test harness.
- `build_pipecat_pipeline` builds the realtime voice pipeline once STT/TTS services are selected and installed.

## StackChan Body

The body protocol is newline-delimited JSON over TCP in v1:

- PC -> StackChan: `audio.out`, `body.set_expression`, `body.look_at`, `body.gesture`, `body.leds`, `body.status`, `mobility.intent`.
- StackChan -> PC: `audio.in`, `touch`, `battery`, `imu`, `proximity`, `status`.

Audio is represented as base64 PCM frames first. That is not the final fastest transport, but it is easy to inspect while the custom CoreS3 firmware comes online.

## Sandcode

Stacky uses Sandcode's mobile host, not scraped terminal output.

- Start host: `C:\Users\nicol\SANDCODE\ios\host\sandcode-mobile-host.mjs`.
- Create session: `POST /api/sessions`.
- Send prompt/cancel/listen: WebSocket with token.
- Required mode: `permissionMode: "autonomousAgent"`.

Stacky only speaks short Danish updates from Sandcode events.
