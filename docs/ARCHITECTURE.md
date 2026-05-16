# Stacky Architecture

Stacky is split into PC brain and StackChan body.

## PC Brain

- `StackyBrain` builds a Danish system prompt from `data/stacky/soul.yaml`.
- `MemoryStore` owns a fresh SQLite DB at `data/stacky/memory.sqlite`.
- `StackySelfModel` owns fresh personality runtime state at `data/stacky/personality/`.
- The self-model tracks continuity, Nicolai's current interaction pattern, trusted style feedback, and Stacky convictions.
- Trusted text/chat turns can evolve the self-model. Untrusted StackChan voice turns only update lightweight counters until STT is reliable enough.
- `LMStudioClient` calls LM Studio through its OpenAI-compatible `/v1/chat/completions` endpoint.
- `LocalTextVoiceRuntime` is the current test harness.
- `build_pipecat_pipeline` builds the realtime voice pipeline once STT/TTS services are selected and installed.
- Web search is planned as an early PC-brain tool, but it is not active yet. The intended shape is a provider/router that only searches for explicit or current-knowledge questions and records citations in the session.

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
