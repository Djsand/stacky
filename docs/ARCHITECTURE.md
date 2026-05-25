# Stacky Architecture

Stacky is split into PC brain and StackChan body.

## PC Brain

- `StackyBrain` builds a Danish system prompt from `data/stacky/soul.yaml`.
- `MemoryStore` owns a fresh SQLite DB at `data/stacky/memory.sqlite`.
- `MemoryMapStore` owns `data/stacky/memory_map.json`, a small writeable red-thread index for capabilities, decisions, and high-signal preferences. It is not raw transcript storage.
- `StackySelfModel` owns fresh personality runtime state at `data/stacky/personality/`.
- The self-model tracks continuity, Nicolai's current interaction pattern, trusted style feedback, Stacky convictions, presence mode, Stacky's own lightweight mood, and a sparse read-only sense diary.
- Trusted text/chat turns can evolve the self-model. Untrusted StackChan voice turns only update lightweight counters until STT is reliable enough.
- Global monitor observations enter the self-model as `sanseinput`, not commands. Only important events such as long focus, long silence, or agent-health trouble become diary items.
- `LMStudioClient` calls LM Studio through its OpenAI-compatible `/v1/chat/completions` endpoint.
- `LocalTextVoiceRuntime` is the current test harness.
- `build_pipecat_pipeline` builds the realtime voice pipeline once STT/TTS services are selected and installed.
- Web search is planned as an early PC-brain tool, but it is not active yet. The intended shape is a provider/router that only searches for explicit or current-knowledge questions and records citations in the session.

## StackChan Body

The body protocol is newline-delimited JSON over TCP in v1:

- PC -> StackChan: `audio.out`, `body.set_expression`, `body.look_at`, `body.gesture`, `body.leds`, `body.status`, `mobility.intent`.
- StackChan -> PC: `audio.in`, `touch`, `battery`, `imu`, `proximity`, `status`.

Audio is represented as base64 PCM frames first. That is not the final fastest transport, but it is easy to inspect while the custom CoreS3 firmware comes online.

`BodyDirector` adds restrained personality: small semantic reply motions, sparse autonomous presence ticks, and LEDs shaped by presence mode and Stacky's current mood. The StackChan bridge keeps playback asynchronous so these body signals can run while audio is playing. These signals never imply permission to act.

## Sandcode

Stacky uses Sandcode's mobile host, not scraped terminal output.

- Start host: `C:\Users\nicol\SANDCODE\ios\host\sandcode-mobile-host.mjs`.
- Create session: `POST /api/sessions`.
- Send prompt/cancel/listen: WebSocket with token.
- Required mode: `permissionMode: "autonomousAgent"`.

Stacky only speaks short Danish updates from Sandcode events, framed as the agent working behind the scenes so Stacky does not turn into a code-assistant persona.

Long Sandcode runs also emit sparse heartbeat updates if no tool or assistant event has arrived for a while. This keeps Nicolai informed without turning Stacky into a log reader.
