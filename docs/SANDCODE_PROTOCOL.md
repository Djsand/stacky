# Sandcode Mobile Host Contract

Stacky starts or connects to Sandcode mobile host on `127.0.0.1:7390` with a private token.

## Session Creation

Stacky sends:

```json
{
  "cwd": "C:/path/to/project",
  "provider": "ChatGPT Codex",
  "model": "gpt-5.5",
  "permissionMode": "autonomousAgent",
  "effort": "max",
  "chatOnly": false
}
```

The initial id may be `mobile-*`. Stacky must replace it when the host emits `session_rekey`.

## Spoken Policy

- Ignore streaming deltas for speech.
- Speak final `assistant_message` as a short Danish summary.
- Speak `tool_call` and completed `tool_update` as short status.
- Speak `error` and `turn_cancelled`.
- Never read full logs aloud.
