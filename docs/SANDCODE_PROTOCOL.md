# Sandcode Mobile Host Contract

Stacky starts or connects to the configured Sandcode mobile host, defaulting to
`127.0.0.1:7390` with token `sandcode-local`.

Local Sandcode HTTP and websocket calls bypass proxy environment variables. Health
checks use a short timeout so voice mode fails quickly instead of blocking a live
conversation on a dead or wrong host.

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
- Speak final `assistant_message` as a short Danish "Agenten melder..." summary.
- Speak `tool_call` and completed `tool_update` as short "agenten arbejder..." status.
- If the session is silent for a while, speak a sparse "Agenten arbejder stadig..." heartbeat with elapsed time and the latest short status.
- Speak `error` and `turn_cancelled`.
- Never read full logs aloud.
- Do not make Stacky sound like the coding assistant. Stacky sends the agent behind the curtain and stays present as Stacky.
