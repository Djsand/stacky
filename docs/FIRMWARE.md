# StackChan Custom Firmware

The first firmware target is `firmware/stacky_cores3`.

## Build

Install PlatformIO, then copy the Wi-Fi example:

```powershell
Copy-Item firmware\stacky_cores3\include\wifi_secrets.example.h firmware\stacky_cores3\include\wifi_secrets.h
pio run -d firmware\stacky_cores3
```

## Current Scope

- Connect to Wi-Fi.
- Connect to the PC body hub over TCP.
- Draw basic Stacky expressions.
- Send touch and status events.
- Stream CoreS3 mic chunks to the PC as `audio.in` PCM16/base64 events.
- Play PC-generated Stacky speech on the CoreS3 speaker via chunked `audio.start` / `audio.chunk` / `audio.end`.
- Reserve `mobility.intent` protocol handling.

CoreS3 cannot use the internal mic and speaker at the same time, so firmware pauses mic streaming while Stacky speaks and re-enables it afterwards. Firmware `0.2.0` uses small base64 PCM chunks and rotating speaker buffers instead of one large `audio.out` payload. Wheel commands stay disabled until the physical wheel build is calibrated.
