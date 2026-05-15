# Stacky Handoff

## Stop Point

Stop debugging here. The last active work was on Stacky audio and mic reliability for the M5Stack StackChan/CoreS3 body. Old handsfree processes were stopped before this handoff was written.

## Project

Stacky is a fresh local Danish AI friend for Nicolai. It runs mainly on the Windows PC in `C:\Users\nicol\stackchan`, with StackChan/CoreS3 as the body: mic, speaker, face, servos, touch, LEDs, and later wheels.

Do not import or reuse Moss identity, Moss memories, Moss sessions, or the Moss name. Stacky has its own state in `data/stacky/soul.yaml`; runtime memory DB files are local and ignored by Git.

## Current Status

- Python package: `src/stacky`
- Firmware: `firmware/stacky_cores3`
- Tests: `tests`
- Body server port: `8765`
- StackChan IP seen during tests: `192.168.50.2`
- PC IP used during tests: `192.168.50.208`
- USB serial seen during tests: `COM3`
- Latest flashed firmware version during this session: `0.3.15`

The StackChan speaker path can play Stacky speech, but playback quality has been fragile during streaming. The best current direction is the raw/combined StackChan audio path rather than PC speaker fallback.

The Danish STT path is still the biggest unresolved problem. Stacky is currently using the local wav2vec2 Danish STT engine, not Whisper, by default:

```text
CoRal-project/roest-v3-wav2vec2-315m
```

Recent logs showed bad transcripts even when Nicolai spoke clearly. The likely root cause is the StackChan/CoreS3 mic pipeline or framing/levels, not simply the STT model.

## Latest Changes

- `src/stacky/body/controller.py`
  - Added raw binary `audio.in` support after a JSON header.
  - Added pending audio state so mic PCM can arrive without base64 overhead.

- `src/stacky/body/protocol.py`
  - `decode_pcm_payload` now accepts raw `bytes`/`bytearray` PCM as well as base64 data.

- `src/stacky/voice/stt.py`
  - Added STT AGC for wav2vec2 input.
  - `Wav2Vec2DanishSTT` now normalizes quiet StackChan mic captures before transcription.

- `firmware/stacky_cores3/src/main.cpp`
  - Mic frames are now sent as JSON header plus raw PCM bytes.
  - Audio-in frame size is 640 samples at 16 kHz, about 40 ms.
  - Mic config was changed to higher gain and reduced firmware noise filtering.

- `tests/test_body_controller.py`
  - Added raw incoming audio tests.

- `tests/test_body_protocol.py`
  - Added raw PCM decode test.

- `tests/test_stt.py`
  - Added AGC coverage.

## Verified Before Stop

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

Result: `57 tests OK`

```powershell
.\.venv\Scripts\pio.exe run -d firmware\stacky_cores3
```

Result: build success

```powershell
.\.venv\Scripts\pio.exe run -d firmware\stacky_cores3 -t upload
```

Result: firmware upload success

## Usual Commands

Run tests:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

Build firmware:

```powershell
.\.venv\Scripts\pio.exe run -d firmware\stacky_cores3
```

Flash firmware:

```powershell
.\.venv\Scripts\pio.exe run -d firmware\stacky_cores3 -t upload
```

Run handsfree with StackChan speaker:

```powershell
.\.venv\Scripts\python.exe -m stacky handsfree --tts-engine supertonic --speaker stackchan
```

Run listen-only mic debugging:

```powershell
.\.venv\Scripts\python.exe -m stacky handsfree --listen-only --debug-audio --body-timeout 35
```

Stop stuck Stacky/Pio processes:

```powershell
Get-CimInstance Win32_Process |
  Where-Object { $_.CommandLine -match 'stacky handsfree|pio monitor' } |
  ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
```

## Voice

Local APIs only. Avoid paid cloud TTS/STT by default.

Current preferred TTS is Supertonic with the `stacky` profile:

- Voice: `F2`
- Language: Danish
- Speed: `1.18`
- Steps: `8`
- Max chunk length: `240`
- Silence duration: `0.035`

Pronunciation adapter currently rewrites:

- `Nicolai` to `Nikolai`
- `Stacky` to `Stækki`
- `StackChan` to `Stack-tjan`
- `Sandcode` to `Sand-kode`

The user chose the first tuning profile as the current best one. The voice is acceptable, but still sings names a little and needs naturalness work later.

## Open Problem: Mic/STT

The current failure mode is not that Stacky uses Whisper. It does not, unless explicitly selected. The default path is wav2vec2 Danish STT.

Observed symptoms:

- Nicolai has to speak too loudly.
- Transcripts are wrong or fragmented.
- Captured turns were often very short, around 0.6 to 1.8 seconds.
- Earlier WAVs had low RMS and sometimes only partial speech.
- Gain in Python alone did not fix recognition.

The latest firmware raw mic transport and gain changes were flashed, but not fully validated because Nicolai asked to stop here.

Next debugging step should be:

1. Ensure no old `stacky handsfree` or `pio monitor` process is running.
2. Run listen-only with debug audio after firmware `0.3.15`.
3. Speak one clear Danish sentence at normal volume.
4. Inspect console RMS, peak, duration, transcript, and `artifacts/handsfree_turns`.
5. If raw framing is broken, compare against the old base64 mic path.
6. If audio clips, reduce `micCfg.magnification`.
7. If levels look healthy but transcripts remain bad, test another Danish STT engine.

## StackChan Mic Research

Useful links gathered during the mic investigation:

- M5Stack StackChan docs: https://docs.m5stack.com/en/stackchan
- StackChan Mic Arduino docs: https://docs.m5stack.com/en/arduino/stackchan/mic
- M5Stack CoreS3 docs: https://docs.m5stack.com/en/core/CoreS3
- ESPHome CoreS3 satellite config: https://raw.githubusercontent.com/m5stack/esphome-yaml/main/common/cores3-satellite-base.yaml
- ESPHome CoreS3 device page: https://devices.esphome.io/devices/m5stack-cores3/
- LiveKit shared I2S thread: https://community.livekit.io/t/m5stack-cores3-aw88298-es7210-shared-i2s-no-speaker-output-capture-works/561

Important clue: the ESPHome/CoreS3 voice configs use ES7210 mic gain around `36`, automatic gain around `31dBFS`, and volume multiplier around `2.0`. Stacky firmware should probably keep moving closer to that style if the current M5Unified mic path remains too quiet.

## Secrets And Local Files

Do not commit:

- `configs/stacky.toml`
- `firmware/stacky_cores3/include/wifi_secrets.h`
- `data/stacky/*.sqlite*`
- `artifacts/`
- `models/`
- `.venv/`
- `.sandcode/`

The repo should include examples only, such as `configs/stacky.example.toml` and `firmware/stacky_cores3/include/wifi_secrets.example.h`.

## GitHub Handoff Intent

Create a private GitHub repo for this project and push the current source, firmware, tests, docs, and this handoff. Keep it private until secrets/history have been reviewed again.
