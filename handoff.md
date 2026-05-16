# Stacky Handoff

## Latest Official Stacky Firmware Update

Official firmware migration is now a real Stacky firmware variant, not a loose bridge started from `app_main`. `AppStacky` has been added as a first-class Mooncake app, the firmware boots directly into it, and the Stacky body service is owned by that app lifecycle.

Active body base:

- Branch: `official-firmware-base`
- Official submodule: `vendor/m5stack-stackchan`
- Repro patch: `patches/official-stackchan/0001-stacky-bridge.patch`
- Flashed firmware: official StackChan `1.4.1` with `AppStacky` and bridge `official-0.1.6`
- Bridge `official-0.1.6` includes `body.look_at` and `body.gesture` head motion. Social center is `yaw=90`, `pitch=260`.
- PC body server: `192.168.50.208:8765`
- StackChan IP in tests: `192.168.50.2`

Firmware variant changes:

- New app: `firmware/main/apps/app_stacky/app_stacky.h`
- New app: `firmware/main/apps/app_stacky/app_stacky.cpp`
- `firmware/main/main.cpp` installs and opens only `AppStacky` at boot.
- `StackyBridge` now has `start()` / `stop()` lifecycle and is started from `AppStacky::onOpen()`.
- `StackyBridge` now sets CoreS3 codec output volume to 100 during playback.
- `StackyBridge` now accepts `audio.volume`, reports `speakerVolume`, uses mic gain 60, and drops the first 12 mic frames after input enable to avoid clipped warmup transients.
- Python StackChan TTS output has tunable loudness: `--stackchan-target-rms` and `--stackchan-max-gain` default to `9000` / `4.0`.
- Handsfree catches Danish volume commands before the LLM, e.g. `skru op`, `skru ned`, `sæt volumen til 60 procent`.
- Launcher/setup/app-center are no longer the boot path for Stacky.

Bridge support:

- `audio.in` raw PCM16 mono at 24 kHz from StackChan to PC
- `audio.start` / binary `audio.raw` / `audio.end` playback from PC to StackChan
- `audio.tone`, `audio.stop`, `audio.hold`, `body.status`, `body.set_expression`

Validated 2026-05-16:

- ESP-IDF build passed with short aliases `S:` and `I:`
- Flash to `COM3` passed
- `scripts\mic_listener.py --output artifacts\official_app_stacky_mic.wav --seconds 10` connected to `official-0.1.3` and captured mic frames
- `python -m stacky speaker-tone --body-timeout 35 --frequency 880 --duration-ms 250` returned success
- Full Python tests pass: `.\.venv\Scripts\python.exe -m unittest discover -s tests` -> 66 OK
- Python handsfree now analyzes raw turn audio before STT. Sparse/percussive sounds such as keyboard clicks are rejected before transcription, LLM response, and memory write.
- `StackyBrain` now includes the last 6 live conversation turns in the system context, so short bad transcripts do not wipe immediate conversational context.
- STT is still the weakest component. If raw speech-like turns pass the gate but transcripts are wrong, replace or add a better Danish STT backend rather than loosening filters.
- Added `python -m stacky stt-bench` to benchmark saved StackChan WAV turns without running the live body loop.
- Low-latency STT benchmark on StackChan turns:
  - `CoRal-project/roest-v3-wav2vec2-315m`: usable latency after load, about `rtf=0.16-0.18` on local CPU, but still mishears noisy/quiet turns.
  - `saattrupdan/wav2vec2-xls-r-300m-ftspeech`: low latency, but not usable on current StackChan mic captures; it returned `r` on tested clips.
  - `Qwen/Qwen3-ASR-0.6B` via `qwen-asr`: too slow and wrong in the quick local CPU test (`1.58s` audio -> `3.84s` inference, transcript `Se dig ved leon.`). `qwen-asr` also conflicts with Roest/Chatterbox by pinning `transformers==4.57.6`, so it was removed from the main venv and `transformers==5.2.0` restored.
- Handsfree latency tuning:
  - Default handsfree Supertonic profile is now `quick`.
  - Default end silence is now `450ms` instead of `650ms`.
  - Live replies default to about `150` spoken characters (`260` when details are requested).
  - Handsfree logs `[timing] stt=... brain=... tts_send=... total=...` after accepted turns.

Next checkpoint: run v0.1.3 handsfree with `--debug-audio`, test normal speech vs. keyboard noise, and inspect `[audio]` lines before changing STT thresholds.

Important: opening `COM3` with `scripts/serial_log.py` resets the CoreS3 via USB-serial. A boot log starting with `rst:0x15 (USB_UART_CHIP_RESET)` after playback is not proof of an audio crash.

## Stop Point

Stop here after official Stacky app variant v0.1.3. The old custom Arduino speaker-crash investigation is no longer the active path unless the same failure reproduces on official firmware.

## Current Direction

Stop fighting the custom Arduino audio stack. The active branch is `official-firmware-base`, which imports the official M5Stack StackChan firmware as a submodule at `vendor/m5stack-stackchan`. The current goal is to harden `AppStacky` as the permanent firmware shell.

Update: ESP-IDF v5.5.4 is now installed at `C:\Users\nicol\esp\esp-idf-v5.5.4`. Official firmware builds after applying the official XiaoZhi patch manually and building through short drive aliases. Official firmware was flashed to CoreS3 on `COM3`, and a 25-second serial boot log showed stable boot without reboot.

Migration details are in `docs/OFFICIAL_FIRMWARE_MIGRATION.md`.

## Project

Stacky is a fresh local Danish AI friend for Nicolai. It runs mainly on the Windows PC in `C:\Users\nicol\stackchan`, with StackChan/CoreS3 as the body: mic, speaker, face, servos, touch, LEDs, and later wheels.

Do not import or reuse Moss identity, Moss memories, Moss sessions, or the Moss name. Stacky has its own state in `data/stacky/soul.yaml`; runtime memory DB files are local and ignored by Git.

## Current Status

- Python package: `src/stacky`
- Firmware: `firmware/stacky_cores3` (now version `0.3.22`)
- Official firmware body base: `vendor/m5stack-stackchan/firmware` plus `patches/official-stackchan/0001-stacky-bridge.patch`
- Tests: `tests`
- Body server port: `8765`
- StackChan IP seen during tests: `192.168.50.2`
- PC IP used during tests: `192.168.50.208`
- USB serial seen during tests: `/dev/cu.usbmodem1101` or `cu.usbmodem101` on Mac, `COM3` on Windows
- Latest flashed firmware version: official StackChan `1.4.1` with `AppStacky` / Stacky bridge `official-0.1.6`

The old custom Arduino firmware remains as fallback/reference. The active hardware path is official ESP-IDF firmware; old speaker-crash assumptions should be re-tested before using them.

## Mac/Windows Workflow (NEW)

The Mac at `/Users/nicolai/stacky` is now a working dev mirror of the Windows project. Set up tonight:

- **PlatformIO installed on Mac**: `/Users/nicolai/Library/Python/3.13/bin/pio` — can flash firmware directly via USB without Windows
- **SSH key auth Mac → Windows**: `ssh nicolai@192.168.50.208` works without password. Key in `C:\ProgramData\ssh\administrators_authorized_keys` (admin auth path)
- **Wi-Fi creds copied**: Mac's `firmware/stacky_cores3/include/wifi_secrets.h` has real credentials (gitignored)

Typical iteration loop:
1. Edit code on Mac
2. `scp` changed files to Windows OR `pio run -t upload` to flash StackChan directly from Mac
3. `ssh nicolai@192.168.50.208 'cd C:\Users\nicol\stackchan && .\.venv\Scripts\python.exe -m stacky handsfree ...'` to run Python stack
4. Read serial from Mac (StackChan is USB-tethered to Mac during dev): `python3 /tmp/serial_log.py /dev/cu.usbmodem101 N /tmp/log.log`

## Latest Changes (Mac-side, uncommitted)

### Firmware 0.3.17 — mic fix (the big win)

- `firmware/stacky_cores3/src/main.cpp` setup:
  - **Removed `micCfg.magnification = 16;`** — this was the bug. M5Unified default (`magnification=2`) works; setting to 16 caused the mic to output only noise floor (RMS ~260, peak ~1700, no speech modulation). Pre-0.3.15 default mic worked; the 0.3.15 attempt at higher gain killed it.
  - Removed `micCfg.noise_filter_level = 0;` — was a no-op anyway (`0` is the default).
- `firmware/stacky_cores3/src/main.cpp` `sendStatus()`: Added periodic Serial diagnostic line so mic state is observable without USB-monitor on boot:
  ```
  mic_state running=1 gain=2 sr=16000 nf=0 recording=0 ap=0 ah=0 pending=0 session=0 spk=0
  ```
  Flags: running (M5.Mic), recording, ap=audioPlaying, ah=audioOutputHold, session=audioOutSessionOpen, spk=speaker playing.

### Firmware 0.3.18 / 0.3.19 — speaker I2S race (attempted fix, INCOMPLETE)

CoreS3 shares I2S port 1 between mic (ES7210, data pin 14) and speaker (AW88298, data pin 13). When `M5.Speaker.begin()` runs, it calls `_i2s_driver_uninstall(port)` before re-installing the port for output (Speaker_Class.cpp:200). If the mic FreeRTOS task is still in `i2s_read()` when this happens, mic_task crashes with Guru Meditation Error / LoadProhibited at EXCVADDR `0x1c` (addr2line resolved: `i2s_read` called from `m5::Mic_Class::mic_task` at Mic_Class.cpp:232/598).

Fix attempt in 0.3.19:
- `pauseMicForSharedI2S()`: always call `M5.Mic.end()` (no `isRunning()` check — that check was racy). Default settle increased from `60ms` to `150ms`.
- `prepareSpeakerForAudio()`: added `delay(120)` before `M5.Speaker.begin()` to ensure mic_task fully exits before the I2S port is yanked.

**This was not enough.** ESP32 still reboots when Python sends audio chunks to the speaker. User confirmed by direct observation (screen reset, expression resetting to "neutral" startup default in Python's status events). The next fix attempt should be informed by serial captured DURING the crash — earlier in this session we couldn't catch it because our serial logger wasn't running at the right moment.

### Python: ffmpeg fallback in `output.py`

`_prepare_stackchan_wav` in `src/stacky/voice/output.py` called `subprocess.run(["ffmpeg", ...])` and got `FileNotFoundError` even though ffmpeg is installed at `C:\Users\nicol\miniconda3\Library\bin\ffmpeg.exe` — PATH issue when subprocess runs from the venv. Added `shutil.which()` lookup with explicit fallback path. Tested but the speaker crash happened before we could validate the ffmpeg fix in isolation.

### Python: debug counter in `cli.py`

`on_event` in `_handsfree` now counts audio.in events and prints `[debug] audio.in #N accepting=X` every 200 events. Helps confirm controller thread is dispatching events correctly. Verified Python is receiving events steadily (counter went to 2400+ in tonight's test) — earlier "intet sker" mystery was actually low VAD-trigger from quiet speech, not blocked event flow.

### New: `scripts/mic_listener.py`

Minimal stdlib-only Python listener that accepts a StackChan TCP connection, captures raw PCM `audio.in` frames to a WAV file, and logs per-frame RMS/peak. Built to isolate mic-capture quality from the full STT stack. Was not used in the end (we tested via real `handsfree --listen-only` instead) but it's a clean fallback debug tool.

## Verified During This Session

```powershell
# On Windows via SSH from Mac:
.\.venv\Scripts\python.exe -m stacky handsfree --listen-only --debug-audio --body-timeout 60
# → mic captures speech, STT returns 'hej med dig' with logprob=-0.07 (high confidence)
```

```powershell
.\.venv\Scripts\python.exe -m stacky handsfree --tts-engine supertonic --speaker stackchan --debug-audio --min-speech-ms 100 --vad-threshold 100
# → full loop fires:
#   Nicolai: hej med dig
#   Stacky: Hej med dig. Hvordan går det i dag?
# → then ESP32 reboot when audio chunks start streaming to firmware speaker
```

```bash
# On Mac:
/Users/nicolai/Library/Python/3.13/bin/pio run -d firmware/stacky_cores3 -t upload --upload-port /dev/cu.usbmodem1101
# → builds and flashes 0.3.19 successfully
```

## Open Problems (ranked by priority)

### 1. Speaker chunked playback crashes ESP32 (BLOCKING)

When Python streams audio.raw chunks to firmware and calls `audio.end`, firmware tries to play via `M5.Speaker.playWav` → reboots. The 0.3.19 fix (always-Mic.end + 150ms settle + 120ms pre-Speaker delay) did NOT solve it. Need to:

1. Start serial logger BEFORE running handsfree
2. Capture the crash output (Guru Meditation register dump or similar)
3. Use `addr2line` against the elf in `.pio/build/m5stack-cores3/firmware.elf` to symbolize crash PC
4. Determine if it's still the same `mic_task / i2s_read` crash (mic not fully exited) or a NEW crash path (probably in `M5.Speaker` or chunked buffer handling)

Tools ready: addr2line at `/Users/nicolai/.platformio/packages/toolchain-xtensa-esp32s3/bin/xtensa-esp32s3-elf-addr2line`. Use `pio device monitor` or `python3 scripts/mic_listener.py` style serial reader.

### 2. Mic gain still too low at default magnification=2

User had to "fandeme råbe" to register speech on second test (peak only ~580 vs 7000+ seen in earlier test). The Python-side AGC and the "for lavt mic-niveau" rejection filter combine to drop many real-but-quiet transcripts.

Plan:
1. Bump firmware `micCfg.magnification` from default 2 → 4. **Do not go higher than 8** — 16 broke it entirely.
2. Re-test. If signal is healthier (peak >3000 at normal voice), keep 4. Else try 6.
3. Possibly lower the Python-side "for lavt mic-niveau" threshold in `_accept_stt_result` (find in `src/stacky/cli.py`).

### 3. "ignorerer STT (for lavt mic-niveau)" filter is too aggressive

Even when STT produced reasonable text (`'der har mere soreterede'` was garbage, but earlier `'hej med dig'` was good), the filter rejects based on absolute RMS. Should probably be replaced by a logprob-based check (we already see `logprob=-0.07` is excellent, `-0.61` is borderline, `-1.07` is junk).

## Usual Commands

Run tests:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

Build firmware from Mac (NEW):

```bash
/Users/nicolai/Library/Python/3.13/bin/pio run -d firmware/stacky_cores3
```

Flash firmware from Mac via USB (NEW):

```bash
/Users/nicolai/Library/Python/3.13/bin/pio run -d firmware/stacky_cores3 -t upload --upload-port /dev/cu.usbmodem1101
```

Build firmware on Windows (still works):

```powershell
.\.venv\Scripts\pio.exe run -d firmware\stacky_cores3
```

Run listen-only mic debugging via SSH:

```bash
ssh nicolai@192.168.50.208 'cd C:\Users\nicol\stackchan && .\.venv\Scripts\python.exe -m stacky handsfree --listen-only --debug-audio --body-timeout 60'
```

Run full handsfree via SSH (currently crashes at speaker):

```bash
ssh nicolai@192.168.50.208 'cd C:\Users\nicol\stackchan && .\.venv\Scripts\python.exe -m stacky handsfree --tts-engine supertonic --speaker stackchan --debug-audio --min-speech-ms 100 --vad-threshold 100'
```

Read serial from StackChan via USB on Mac:

```bash
python3 /tmp/serial_log.py /dev/cu.usbmodem101 30 /tmp/serial.log
# script source: see /tmp/serial_log.py — short stdlib + pyserial reader
```

Stop stuck Python on Windows via SSH:

```bash
ssh nicolai@192.168.50.208 'powershell -Command "Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force"'
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

The voice is acceptable, but still sings names a little and needs naturalness work later.

## Mic Research Done This Session

### Architecture diagnosis (M5Unified internals)

- `M5.Mic` uses a FreeRTOS task `mic_task` pinned via `xTaskCreatePinnedToCore`
- The task loops calling `i2s_read(port, buf, len, &result, 100ms)` on the shared I2S port
- `M5.Mic.end()` sets `_task_running = false` and waits via `do { vTaskDelay(1); } while (_task_handle);` for the task to set `_task_handle = nullptr` before returning
- But: if the task is mid-`i2s_read` (100ms timeout), end() waits — and during that wait, another caller can race
- `M5.Speaker.begin()` calls `_i2s_driver_uninstall(port)` at Speaker_Class.cpp:200 before re-installing for output. This is the dangerous moment.

### XiaoZhi protocol research (for future reference)

The new official M5StackChan ships with XiaoZhi firmware which uses a WebSocket protocol v3 (or MQTT+UDP hybrid) with OPUS audio at 16kHz mono input / 24kHz mono output, 60ms frames. This is a different paradigm from our PCM-over-TCP custom protocol. Not relevant for current Stacky (we use our own firmware), but useful context if we ever consider migrating to a more standard protocol.

### Useful links

- M5Stack StackChan docs: https://docs.m5stack.com/en/stackchan
- StackChan Mic Arduino docs: https://docs.m5stack.com/en/arduino/stackchan/mic
- M5Stack CoreS3 docs: https://docs.m5stack.com/en/core/CoreS3
- ESPHome CoreS3 satellite config: https://raw.githubusercontent.com/m5stack/esphome-yaml/main/common/cores3-satellite-base.yaml
- ESPHome CoreS3 device page: https://devices.esphome.io/devices/m5stack-cores3/
- LiveKit shared I2S thread: https://community.livekit.io/t/m5stack-cores3-aw88298-es7210-shared-i2s-no-speaker-output-capture-works/561
- m5stack-avatar (better face rendering than current custom drawFace): https://github.com/stack-chan/m5stack-avatar
- kisaragi-mochi/stackchan-mcp (MCP gateway with full tool list — useful protocol reference): https://github.com/kisaragi-mochi/stackchan-mcp

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

## Recommended Next Session Order

1. **Capture the speaker crash in serial** — start `python3 /tmp/serial_log.py /dev/cu.usbmodem101 60 /tmp/crash.log` BEFORE running handsfree. Trigger one turn. Read the Guru Meditation dump from the log.
2. **Symbolize with addr2line** — use `xtensa-esp32s3-elf-addr2line -pfiaC -e firmware.elf <PC>` to see exactly where it dies.
3. **Apply targeted fix** based on actual crash location — could be in mic_task still, could be in `M5.Speaker` setup, could be in chunked audio buffer handling.
4. **Bump mic magnification** to 4 (don't go higher — 16 was catastrophic).
5. **Tune "for lavt mic-niveau" filter** to use logprob instead of absolute RMS.
6. **Final validation**: full loop with Nicolai speaking at normal voice, Stacky responding through StackChan speaker without crash.
