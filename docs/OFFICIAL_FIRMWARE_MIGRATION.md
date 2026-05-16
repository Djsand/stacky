# Stacky Official Firmware Migration

This branch moves Stacky away from the fragile Arduino/M5Unified firmware spike and toward the official M5Stack StackChan firmware as the body base.

The goal is not to import any old identity or memory. Stacky remains a fresh local Danish AI companion. The official firmware is only the hardware/body base for microphone, speaker, display, touch, LEDs, head servos, and later wheels.

## Current State

- Branch: `official-firmware-base`
- Official upstream is tracked as a git submodule:
  - `vendor/m5stack-stackchan`
  - upstream: `https://github.com/m5stack/StackChan.git`
  - revision at import: `da156e1`
- The previous custom Arduino firmware is still in `firmware/stacky_cores3` for fallback/reference only.
- ESP-IDF v5.5.4 is installed locally at `C:\Users\nicol\esp\esp-idf-v5.5.4`.
- Official firmware builds on this PC after applying the official XiaoZhi patch manually and building through short drive aliases.
- Official firmware was flashed to CoreS3 on `COM3`.
- Boot was serial-logged for 25 seconds without a reboot. Log: `artifacts/official_firmware_boot.log`.
- Stacky firmware variant `official-0.1.7` has been added as a local patch on top of official firmware.
- Repro patch: `patches/official-stackchan/0001-stacky-bridge.patch`
- Latest flashed body version: official StackChan `1.4.1` plus `AppStacky` / bridge `official-0.1.7`.

## Stacky App Variant Official-0.1.7

The first official-firmware customization is now a real app variant. Stacky boots directly into a custom Mooncake app instead of launching the official launcher/setup path. It keeps the existing Python body-controller protocol so the PC runtime does not need a rewrite before we validate hardware stability.

Changed official firmware files:

- `firmware/main/main.cpp`
- `firmware/main/apps/apps.h`
- `firmware/main/apps/app_stacky/app_stacky.h`
- `firmware/main/apps/app_stacky/app_stacky.cpp`
- `firmware/main/hal/stacky_bridge.h`
- `firmware/main/hal/stacky_bridge.cpp`

Runtime shape:

- `main.cpp` installs and opens only `AppStacky`.
- `AppStacky::onOpen()` creates the Stacky face and starts `StackyBridge`.
- `AppStacky::onClose()` stops `StackyBridge` and resets the avatar.
- `StackyBridge` has explicit `start()` / `stop()` lifecycle and updates the Stacky face for listening/thinking/speaking states.
- `StackyBridge` sets CoreS3 codec output volume to 100 during playback; PC-side StackChan PCM boost is also tunable with `--stackchan-target-rms` and `--stackchan-max-gain`.
- `StackyBridge` accepts `audio.volume`, reports `speakerVolume`, uses mic gain 60, and discards the first 12 mic frames after input enable to avoid clipped warmup transients poisoning VAD/STT.
- `StackyBridge` accepts `body.look_at` and `body.gesture` for head-servo motion. Gestures currently include `center`, `look_left`, `look_right`, `look_up`, `look_down`, `nod`, and `shake`.
- `StackyBridge` accepts `body.motion_config` for runtime head center/range calibration and reports `centerYaw` / `centerPitch`.
- `StackyBridge` uses a Stacky-specific default social head center (`yaw=90`, `pitch=260`) instead of raw servo home so up/down motion has visible travel in both directions. Python can override this at runtime from `data/stacky/body_calibration.json`.
- The official launcher/setup/app-center flow is not the Stacky boot path.

The bridge reads local Wi-Fi/host config from `stacky_local_config.h` when present, otherwise from the existing gitignored `firmware/stacky_cores3/include/wifi_secrets.h` compatibility header.

Supported protocol:

- `audio.in`: raw PCM16 mono mic frames at 24 kHz from StackChan to PC
- `audio.start` / binary `audio.raw` / `audio.end`: buffered PCM16 playback from PC to StackChan
- `audio.tone`: local firmware-generated tone for speaker smoke tests
- `audio.stop`, `audio.hold`, `body.status`, `body.set_expression`, `body.look_at`, `body.gesture`, `body.motion_config`

Validated locally:

```powershell
.\.venv\Scripts\python.exe scripts\mic_listener.py --output artifacts\official_bridge_mic.wav --seconds 25
# captured 24.94s @ 24000 Hz, 1247 frames, no clipping

.\.venv\Scripts\python.exe -m stacky speaker-tone --body-timeout 45 --frequency 880 --duration-ms 350
# command returned success

.\.venv\Scripts\python.exe -m stacky speaker-test --body-timeout 45 --tts-engine supertonic --text "Hej Nicolai. Det her er official Stacky firmware. Jeg taler gennem StackChan nu."
# command returned success

.\.venv\Scripts\python.exe -m stacky speaker-test --body-timeout 45 --tts-engine supertonic --stackchan-target-rms 9000 --stackchan-max-gain 4.0 --text "Hej Nicolai. Det her er en tydeligere volumen-test."
# louder StackChan speaker tuning without reflashing

.\.venv\Scripts\python.exe scripts\mic_listener.py --output artifacts\official_app_stacky_mic.wav --seconds 10
# connected to firmware=official-0.1.3 and captured mic frames

.\.venv\Scripts\python.exe -m stacky speaker-tone --body-timeout 35 --frequency 880 --duration-ms 250
# command returned success after AppStacky refactor
```

Note: opening `COM3` with `scripts/serial_log.py` resets CoreS3. A serial log that begins with `rst:0x15 (USB_UART_CHIP_RESET)` after playback is not evidence of an audio crash.

## Why Switch Bases

The custom Arduino firmware proved that WebSocket control, face state, basic audio input, and speaker output are possible, but it stayed unstable:

- microphone level and STT segmentation changed unpredictably between firmware attempts,
- speaker playback could crash or underrun,
- audio streaming was fragile and sounded broken under real use,
- official StackChan already has a more complete ESP-IDF body stack.

The official firmware includes:

- an ESP-IDF build (`firmware/README.md`),
- `CoreS3AudioCodec` for CoreS3 duplex audio,
- an integrated mic test path,
- body HAL for display, head control, LEDs, touch, battery, camera hooks, and XiaoZhi mode,
- MCP-style body tools for robot/head functions.

## Official Build Path

Official firmware lives here:

```powershell
cd C:\Users\nicol\stackchan\vendor\m5stack-stackchan\firmware
```

Install ESP-IDF v5.5.x first. The official README names ESP-IDF v5.5.4, and `idf_component.yml` requires at least 5.5.2.

Then fetch official dependencies:

```powershell
python .\fetch_repos.py
```

Build and flash:

```powershell
idf.py set-target esp32s3
idf.py build
idf.py -p COM_PORT flash monitor
```

Replace `COM_PORT` with the CoreS3 serial port.

## Windows Build Notes

On this PC, direct build from the long repo path hit Windows' command line length limit during the final link step. Use short drive aliases for repeat builds:

```powershell
subst S: C:\Users\nicol\stackchan\vendor\m5stack-stackchan\firmware
subst I: C:\Users\nicol\esp\esp-idf-v5.5.4
. I:\export.ps1
$env:IDF_PATH = 'I:\'
Set-Location S:\
idf.py build
```

The official `fetch_repos.py` currently reports that `patches/xiaozhi-esp32.patch` cannot be applied cleanly to `xiaozhi-esp32`. The practical workaround used here was:

```powershell
Set-Location C:\Users\nicol\stackchan\vendor\m5stack-stackchan\firmware\xiaozhi-esp32
git apply --reject --whitespace=nowarn ..\patches\xiaozhi-esp32.patch
```

Then manually add the rejected `TryReadRegs` implementation to `main\boards\common\i2c_device.cc`:

```cpp
esp_err_t I2cDevice::TryReadRegs(uint8_t reg, uint8_t* buffer, size_t length, int timeout_ms) {
    return i2c_master_transmit_receive(i2c_device_, &reg, 1, buffer, length, timeout_ms);
}
```

After that, `idf.py build` completed and produced `build\stack-chan.bin` plus `build\generated_assets.bin`.

Apply Stacky's local official-firmware bridge patch from the parent repo:

```powershell
Set-Location C:\Users\nicol\stackchan\vendor\m5stack-stackchan
git apply ..\..\patches\official-stackchan\0001-stacky-bridge.patch
```

Flash command used:

```powershell
idf.py -p COM3 flash
```

## First Validation

Before adding Stacky custom code, validate stock official firmware:

1. Flash official firmware cleanly.
2. Confirm boot does not loop.
3. Run or trigger its built-in audio/mic test.
4. Confirm the StackChan speaker plays audio without crashing.
5. Confirm the mic path captures voice at normal speaking volume.
6. Confirm face, touch, head servos, and LEDs still work.

If stock official mic/speaker works, stop investing in the Arduino audio firmware.

## Files Of Interest

Official audio and body files to inspect/edit first:

- `vendor/m5stack-stackchan/firmware/main/hal/board/cores3_audio_codec.h`
- `vendor/m5stack-stackchan/firmware/main/hal/board/cores3_audio_codec.cc`
- `vendor/m5stack-stackchan/firmware/main/hal/audio.cpp`
- `vendor/m5stack-stackchan/firmware/main/hal/hal.cpp`
- `vendor/m5stack-stackchan/firmware/main/hal/hal.h`
- `vendor/m5stack-stackchan/firmware/main/hal/hal_mcp.cpp`
- `vendor/m5stack-stackchan/firmware/main/hal/board/hal_bridge.h`
- `vendor/m5stack-stackchan/firmware/main/hal/board/stackchan.cc`
- `vendor/m5stack-stackchan/firmware/main/main.cpp`
- `vendor/m5stack-stackchan/firmware/main/idf_component.yml`
- `vendor/m5stack-stackchan/firmware/repos.json`

## Stacky Integration Plan

Keep the PC side as the brain:

- `src/stacky` remains the local Windows brain/runtime.
- LM Studio remains the local LLM endpoint.
- Danish remains the default spoken language.
- Memory remains Stacky-only and fresh.
- Sandcode/Codex orchestration stays on the PC side.

Make official firmware the body:

- add a Stacky local mode to the ESP-IDF firmware,
- expose a PC transport over WebSocket/TCP,
- stream mic audio from StackChan to PC,
- stream generated TTS audio from PC to StackChan,
- send body commands from PC to firmware:
  - `body.status`
  - `body.set_expression`
  - `body.look_at`
  - `body.gesture`
  - `body.leds`
  - `body.audio_out`
- preserve the official audio codec path instead of reimplementing low-level I2S in Arduino.

## Transport Options

Preferred path:

- reuse or adapt the official XiaoZhi/audio transport if it already handles robust streaming and buffering,
- point it at the local Stacky PC service instead of a cloud service,
- translate official events into Stacky's existing hands-free pipeline.

Fast fallback path:

- add a simple local Stacky WebSocket mode:
  - binary `audio.in` frames from device to PC,
  - binary `audio.out` frames from PC to device,
  - JSON control frames for body state and status,
  - ring-buffered playback on firmware to avoid broken/choppy audio.

## Immediate Next Steps

1. Confirm on-device screen boots straight into Stacky face, not launcher/setup.
2. Run `handsfree --listen-only --debug-audio --body-timeout 45` against `AppStacky`.
3. If listen-only is acceptable, run full `handsfree --tts-engine supertonic --speaker stackchan` against `AppStacky`.
4. If playback is choppy or silent, improve `StackyBridge` buffering inside the app-owned service.
5. Only after the body transport is stable, return to Danish STT/TTS quality tuning.

## Decision Rule

Do not keep fighting the custom Arduino audio path unless official stock firmware also fails the same mic/speaker tests. The official firmware is now the primary body path.
