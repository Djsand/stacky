# Stacky Handoff

## Current Snapshot

Updated 2026-05-16 after the StackChan mic-gain/auto-channel hotfix and second STT research pass.

- Branch: `official-firmware-base`
- Latest branch head: run `git log --oneline -1` after pull.
- Recent implementation work:
  - Danish STT hotwords, live transcript correction, stricter clipped-noise gate, benchmark live-gate mode.
  - StackChan firmware `official-0.1.10` with PC-controlled mic gain.
  - Handsfree/capture default `--mic-channel auto` instead of fixed channel `0`.
  - Short spoken fallback when the brain endpoint is down, so TTS does not read a long exception.
  - `body-server` now parses raw `audio.in` frames safely and no longer sends a status feedback loop.
  - `efcff5c Fix VAD noise floor endpointing`
  - `f50b3b3 Prioritize Danish Røst STT evaluation`
  - `ac5cd41 Add StackChan STT dataset benchmark loop`
  - `2a26d35 Add Stacky infinite sessions and safe memory gating`
- Remote: `origin https://github.com/Djsand/stacky.git`
- Push status: `official-firmware-base` pushed to `origin`
- Local `git status --short`: expected to show only `m vendor/m5stack-stackchan` while the official patch is applied for build/flash.
- To recreate the Stacky firmware files on another PC, run `.\scripts\apply-official-firmware-patch.ps1`. The script initializes `vendor/m5stack-stackchan` and applies `patches/official-stackchan/0001-stacky-bridge.patch`.
- No live hands-free server should be assumed running.

Latest verification:

- `.\.venv\Scripts\python.exe -m pytest tests` -> `133 passed`
- `.\.venv\Scripts\python.exe -m pip check` -> no broken requirements
- `git diff --check -- . ':!patches/official-stackchan/0001-stacky-bridge.patch'` -> clean apart from CRLF warnings. The patch file itself contains normal unified-diff context blank lines that `git diff --check` reports as trailing whitespace when treating the patch as a text file.
- ESP-IDF build passed via `S:\` / `I:\`.
- Flash to `COM3` passed for `official-0.1.10`.
- `python -m stacky body-server --duration 4` connected and reported `firmware=official-0.1.10`, `micGain=75.0`, `micChannels=2`, and `audio.in raw 24000 Hz 2 ch 1920 bytes`.

Runtime state:

- Runtime DB is ignored: `data/stacky/memory.sqlite`
- Runtime DB backup is ignored: `data/stacky/memory.sqlite.backup-20260516-164456`
- Runtime session files are ignored: `data/stacky/sessions/`
- Runtime body calibration is ignored: `data/stacky/body_calibration.json`

Next engineering priority:

1. Test the new live STT path on real StackChan hardware:
   - `python -m stacky handsfree --tts-engine supertonic --speaker stackchan`
   - Expected startup should print `Using StackChan mic channel: auto` and `Setting StackChan mic gain: 75`.
   - speak normal Danish first, then fast, then slightly mumbled.
2. If live behavior is still wrong, run listen-only with channel probes before changing model code:
   - `python -m stacky handsfree --listen-only --debug-audio --mic-channel auto`
   - `python -m stacky handsfree --listen-only --debug-audio --mic-channel 1`
   - `python -m stacky handsfree --listen-only --debug-audio --mic-channel mix`
3. If a transcript is wrong but the audio is real speech, capture the failing turn WAV from `artifacts/handsfree_turns/` and add a narrow correction/test before changing models.
4. Use the STT dataset loop before changing models/filters:
   - `python -m stacky stt-capture --limit 12 --debug-audio`
   - `python -m stacky stt-capture --phrases-file .\artifacts\stt_phrases.txt --noise-count 3 --debug-audio`
   - `python -m stacky stt-bench --dataset .\artifacts\stt_dataset\stackchan\manifest.jsonl --report .\artifacts\stt_dataset\stt-report.jsonl`
5. Capture and benchmark channel 1 separately before changing more ASR code:
   - `python -m stacky stt-capture --limit 10 --speech-style normal --speech-style fast --speech-style mumble --noise-count 5 --mic-channel 1 --output-dir .\artifacts\stt_dataset\stackchan-ch1 --debug-audio`
   - `python -m stacky stt-bench --dataset .\artifacts\stt_dataset\stackchan-ch1\manifest.jsonl --engine roest-v3 --report .\artifacts\stt_dataset\stt-ch1-roest-v3.jsonl`
6. Voice remains an untrusted session source. Do not write handsfree transcripts into memory/infinite sessions until live STT is stable across normal daily speech.
7. Once STT is reliable, flip hands-free voice from untrusted to trusted session persistence deliberately, with a test.

## Latest Mic/STT Hotfix

- Firmware patch regenerated for `official-0.1.10`.
- Firmware default input gain is now `75`; Python sends `audio.input_gain` at connect time.
- `handsfree` and `stt-capture` default to `--mic-channel auto`; `auto`/`best` selects the loudest channel per PCM chunk when firmware sends stereo mic audio.
- Signal quality now accepts shorter soft speech runs (`180ms`) so quiet real speech is less likely to be thrown away before STT.
- Observed live failures now have conservative transcript corrections:
  - `hej den i` / `hej d` -> `Hej Stacky`
  - `d gik op ad` / `gik op ad` -> `Kig op.`
- Brain connection failures keep the full exception in text logs but only speak: `Min brain-model svarer ikke lige nu. Jeg lytter stadig.`
- `body-server` is safe for raw binary `audio.in` frames and prints only the first few/every 100th audio frame.

Second STT candidate check:

- Qwen3-ASR GGUF via latest llama.cpp b9181 HIP/ROCm loaded and used the Strix Halo GPU, but was not better on StackChan clips:
  - 0.6B on `Hej Stacky` -> `Hai, Stegi.`
  - 1.7B on `Hej Stacky` -> `Hej Stegi.`
  - 1.7B on the next normal clips produced wrong/foreign-language text, so it is not a usable replacement.
- Parakeet TDT 0.6B v3 ONNX was very fast, but failed Danish StackChan audio (`I`, English hallucinations).
- Canary 1B v2 ONNX was fast and accepted `language='da'`, but still mistranscribed StackChan audio badly.
- Practical conclusion remains: fix StackChan mic/channel/preprocessing first. Blind model swapping is not supported by the benchmark evidence.

## Latest Danish STT Research/Fix

Research summary:

- Better model cards did not translate to better StackChan mic performance.
- Tested/checked candidates include Røst v2/v3 wav2vec2, Røst Whisper 1.5B, Whisper large-v3-turbo, Qwen3-ASR, Saga, Milo, MediaCatch XLS-R, NbAiLab wav2vec2, and NVIDIA Parakeet Danish.
- Practical conclusion: keep Røst v3 wav2vec2 as the low-latency base and add Stacky-specific decoding/correction around it.
- Qwen/Saga/Milo can be revisited only if a proper accelerated backend is available. CPU/Windows tests were not usable enough.

Implemented:

- `Wav2Vec2DanishSTT` now uses pyctcdecode hotwords by default, weight `5.0`.
- Hotwords cover Stacky, Nicolai, common volume/motion/calibration phrases, latency, Sandcode, Home Assistant.
- New `src/stacky/voice/transcript_correction.py` applies conservative live correction for observed Stacky intents:
  - `hej stakke` / `hej op i` -> `Hej Stacky`
  - `oligopoly` / similar -> `Skru lidt op for lyden.`
  - `lidt til hojre` / `lidt for her` -> `Kig lidt til højre.`
  - common Nicolai/Stacky spelling variants
- Handsfree now logs raw-to-corrected STT when it changes a transcript.
- Handsfree only treats exact/phrase corrections as trusted; short unclear uncorrected fragments are rejected instead of sent to the LLM.
- `stt-bench --correct-transcripts` scores the live post-correction path.
- `stt-bench --live-gate` recomputes signal quality and measures the same pre-STT noise rejection used by handsfree.
- VAD now rejects clipped/percussive StackChan noise with `peak>=32000`, high crest factor, and high p95 RMS.

Current benchmark on `artifacts/stt_dataset/stackchan/manifest.jsonl`:

- Previous Røst v3 full channel-0 dataset: `mean_wer=72.2%`, `mean_cer=49.5%`, `rtf=0.15`.
- Røst v3 + hotwords only, first 10 normal clips: `mean_wer=45.8%`, `mean_cer=26.0%`, `rtf=0.13`.
- Røst v3 + hotwords + live correction, first 10 normal clips: `mean_wer=18.4%`, `mean_cer=11.4%`, `rtf=0.13`.
- Current full live pipeline with correction + live gate:
  - total: `mean_wer=27.4%`, `mean_cer=17.6%`, `rtf=0.11`
  - normal: `mean_wer=15.1%`, `mean_cer=9.5%`
  - fast: `mean_wer=38.4%`, `mean_cer=19.7%`
  - mumble: `mean_wer=42.4%`, `mean_cer=32.5%`
  - noise: all 5 noise clips rejected before STT

Reality check:

- This is a better usable local path, not perfect general Danish dictation.
- Stacky control phrases, names, greetings, volume, center, and right-look commands are now much stronger.
- Long fast/mumbled free text is still weak. For that, the real fix is a larger Nicolai/StackChan dataset and fine-tuning/adaptation, not another blind model swap.

## Latest STT/Mic Channel Update

- The user captured 35 StackChan clips: 10 normal, 10 fast, 10 mumble, 5 noise.
- `stt-bench` now tests all clips by default (`--limit 0`) and prints per-`speechStyle` summaries.
- Røst v3 on the full channel-0 dataset: `mean_wer=72.2%`, `mean_cer=49.5%`, `rtf=0.15`. This is not usable for daily voice.
- Røst v2 315M/1B/2B were also poor on the first 8 normal clips; bigger Røst did not fix it.
- Qwen3-ASR 0.6B was tested in an isolated venv at `C:\Users\nicol\.stacky-qwen-asr` to avoid main-venv dependency conflicts. On the first 10 normal channel-0 clips it produced `mean_wer=100.0%`, `mean_cer=71.5%`, and was slower than Røst after warm-up.
- Conclusion: do not keep swapping models blindly. The current evidence points to StackChan audio/channel quality or mic-path selection. Official firmware now streams both input channels so channel 1 can be measured.
- `TurnSignalQuality` now rejects clipped/percussive noise with `peak>=32000`, high crest factor, and short active runs.
- `handsfree` and `stt-capture` accept `--mic-channel auto|best|0|1|mix|all`; default is now `auto`.
- `scripts/mic_listener.py` now handles multichannel WAV output and prints `micChannels` from status.

## Latest Session And Memory Update

Stacky's session/memory layer has been rebuilt after inspecting the proven architecture in `D:\moss\core`. Only implementation patterns were copied. No old identity, old memories, old sessions, or old names were imported.

New Stacky-native runtime pieces:

- `src/stacky/sessions.py` adds an append-only infinite thread at `data/stacky/sessions/stacky-infinite-thread.jsonl`.
- The session store rolls old JSONL files, stitches newest-to-oldest up to a token budget, then presents them chronologically to the brain.
- Stitching injects time-gap markers and escaped recalled memories as system context.
- Repetitive assistant replies are condensed so repeated filler does not dominate context.
- Trusted text/chat turns persist to the infinite thread.
- Hands-free StackChan voice currently uses existing session context but does **not** write transcripts into the infinite thread, long-term memory, or short-term live context. This is intentional until STT is reliable enough for daily use.
- `MemoryStore.recall()` excludes `dialogue` memories by default.
- `StackyBrain.respond()` no longer writes raw dialogue into long-term memory by default.
- Existing polluted dialogue memories were removed from `data/stacky/memory.sqlite` after backup.
- `src/stacky/voice/stt_eval.py` contains the repeatable StackChan STT dataset/benchmark helpers.

Memory cleanup performed 2026-05-16:

- Backup: `data/stacky/memory.sqlite.backup-20260516-164456`
- Removed rows tagged `dialogue`: `142`
- Remaining memory rows: `1`

Current priority is not head-motion polish. The blocking product problem is still input/session quality: Danish STT must be stable enough that voice can safely become a trusted session source again.

## Latest Official Stacky Firmware Update

Official firmware migration is now a real Stacky firmware variant, not a loose bridge started from `app_main`. `AppStacky` has been added as a first-class Mooncake app, the firmware boots directly into it, and the Stacky body service is owned by that app lifecycle.

Active body base:

- Branch: `official-firmware-base`
- Official submodule: `vendor/m5stack-stackchan`
- Repro patch: `patches/official-stackchan/0001-stacky-bridge.patch`
- Patch apply script: `scripts/apply-official-firmware-patch.ps1`
- Latest body patch: official StackChan `1.4.1` with `AppStacky` and bridge `official-0.1.9`
- Bridge `official-0.1.9` includes `body.look_at`, `body.gesture`, runtime `body.motion_config` head calibration, dual-channel mic streaming, and Stacky-branded boot screen. Default social center is `yaw=90`, `pitch=260`.
- PC body server: `192.168.50.208:8765`
- StackChan IP in tests: `192.168.50.2`

Firmware variant changes:

- New app: `firmware/main/apps/app_stacky/app_stacky.h`
- New app: `firmware/main/apps/app_stacky/app_stacky.cpp`
- `firmware/main/main.cpp` installs and opens only `AppStacky` at boot.
- `StackyBridge` now has `start()` / `stop()` lifecycle and is started from `AppStacky::onOpen()`.
- `StackyBridge` now sets CoreS3 codec output volume to 100 during playback.
- `StackyBridge` now accepts `audio.volume`, reports `speakerVolume`, uses mic gain 60, and drops the first 12 mic frames after input enable to avoid clipped warmup transients.
- `StackyBridge` now accepts `body.motion_config` and reports `centerYaw` / `centerPitch`.
- Boot screen now says `STACKY`, shows `official-0.1.9`, and uses a small LVGL-drawn Stacky logo mark.
- Python StackChan TTS output has tunable loudness: `--stackchan-target-rms` and `--stackchan-max-gain` default to `9000` / `4.0`.
- Handsfree catches Danish volume commands before the LLM, e.g. `skru op`, `skru ned`, `sæt volumen til 60 procent`.
- Handsfree catches Danish center calibration commands before the LLM: `lidt mere til højre`, `lidt mere til venstre`, `lidt op`, `lidt ned`, `gem den her position som center`.
- Runtime head calibration persists to `data/stacky/body_calibration.json`.
- Handsfree VAD default is now `--vad-threshold 280 --start-speech-ms 120 --min-speech-ms 220`; turn quality analysis rejects high-frequency noise by zero-crossing rate before STT.
- Launcher/setup/app-center are no longer the boot path for Stacky.

Bridge support:

- `audio.in` raw PCM16 at 24 kHz from StackChan to PC; firmware `official-0.1.9` sends all codec input channels and Python selects `--mic-channel`.
- `audio.start` / binary `audio.raw` / `audio.end` playback from PC to StackChan
- `audio.tone`, `audio.stop`, `audio.hold`, `body.status`, `body.set_expression`, `body.look_at`, `body.gesture`, `body.motion_config`

Validated 2026-05-16:

- ESP-IDF build passed with short aliases `S:` and `I:`
- Flash to `COM3` passed
- `scripts\mic_listener.py --output artifacts\official_app_stacky_mic.wav --seconds 10` connected to `official-0.1.3` and captured mic frames
- `python -m stacky speaker-tone --body-timeout 35 --frequency 880 --duration-ms 250` returned success
- Full Python tests pass: `.\.venv\Scripts\python.exe -m unittest discover -s tests` -> 101 OK
- Python handsfree now analyzes raw turn audio before STT. Sparse/percussive sounds such as keyboard clicks are rejected before transcription, LLM response, and memory write.
- `StackyBrain` now includes the last 6 live conversation turns in the system context, so short bad transcripts do not wipe immediate conversational context.
- STT is still the weakest component. If raw speech-like turns pass the gate but transcripts are wrong, replace or add a better Danish STT backend rather than loosening filters.
- Added `python -m stacky stt-bench` to benchmark saved StackChan WAV turns without running the live body loop.
- Low-latency STT benchmark on StackChan turns:
  - Default candidates are now Danish-specific Røst models: `CoRal-project/roest-v3-wav2vec2-315m` and `CoRal-project/roest-v2-wav2vec2-315m`.
  - Heavy candidates add `CoRal-project/roest-v2-wav2vec2-1B` and `CoRal-project/roest-v2-wav2vec2-2B` before Qwen/Saga/Milo experiments.
  - `saattrupdan/wav2vec2-xls-r-300m-ftspeech` remains available by explicit `--engine ftspeech`, but is not a default after poor StackChan mic results.
- `stt-capture` now supports repeated speech styles: `--speech-style normal --speech-style fast --speech-style mumble --speech-style quiet`. Use this to build a Nicolai/StackChan dataset for fast or slightly mumbled Danish.
  - `Qwen/Qwen3-ASR-0.6B` via `qwen-asr`: too slow and wrong in the quick local CPU test (`1.58s` audio -> `3.84s` inference, transcript `Se dig ved leon.`). `qwen-asr` also conflicts with Roest/Chatterbox by pinning `transformers==4.57.6`, so it was removed from the main venv and `transformers==5.2.0` restored.
- Handsfree latency tuning:
  - Default handsfree Supertonic profile is now `quick`.
  - Default end silence is `850ms`; current live test uses `900ms` to avoid splitting Danish phrases.
  - Live replies default to about `150` spoken characters (`260` when details are requested).
  - Handsfree logs `[timing] stt=... brain=... tts_send=... total=...` after accepted turns.

No live hands-free server should be assumed running after the session/memory pivot. Start it manually for the next hardware test. Previous logs are in `artifacts/logs/handsfree-current.log`.

Latest observed VAD behavior: high-frequency peaks were rejected before STT with `reason='højfrekvent støj'`; accepted speech-like turns used internal thresholds around `thr=506` instead of the previous hard 650.

Important: opening `COM3` with `scripts/serial_log.py` resets the CoreS3 via USB-serial. A boot log starting with `rst:0x15 (USB_UART_CHIP_RESET)` after playback is not proof of an audio crash.

## Stop Point

Stop here after official Stacky app variant `official-0.1.9`. The old custom Arduino speaker-crash investigation is no longer the active path unless the same failure reproduces on official firmware.

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
- Latest body patch version: official StackChan `1.4.1` with `AppStacky` / Stacky bridge `official-0.1.9`

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
.\.venv\Scripts\python.exe -m stacky handsfree --tts-engine supertonic --speaker stackchan --debug-audio --vad-threshold 280 --start-speech-ms 140 --min-speech-ms 220 --end-silence-ms 900
# → full loop runs on official firmware; high-frequency false turns are rejected before STT.
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

Run full handsfree via SSH:

```bash
ssh nicolai@192.168.50.208 'cd C:\Users\nicol\stackchan && .\.venv\Scripts\python.exe -m stacky handsfree --tts-engine supertonic --speaker stackchan --debug-audio --vad-threshold 280 --start-speech-ms 140 --min-speech-ms 220 --end-silence-ms 900'
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
