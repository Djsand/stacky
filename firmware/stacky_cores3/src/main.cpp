#include <Arduino.h>
#include <M5Unified.h>
#include <WiFi.h>
#include <esp_heap_caps.h>
#include <mbedtls/base64.h>

#include "wifi_secrets.h"

WiFiClient client;
String incoming;
String expressionName = "neutral";
String statusLine = "";
unsigned long lastStatusAt = 0;
static constexpr uint32_t AUDIO_IN_SAMPLE_RATE = 16000;
static constexpr size_t AUDIO_IN_SAMPLES = 640;
static constexpr size_t AUDIO_IN_BYTES = AUDIO_IN_SAMPLES * sizeof(int16_t);
static constexpr size_t AUDIO_OUT_CHUNK_SAMPLES = 8192;
static constexpr size_t AUDIO_OUT_CHUNK_BYTES = AUDIO_OUT_CHUNK_SAMPLES * sizeof(int16_t);
static constexpr size_t AUDIO_OUT_BUFFER_COUNT = 3;
static constexpr size_t AUDIO_RAW_READ_BYTES = 4096;
static constexpr size_t WAV_HEADER_BYTES = 44;
static constexpr size_t MAX_COMMAND_LINE = 32768;
static constexpr size_t MAX_AUDIO_OUT_BYTES = 1024 * 1024;
static constexpr uint16_t SPEAKER_VOLUME = 192;
int16_t micBuffer[AUDIO_IN_SAMPLES];
int16_t audioOutChunkBuffers[AUDIO_OUT_BUFFER_COUNT][AUDIO_OUT_CHUNK_SAMPLES];
uint8_t audioRawReadBuffer[AUDIO_RAW_READ_BYTES];
uint8_t* audioOutBytes = nullptr;
size_t audioOutLen = 0;
size_t audioOutCapacity = 0;
size_t audioOutBufferIndex = 0;
size_t pendingAudioRawBytes = 0;
size_t pendingAudioRawRead = 0;
int pendingAudioRawSeq = -1;
bool pendingAudioRawOk = true;
uint32_t audioOutSampleRate = 24000;
bool audioOutSessionOpen = false;
bool audioPlaying = false;
bool audioOutputHold = false;
uint32_t audioInSeq = 0;
uint32_t audioPlayCounter = 0;
uint32_t audioPlaybackDeadlineMs = 0;
uint32_t audioChunkQueued = 0;
uint32_t audioChunkDropped = 0;

void showStatusLine(const String& text, uint16_t color = TFT_DARKGREY) {
  statusLine = text;
  M5.Display.fillRect(0, 204, 320, 36, TFT_BLACK);
  M5.Display.setTextColor(color, TFT_BLACK);
  M5.Display.setTextDatum(middle_center);
  M5.Display.setTextSize(2);
  M5.Display.drawString(text, 160, 222);
}

void drawFace(const String& expression) {
  M5.Display.fillScreen(TFT_BLACK);
  M5.Display.setTextColor(TFT_WHITE, TFT_BLACK);
  M5.Display.setTextDatum(middle_center);
  M5.Display.setTextSize(2);
  M5.Display.drawString("Stacky", 160, 35);

  int eyeY = 112;
  int mouthY = 170;
  if (expression == "thinking") {
    M5.Display.fillCircle(112, eyeY, 12, TFT_CYAN);
    M5.Display.fillCircle(208, eyeY, 12, TFT_CYAN);
    M5.Display.drawLine(132, mouthY, 188, mouthY, TFT_CYAN);
  } else if (expression == "happy") {
    M5.Display.fillCircle(112, eyeY, 16, TFT_GREEN);
    M5.Display.fillCircle(208, eyeY, 16, TFT_GREEN);
    M5.Display.drawArc(160, mouthY - 8, 44, 28, 20, 160, TFT_GREEN);
  } else if (expression == "listening") {
    M5.Display.drawCircle(112, eyeY, 18, TFT_YELLOW);
    M5.Display.drawCircle(208, eyeY, 18, TFT_YELLOW);
    M5.Display.fillCircle(112, eyeY, 7, TFT_YELLOW);
    M5.Display.fillCircle(208, eyeY, 7, TFT_YELLOW);
    M5.Display.drawLine(140, mouthY, 180, mouthY, TFT_YELLOW);
  } else {
    M5.Display.fillCircle(112, eyeY, 14, TFT_WHITE);
    M5.Display.fillCircle(208, eyeY, 14, TFT_WHITE);
    M5.Display.drawLine(140, mouthY, 180, mouthY, TFT_WHITE);
  }
  if (statusLine.length() > 0) {
    showStatusLine(statusLine);
  }
}

String extractJsonString(const String& raw, const String& key, const String& fallback) {
  String needle = "\"" + key + "\"";
  int start = raw.indexOf(needle);
  if (start < 0) return fallback;
  start += needle.length();
  start = raw.indexOf(":", start);
  if (start < 0) return fallback;
  start = raw.indexOf("\"", start);
  if (start < 0) return fallback;
  start += 1;
  int end = raw.indexOf("\"", start);
  if (end < 0) return fallback;
  return raw.substring(start, end);
}

int extractJsonInt(const String& raw, const String& key, int fallback) {
  String needle = "\"" + key + "\"";
  int start = raw.indexOf(needle);
  if (start < 0) return fallback;
  start += needle.length();
  start = raw.indexOf(":", start);
  if (start < 0) return fallback;
  start += 1;
  while (start < raw.length() && raw[start] == ' ') start++;
  int end = start;
  while (end < raw.length() && isDigit(raw[end])) end++;
  if (end <= start) return fallback;
  return raw.substring(start, end).toInt();
}

void releaseAudioOut() {
  if (audioOutBytes != nullptr) {
    free(audioOutBytes);
    audioOutBytes = nullptr;
  }
  audioOutLen = 0;
  audioOutCapacity = 0;
}

uint8_t* allocateAudioOutBuffer(size_t capacity) {
  uint8_t* buffer = static_cast<uint8_t*>(
    heap_caps_malloc(capacity, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT)
  );
  if (buffer == nullptr) {
    buffer = static_cast<uint8_t*>(heap_caps_malloc(capacity, MALLOC_CAP_8BIT));
  }
  return buffer;
}

void sendAudioPlaybackDone(const char* reason) {
  if (!client.connected()) return;
  client.print("{\"type\":\"audio.play_done\",\"payload\":{\"id\":");
  client.print(audioPlayCounter);
  client.print(",\"reason\":\"");
  client.print(reason);
  client.print("\"},\"ts\":");
  client.print(millis() / 1000.0, 3);
  client.print("}\n");
}

void markAudioPlaybackStarted(size_t pcmBytes) {
  uint32_t bytesPerSecond = audioOutSampleRate * sizeof(int16_t);
  if (bytesPerSecond == 0) bytesPerSecond = 48000;
  uint32_t durationMs = static_cast<uint32_t>((pcmBytes * 1000ULL) / bytesPerSecond);
  if (durationMs < 100) durationMs = 100;
  audioPlaybackDeadlineMs = millis() + durationMs + 280;
}

void pauseMicForSharedI2S(uint32_t settleMs = 150) {
  uint32_t deadline = millis() + 500;
  while (M5.Mic.isRecording() && millis() < deadline) {
    delay(2);
  }
  delay(20);
  // 0.3.19: always call Mic.end() (no-op if already stopped). isRunning() check was
  // racy — when it returned false but task was still in i2s_read, Speaker.begin()'s
  // i2s_driver_uninstall would nuke the port under the task, crashing in i2s_read.
  Serial.printf("Mic.end() pre-shared-i2s\n");
  M5.Mic.end();
  Serial.printf("Mic.end() returned, settling %ums\n", static_cast<unsigned>(settleMs));
  delay(settleMs);
}

bool ensureMicReady() {
  if (M5.Mic.isRunning()) return true;
  bool ok = M5.Mic.begin();
  Serial.printf("mic.begin=%s\n", ok ? "true" : "false");
  delay(12);
  return ok;
}

void pauseMicForAudio() {
  audioPlaying = true;
  pauseMicForSharedI2S();
}

void stopAudioPlayback() {
  bool wasPlaying = audioPlaying || M5.Speaker.isPlaying();
  audioOutputHold = false;
  audioOutSessionOpen = false;
  audioChunkQueued = 0;
  audioChunkDropped = 0;
  if (M5.Speaker.isEnabled()) {
    M5.Speaker.stop();
    M5.Speaker.end();
  }
  audioPlaying = false;
  audioPlaybackDeadlineMs = 0;
  releaseAudioOut();
  showStatusLine("Lytter", TFT_YELLOW);
  if (wasPlaying) {
    sendAudioPlaybackDone("stopped");
  }
  pendingAudioRawBytes = 0;
  pendingAudioRawRead = 0;
  pendingAudioRawSeq = -1;
  pendingAudioRawOk = true;
}

void finishAudioPlaybackIfDone() {
  if (!audioPlaying) return;
  if (audioOutSessionOpen) return;
  bool stillPlaying = M5.Speaker.isPlaying();
  bool beforeDeadline = audioPlaybackDeadlineMs != 0
    && static_cast<int32_t>(millis() - audioPlaybackDeadlineMs) < 0;
  if (stillPlaying && beforeDeadline) return;
  if (stillPlaying) {
    Serial.printf(
      "audio.finish deadline id=%u channels=%u playing=%s\n",
      static_cast<unsigned>(audioPlayCounter),
      static_cast<unsigned>(M5.Speaker.getPlayingChannels()),
      M5.Speaker.isPlaying() ? "true" : "false"
    );
    M5.Speaker.stop();
  } else {
    Serial.printf(
      "audio.finish natural id=%u channels=%u\n",
      static_cast<unsigned>(audioPlayCounter),
      static_cast<unsigned>(M5.Speaker.getPlayingChannels())
    );
  }
  M5.Speaker.end();
  audioPlaying = false;
  audioPlaybackDeadlineMs = 0;
  releaseAudioOut();
  showStatusLine(audioOutputHold ? "Modtager lyd" : "Lytter", audioOutputHold ? TFT_CYAN : TFT_YELLOW);
  sendAudioPlaybackDone("finished");
}

void prepareSpeakerForAudio(uint32_t sampleRate) {
  audioOutSampleRate = sampleRate > 0 ? sampleRate : 24000;
  pauseMicForAudio();
  // 0.3.19: extra settle before Speaker.begin's i2s_driver_uninstall to ensure
  // mic_task has fully exited its i2s_read syscall (up to 100ms timeout in M5Unified)
  delay(120);
  if (!M5.Speaker.isRunning()) {
    bool speakerReady = M5.Speaker.begin();
    Serial.printf("speaker.begin=%s rate=%u\n", speakerReady ? "true" : "false", audioOutSampleRate);
  }
  M5.Speaker.setVolume(SPEAKER_VOLUME);
  M5.Speaker.setAllChannelVolume(255);
  showStatusLine("Taler", TFT_GREEN);
}

void prepareAudioReceive(uint32_t sampleRate) {
  audioOutSampleRate = sampleRate > 0 ? sampleRate : 24000;
  pauseMicForAudio();
  showStatusLine("Modtager lyd", TFT_CYAN);
}

void handleAudioHold(const String& line) {
  audioOutputHold = line.indexOf("\"active\":true") >= 0;
  if (audioOutputHold) {
    pauseMicForSharedI2S(20);
    showStatusLine("Modtager lyd", TFT_CYAN);
  } else if (!audioPlaying && !audioOutSessionOpen) {
    showStatusLine("Lytter", TFT_YELLOW);
  }
}

bool waitMicRecordingState(bool expected, uint32_t timeoutMs) {
  uint32_t deadline = millis() + timeoutMs;
  while (M5.Mic.isRecording() != expected && millis() < deadline) {
    M5.update();
    delay(1);
  }
  return M5.Mic.isRecording() == expected;
}

bool reserveAudioOut(size_t capacity) {
  if (capacity == 0) return true;
  if (capacity + WAV_HEADER_BYTES > MAX_AUDIO_OUT_BYTES) return false;
  releaseAudioOut();
  audioOutBytes = allocateAudioOutBuffer(capacity + WAV_HEADER_BYTES);
  if (audioOutBytes == nullptr) {
    audioOutCapacity = 0;
    return false;
  }
  audioOutCapacity = capacity + WAV_HEADER_BYTES;
  audioOutLen = 0;
  return true;
}

bool appendAudioOut(const uint8_t* data, size_t len) {
  if (len == 0) return true;
  size_t needed = audioOutLen + len;
  if (needed + WAV_HEADER_BYTES > MAX_AUDIO_OUT_BYTES) return false;
  if (needed + WAV_HEADER_BYTES > audioOutCapacity) {
    size_t nextCapacity = audioOutCapacity > 0 ? audioOutCapacity * 2 : 4096;
    while (nextCapacity < needed + WAV_HEADER_BYTES) {
      nextCapacity *= 2;
    }
    if (nextCapacity > MAX_AUDIO_OUT_BYTES) {
      nextCapacity = MAX_AUDIO_OUT_BYTES;
    }
    uint8_t* next = allocateAudioOutBuffer(nextCapacity);
    if (next == nullptr) return false;
    if (audioOutBytes != nullptr && audioOutLen > 0) {
      memcpy(next, audioOutBytes, audioOutLen);
      free(audioOutBytes);
    }
    audioOutBytes = next;
    audioOutCapacity = nextCapacity;
  }
  memcpy(audioOutBytes + audioOutLen, data, len);
  audioOutLen = needed;
  return true;
}

void writeLe16(uint8_t* target, uint16_t value) {
  target[0] = static_cast<uint8_t>(value & 0xFF);
  target[1] = static_cast<uint8_t>((value >> 8) & 0xFF);
}

void writeLe32(uint8_t* target, uint32_t value) {
  target[0] = static_cast<uint8_t>(value & 0xFF);
  target[1] = static_cast<uint8_t>((value >> 8) & 0xFF);
  target[2] = static_cast<uint8_t>((value >> 16) & 0xFF);
  target[3] = static_cast<uint8_t>((value >> 24) & 0xFF);
}

bool wrapAudioOutAsWav() {
  audioOutLen -= audioOutLen % sizeof(int16_t);
  if (audioOutBytes == nullptr || audioOutLen < sizeof(int16_t)) return false;
  size_t pcmBytes = audioOutLen;
  size_t wavBytes = pcmBytes + WAV_HEADER_BYTES;
  if (wavBytes > MAX_AUDIO_OUT_BYTES) return false;
  if (wavBytes > audioOutCapacity) return false;

  memmove(audioOutBytes + WAV_HEADER_BYTES, audioOutBytes, pcmBytes);
  memcpy(audioOutBytes, "RIFF", 4);
  writeLe32(audioOutBytes + 4, static_cast<uint32_t>(wavBytes - 8));
  memcpy(audioOutBytes + 8, "WAVEfmt ", 8);
  writeLe32(audioOutBytes + 16, 16);
  writeLe16(audioOutBytes + 20, 1);
  writeLe16(audioOutBytes + 22, 1);
  writeLe32(audioOutBytes + 24, audioOutSampleRate);
  writeLe32(audioOutBytes + 28, audioOutSampleRate * 2);
  writeLe16(audioOutBytes + 32, 2);
  writeLe16(audioOutBytes + 34, 16);
  memcpy(audioOutBytes + 36, "data", 4);
  writeLe32(audioOutBytes + 40, static_cast<uint32_t>(pcmBytes));

  audioOutLen = wavBytes;
  return true;
}

bool playBufferedAudioOut() {
  size_t pcmBytes = audioOutLen - (audioOutLen % sizeof(int16_t));
  if (!wrapAudioOutAsWav()) {
    return false;
  }
  prepareSpeakerForAudio(audioOutSampleRate);
  bool ok = M5.Speaker.playWav(
    audioOutBytes,
    audioOutLen,
    1,
    0,
    true
  );
  if (ok) {
    audioPlayCounter++;
    markAudioPlaybackStarted(pcmBytes);
  }
  Serial.printf(
    "audio.play wav ok=%s pcm16Bytes=%u wavBytes=%u rate=%u playing=%s volume=%u\n",
    ok ? "true" : "false",
    static_cast<unsigned>(pcmBytes),
    static_cast<unsigned>(audioOutLen),
    static_cast<unsigned>(audioOutSampleRate),
    M5.Speaker.isPlaying() ? "true" : "false",
    static_cast<unsigned>(M5.Speaker.getVolume())
  );
  return ok;
}

void handleAudioOut(const String& line) {
  String encoded = extractJsonString(line, "data", "");
  if (encoded.length() == 0) return;
  int sampleRate = extractJsonInt(line, "sampleRate", 22050);
  if (sampleRate <= 0) sampleRate = 22050;

  size_t maxDecoded = (encoded.length() * 3) / 4 + 4;
  uint8_t* decoded = allocateAudioOutBuffer(maxDecoded + WAV_HEADER_BYTES);
  if (decoded == nullptr) {
    showStatusLine("Audio RAM fejl", TFT_RED);
    return;
  }

  size_t decodedLen = 0;
  int rc = mbedtls_base64_decode(
    decoded,
    maxDecoded,
    &decodedLen,
    reinterpret_cast<const unsigned char*>(encoded.c_str()),
    encoded.length()
  );
  if (rc != 0 || decodedLen < 2) {
    free(decoded);
    showStatusLine("Audio decode fejl", TFT_RED);
    return;
  }

  pauseMicForAudio();
  if (M5.Speaker.isRunning()) {
    M5.Speaker.stop();
  } else {
    bool speakerReady = M5.Speaker.begin();
    Serial.printf("speaker.begin=%s audio.out rate=%d\n", speakerReady ? "true" : "false", sampleRate);
  }
  M5.Speaker.setVolume(SPEAKER_VOLUME);
  M5.Speaker.setAllChannelVolume(255);
  releaseAudioOut();
  audioOutBytes = decoded;
  audioOutLen = decodedLen;
  audioOutCapacity = maxDecoded + WAV_HEADER_BYTES;
  size_t pcmBytes = audioOutLen - (audioOutLen % sizeof(int16_t));
  if (!wrapAudioOutAsWav()) {
    stopAudioPlayback();
    return;
  }
  audioPlaying = M5.Speaker.playWav(
    audioOutBytes,
    audioOutLen,
    1,
    0,
    true
  );
  if (audioPlaying) {
    audioPlayCounter++;
    markAudioPlaybackStarted(pcmBytes);
  }
  Serial.printf(
    "audio.play directWav ok=%s pcm16Bytes=%u wavBytes=%u rate=%d playing=%s volume=%u\n",
    audioPlaying ? "true" : "false",
    static_cast<unsigned>(pcmBytes),
    static_cast<unsigned>(audioOutLen),
    sampleRate,
    M5.Speaker.isPlaying() ? "true" : "false",
    static_cast<unsigned>(M5.Speaker.getVolume())
  );
  if (!audioPlaying) {
    stopAudioPlayback();
    return;
  }
  showStatusLine("Taler", TFT_GREEN);
}

void handleAudioTone(const String& line) {
  int frequency = extractJsonInt(line, "frequency", 880);
  int durationMs = extractJsonInt(line, "durationMs", 180);
  if (frequency < 80) frequency = 80;
  if (frequency > 4000) frequency = 4000;
  if (durationMs < 20) durationMs = 20;
  if (durationMs > 1200) durationMs = 1200;
  audioOutSessionOpen = false;
  releaseAudioOut();
  prepareSpeakerForAudio(24000);
  audioPlaying = M5.Speaker.tone(
    static_cast<float>(frequency),
    static_cast<uint32_t>(durationMs),
    0,
    true
  );
  if (audioPlaying) {
    audioPlayCounter++;
    audioPlaybackDeadlineMs = millis() + static_cast<uint32_t>(durationMs) + 120;
  }
  Serial.printf(
    "audio.tone ok=%s frequency=%d durationMs=%d playing=%s volume=%u\n",
    audioPlaying ? "true" : "false",
    frequency,
    durationMs,
    M5.Speaker.isPlaying() ? "true" : "false",
    static_cast<unsigned>(M5.Speaker.getVolume())
  );
  if (!audioPlaying) {
    stopAudioPlayback();
  }
}

void handleAudioStart(const String& line) {
  int sampleRate = extractJsonInt(line, "sampleRate", 24000);
  if (sampleRate <= 0) sampleRate = 24000;
  int totalBytes = extractJsonInt(line, "totalBytes", 0);
  audioOutSessionOpen = true;
  audioOutBufferIndex = 0;
  audioChunkQueued = 0;
  audioChunkDropped = 0;
  releaseAudioOut();
  prepareAudioReceive(static_cast<uint32_t>(sampleRate));
  Serial.printf("audio.start rate=%d totalBytes=%d\n", sampleRate, totalBytes);
  if (totalBytes > 0 && !reserveAudioOut(static_cast<size_t>(totalBytes))) {
    audioChunkDropped++;
    showStatusLine("Audio RAM fejl", TFT_RED);
  }
}

void sendAudioChunkAck(int seq, bool queued) {
  if (!client.connected()) return;
  client.print("{\"type\":\"audio.chunk_ack\",\"payload\":{\"seq\":");
  client.print(seq);
  client.print(",\"queued\":");
  client.print(queued ? "true" : "false");
  client.print(",\"queuedCount\":");
  client.print(audioChunkQueued);
  client.print(",\"droppedCount\":");
  client.print(audioChunkDropped);
  client.print("},\"ts\":");
  client.print(millis() / 1000.0, 3);
  client.print("}\n");
}

void handleAudioChunk(const String& line) {
  String encoded = extractJsonString(line, "data", "");
  if (encoded.length() == 0) return;
  int seq = extractJsonInt(line, "seq", -1);
  if (!audioOutSessionOpen) {
    audioOutSessionOpen = true;
    prepareAudioReceive(audioOutSampleRate);
  }

  size_t decodedLen = 0;
  int rc = mbedtls_base64_decode(
    reinterpret_cast<unsigned char*>(audioOutChunkBuffers[audioOutBufferIndex]),
    AUDIO_OUT_CHUNK_BYTES,
    &decodedLen,
    reinterpret_cast<const unsigned char*>(encoded.c_str()),
    encoded.length()
  );
  if (rc != 0 || decodedLen < 2) {
    showStatusLine("Audio chunk fejl", TFT_RED);
    sendAudioChunkAck(seq, false);
    return;
  }

  decodedLen -= decodedLen % sizeof(int16_t);
  if (!appendAudioOut(reinterpret_cast<const uint8_t*>(audioOutChunkBuffers[audioOutBufferIndex]), decodedLen)) {
    audioChunkDropped++;
    showStatusLine("Audio RAM fejl", TFT_RED);
    sendAudioChunkAck(seq, false);
    return;
  }
  audioChunkQueued++;
  sendAudioChunkAck(seq, true);
  audioPlaying = true;
}

void handleAudioRawStart(const String& line) {
  int seq = extractJsonInt(line, "seq", -1);
  int byteCount = extractJsonInt(line, "bytes", 0);
  if (byteCount <= 0 || byteCount > AUDIO_OUT_CHUNK_BYTES) {
    sendAudioChunkAck(seq, false);
    return;
  }
  if (!audioOutSessionOpen) {
    audioOutSessionOpen = true;
    prepareAudioReceive(audioOutSampleRate);
  }
  pendingAudioRawBytes = static_cast<size_t>(byteCount);
  pendingAudioRawRead = 0;
  pendingAudioRawSeq = seq;
  pendingAudioRawOk = true;
}

void drainPendingAudioRaw() {
  while (pendingAudioRawBytes > 0 && client.available() > 0) {
    size_t remaining = pendingAudioRawBytes - pendingAudioRawRead;
    size_t toRead = remaining;
    if (toRead > AUDIO_RAW_READ_BYTES) toRead = AUDIO_RAW_READ_BYTES;
    int availableBytes = client.available();
    if (toRead > static_cast<size_t>(availableBytes)) {
      toRead = static_cast<size_t>(availableBytes);
    }
    int readBytes = client.read(audioRawReadBuffer, toRead);
    if (readBytes <= 0) break;
    if (pendingAudioRawOk && !appendAudioOut(audioRawReadBuffer, static_cast<size_t>(readBytes))) {
      pendingAudioRawOk = false;
      audioChunkDropped++;
      showStatusLine("Audio RAM fejl", TFT_RED);
    }
    pendingAudioRawRead += static_cast<size_t>(readBytes);
    if (pendingAudioRawRead >= pendingAudioRawBytes) {
      bool queued = pendingAudioRawOk;
      if (queued) {
        audioChunkQueued++;
        audioPlaying = true;
      }
      sendAudioChunkAck(pendingAudioRawSeq, queued);
      pendingAudioRawBytes = 0;
      pendingAudioRawRead = 0;
      pendingAudioRawSeq = -1;
      pendingAudioRawOk = true;
      break;
    }
  }
}

void handleAudioEnd() {
  audioOutSessionOpen = false;
  Serial.printf(
    "audio.end queued=%u dropped=%u bytes=%u capacity=%u\n",
    audioChunkQueued,
    audioChunkDropped,
    static_cast<unsigned>(audioOutLen),
    static_cast<unsigned>(audioOutCapacity)
  );
  if (audioChunkDropped == 0 && playBufferedAudioOut()) {
    audioPlaying = true;
    return;
  }
  stopAudioPlayback();
}

void handleCommand(const String& line) {
  if (line.indexOf("body.set_expression") >= 0) {
    expressionName = extractJsonString(line, "name", "neutral");
    drawFace(expressionName);
    return;
  }
  if (line.indexOf("body.status") >= 0) {
    drawFace(expressionName);
    return;
  }
  if (line.indexOf("mobility.intent") >= 0) {
    // Wheels are deliberately disabled until the physical build is calibrated.
    return;
  }
  if (line.indexOf("audio.stop") >= 0) {
    stopAudioPlayback();
    return;
  }
  if (line.indexOf("audio.hold") >= 0) {
    handleAudioHold(line);
    return;
  }
  if (line.indexOf("audio.raw") >= 0) {
    handleAudioRawStart(line);
    return;
  }
  if (line.indexOf("audio.tone") >= 0) {
    handleAudioTone(line);
    return;
  }
  if (line.indexOf("audio.start") >= 0) {
    handleAudioStart(line);
    return;
  }
  if (line.indexOf("audio.chunk") >= 0) {
    handleAudioChunk(line);
    return;
  }
  if (line.indexOf("audio.end") >= 0) {
    handleAudioEnd();
    return;
  }
  if (line.indexOf("audio.out") >= 0) {
    handleAudioOut(line);
    return;
  }
}

void connectWifi() {
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.printf("Connecting to WiFi SSID: %s\n", WIFI_SSID);
  showStatusLine("WiFi forbinder...", TFT_YELLOW);
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    attempts++;
    if (attempts % 4 == 0) {
      Serial.printf("WiFi status=%d after %d attempts\n", WiFi.status(), attempts);
      showStatusLine("WiFi prover igen...", TFT_YELLOW);
    }
    if (attempts >= 60) {
      Serial.printf("WiFi timeout, restarting WiFi\n");
      showStatusLine("WiFi timeout", TFT_RED);
      WiFi.disconnect(true);
      delay(750);
      WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
      attempts = 0;
      showStatusLine("WiFi forbinder...", TFT_YELLOW);
    }
  }
  Serial.print("WiFi connected: ");
  Serial.println(WiFi.localIP());
  showStatusLine("WiFi OK " + WiFi.localIP().toString(), TFT_GREEN);
}

void connectStacky() {
  while (!client.connected()) {
    Serial.printf("Connecting to Stacky at %s:%d\n", STACKY_HOST, STACKY_PORT);
    showStatusLine("PC forbinder...", TFT_CYAN);
    client.stop();
    client.connect(STACKY_HOST, STACKY_PORT);
    delay(250);
  }
  Serial.println("Connected to Stacky body server");
  showStatusLine("PC forbundet", TFT_GREEN);
}

void sendStatus() {
  auto micCfg = M5.Mic.config();
  Serial.printf(
    "mic_state running=%d gain=%u sr=%u nf=%u recording=%d ap=%d ah=%d pending=%u session=%d spk=%d\n",
    M5.Mic.isRunning() ? 1 : 0,
    static_cast<unsigned>(micCfg.magnification),
    static_cast<unsigned>(micCfg.sample_rate),
    static_cast<unsigned>(micCfg.noise_filter_level),
    M5.Mic.isRecording() ? 1 : 0,
    audioPlaying ? 1 : 0,
    audioOutputHold ? 1 : 0,
    static_cast<unsigned>(pendingAudioRawBytes),
    audioOutSessionOpen ? 1 : 0,
    M5.Speaker.isPlaying() ? 1 : 0
  );
  if (!client.connected()) return;
  String payload = "{\"type\":\"status\",\"payload\":{\"device\":\"stackchan-cores3\",\"expression\":\"";
  payload += expressionName;
  payload += "\",\"firmware\":\"";
  payload += STACKY_FIRMWARE_VERSION;
  payload += "\",\"wheelsEnabled\":false},\"ts\":";
  payload += String(millis() / 1000.0, 3);
  payload += "}\n";
  client.print(payload);
}

void streamMicToStacky() {
  if (!client.connected() || audioPlaying || audioOutputHold) return;
  if (!ensureMicReady()) return;
  if (!M5.Mic.record(micBuffer, AUDIO_IN_SAMPLES, AUDIO_IN_SAMPLE_RATE)) return;
  bool started = waitMicRecordingState(true, 250);
  bool completed = started && waitMicRecordingState(false, 250);
  if (!completed) return;
  if (audioPlaying) return;

  client.print("{\"type\":\"audio.in\",\"payload\":{\"encoding\":\"pcm16le\",\"sampleRate\":");
  client.print(AUDIO_IN_SAMPLE_RATE);
  client.print(",\"channels\":1,\"transport\":\"raw\",\"seq\":");
  client.print(audioInSeq++);
  client.print(",\"bytes\":");
  client.print(AUDIO_IN_BYTES);
  client.print("},\"ts\":");
  client.print(millis() / 1000.0, 3);
  client.print("}\n");
  size_t written = client.write(reinterpret_cast<const uint8_t*>(micBuffer), AUDIO_IN_BYTES);
  if (written != AUDIO_IN_BYTES) {
    Serial.printf(
      "audio.in raw short write written=%u expected=%u\n",
      static_cast<unsigned>(written),
      static_cast<unsigned>(AUDIO_IN_BYTES)
    );
  }
}

void setup() {
  Serial.begin(115200);
  incoming.reserve(MAX_COMMAND_LINE);
  auto cfg = M5.config();
  cfg.internal_mic = true;
  cfg.internal_spk = true;
  M5.begin(cfg);
  M5.Speaker.setVolume(SPEAKER_VOLUME);
  M5.Speaker.end();
  auto micCfg = M5.Mic.config();
  micCfg.sample_rate = AUDIO_IN_SAMPLE_RATE;
  // 0.3.17: revert magnification + noise_filter_level overrides — 0.3.15 values broke mic capture (only noise floor)
  M5.Mic.config(micCfg);
  auto speakerCfg = M5.Speaker.config();
  Serial.printf("Stacky firmware %s\n", STACKY_FIRMWARE_VERSION);
  Serial.printf(
    "CoreS3 audio: mic i2s=%d data=%d bck=%d ws=%d gain=%u; speaker i2s=%d data=%d bck=%d ws=%d\n",
    static_cast<int>(micCfg.i2s_port),
    static_cast<int>(micCfg.pin_data_in),
    static_cast<int>(micCfg.pin_bck),
    static_cast<int>(micCfg.pin_ws),
    static_cast<unsigned>(micCfg.magnification),
    static_cast<int>(speakerCfg.i2s_port),
    static_cast<int>(speakerCfg.pin_data_out),
    static_cast<int>(speakerCfg.pin_bck),
    static_cast<int>(speakerCfg.pin_ws)
  );
  M5.Display.setRotation(1);
  drawFace("neutral");
  ensureMicReady();
  connectWifi();
  connectStacky();
  sendStatus();
}

void loop() {
  M5.update();
  finishAudioPlaybackIfDone();
  if (!client.connected()) {
    showStatusLine("PC mistet", TFT_ORANGE);
    connectStacky();
  }
  while (client.available()) {
    if (pendingAudioRawBytes > 0) {
      drainPendingAudioRaw();
      if (pendingAudioRawBytes > 0) break;
      continue;
    }
    char ch = static_cast<char>(client.read());
    if (ch == '\n') {
      handleCommand(incoming);
      incoming = "";
    } else {
      if (incoming.length() < MAX_COMMAND_LINE) {
        incoming += ch;
      } else {
        incoming = "";
        showStatusLine("Kommando for stor", TFT_RED);
      }
    }
  }
  if (M5.Touch.getCount() > 0) {
    client.print("{\"type\":\"touch\",\"payload\":{\"zone\":\"screen\"}}\n");
    drawFace("listening");
  }
  if (millis() - lastStatusAt > 5000) {
    sendStatus();
    lastStatusAt = millis();
  }
  streamMicToStacky();
  delay(1);
}
