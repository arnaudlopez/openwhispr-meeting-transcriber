#!/usr/bin/env node
import fs from "node:fs";
import fsp from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import {
  applySpeakerLearning,
  getDefaultSpeakerDbPath,
  parseSpeakerLabels,
} from "./speaker-profiles.mjs";
import { diarizeWithFluidAudio } from "./fluidaudio-diarization.mjs";
import {
  alignChunkedTranscriptToDiarization,
  alignTranscriptToDiarization,
} from "./transcript-alignment.mjs";

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const SAMPLE_RATE = 16000;
const F32_BYTES = 4;
const DEFAULT_CHUNK_SECONDS = 15;
const DEFAULT_ONLINE_CHUNK_SECONDS = 600;
const ONLINE_SEND_CHUNK_BYTES = 8000 * F32_BYTES;
const ONLINE_END_TAIL_SECONDS = 0.6;
const DEFAULT_OPENROUTER_MODEL = "openai/gpt-4o-transcribe";
const DEFAULT_OPENROUTER_URL = "https://openrouter.ai/api/v1/audio/transcriptions";
const DEFAULT_OPENROUTER_CONCURRENCY = 3;
const DEFAULT_TURN_GAP_SECONDS = 1.2;
const DEFAULT_TURN_STEP_SECONDS = 0.1;
const DEFAULT_TURN_PADDING_SECONDS = 0.15;
const MIN_TURN_SECONDS = 0.35;
const OPENROUTER_MAX_ATTEMPTS = 4;
const DEFAULT_OPENAI_MODEL = "gpt-4o-transcribe-diarize";
const OPENAI_TEXT_TRANSCRIPTION_MODELS = new Set([
  "gpt-transcribe",
  "gpt-4o-transcribe",
  "gpt-4o-mini-transcribe",
  "gpt-4o-mini-transcribe-2025-12-15",
  "whisper-1",
]);
const DEFAULT_ALIGNMENT_ASR_MODEL = "nemotron-3.5-asr-streaming-0.6b";
const DEFAULT_OPENAI_URL = "https://api.openai.com/v1/audio/transcriptions";
const DEFAULT_OPENAI_BITRATE_KBPS = 40;
const DEFAULT_OPENAI_MAX_UPLOAD_MB = 24;
const DEFAULT_OPENAI_SEGMENT_SECONDS = 15 * 60;
const OPENAI_MAX_ATTEMPTS = 4;
const OPENAI_REQUEST_TIMEOUT_MS = 30 * 60 * 1000;
const MIN_SPEAKER_REFERENCE_SECONDS = 2;
const MAX_SPEAKER_REFERENCE_SECONDS = 8;
const APP_BIN_DIR = "/Applications/OpenWhispr.app/Contents/Resources/bin";
const ELECTRON_NODE = "/Applications/OpenWhispr.app/Contents/MacOS/OpenWhispr";
const DEFAULT_ASR_MODEL = "parakeet-tdt-0.6b-v3";
const PARAKEET_MODELS_DIR = path.join(os.homedir(), ".cache/openwhispr/parakeet-models");
const DIAR_MODELS_DIR = path.join(APP_BIN_DIR, "diarization-models");

function loadLocalEnvironment() {
  if (typeof process.loadEnvFile !== "function") return;
  for (const filename of [".env.local", ".env"]) {
    const envPath = path.join(REPO_ROOT, filename);
    if (!fs.existsSync(envPath)) continue;
    process.loadEnvFile(envPath);
  }
}

loadLocalEnvironment();

function parseArgs(argv) {
  const args = {};
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (!arg.startsWith("--")) continue;
    const key = arg.slice(2);
    const next = argv[i + 1];
    if (!next || next.startsWith("--")) {
      args[key] = true;
    } else {
      args[key] = next;
      i += 1;
    }
  }
  return args;
}

function requirePath(filePath, label) {
  if (!fs.existsSync(filePath)) {
    throw new Error(`${label} not found: ${filePath}`);
  }
  return filePath;
}

function parseNumberOption(value, fallback) {
  const n = Number.parseFloat(value);
  return Number.isFinite(n) ? n : fallback;
}

function normalizeSpeakerId(speakerId) {
  const match = String(speakerId || "").match(/^speaker_0*(\d+)$/);
  if (!match) return String(speakerId || "");
  return `speaker_${Number(match[1])}`;
}

function paddedSpeakerId(speakerId) {
  const match = String(speakerId || "").match(/^speaker_0*(\d+)$/);
  if (!match) return String(speakerId || "");
  return `speaker_${String(Number(match[1])).padStart(2, "0")}`;
}

function shouldReexecWithElectron(args) {
  if (process.versions.electron || process.env.OPENWHISPR_NO_ELECTRON_REEXEC) return false;
  if (!fs.existsSync(ELECTRON_NODE)) return false;
  return !!(args.input && args.output);
}

function run(command, args, options = {}) {
  return new Promise((resolve, reject) => {
    const proc = spawn(command, args, {
      stdio: ["ignore", "pipe", "pipe"],
      ...options,
    });
    let stdout = "";
    let stderr = "";
    proc.stdout.on("data", (data) => {
      stdout += data.toString();
    });
    proc.stderr.on("data", (data) => {
      stderr += data.toString();
    });
    proc.on("error", reject);
    proc.on("close", (code) => {
      if (code !== 0) {
        reject(new Error(`${path.basename(command)} exited ${code}: ${stderr.slice(-1200)}`));
        return;
      }
      resolve({ stdout, stderr });
    });
  });
}

function runInherited(command, args, options = {}) {
  return new Promise((resolve, reject) => {
    const proc = spawn(command, args, {
      stdio: "inherit",
      ...options,
    });
    proc.on("error", reject);
    proc.on("close", resolve);
  });
}

async function convertAudio(inputPath, wavPath, rawPath) {
  await run("ffmpeg", [
    "-hide_banner",
    "-loglevel",
    "error",
    "-i",
    inputPath,
    "-ar",
    String(SAMPLE_RATE),
    "-ac",
    "1",
    "-c:a",
    "pcm_s16le",
    "-y",
    wavPath,
  ]);

  await run("ffmpeg", [
    "-hide_banner",
    "-loglevel",
    "error",
    "-i",
    inputPath,
    "-ar",
    String(SAMPLE_RATE),
    "-ac",
    "1",
    "-f",
    "f32le",
    "-y",
    rawPath,
  ]);
}

function parseDiarization(stdout) {
  const segments = [];
  const lineRegex = /^(\d+(?:\.\d+)?)\s+--\s+(\d+(?:\.\d+)?)\s+(speaker_\d+)$/;
  for (const line of stdout.split("\n")) {
    const match = line.trim().match(lineRegex);
    if (!match) continue;
    segments.push({
      start: Number(match[1]),
      end: Number(match[2]),
      speaker: match[3],
    });
  }
  return segments;
}

async function diarizeSherpa(wavPath, expectedSpeakers) {
  const binary = requirePath(
    path.join(APP_BIN_DIR, "sherpa-onnx-diarize-darwin-arm64"),
    "diarization binary"
  );
  const segmentation = requirePath(
    path.join(DIAR_MODELS_DIR, "sherpa-onnx-pyannote-segmentation-3-0/model.onnx"),
    "segmentation model"
  );
  const embedding = requirePath(
    path.join(DIAR_MODELS_DIR, "3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx"),
    "speaker embedding model"
  );

  const args = [
    `--segmentation.pyannote-model=${segmentation}`,
    `--embedding.model=${embedding}`,
    "--segmentation.provider=coreml",
    "--embedding.provider=coreml",
    "--segmentation.num-threads=4",
    "--embedding.num-threads=4",
    "--min-duration-on=0.2",
    "--min-duration-off=0.5",
  ];

  const parsedSpeakers =
    expectedSpeakers === "auto" ? 2 : Number.parseInt(expectedSpeakers, 10);
  if (Number.isFinite(parsedSpeakers) && parsedSpeakers > 0) {
    args.push(`--clustering.num-clusters=${parsedSpeakers}`);
  } else {
    args.push("--clustering.cluster-threshold=0.5");
  }
  args.push(wavPath);

  const { stdout } = await run(binary, args);
  return parseDiarization(stdout);
}

function getAsrConfig(modelName, provider = "local") {
  if (provider === "openrouter") {
    return {
      modelName,
      modelDir: null,
      runtime: "openrouter",
    };
  }
  if (provider === "openai") {
    return {
      modelName,
      modelDir: null,
      runtime: "openai",
    };
  }
  const runtime = String(modelName).startsWith("nemotron") ? "online" : "offline";
  return {
    modelName,
    modelDir: path.join(PARAKEET_MODELS_DIR, modelName),
    runtime,
  };
}

function startAsrServer(modelName) {
  const config = getAsrConfig(modelName);
  const binaryName =
    config.runtime === "online"
      ? "sherpa-onnx-online-ws-darwin-arm64"
      : "sherpa-onnx-ws-darwin-arm64";
  const binary = requirePath(path.join(APP_BIN_DIR, binaryName), `${config.runtime} ASR WS`);
  for (const file of ["tokens.txt", "encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx"]) {
    requirePath(path.join(config.modelDir, file), `${modelName} model file ${file}`);
  }

  const port = 6036 + Math.floor(Math.random() * 200);
  const args = [
    `--tokens=${path.join(config.modelDir, "tokens.txt")}`,
    `--encoder=${path.join(config.modelDir, "encoder.int8.onnx")}`,
    `--decoder=${path.join(config.modelDir, "decoder.int8.onnx")}`,
    `--joiner=${path.join(config.modelDir, "joiner.int8.onnx")}`,
    `--port=${port}`,
    "--num-threads=4",
  ];
  if (config.runtime === "online") {
    args.push(
      "--num-work-threads=2",
      "--loop-interval-ms=2",
      `--end-tail-padding=${ONLINE_END_TAIL_SECONDS}`,
      "--warm-up=0"
    );
  }
  const proc = spawn(binary, args, { stdio: ["ignore", "pipe", "pipe"] });

  const ready = new Promise((resolve, reject) => {
    const timeout = setTimeout(() => reject(new Error(`${modelName} server startup timed out`)), 60000);
    let stderr = "";
    proc.stderr.on("data", (data) => {
      stderr += data.toString();
      if (stderr.includes("Listening on:")) {
        clearTimeout(timeout);
        resolve();
      }
    });
    proc.on("error", (err) => {
      clearTimeout(timeout);
      reject(err);
    });
    proc.on("close", (code) => {
      clearTimeout(timeout);
      reject(new Error(`${modelName} server exited during startup: ${code}: ${stderr.slice(-800)}`));
    });
  });

  return { proc, port, ready, ...config };
}

function transcribeOfflineChunk(port, samplesBuffer) {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(`ws://127.0.0.1:${port}`);
    const timeout = setTimeout(() => {
      try {
        ws.close();
      } catch {}
      reject(new Error("Parakeet chunk transcription timed out"));
    }, 300000);

    let result = "";
    ws.binaryType = "arraybuffer";
    ws.addEventListener("open", () => {
      const message = Buffer.alloc(8 + samplesBuffer.length);
      message.writeInt32LE(SAMPLE_RATE, 0);
      message.writeInt32LE(samplesBuffer.length, 4);
      samplesBuffer.copy(message, 8);
      ws.send(message);
    });
    ws.addEventListener("message", (event) => {
      result += String(event.data || "");
      ws.send("Done");
    });
    ws.addEventListener("close", () => {
      clearTimeout(timeout);
      try {
        const parsed = JSON.parse(result);
        resolve((parsed.text || "").trim());
      } catch {
        resolve(result.trim());
      }
    });
    ws.addEventListener("error", () => {
      clearTimeout(timeout);
      reject(new Error("Parakeet websocket error"));
    });
  });
}

function transcribeOnlineChunk(port, samplesBuffer) {
  return new Promise((resolve, reject) => {
    const audioSeconds = samplesBuffer.length / F32_BYTES / SAMPLE_RATE;
    const timeout = setTimeout(
      () => {
        try {
          ws.close();
        } catch {}
        reject(new Error("Nemotron streaming transcription timed out"));
      },
      Math.max(300000, audioSeconds * 2000)
    );

    const ws = new WebSocket(`ws://127.0.0.1:${port}`);
    const results = [];
    let serverDone = false;
    ws.binaryType = "arraybuffer";
    ws.addEventListener("open", () => {
      for (let offset = 0; offset < samplesBuffer.length; offset += ONLINE_SEND_CHUNK_BYTES) {
        ws.send(samplesBuffer.subarray(offset, offset + ONLINE_SEND_CHUNK_BYTES));
      }
      ws.send("Done");
    });
    ws.addEventListener("message", (event) => {
      const message = String(event.data || "");
      if (message === "Done!") {
        serverDone = true;
        ws.close();
        return;
      }
      try {
        const parsed = JSON.parse(message);
        if (parsed?.is_final && String(parsed.text || "").trim()) results.push(parsed);
      } catch {}
    });
    ws.addEventListener("close", () => {
      clearTimeout(timeout);
      if (!serverDone) {
        reject(new Error("Nemotron stream closed before transcription completed"));
        return;
      }
      resolve(results);
    });
    ws.addEventListener("error", () => {
      clearTimeout(timeout);
      reject(new Error("Nemotron websocket error"));
    });
  });
}

function resolveSpeakerForInterval(diarization, start, end) {
  let best = null;
  let bestOverlap = 0;
  const midpoint = start + (end - start) / 2;
  for (const seg of diarization) {
    const overlap = Math.max(0, Math.min(end, seg.end) - Math.max(start, seg.start));
    if (overlap > bestOverlap) {
      best = seg.speaker;
      bestOverlap = overlap;
    }
  }
  if (best) return best;
  const nearest = diarization.find((seg) => midpoint >= seg.start && midpoint <= seg.end);
  return nearest?.speaker || "speaker_0";
}

function onlineResultInterval(result, streamStart) {
  const relativeStart = Number(result.start_time || 0);
  const timestamps = Array.isArray(result.timestamps)
    ? result.timestamps.filter((value) => Number.isFinite(Number(value))).map(Number)
    : [];
  const relativeEnd =
    relativeStart +
    (timestamps.length
      ? Math.max(...timestamps) + ONLINE_END_TAIL_SECONDS
      : Math.max(1, String(result.text || "").split(/\s+/).length * 0.35));
  return {
    start: streamStart + relativeStart,
    end: streamStart + relativeEnd,
  };
}

async function transcribeLocal(rawPath, diarization, chunkSeconds, asrModel) {
  const raw = await fsp.readFile(rawPath);
  const chunkBytes = chunkSeconds * SAMPLE_RATE * F32_BYTES;
  const server = startAsrServer(asrModel);
  await server.ready;
  const segments = [];
  try {
    for (let offset = 0; offset < raw.length; offset += chunkBytes) {
      const end = Math.min(offset + chunkBytes, raw.length);
      const chunk = raw.subarray(offset, end);
      const startSeconds = offset / F32_BYTES / SAMPLE_RATE;
      const endSeconds = end / F32_BYTES / SAMPLE_RATE;
      console.log(
        `[ASR ${asrModel}] ${formatTimestamp(startSeconds)}-${formatTimestamp(endSeconds)}`
      );
      if (server.runtime === "online") {
        const results = await transcribeOnlineChunk(server.port, chunk);
        for (const result of results) {
          const interval = onlineResultInterval(result, startSeconds);
          segments.push({
            id: `segment-${segments.length}`,
            timestamp: Number(interval.start.toFixed(2)),
            endTimestamp: Number(interval.end.toFixed(2)),
            text: String(result.text || "").trim(),
            source: "system",
            speaker: resolveSpeakerForInterval(diarization, interval.start, interval.end),
            asrModel,
          });
        }
      } else {
        const text = await transcribeOfflineChunk(server.port, chunk);
        if (!text) continue;
        const speaker = resolveSpeakerForInterval(diarization, startSeconds, endSeconds);
        segments.push({
          id: `segment-${segments.length}`,
          timestamp: Number(startSeconds.toFixed(2)),
          endTimestamp: Number(endSeconds.toFixed(2)),
          text,
          source: "system",
          speaker,
          asrModel,
        });
      }
    }
  } finally {
    server.proc.kill("SIGTERM");
  }
  return segments;
}

function parsePcmWav(buffer) {
  if (
    buffer.length < 44 ||
    buffer.toString("ascii", 0, 4) !== "RIFF" ||
    buffer.toString("ascii", 8, 12) !== "WAVE"
  ) {
    throw new Error("Converted WAV is invalid");
  }

  let offset = 12;
  let format = null;
  let pcm = null;
  while (offset + 8 <= buffer.length) {
    const chunkId = buffer.toString("ascii", offset, offset + 4);
    const chunkSize = buffer.readUInt32LE(offset + 4);
    const dataStart = offset + 8;
    const dataEnd = Math.min(dataStart + chunkSize, buffer.length);
    if (chunkId === "fmt " && chunkSize >= 16) {
      format = {
        audioFormat: buffer.readUInt16LE(dataStart),
        channels: buffer.readUInt16LE(dataStart + 2),
        sampleRate: buffer.readUInt32LE(dataStart + 4),
        bitsPerSample: buffer.readUInt16LE(dataStart + 14),
      };
    } else if (chunkId === "data") {
      pcm = buffer.subarray(dataStart, dataEnd);
    }
    offset = dataStart + chunkSize + (chunkSize % 2);
  }

  if (!format || !pcm) throw new Error("Converted WAV is missing fmt or data chunks");
  if (
    format.audioFormat !== 1 ||
    format.channels !== 1 ||
    format.sampleRate !== SAMPLE_RATE ||
    format.bitsPerSample !== 16
  ) {
    throw new Error(
      `Expected mono PCM16 ${SAMPLE_RATE} Hz WAV, got format=${format.audioFormat}, channels=${format.channels}, rate=${format.sampleRate}, bits=${format.bitsPerSample}`
    );
  }
  return {
    pcm,
    durationSeconds: pcm.length / 2 / SAMPLE_RATE,
  };
}

function buildWavBuffer(pcm) {
  const header = Buffer.alloc(44);
  header.write("RIFF", 0, "ascii");
  header.writeUInt32LE(36 + pcm.length, 4);
  header.write("WAVE", 8, "ascii");
  header.write("fmt ", 12, "ascii");
  header.writeUInt32LE(16, 16);
  header.writeUInt16LE(1, 20);
  header.writeUInt16LE(1, 22);
  header.writeUInt32LE(SAMPLE_RATE, 24);
  header.writeUInt32LE(SAMPLE_RATE * 2, 28);
  header.writeUInt16LE(2, 32);
  header.writeUInt16LE(16, 34);
  header.write("data", 36, "ascii");
  header.writeUInt32LE(pcm.length, 40);
  return Buffer.concat([header, pcm]);
}

function sliceWav(pcm, startSeconds, endSeconds) {
  const startSample = Math.max(0, Math.floor(startSeconds * SAMPLE_RATE));
  const endSample = Math.min(pcm.length / 2, Math.ceil(endSeconds * SAMPLE_RATE));
  return buildWavBuffer(pcm.subarray(startSample * 2, endSample * 2));
}

function dominantSpeakerAt(diarization, time) {
  let best = null;
  let bestDuration = -1;
  for (const segment of diarization) {
    if (segment.start > time || segment.end <= time) continue;
    const duration = segment.end - segment.start;
    if (duration > bestDuration) {
      best = segment.speaker;
      bestDuration = duration;
    }
  }
  return best;
}

function buildSpeakerTurns(
  diarization,
  durationSeconds,
  {
    gapSeconds = DEFAULT_TURN_GAP_SECONDS,
    stepSeconds = DEFAULT_TURN_STEP_SECONDS,
    paddingSeconds = DEFAULT_TURN_PADDING_SECONDS,
  } = {}
) {
  const valid = diarization
    .filter(
      (segment) =>
        segment &&
        Number.isFinite(Number(segment.start)) &&
        Number.isFinite(Number(segment.end)) &&
        Number(segment.end) > Number(segment.start) &&
        segment.speaker
    )
    .map((segment) => ({
      start: Math.max(0, Number(segment.start)),
      end: Math.min(durationSeconds, Number(segment.end)),
      speaker: String(segment.speaker),
    }))
    .filter((segment) => segment.end > segment.start);

  if (!valid.length) {
    const fallback = [];
    for (let start = 0; start < durationSeconds; start += 60) {
      fallback.push({
        start,
        end: Math.min(start + 60, durationSeconds),
        speaker: "speaker_0",
      });
    }
    return fallback.map((turn) => ({
      ...turn,
      audioStart: turn.start,
      audioEnd: turn.end,
    }));
  }

  const rawTurns = [];
  let currentSpeaker = null;
  let currentStart = 0;
  const bins = Math.ceil(durationSeconds / stepSeconds);
  for (let index = 0; index <= bins; index += 1) {
    const binStart = index * stepSeconds;
    const speaker =
      index < bins
        ? dominantSpeakerAt(valid, Math.min(durationSeconds, binStart + stepSeconds / 2))
        : null;
    if (speaker === currentSpeaker) continue;
    if (currentSpeaker) {
      rawTurns.push({
        start: currentStart,
        end: Math.min(binStart, durationSeconds),
        speaker: currentSpeaker,
      });
    }
    currentSpeaker = speaker;
    currentStart = binStart;
  }

  const bridged = [];
  for (const turn of rawTurns) {
    const previous = bridged.at(-1);
    if (
      previous &&
      previous.speaker === turn.speaker &&
      turn.start - previous.end <= gapSeconds
    ) {
      previous.end = turn.end;
    } else {
      bridged.push({ ...turn });
    }
  }

  const turns = bridged.filter((turn) => turn.end - turn.start >= MIN_TURN_SECONDS);
  return turns.map((turn, index) => {
    const previous = turns[index - 1];
    const next = turns[index + 1];
    const availableBefore = previous ? Math.max(0, turn.start - previous.end) : turn.start;
    const availableAfter = next
      ? Math.max(0, next.start - turn.end)
      : Math.max(0, durationSeconds - turn.end);
    return {
      ...turn,
      audioStart: Math.max(0, turn.start - Math.min(paddingSeconds, availableBefore / 2)),
      audioEnd: Math.min(
        durationSeconds,
        turn.end + Math.min(paddingSeconds, availableAfter / 2)
      ),
    };
  });
}

function wait(milliseconds) {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}

async function requestOpenRouterTranscription({
  apiKey,
  url,
  model,
  language,
  wavBuffer,
}) {
  let lastError = null;
  for (let attempt = 1; attempt <= OPENROUTER_MAX_ATTEMPTS; attempt += 1) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 300000);
    try {
      const response = await fetch(url, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${apiKey}`,
          "Content-Type": "application/json",
          "X-OpenRouter-Title": "OpenWhispr Transcribe",
        },
        body: JSON.stringify({
          model,
          input_audio: {
            data: wavBuffer.toString("base64"),
            format: "wav",
          },
          language,
          temperature: 0,
        }),
        signal: controller.signal,
      });
      const responseText = await response.text();
      let data = null;
      try {
        data = JSON.parse(responseText);
      } catch {}

      if (!response.ok) {
        const message =
          data?.error?.message ||
          data?.error ||
          responseText.slice(0, 500) ||
          `HTTP ${response.status}`;
        const error = new Error(`OpenRouter ${response.status}: ${message}`);
        error.retryable = response.status === 429 || response.status >= 500;
        throw error;
      }
      const text = String(data?.text || "").trim();
      if (!text) throw new Error("OpenRouter returned no transcript text");
      return {
        text,
        usage: data?.usage || null,
        generationId: response.headers.get("x-generation-id") || null,
      };
    } catch (error) {
      lastError = error;
      const retryable =
        error?.name === "AbortError" ||
        error?.retryable === true ||
        error?.cause?.code === "ECONNRESET";
      if (!retryable || attempt === OPENROUTER_MAX_ATTEMPTS) throw error;
      await wait(750 * 2 ** (attempt - 1));
    } finally {
      clearTimeout(timeout);
    }
  }
  throw lastError || new Error("OpenRouter transcription failed");
}

function turnCacheKey(turn) {
  return [
    turn.speaker,
    Number(turn.start).toFixed(2),
    Number(turn.end).toFixed(2),
    Number(turn.audioStart).toFixed(2),
    Number(turn.audioEnd).toFixed(2),
  ].join(":");
}

async function loadOpenRouterCache(cachePath, inputPath, model) {
  if (!cachePath || !fs.existsSync(cachePath)) {
    return { version: 1, input: inputPath, model, entries: {} };
  }
  try {
    const cache = JSON.parse(await fsp.readFile(cachePath, "utf8"));
    if (
      cache?.version === 1 &&
      cache?.input === inputPath &&
      cache?.model === model &&
      cache?.entries &&
      typeof cache.entries === "object"
    ) {
      return cache;
    }
  } catch {}
  return { version: 1, input: inputPath, model, entries: {} };
}

async function writeOpenRouterCache(cachePath, cache) {
  if (!cachePath) return;
  await fsp.mkdir(path.dirname(cachePath), { recursive: true });
  const temporaryPath = `${cachePath}.tmp`;
  await fsp.writeFile(temporaryPath, JSON.stringify(cache, null, 2), "utf8");
  await fsp.rename(temporaryPath, cachePath);
}

async function transcribeOpenRouter({
  wavPath,
  inputPath,
  diarization,
  model,
  apiKey,
  url,
  language,
  concurrency,
  turnGapSeconds,
  cachePath,
}) {
  if (!apiKey) {
    throw new Error("OPENROUTER_API_KEY is not configured");
  }
  const wav = parsePcmWav(await fsp.readFile(wavPath));
  const turns = buildSpeakerTurns(diarization, wav.durationSeconds, {
    gapSeconds: turnGapSeconds,
  });
  const cache = await loadOpenRouterCache(cachePath, inputPath, model);
  const results = new Array(turns.length);
  const workerCount = Math.max(1, Math.min(8, Number.parseInt(concurrency, 10) || 1));
  let nextIndex = 0;
  let cacheWrite = Promise.resolve();
  let fatalError = null;

  const persistCache = () => {
    cache.updatedAt = new Date().toISOString();
    cacheWrite = cacheWrite.then(() => writeOpenRouterCache(cachePath, cache));
    return cacheWrite;
  };

  const worker = async () => {
    while (true) {
      if (fatalError) return;
      const index = nextIndex;
      nextIndex += 1;
      if (index >= turns.length) return;
      const turn = turns[index];
      const key = turnCacheKey(turn);
      const cached = cache.entries[key];
      if (cached?.text) {
        results[index] = cached;
        console.log(
          `[ASR ${model}] ${index + 1}/${turns.length} ${formatTimestamp(turn.start)}-${formatTimestamp(turn.end)} ${turn.speaker} (cache)`
        );
        continue;
      }

      console.log(
        `[ASR ${model}] ${index + 1}/${turns.length} ${formatTimestamp(turn.start)}-${formatTimestamp(turn.end)} ${turn.speaker}`
      );
      let response;
      try {
        response = await requestOpenRouterTranscription({
          apiKey,
          url,
          model,
          language,
          wavBuffer: sliceWav(wav.pcm, turn.audioStart, turn.audioEnd),
        });
      } catch (error) {
        fatalError ||= error;
        return;
      }
      const entry = {
        ...turn,
        text: response.text,
        usage: response.usage,
        generationId: response.generationId,
      };
      results[index] = entry;
      cache.entries[key] = entry;
      await persistCache();
    }
  };

  await Promise.all(Array.from({ length: Math.min(workerCount, turns.length) }, () => worker()));
  await cacheWrite;
  if (fatalError) throw fatalError;

  const segments = results
    .filter((result) => result?.text)
    .map((result, index) => ({
      id: `segment-${index}`,
      timestamp: Number(result.start.toFixed(2)),
      endTimestamp: Number(result.end.toFixed(2)),
      text: result.text,
      source: "openrouter",
      speaker: result.speaker,
      asrModel: model,
      asrUsage: result.usage,
      generationId: result.generationId,
    }));
  const usage = results.reduce(
    (total, result) => {
      const value = result?.usage || {};
      for (const key of ["cost", "input_tokens", "output_tokens", "total_tokens", "seconds"]) {
        const number = Number(value[key]);
        if (Number.isFinite(number)) total[key] += number;
      }
      return total;
    },
    { cost: 0, input_tokens: 0, output_tokens: 0, total_tokens: 0, seconds: 0 }
  );

  return {
    segments,
    metadata: {
      provider: "openrouter",
      model,
      language,
      endpoint: url,
      turns: turns.length,
      concurrency: workerCount,
      cachePath,
      usage,
    },
  };
}

function openAIAudioMimeType(filePath) {
  const extension = path.extname(filePath).toLowerCase();
  const mimeTypes = {
    ".flac": "audio/flac",
    ".m4a": "audio/mp4",
    ".mp3": "audio/mpeg",
    ".mp4": "audio/mp4",
    ".mpeg": "audio/mpeg",
    ".mpga": "audio/mpeg",
    ".ogg": "audio/ogg",
    ".wav": "audio/wav",
    ".webm": "audio/webm",
  };
  return mimeTypes[extension] || "application/octet-stream";
}

function isOpenAISupportedAudio(filePath) {
  return openAIAudioMimeType(filePath) !== "application/octet-stream";
}

async function prepareOpenAIUpload({
  inputPath,
  workDir,
  bitrateKbps,
  maxUploadMb,
}) {
  const maxBytes = Math.floor(maxUploadMb * 1_000_000);
  const sourceStats = await fsp.stat(inputPath);
  if (sourceStats.size <= maxBytes && isOpenAISupportedAudio(inputPath)) {
    return {
      path: inputPath,
      bytes: sourceStats.size,
      converted: false,
      bitrateKbps: null,
      maxBytes,
    };
  }

  const requestedBitrate = Math.max(16, Math.floor(bitrateKbps));
  const bitrates = [...new Set([requestedBitrate, 32, 24, 16].filter((value) => value <= requestedBitrate))];
  const uploadPath = path.join(workDir, "openai-upload.webm");
  let uploadStats = null;
  let finalBitrate = null;
  for (const bitrate of bitrates) {
    await run("ffmpeg", [
      "-hide_banner",
      "-loglevel",
      "error",
      "-i",
      inputPath,
      "-vn",
      "-map_metadata",
      "-1",
      "-ac",
      "1",
      "-ar",
      String(SAMPLE_RATE),
      "-c:a",
      "libopus",
      "-b:a",
      `${bitrate}k`,
      "-vbr",
      "on",
      "-compression_level",
      "10",
      "-application",
      "voip",
      "-y",
      uploadPath,
    ]);
    uploadStats = await fsp.stat(uploadPath);
    finalBitrate = bitrate;
    if (uploadStats.size <= maxBytes) break;
  }

  if (!uploadStats || uploadStats.size > maxBytes) {
    throw new Error(
      `OpenAI upload remains above ${maxUploadMb} MB after speech compression; split the recording before transcription`
    );
  }

  return {
    path: uploadPath,
    bytes: uploadStats.size,
    converted: true,
    bitrateKbps: finalBitrate,
    maxBytes,
  };
}

function mergeSpeechIntervals(diarization, durationSeconds) {
  const intervals = (diarization || [])
    .map((segment) => ({
      start: Math.max(0, Number(segment.start)),
      end: Math.min(durationSeconds, Number(segment.end)),
    }))
    .filter(
      (segment) =>
        Number.isFinite(segment.start) &&
        Number.isFinite(segment.end) &&
        segment.end > segment.start
    )
    .sort((left, right) => left.start - right.start || left.end - right.end);
  const merged = [];
  for (const interval of intervals) {
    const previous = merged.at(-1);
    if (!previous || interval.start > previous.end) {
      merged.push({ ...interval });
    } else {
      previous.end = Math.max(previous.end, interval.end);
    }
  }
  return merged;
}

function chooseOpenAITextBoundaries({
  diarization,
  durationSeconds,
  segmentSeconds,
}) {
  const speech = mergeSpeechIntervals(diarization, durationSeconds);
  const gaps = [];
  for (let index = 1; index < speech.length; index += 1) {
    const start = speech[index - 1].end;
    const end = speech[index].start;
    if (end - start >= 0.25) {
      gaps.push({ start, end, midpoint: start + (end - start) / 2 });
    }
  }

  const boundaries = [0];
  let previous = 0;
  while (durationSeconds - previous > segmentSeconds) {
    const target = previous + segmentSeconds;
    const minimum = previous + segmentSeconds * 0.72;
    const maximum = Math.min(durationSeconds, previous + segmentSeconds * 1.1);
    const candidate = gaps
      .filter((gap) => gap.midpoint >= minimum && gap.midpoint <= maximum)
      .sort(
        (left, right) =>
          Math.abs(left.midpoint - target) - Math.abs(right.midpoint - target) ||
          right.end - right.start - (left.end - left.start)
      )[0];
    const boundary = candidate?.midpoint || target;
    if (boundary <= previous + 1) break;
    boundaries.push(boundary);
    previous = boundary;
  }
  boundaries.push(durationSeconds);
  return boundaries;
}

async function prepareOpenAITextUploads({
  inputPath,
  workDir,
  bitrateKbps,
  maxUploadMb,
  durationSeconds,
  diarization,
  segmentSeconds,
}) {
  if (durationSeconds <= segmentSeconds) {
    const upload = await prepareOpenAIUpload({
      inputPath,
      workDir,
      bitrateKbps,
      maxUploadMb,
    });
    return [{ ...upload, start: 0, end: durationSeconds }];
  }

  const maxBytes = Math.floor(maxUploadMb * 1_000_000);
  const requestedBitrate = Math.max(16, Math.floor(bitrateKbps));
  const bitrates = [
    ...new Set([requestedBitrate, 32, 24, 16].filter((value) => value <= requestedBitrate)),
  ];
  const boundaries = chooseOpenAITextBoundaries({
    diarization,
    durationSeconds,
    segmentSeconds,
  });
  const uploads = [];
  for (let index = 0; index < boundaries.length - 1; index += 1) {
    const start = boundaries[index];
    const end = boundaries[index + 1];
    const uploadPath = path.join(
      workDir,
      `openai-upload-${String(index + 1).padStart(3, "0")}.webm`
    );
    let uploadStats = null;
    let finalBitrate = null;
    for (const bitrate of bitrates) {
      await run("ffmpeg", [
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        inputPath,
        "-ss",
        start.toFixed(3),
        "-t",
        (end - start).toFixed(3),
        "-vn",
        "-map_metadata",
        "-1",
        "-ac",
        "1",
        "-ar",
        String(SAMPLE_RATE),
        "-c:a",
        "libopus",
        "-b:a",
        `${bitrate}k`,
        "-vbr",
        "on",
        "-compression_level",
        "10",
        "-application",
        "voip",
        "-y",
        uploadPath,
      ]);
      uploadStats = await fsp.stat(uploadPath);
      finalBitrate = bitrate;
      if (uploadStats.size <= maxBytes) break;
    }
    if (!uploadStats || uploadStats.size > maxBytes) {
      throw new Error(
        `OpenAI part ${index + 1} remains above ${maxUploadMb} MB; reduce --openai-segment-seconds`
      );
    }
    uploads.push({
      path: uploadPath,
      bytes: uploadStats.size,
      converted: true,
      bitrateKbps: finalBitrate,
      maxBytes,
      start,
      end,
    });
  }
  return uploads;
}

function overlapSeconds(leftStart, leftEnd, rightStart, rightEnd) {
  return Math.max(0, Math.min(leftEnd, rightEnd) - Math.max(leftStart, rightStart));
}

function chooseSpeakerReference(diarization, speakerId, durationSeconds) {
  const speakerTurns = buildSpeakerTurns(diarization, durationSeconds, {
    gapSeconds: 0.5,
    paddingSeconds: 0,
  });
  const candidates = speakerTurns
    .filter((segment) => segment.speaker === speakerId)
    .map((segment) => {
      const segmentStart = Math.max(0, Number(segment.start));
      const segmentEnd = Math.min(durationSeconds, Number(segment.end));
      if (!Number.isFinite(segmentStart) || !Number.isFinite(segmentEnd)) return null;
      const availableDuration = segmentEnd - segmentStart;
      if (availableDuration < MIN_SPEAKER_REFERENCE_SECONDS) return null;

      const clipDuration = Math.min(MAX_SPEAKER_REFERENCE_SECONDS, availableDuration);
      const midpoint = segmentStart + availableDuration / 2;
      const start = Math.max(segmentStart, midpoint - clipDuration / 2);
      const end = Math.min(segmentEnd, start + clipDuration);
      const competingOverlap = diarization
        .filter((other) => other.speaker !== speakerId)
        .reduce(
          (total, other) =>
            total + overlapSeconds(start, end, Number(other.start), Number(other.end)),
          0
        );
      return {
        start,
        end,
        duration: end - start,
        competingOverlap,
        score: end - start - competingOverlap * 5,
      };
    })
    .filter(Boolean)
    .sort((left, right) => right.score - left.score || right.duration - left.duration);
  return candidates[0] || null;
}

function buildKnownSpeakerReferences({ wav, diarization, speakerLabels }) {
  const references = [];
  const seenSpeakers = new Set();
  const seenNames = new Set();
  for (const segment of diarization) {
    const speakerId = String(segment.speaker || "");
    if (!speakerId || seenSpeakers.has(speakerId)) continue;
    seenSpeakers.add(speakerId);
    const name =
      speakerLabels.get(speakerId) ||
      speakerLabels.get(normalizeSpeakerId(speakerId)) ||
      speakerLabels.get(paddedSpeakerId(speakerId));
    if (!name || seenNames.has(name)) continue;
    const clip = chooseSpeakerReference(diarization, speakerId, wav.durationSeconds);
    if (!clip) continue;
    const wavBuffer = sliceWav(wav.pcm, clip.start, clip.end);
    references.push({
      speakerId,
      name,
      start: Number(clip.start.toFixed(2)),
      end: Number(clip.end.toFixed(2)),
      competingOverlap: Number(clip.competingOverlap.toFixed(2)),
      dataUrl: `data:audio/wav;base64,${wavBuffer.toString("base64")}`,
    });
    seenNames.add(name);
    if (references.length >= 4) break;
  }
  return references;
}

function parseKnownSpeakerReferencePaths(value) {
  if (!value || value === true) return [];
  const text = String(value).trim();
  if (!text) return [];
  const parsed = text.startsWith("{")
    ? Object.entries(JSON.parse(text))
    : text.split(",").map((entry) => {
        const [name, ...pathParts] = entry.split("=");
        return [name, pathParts.join("=")];
      });
  return parsed
    .map(([name, filePath]) => ({
      name: String(name || "").trim(),
      filePath: String(filePath || "").trim(),
    }))
    .filter((reference) => reference.name && reference.filePath)
    .map((reference) => ({
      name: reference.name,
      path: path.resolve(reference.filePath),
    }));
}

function speakerIdForName(speakerLabels, name) {
  for (const [speakerId, displayName] of speakerLabels) {
    if (displayName !== name || !/^speaker_0*\d+$/.test(speakerId)) continue;
    return normalizeSpeakerId(speakerId);
  }
  return null;
}

async function loadExplicitSpeakerReferences({ value, speakerLabels }) {
  const references = [];
  for (const reference of parseKnownSpeakerReferencePaths(value)) {
    requirePath(reference.path, `known speaker reference for ${reference.name}`);
    if (!isOpenAISupportedAudio(reference.path)) {
      throw new Error(`Unsupported known speaker reference format: ${reference.path}`);
    }
    const buffer = await fsp.readFile(reference.path);
    references.push({
      speakerId: speakerIdForName(speakerLabels, reference.name),
      name: reference.name,
      path: reference.path,
      start: null,
      end: null,
      competingOverlap: null,
      dataUrl: `data:${openAIAudioMimeType(reference.path)};base64,${buffer.toString("base64")}`,
    });
    if (references.length >= 4) break;
  }
  return references;
}

function alignOpenAISpeakersToLocal(segments, diarization, speakerLabels) {
  const openAISpeakers = [...new Set(segments.map((segment) => segment.openaiSpeaker).filter(Boolean))];
  const localSpeakers = [
    ...new Set(
      diarization
        .map((segment) => normalizeSpeakerId(segment.speaker))
        .filter(Boolean)
    ),
  ];
  const speakerMap = new Map();
  const usedLocalSpeakers = new Set();
  for (const openaiSpeaker of openAISpeakers) {
    const localSpeaker = speakerIdForName(speakerLabels, openaiSpeaker);
    if (!localSpeaker) continue;
    speakerMap.set(openaiSpeaker, localSpeaker);
    usedLocalSpeakers.add(localSpeaker);
  }

  const candidates = [];
  for (const openaiSpeaker of openAISpeakers) {
    if (speakerMap.has(openaiSpeaker)) continue;
    const apiSegments = segments.filter((segment) => segment.openaiSpeaker === openaiSpeaker);
    for (const localSpeaker of localSpeakers) {
      if (usedLocalSpeakers.has(localSpeaker)) continue;
      const localSegments = diarization.filter(
        (segment) => normalizeSpeakerId(segment.speaker) === localSpeaker
      );
      let overlap = 0;
      for (const apiSegment of apiSegments) {
        for (const localSegment of localSegments) {
          overlap += overlapSeconds(
            apiSegment.timestamp,
            apiSegment.endTimestamp,
            Number(localSegment.start),
            Number(localSegment.end)
          );
        }
      }
      candidates.push({ openaiSpeaker, localSpeaker, overlap });
    }
  }

  candidates.sort((left, right) => right.overlap - left.overlap);
  for (const candidate of candidates) {
    if (candidate.overlap <= 0) continue;
    if (speakerMap.has(candidate.openaiSpeaker) || usedLocalSpeakers.has(candidate.localSpeaker)) {
      continue;
    }
    speakerMap.set(candidate.openaiSpeaker, candidate.localSpeaker);
    usedLocalSpeakers.add(candidate.localSpeaker);
  }

  return {
    segments: segments.map((segment) => ({
      ...segment,
      speaker:
        speakerMap.get(segment.openaiSpeaker) ||
        resolveSpeakerForInterval(diarization, segment.timestamp, segment.endTimestamp) ||
        segment.openaiSpeaker,
    })),
    speakerMap: Object.fromEntries(speakerMap),
  };
}

function estimateOpenAITranscriptionCost(model, usage) {
  const rates = String(model).includes("mini")
    ? { input: 1.25, output: 5 }
    : { input: 2.5, output: 10 };
  const inputTokens = Number(usage?.input_tokens);
  const outputTokens = Number(usage?.output_tokens);
  if (!Number.isFinite(inputTokens) || !Number.isFinite(outputTokens)) return null;
  return (inputTokens * rates.input + outputTokens * rates.output) / 1_000_000;
}

async function requestOpenAITranscription({
  apiKey,
  url,
  model,
  language,
  upload,
  references,
}) {
  let lastError = null;
  for (let attempt = 1; attempt <= OPENAI_MAX_ATTEMPTS; attempt += 1) {
    const form = new FormData();
    const uploadBuffer = await fsp.readFile(upload.path);
    form.append(
      "file",
      new Blob([uploadBuffer], { type: openAIAudioMimeType(upload.path) }),
      path.basename(upload.path)
    );
    form.append("model", model);
    form.append("response_format", "diarized_json");
    form.append("chunking_strategy", "auto");
    if (language) form.append("language", language);
    for (const reference of references) {
      form.append("known_speaker_names[]", reference.name);
      form.append("known_speaker_references[]", reference.dataUrl);
    }

    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), OPENAI_REQUEST_TIMEOUT_MS);
    try {
      const response = await fetch(url, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${apiKey}`,
        },
        body: form,
        signal: controller.signal,
      });
      const responseText = await response.text();
      let data = null;
      try {
        data = JSON.parse(responseText);
      } catch {}

      if (!response.ok) {
        const message =
          data?.error?.message ||
          data?.error ||
          responseText.slice(0, 700) ||
          `HTTP ${response.status}`;
        const error = new Error(`OpenAI ${response.status}: ${message}`);
        error.retryable = response.status === 429 || response.status >= 500;
        throw error;
      }
      if (!data || !Array.isArray(data.segments)) {
        throw new Error("OpenAI returned no diarized transcript segments");
      }
      return data;
    } catch (error) {
      lastError = error;
      const retryable =
        error?.name === "AbortError" ||
        error?.retryable === true ||
        error?.cause?.code === "ECONNRESET";
      if (!retryable || attempt === OPENAI_MAX_ATTEMPTS) throw error;
      await wait(1000 * 2 ** (attempt - 1));
    } finally {
      clearTimeout(timeout);
    }
  }
  throw lastError || new Error("OpenAI transcription failed");
}

async function requestOpenAITextTranscription({
  apiKey,
  url,
  model,
  language,
  upload,
}) {
  let lastError = null;
  for (let attempt = 1; attempt <= OPENAI_MAX_ATTEMPTS; attempt += 1) {
    const form = new FormData();
    const uploadBuffer = await fsp.readFile(upload.path);
    form.append(
      "file",
      new Blob([uploadBuffer], { type: openAIAudioMimeType(upload.path) }),
      path.basename(upload.path)
    );
    form.append("model", model);
    form.append("response_format", "json");
    if (language) {
      if (model === "gpt-transcribe") form.append("languages[]", language);
      else form.append("language", language);
    }

    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), OPENAI_REQUEST_TIMEOUT_MS);
    try {
      const response = await fetch(url, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${apiKey}`,
        },
        body: form,
        signal: controller.signal,
      });
      const responseText = await response.text();
      let data = null;
      try {
        data = JSON.parse(responseText);
      } catch {}
      if (!response.ok) {
        const message =
          data?.error?.message ||
          data?.error ||
          responseText.slice(0, 700) ||
          `HTTP ${response.status}`;
        const error = new Error(`OpenAI ${response.status}: ${message}`);
        error.retryable = response.status === 429 || response.status >= 500;
        throw error;
      }
      if (!data || !String(data.text || "").trim()) {
        throw new Error("OpenAI returned no transcript text");
      }
      return data;
    } catch (error) {
      lastError = error;
      const retryable =
        error?.name === "AbortError" ||
        error?.retryable === true ||
        error?.cause?.code === "ECONNRESET";
      if (!retryable || attempt === OPENAI_MAX_ATTEMPTS) throw error;
      await wait(1000 * 2 ** (attempt - 1));
    } finally {
      clearTimeout(timeout);
    }
  }
  throw lastError || new Error("OpenAI transcription failed");
}

function aggregateOpenAIUsage(results) {
  const usage = {};
  for (const result of results) {
    for (const [key, value] of Object.entries(result?.usage || {})) {
      if (Number.isFinite(Number(value)) && value !== "") {
        usage[key] = (Number(usage[key]) || 0) + Number(value);
      } else if (usage[key] === undefined && typeof value === "string") {
        usage[key] = value;
      }
    }
  }
  return Object.keys(usage).length ? usage : null;
}

async function writeJsonAtomically(filePath, value) {
  await fsp.mkdir(path.dirname(filePath), { recursive: true });
  const temporaryPath = `${filePath}.tmp-${process.pid}`;
  await fsp.writeFile(temporaryPath, JSON.stringify(value, null, 2), "utf8");
  await fsp.rename(temporaryPath, filePath);
}

async function requestOpenAITextChunks({
  apiKey,
  url,
  model,
  language,
  uploads,
  inputPath,
  cachePath,
}) {
  const sourceStats = await fsp.stat(inputPath);
  const identity = {
    inputPath: path.resolve(inputPath),
    sourceBytes: sourceStats.size,
    sourceMtimeMs: Math.round(sourceStats.mtimeMs),
    model,
    language,
  };
  let cache = { ...identity, chunks: [] };
  if (cachePath && fs.existsSync(cachePath)) {
    try {
      const candidate = JSON.parse(await fsp.readFile(cachePath, "utf8"));
      const matches = Object.entries(identity).every(
        ([key, value]) => candidate?.[key] === value
      );
      if (matches && Array.isArray(candidate.chunks)) cache = candidate;
    } catch {}
  }

  const results = [];
  for (let index = 0; index < uploads.length; index += 1) {
    const upload = uploads[index];
    const cached = cache.chunks.find(
      (chunk) =>
        Math.abs(Number(chunk.start) - upload.start) < 0.01 &&
        Math.abs(Number(chunk.end) - upload.end) < 0.01 &&
        String(chunk.text || "").trim()
    );
    if (cached) {
      console.log(
        `[ASR ${model}] cached cloud part ${index + 1}/${uploads.length} ${formatTimestamp(upload.start)}-${formatTimestamp(upload.end)}`
      );
      results.push(cached);
      continue;
    }

    console.log(
      `[ASR ${model}] cloud part ${index + 1}/${uploads.length} ${formatTimestamp(upload.start)}-${formatTimestamp(upload.end)} (${(upload.bytes / 1_000_000).toFixed(2)} MB)`
    );
    const data = await requestOpenAITextTranscription({
      apiKey,
      url,
      model,
      language,
      upload,
    });
    const result = {
      start: Number(upload.start.toFixed(3)),
      end: Number(upload.end.toFixed(3)),
      bytes: upload.bytes,
      bitrateKbps: upload.bitrateKbps,
      text: String(data.text || "").trim(),
      usage: data.usage || null,
      languages: data.languages || [],
    };
    cache.chunks = cache.chunks.filter(
      (chunk) =>
        !(
          Math.abs(Number(chunk.start) - upload.start) < 0.01 &&
          Math.abs(Number(chunk.end) - upload.end) < 0.01
        )
    );
    cache.chunks.push(result);
    cache.chunks.sort((left, right) => left.start - right.start);
    if (cachePath) await writeJsonAtomically(cachePath, cache);
    results.push(result);
  }

  return {
    text: results.map((result) => result.text).filter(Boolean).join("\n\n"),
    usage: aggregateOpenAIUsage(results),
    languages: [...new Set(results.flatMap((result) => result.languages || []))],
    cachePath: cachePath ? path.resolve(cachePath) : null,
    textChunks: results,
    chunks: results.map(({ text, ...metadata }) => metadata),
  };
}

async function loadCachedOpenAIText(filePath) {
  const resolved = requirePath(path.resolve(filePath), "cached OpenAI transcript JSON");
  const cached = JSON.parse(await fsp.readFile(resolved, "utf8"));
  const cachedChunks = Array.isArray(cached.chunks)
    ? cached.chunks
    : Array.isArray(cached.asrMetadata?.transcriptChunks)
      ? cached.asrMetadata.transcriptChunks
      : [];
  const text =
    String(cached.text || "").trim() ||
    (Array.isArray(cached.transcriptSegments)
      ? cached.transcriptSegments
          .map((segment) => String(segment.text || "").trim())
          .filter(Boolean)
          .join(" ")
      : "") ||
    cachedChunks
      .map((chunk) => String(chunk.text || "").trim())
      .filter(Boolean)
      .join("\n\n");
  if (!text) throw new Error(`No transcript text found in ${resolved}`);
  const chunkUsage = cachedChunks.length
    ? aggregateOpenAIUsage(cachedChunks)
    : null;
  const chunkUpload = cachedChunks.length
    ? {
        bytes: cachedChunks.reduce(
          (total, chunk) => total + (Number(chunk.bytes) || 0),
          0
        ),
        converted: true,
        bitrateKbps: cachedChunks[0]?.bitrateKbps || null,
        parts: cachedChunks.map(({ text: _text, ...metadata }) => metadata),
      }
    : null;
  return {
    text,
    usage: cached.usage || cached.asrMetadata?.usage || chunkUsage || null,
    languages: cached.languages || cached.asrMetadata?.languages || [],
    cachePath: resolved,
    textChunks: cachedChunks,
    roughTranscriptSegments:
      cached.roughTranscriptSegments ||
      cached.asrMetadata?.roughTranscriptSegments ||
      null,
    upload: cached.upload || cached.asrMetadata?.upload || chunkUpload || null,
  };
}

async function transcribeOpenAI({
  wavPath,
  rawPath,
  inputPath,
  workDir,
  diarization,
  speakerLabels,
  model,
  apiKey,
  url,
  language,
  bitrateKbps,
  maxUploadMb,
  knownSpeakerReferencePaths,
  useAutoSpeakerReferences,
  alignmentAsrModel = DEFAULT_ALIGNMENT_ASR_MODEL,
  cachedTranscriptPath,
  segmentSeconds = DEFAULT_OPENAI_SEGMENT_SECONDS,
  chunkCachePath,
}) {
  if (!apiKey && !cachedTranscriptPath) {
    throw new Error("OPENAI_API_KEY is not configured");
  }
  const wav = parsePcmWav(await fsp.readFile(wavPath));
  if (OPENAI_TEXT_TRANSCRIPTION_MODELS.has(model)) {
    const uploads = cachedTranscriptPath
      ? []
      : await prepareOpenAITextUploads({
          inputPath,
          workDir,
          bitrateKbps,
          maxUploadMb,
          durationSeconds: wav.durationSeconds,
          diarization,
          segmentSeconds,
        });
    const totalUploadBytes = uploads.reduce(
      (total, upload) => total + upload.bytes,
      0
    );
    console.log(
      `[ASR ${model}] ${
        cachedTranscriptPath
          ? "cached text"
          : `${uploads.length} cloud part(s), ${(totalUploadBytes / 1_000_000).toFixed(2)} MB total`
      }; local alignment with ${alignmentAsrModel}`
    );
    const cachedData = cachedTranscriptPath
      ? await loadCachedOpenAIText(cachedTranscriptPath)
      : null;
    if (
      cachedData &&
      (!cachedData.upload?.parts?.length || !cachedData.usage) &&
      chunkCachePath &&
      fs.existsSync(chunkCachePath)
    ) {
      const cloudCache = await loadCachedOpenAIText(chunkCachePath);
      if (!cachedData.upload?.parts?.length) cachedData.upload = cloudCache.upload;
      if (!cachedData.usage) cachedData.usage = cloudCache.usage;
    }
    const [data, roughSegments] = await Promise.all([
      cachedData ||
        requestOpenAITextChunks({
            apiKey,
            url,
            model,
            language,
            uploads,
            inputPath,
            cachePath: chunkCachePath,
          }),
      cachedData?.roughTranscriptSegments?.length
        ? Promise.resolve(cachedData.roughTranscriptSegments)
        : transcribeLocal(
            rawPath,
            diarization,
            DEFAULT_ONLINE_CHUNK_SECONDS,
            alignmentAsrModel
          ),
    ]);
    const aligned =
      data.textChunks?.length > 1
        ? alignChunkedTranscriptToDiarization({
            chunks: data.textChunks,
            roughSegments,
            diarization,
            asrModel: model,
            alignmentModel: alignmentAsrModel,
          })
        : alignTranscriptToDiarization({
            text: data.text,
            roughSegments,
            diarization,
            asrModel: model,
            alignmentModel: alignmentAsrModel,
          });
    if (aligned.metadata.matchedRatio < 0.25) {
      throw new Error(
        `Transcript alignment confidence is too low (${aligned.metadata.matchedRatio})`
      );
    }
    const estimatedCost = estimateOpenAITranscriptionCost(model, data.usage);
    const usage =
      data.usage && Number.isFinite(estimatedCost)
        ? { ...data.usage, cost: estimatedCost }
        : data.usage || null;
    return {
      segments: aligned.segments,
      metadata: {
        provider: "openai",
        model,
        language,
        endpoint: url,
        duration: wav.durationSeconds,
        usage,
        languages: data.languages || [],
        upload: {
          bytes:
            totalUploadBytes ||
            data.upload?.bytes ||
            (await fsp.stat(inputPath)).size,
          converted:
            uploads.some((upload) => upload.converted) ||
            data.upload?.converted ||
            false,
          bitrateKbps:
            uploads[0]?.bitrateKbps || data.upload?.bitrateKbps || null,
          parts:
            data.chunks ||
            data.upload?.parts ||
            uploads.map(({ path: _path, ...upload }) => upload),
        },
        alignment: aligned.metadata,
        transcriptChunks: (data.textChunks || []).map((chunk) => ({
          start: Number(chunk.start),
          end: Number(chunk.end),
          text: String(chunk.text || "").trim(),
        })),
        transcriptCachePath: data.cachePath || null,
        roughTranscriptSegments: roughSegments,
      },
    };
  }

  const upload = await prepareOpenAIUpload({
    inputPath,
    workDir,
    bitrateKbps,
    maxUploadMb,
  });
  const explicitReferences = await loadExplicitSpeakerReferences({
    value: knownSpeakerReferencePaths,
    speakerLabels,
  });
  const references = explicitReferences.length
    ? explicitReferences
    : useAutoSpeakerReferences
      ? buildKnownSpeakerReferences({ wav, diarization, speakerLabels })
      : [];
  console.log(
    `[ASR ${model}] continuous upload ${(upload.bytes / 1_000_000).toFixed(2)} MB, ${references.length} known speaker reference(s)`
  );
  const data = await requestOpenAITranscription({
    apiKey,
    url,
    model,
    language,
    upload,
    references,
  });
  const rawSegments = data.segments
    .map((segment, index) => {
      const start = Number(segment.start);
      const end = Number(segment.end);
      const text = String(segment.text || "").trim();
      const openaiSpeaker = String(segment.speaker || "");
      if (!Number.isFinite(start) || !Number.isFinite(end) || !text) return null;
      return {
        id: segment.id || `segment-${index}`,
        timestamp: Number(start.toFixed(2)),
        endTimestamp: Number(end.toFixed(2)),
        text,
        source: "openai",
        speaker: openaiSpeaker,
        openaiSpeaker,
        asrModel: model,
      };
    })
    .filter(Boolean);
  const aligned = alignOpenAISpeakersToLocal(rawSegments, diarization, speakerLabels);
  const estimatedCost = estimateOpenAITranscriptionCost(model, data.usage);
  const usage =
    data.usage && Number.isFinite(estimatedCost)
      ? { ...data.usage, cost: estimatedCost }
      : data.usage || null;
  return {
    segments: aligned.segments,
    metadata: {
      provider: "openai",
      model,
      language,
      endpoint: url,
      duration: Number(data.duration) || wav.durationSeconds,
      usage,
      upload: {
        bytes: upload.bytes,
        converted: upload.converted,
        bitrateKbps: upload.bitrateKbps,
      },
      knownSpeakerReferences: references.map((reference) => ({
        speakerId: reference.speakerId,
        name: reference.name,
        start: reference.start,
        end: reference.end,
        competingOverlap: reference.competingOverlap,
        path: reference.path || null,
      })),
      speakerMap: aligned.speakerMap,
    },
  };
}

function speakerLabel(speakerId, speakerLearning = null) {
  const resolved =
    speakerLearning?.matches?.[speakerId] ||
    speakerLearning?.matches?.[normalizeSpeakerId(speakerId)] ||
    speakerLearning?.matches?.[paddedSpeakerId(speakerId)];
  if (resolved?.state === "matched" || resolved?.state === "learned") {
    return resolved.displayName;
  }
  const match = String(speakerId || "").match(/speaker_0*(\d+)/);
  if (!match) return String(speakerId || "Speaker 1");
  return `Speaker ${Number(match[1]) + 1}`;
}

function formatTimestamp(seconds) {
  const total = Math.max(0, Math.floor(seconds || 0));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

function formatMarkdown({
  title,
  inputPath,
  transcriptSegments,
  diarizationSegments,
  degraded,
  speakerLearning,
  asrProvider,
  asrModel,
  asrMetadata,
  diarizationProvider,
  diarizationMetadata,
}) {
  const lines = [
    `# ${title}`,
    "",
    `**Source:** ${inputPath}`,
    `**Generated:** ${new Date().toISOString()}`,
    `**ASR provider:** ${asrProvider}`,
    `**ASR model:** ${asrModel}`,
    `**Diarization provider:** ${diarizationProvider}`,
    `**Diarization pipeline:** ${diarizationMetadata?.pipeline || diarizationMetadata?.provider || "unknown"}`,
    `**Diarization segments:** ${diarizationSegments.length}`,
    `**Transcript segments:** ${transcriptSegments.length}`,
  ];
  if (Number.isFinite(asrMetadata?.usage?.cost)) {
    lines.push(`**ASR API cost:** $${asrMetadata.usage.cost.toFixed(6)}`);
  }
  if (degraded) lines.push(`**Status:** ${degraded}`);
  if (speakerLearning) {
    const named = Object.values(speakerLearning.matches || {}).filter(
      (match) => match.state === "matched" || match.state === "learned"
    );
    lines.push(`**Speaker profiles:** ${named.length} named`);
  }
  lines.push("", "---", "");

  for (const seg of transcriptSegments) {
    lines.push(`**${speakerLabel(seg.speaker, speakerLearning)}** \`${formatTimestamp(seg.timestamp)}\``);
    lines.push(seg.text);
    lines.push("");
  }

  return `${lines.join("\n").trim()}\n`;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (shouldReexecWithElectron(args)) {
    const code = await runInherited(ELECTRON_NODE, [process.argv[1], ...process.argv.slice(2)], {
      env: {
        ...process.env,
        ELECTRON_RUN_AS_NODE: "1",
        OPENWHISPR_NO_ELECTRON_REEXEC: "1",
      },
    });
    process.exit(code ?? 0);
  }
  const input = args.input;
  const output = args.output;
  if (!input || !output) {
    throw new Error("Usage: node scripts/local-diarized-markdown.mjs --input <audio> --output <md> [--expected-speakers auto|N]");
  }
  requirePath(input, "input audio");

  const expectedSpeakers = args["expected-speakers"] || "auto";
  const diarizationProvider = args["diarization-provider"] || "sherpa";
  if (!["sherpa", "fluidaudio"].includes(diarizationProvider)) {
    throw new Error(`Unsupported --diarization-provider: ${diarizationProvider}`);
  }
  const asrProvider =
    args["asr-provider"] ||
    (String(args["asr-model"] || "").includes("/")
      ? "openrouter"
      : String(args["asr-model"] || "").startsWith("gpt-")
        ? "openai"
        : "local");
  if (!["local", "openrouter", "openai"].includes(asrProvider)) {
    throw new Error(`Unsupported --asr-provider: ${asrProvider}`);
  }
  const asrModel =
    args["asr-model"] ||
    (asrProvider === "openrouter"
      ? DEFAULT_OPENROUTER_MODEL
      : asrProvider === "openai"
        ? DEFAULT_OPENAI_MODEL
        : DEFAULT_ASR_MODEL);
  const asrRuntime = getAsrConfig(asrModel, asrProvider).runtime;
  const chunkSeconds = Number.parseInt(
    args["chunk-seconds"] ||
      (asrRuntime === "online" ? DEFAULT_ONLINE_CHUNK_SECONDS : DEFAULT_CHUNK_SECONDS),
    10
  );
  const learnSpeakers = !!args["learn-speakers"];
  const speakerDb = args["speaker-db"] || getDefaultSpeakerDbPath();
  const speakerLabels = parseSpeakerLabels(args["speaker-labels"]);
  const noDbWrite = !!args["no-db-write"];
  const matchThreshold = parseNumberOption(args["match-threshold"], 0.6);
  const suggestThreshold = parseNumberOption(args["suggest-threshold"], 0.5);
  const workDir = await fsp.mkdtemp(path.join(os.tmpdir(), "openwhispr-md-"));
  const wavPath = path.join(workDir, "input.wav");
  const rawPath = path.join(workDir, "input.f32le");

  try {
    await fsp.mkdir(path.dirname(output), { recursive: true });
    await convertAudio(input, wavPath, rawPath);

    let diarizationSegments = [];
    let diarizationMetadata = null;
    let degraded = "";
    const diarizationJson = args["diarization-json"];
    if (diarizationJson) {
      const reused = JSON.parse(await fsp.readFile(requirePath(diarizationJson, "diarization JSON"), "utf8"));
      if (!Array.isArray(reused.diarizationSegments)) {
        throw new Error(`No diarizationSegments array in ${diarizationJson}`);
      }
      diarizationSegments = reused.diarizationSegments;
      diarizationMetadata = {
        ...(reused.diarizationMetadata || {}),
        reusedFrom: path.resolve(diarizationJson),
      };
      console.log(`Reused ${diarizationSegments.length} diarization segments from ${diarizationJson}`);
    } else {
      try {
        if (diarizationProvider === "fluidaudio") {
          const fluidAudioResult = await diarizeWithFluidAudio({
            wavPath,
            workDir,
            expectedSpeakers,
            knownSpeakerReferencePaths: args["known-speaker-references"],
            speakerLabels,
            cliPath: args["fluidaudio-cli"],
            minimumSegmentSeconds: parseNumberOption(
              args["diarization-min-segment-seconds"],
              0.3
            ),
          });
          diarizationSegments = fluidAudioResult.segments;
          diarizationMetadata = fluidAudioResult.metadata;
        } else {
          diarizationSegments = await diarizeSherpa(wavPath, expectedSpeakers);
          diarizationMetadata = {
            provider: "sherpa",
            pipeline: "pyannote-segmentation-3.0-campplus-clustering",
          };
        }
      } catch (err) {
        degraded = `Diarization failed; neutral single-speaker fallback used (${err.message}).`;
      }
    }

    let transcriptSegments = [];
    let asrMetadata = null;
    try {
      if (asrRuntime === "openai") {
        const openAIResult = await transcribeOpenAI({
          wavPath,
          rawPath,
          inputPath: input,
          workDir,
          diarization: diarizationSegments,
          speakerLabels,
          model: asrModel,
          apiKey: process.env.OPENAI_API_KEY,
          url: args["openai-url"] || DEFAULT_OPENAI_URL,
          language: args.language || "fr",
          bitrateKbps: parseNumberOption(
            args["openai-bitrate-kbps"],
            DEFAULT_OPENAI_BITRATE_KBPS
          ),
          maxUploadMb: parseNumberOption(
            args["openai-max-upload-mb"],
            DEFAULT_OPENAI_MAX_UPLOAD_MB
          ),
          knownSpeakerReferencePaths: args["known-speaker-references"],
          useAutoSpeakerReferences: !!args["openai-auto-speaker-references"],
          alignmentAsrModel:
            args["alignment-asr-model"] || DEFAULT_ALIGNMENT_ASR_MODEL,
          cachedTranscriptPath: args["openai-transcript-json"],
          segmentSeconds: Math.max(
            60,
            parseNumberOption(
              args["openai-segment-seconds"],
              DEFAULT_OPENAI_SEGMENT_SECONDS
            )
          ),
          chunkCachePath:
            args["openai-chunk-cache"] ||
            output.replace(/\.md$/i, "-openai-chunks.json"),
        });
        transcriptSegments = openAIResult.segments;
        asrMetadata = openAIResult.metadata;
      } else if (asrRuntime === "openrouter") {
        const openRouterResult = await transcribeOpenRouter({
          wavPath,
          inputPath: input,
          diarization: diarizationSegments,
          model: asrModel,
          apiKey: process.env.OPENROUTER_API_KEY,
          url: args["openrouter-url"] || DEFAULT_OPENROUTER_URL,
          language: args.language || "fr",
          concurrency: args.concurrency || DEFAULT_OPENROUTER_CONCURRENCY,
          turnGapSeconds: parseNumberOption(
            args["turn-gap-seconds"],
            DEFAULT_TURN_GAP_SECONDS
          ),
          cachePath:
            args["asr-cache"] || output.replace(/\.md$/i, "-openrouter-cache.json"),
        });
        transcriptSegments = openRouterResult.segments;
        asrMetadata = openRouterResult.metadata;
      } else {
        transcriptSegments = await transcribeLocal(
          rawPath,
          diarizationSegments,
          chunkSeconds,
          asrModel
        );
      }
    } catch (err) {
      if (asrRuntime === "openrouter" || asrRuntime === "openai") throw err;
      degraded = degraded
        ? `${degraded} ASR failed; diarization-only fallback used (${err.message}).`
        : `ASR failed; diarization-only fallback used (${err.message}).`;
    }

    if (!transcriptSegments.length && diarizationSegments.length) {
      transcriptSegments = diarizationSegments.slice(0, 20).map((seg, index) => ({
        id: `diarization-only-${index}`,
        timestamp: seg.start,
        text: "[speech detected]",
        source: "system",
        speaker: seg.speaker,
      }));
    }

    if (!transcriptSegments.length) {
      transcriptSegments = [
        {
          id: "fallback-0",
          timestamp: 0,
          text: "[audio processed, but no speech text was produced]",
          source: "system",
          speaker: "speaker_0",
        },
        {
          id: "fallback-1",
          timestamp: 15,
          text: "[explicit degraded fallback]",
          source: "system",
          speaker: "speaker_0",
        },
      ];
      degraded ||= "No ASR or diarization segments were produced; explicit fallback written.";
    }

    let speakerLearning = null;
    try {
      if (diarizationSegments.length && speakerDb) {
        speakerLearning = await applySpeakerLearning({
          wavPath,
          diarizationSegments,
          dbPath: speakerDb,
          speakerLabels,
          learnSpeakers,
          noDbWrite,
          interactive: learnSpeakers && !speakerLabels.size,
          matchThreshold,
          suggestThreshold,
        });
        for (const segment of transcriptSegments) {
          const match =
            speakerLearning.matches?.[segment.speaker] ||
            speakerLearning.matches?.[normalizeSpeakerId(segment.speaker)] ||
            speakerLearning.matches?.[paddedSpeakerId(segment.speaker)];
          if (match?.state === "matched" || match?.state === "learned") {
            segment.speakerName = match.displayName;
            segment.profileId = match.profileId ?? null;
            segment.speakerConfidence = match.confidence ?? null;
          } else if (match?.state === "suggested") {
            segment.suggestedName = match.displayName;
            segment.suggestedProfileId = match.profileId ?? null;
            segment.speakerConfidence = match.confidence ?? null;
          }
        }
      }
    } catch (err) {
      degraded = degraded
        ? `${degraded} Speaker learning skipped (${err.message}).`
        : `Speaker learning skipped (${err.message}).`;
    }

    const title = path.basename(input).replace(/\.[^.]+$/, "");
    const markdown = formatMarkdown({
      title,
      inputPath: input,
      transcriptSegments,
      diarizationSegments,
      degraded,
      speakerLearning,
      asrProvider,
      asrModel,
      asrMetadata,
      diarizationProvider,
      diarizationMetadata,
    });
    await fsp.writeFile(output, markdown, "utf8");
    const jsonOutput = output.replace(/\.md$/i, ".json");
    await fsp.writeFile(
      jsonOutput,
      JSON.stringify({ input, generated_at: new Date().toISOString(), asrProvider, asrModel, asrMetadata, diarizationProvider, diarizationMetadata, degraded, speakerLearning, diarizationSegments, transcriptSegments }, null, 2),
      "utf8"
    );
    console.log(JSON.stringify({ success: true, output, jsonOutput, transcriptSegments: transcriptSegments.length, diarizationSegments: diarizationSegments.length, speakerLearning: speakerLearning ? { dbPath: speakerLearning.dbPath, speakers: Object.keys(speakerLearning.speakers || {}).length, learned: Object.keys(speakerLearning.learned || {}).length } : null, degraded }, null, 2));
  } finally {
    await fsp.rm(workDir, { recursive: true, force: true }).catch(() => {});
  }
}

main().catch((err) => {
  console.error(err.stack || err.message);
  process.exit(1);
});
