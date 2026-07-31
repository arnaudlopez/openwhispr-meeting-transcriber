import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { createRequire } from "node:module";
import readline from "node:readline/promises";

const APP_NODE_MODULES = "/Applications/OpenWhispr.app/Contents/Resources/app.asar.unpacked/node_modules";
const APP_ASAR = "/Applications/OpenWhispr.app/Contents/Resources/app.asar";
const APP_BIN_DIR = "/Applications/OpenWhispr.app/Contents/Resources/bin";
const MODEL_FILE = "3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx";
const SAMPLE_RATE = 16000;
const EMBEDDING_DIM = 512;
const MIN_SEGMENT_SECONDS = 1.5;
const MAX_EMBEDDING_SECONDS = 8;
const FBANK_FRAME_LENGTH_MS = 25;
const FBANK_FRAME_SHIFT_MS = 10;
const FBANK_NUM_MELS = 80;
const FBANK_FRAME_LENGTH = Math.round((SAMPLE_RATE * FBANK_FRAME_LENGTH_MS) / 1000);
const FBANK_FRAME_SHIFT = Math.round((SAMPLE_RATE * FBANK_FRAME_SHIFT_MS) / 1000);
const FBANK_FFT_SIZE = 512;
const DEFAULT_DB_PATH = path.join(
  os.homedir(),
  "Library/Application Support/open-whispr/transcriptions.db"
);

const appRequire = createRequire(
  process.versions.electron
    ? path.join(APP_ASAR, "package.json")
    : path.join(APP_NODE_MODULES, "package.json")
);
let ort = null;
let speakerSession = null;
let speakerInputName = null;
let sqliteCtor = null;
let melFilterbank = null;

export function getDefaultSpeakerDbPath() {
  return DEFAULT_DB_PATH;
}

export function getDefaultSpeakerModelPath() {
  return path.join(APP_BIN_DIR, "diarization-models", MODEL_FILE);
}

function loadOrt() {
  if (!ort) ort = appRequire("onnxruntime-node");
  return ort;
}

function loadSqlite() {
  if (!sqliteCtor) sqliteCtor = appRequire("better-sqlite3");
  return sqliteCtor;
}

function ensureFile(filePath, label) {
  if (!fs.existsSync(filePath)) throw new Error(`${label} not found: ${filePath}`);
  return filePath;
}

function normalizeSpeakerId(speakerId) {
  const match = String(speakerId || "").match(/^speaker_0*(\d+)$/);
  if (!match) return String(speakerId || "");
  return `speaker_${Number(match[1])}`;
}

export function parseSpeakerLabels(value) {
  const labels = new Map();
  if (!value || value === true) return labels;
  const trimmed = String(value).trim();
  if (!trimmed) return labels;

  if (trimmed.startsWith("{")) {
    const parsed = JSON.parse(trimmed);
    for (const [speaker, name] of Object.entries(parsed)) {
      const finalName = String(name || "").trim();
      if (!finalName) continue;
      labels.set(speaker, finalName);
      labels.set(normalizeSpeakerId(speaker), finalName);
    }
    return labels;
  }

  for (const part of trimmed.split(",")) {
    const [rawSpeaker, ...nameParts] = part.split("=");
    const speaker = rawSpeaker.trim();
    const name = nameParts.join("=").trim();
    if (!speaker || !name) continue;
    labels.set(speaker, name);
    labels.set(normalizeSpeakerId(speaker), name);
  }
  return labels;
}

function getLabelForSpeaker(labels, speakerId) {
  return labels.get(speakerId) || labels.get(normalizeSpeakerId(speakerId)) || null;
}

function getMelFilterbank() {
  if (melFilterbank) return melFilterbank;
  const numBins = FBANK_FFT_SIZE / 2 + 1;
  const melLow = 1127 * Math.log(1 + 20 / 700);
  const melHigh = 1127 * Math.log(1 + SAMPLE_RATE / 2 / 700);
  const melPoints = new Float64Array(FBANK_NUM_MELS + 2);

  for (let i = 0; i < melPoints.length; i += 1) {
    const mel = melLow + ((melHigh - melLow) * i) / (FBANK_NUM_MELS + 1);
    melPoints[i] = 700 * (Math.exp(mel / 1127) - 1);
  }

  const binPoints = new Float64Array(melPoints.length);
  for (let i = 0; i < melPoints.length; i += 1) {
    binPoints[i] = Math.floor(((FBANK_FFT_SIZE + 1) * melPoints[i]) / SAMPLE_RATE);
  }

  melFilterbank = new Array(FBANK_NUM_MELS);
  for (let m = 0; m < FBANK_NUM_MELS; m += 1) {
    const filter = new Float32Array(numBins);
    const left = binPoints[m];
    const center = binPoints[m + 1];
    const right = binPoints[m + 2];
    for (let k = 0; k < numBins; k += 1) {
      if (k >= left && k <= center && center > left) {
        filter[k] = (k - left) / (center - left);
      } else if (k > center && k <= right && right > center) {
        filter[k] = (right - k) / (right - center);
      }
    }
    melFilterbank[m] = filter;
  }
  return melFilterbank;
}

function realFFT(frame) {
  const n = FBANK_FFT_SIZE;
  const re = new Float64Array(n);
  const im = new Float64Array(n);
  for (let i = 0; i < frame.length && i < n; i += 1) re[i] = frame[i];

  for (let i = 1, j = 0; i < n; i += 1) {
    let bit = n >> 1;
    while (j & bit) {
      j ^= bit;
      bit >>= 1;
    }
    j ^= bit;
    if (i < j) [re[i], re[j]] = [re[j], re[i]];
  }

  for (let len = 2; len <= n; len <<= 1) {
    const angle = (-2 * Math.PI) / len;
    const wRe = Math.cos(angle);
    const wIm = Math.sin(angle);
    for (let i = 0; i < n; i += len) {
      let curRe = 1;
      let curIm = 0;
      for (let j = 0; j < len / 2; j += 1) {
        const uRe = re[i + j];
        const uIm = im[i + j];
        const vRe = re[i + j + len / 2] * curRe - im[i + j + len / 2] * curIm;
        const vIm = re[i + j + len / 2] * curIm + im[i + j + len / 2] * curRe;
        re[i + j] = uRe + vRe;
        im[i + j] = uIm + vIm;
        re[i + j + len / 2] = uRe - vRe;
        im[i + j + len / 2] = uIm - vIm;
        const tmpRe = curRe * wRe - curIm * wIm;
        curIm = curRe * wIm + curIm * wRe;
        curRe = tmpRe;
      }
    }
  }

  const power = new Float32Array(n / 2 + 1);
  for (let i = 0; i < power.length; i += 1) power[i] = re[i] * re[i] + im[i] * im[i];
  return power;
}

function computeFbank(samples) {
  const numFrames = Math.max(
    0,
    Math.floor((samples.length - FBANK_FRAME_LENGTH) / FBANK_FRAME_SHIFT) + 1
  );
  if (numFrames === 0) return null;

  const hamming = new Float32Array(FBANK_FRAME_LENGTH);
  for (let i = 0; i < FBANK_FRAME_LENGTH; i += 1) {
    hamming[i] = 0.54 - 0.46 * Math.cos((2 * Math.PI * i) / (FBANK_FRAME_LENGTH - 1));
  }

  const features = new Float32Array(numFrames * FBANK_NUM_MELS);
  const melBank = getMelFilterbank();
  for (let f = 0; f < numFrames; f += 1) {
    const start = f * FBANK_FRAME_SHIFT;
    const frame = new Float32Array(FBANK_FRAME_LENGTH);
    for (let i = 0; i < FBANK_FRAME_LENGTH; i += 1) {
      frame[i] = (samples[start + i] || 0) * hamming[i];
    }
    const power = realFFT(frame);
    for (let m = 0; m < FBANK_NUM_MELS; m += 1) {
      let energy = 0;
      const filter = melBank[m];
      for (let k = 0; k < power.length; k += 1) energy += filter[k] * power[k];
      features[f * FBANK_NUM_MELS + m] = Math.log(Math.max(energy, 1e-10));
    }
  }
  return { features, numFrames };
}

function parseWavHeader(buf) {
  let offset = 12;
  let sampleRate = SAMPLE_RATE;
  let dataOffset = 44;
  while (offset < buf.length - 8) {
    const chunkId = buf.toString("ascii", offset, offset + 4);
    const chunkSize = buf.readUInt32LE(offset + 4);
    if (chunkId === "fmt ") sampleRate = buf.readUInt32LE(offset + 12);
    if (chunkId === "data") {
      dataOffset = offset + 8;
      break;
    }
    offset += 8 + chunkSize;
  }
  return { sampleRate, dataOffset };
}

function wavSamples(buf, startSec, endSec) {
  const { sampleRate, dataOffset } = parseWavHeader(buf);
  const cappedSeconds = Math.min(endSec - startSec, MAX_EMBEDDING_SECONDS);
  const cappedStartSec = Math.max(startSec, endSec - cappedSeconds);
  const startSample = Math.floor(cappedStartSec * sampleRate);
  const endSample = Math.floor(endSec * sampleRate);
  const samples = new Float32Array(Math.max(0, endSample - startSample));
  for (let i = 0; i < samples.length; i += 1) {
    const pos = dataOffset + (startSample + i) * 2;
    if (pos + 1 >= buf.length) break;
    samples[i] = buf.readInt16LE(pos) / 32768;
  }
  return samples;
}

async function ensureSpeakerSession(modelPath = getDefaultSpeakerModelPath()) {
  if (speakerSession) return;
  ensureFile(modelPath, "speaker embedding model");
  const runtime = loadOrt();
  speakerSession = await runtime.InferenceSession.create(modelPath, {
    intraOpNumThreads: 4,
    executionMode: "sequential",
  });
  speakerInputName = speakerSession.inputNames[0];
}

async function extractEmbeddingFromSamples(samples) {
  if (samples.length < SAMPLE_RATE * MIN_SEGMENT_SECONDS) return null;
  await ensureSpeakerSession();
  const limited =
    samples.length > SAMPLE_RATE * MAX_EMBEDDING_SECONDS
      ? samples.subarray(samples.length - SAMPLE_RATE * MAX_EMBEDDING_SECONDS)
      : samples;
  const fbank = computeFbank(limited);
  if (!fbank) return null;
  const runtime = loadOrt();
  const feeds = {
    [speakerInputName]: new runtime.Tensor("float32", fbank.features, [
      1,
      fbank.numFrames,
      FBANK_NUM_MELS,
    ]),
  };
  const results = await speakerSession.run(feeds);
  return new Float32Array(results[Object.keys(results)[0]].data);
}

export function computeCentroid(embeddings) {
  const centroid = new Float32Array(EMBEDDING_DIM);
  if (!embeddings.length) return centroid;
  for (const emb of embeddings) {
    for (let i = 0; i < EMBEDDING_DIM; i += 1) centroid[i] += emb[i];
  }
  for (let i = 0; i < EMBEDDING_DIM; i += 1) centroid[i] /= embeddings.length;
  return centroid;
}

export function cosineSimilarity(a, b) {
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i += 1) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  return denom === 0 ? 0 : dot / denom;
}

export async function computeSpeakerCentroids(wavPath, diarizationSegments) {
  const wav = fs.readFileSync(wavPath);
  const bySpeaker = new Map();
  for (const seg of diarizationSegments) {
    if (!seg?.speaker || seg.end - seg.start < MIN_SEGMENT_SECONDS) continue;
    const list = bySpeaker.get(seg.speaker) || [];
    list.push(seg);
    bySpeaker.set(seg.speaker, list);
  }

  const result = {};
  for (const [speakerId, segments] of bySpeaker.entries()) {
    const bestSegments = [...segments]
      .sort((a, b) => b.end - b.start - (a.end - a.start))
      .slice(0, 3);
    const embeddings = [];
    for (const seg of bestSegments) {
      const emb = await extractEmbeddingFromSamples(wavSamples(wav, seg.start, seg.end));
      if (emb) embeddings.push(emb);
    }
    if (!embeddings.length) continue;
    result[speakerId] = {
      embedding: computeCentroid(embeddings),
      samples: bestSegments.map((seg) => ({
        start: seg.start,
        end: seg.end,
        timestamp: formatTimestamp(seg.start),
      })),
    };
  }
  return result;
}

function formatTimestamp(seconds) {
  const total = Math.max(0, Math.floor(seconds || 0));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

function openDb(dbPath) {
  ensureFile(dbPath, "speaker DB");
  const Database = loadSqlite();
  return new Database(dbPath);
}

export function readSpeakerProfiles(dbPath) {
  const db = openDb(dbPath);
  try {
    return db.prepare("SELECT * FROM speaker_profiles").all();
  } finally {
    db.close();
  }
}

export function matchProfiles(centroids, dbPath, options = {}) {
  const matchThreshold = Number(options.matchThreshold ?? 0.6);
  const suggestThreshold = Number(options.suggestThreshold ?? 0.5);
  const profiles = readSpeakerProfiles(dbPath);
  const matches = {};
  for (const [speakerId, data] of Object.entries(centroids)) {
    let bestProfile = null;
    let bestSimilarity = 0;
    for (const profile of profiles) {
      const profileEmbedding = new Float32Array(
        profile.embedding.buffer,
        profile.embedding.byteOffset,
        profile.embedding.byteLength / 4
      );
      const similarity = cosineSimilarity(data.embedding, profileEmbedding);
      if (similarity > bestSimilarity) {
        bestSimilarity = similarity;
        bestProfile = profile;
      }
    }
    if (bestProfile && bestSimilarity >= matchThreshold) {
      matches[speakerId] = {
        state: "matched",
        profileId: bestProfile.id,
        displayName: bestProfile.display_name,
        confidence: Number(bestSimilarity.toFixed(4)),
      };
    } else if (bestProfile && bestSimilarity >= suggestThreshold) {
      matches[speakerId] = {
        state: "suggested",
        profileId: bestProfile.id,
        displayName: bestProfile.display_name,
        confidence: Number(bestSimilarity.toFixed(4)),
      };
    } else {
      matches[speakerId] = { state: "unknown", confidence: Number(bestSimilarity.toFixed(4)) };
    }
  }
  return matches;
}

export function upsertSpeakerProfile(dbPath, displayName, embedding, profileId = null) {
  const db = openDb(dbPath);
  try {
    let existing = null;
    if (profileId != null) {
      existing = db.prepare("SELECT * FROM speaker_profiles WHERE id = ?").get(profileId);
    }
    if (!existing) {
      existing = db.prepare("SELECT * FROM speaker_profiles WHERE display_name = ?").get(displayName);
    }
    if (existing) {
      const stored = new Float32Array(
        existing.embedding.buffer,
        existing.embedding.byteOffset,
        existing.embedding.byteLength / 4
      );
      const blended = new Float32Array(stored.length);
      for (let i = 0; i < stored.length; i += 1) blended[i] = 0.7 * stored[i] + 0.3 * embedding[i];
      db.prepare(
        "UPDATE speaker_profiles SET display_name = ?, embedding = ?, sample_count = sample_count + 1, updated_at = CURRENT_TIMESTAMP WHERE id = ?"
      ).run(displayName, Buffer.from(blended.buffer), existing.id);
      return db.prepare("SELECT * FROM speaker_profiles WHERE id = ?").get(existing.id);
    }
    const info = db
      .prepare("INSERT INTO speaker_profiles (display_name, embedding) VALUES (?, ?)")
      .run(displayName, Buffer.from(embedding.buffer));
    return db.prepare("SELECT * FROM speaker_profiles WHERE id = ?").get(info.lastInsertRowid);
  } finally {
    db.close();
  }
}

export async function promptForSpeakerLabels(centroids, currentMatches = {}) {
  const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
  const labels = new Map();
  try {
    for (const [speakerId, data] of Object.entries(centroids)) {
      if (currentMatches[speakerId]?.state === "matched") continue;
      const samples = (data.samples || []).map((s) => s.timestamp).slice(0, 3).join(", ");
      const answer = (await rl.question(`Who is ${speakerId}${samples ? ` (samples: ${samples})` : ""}? `)).trim();
      if (answer) labels.set(speakerId, answer);
    }
  } finally {
    rl.close();
  }
  return labels;
}

export async function applySpeakerLearning(options) {
  const {
    wavPath,
    diarizationSegments,
    dbPath = DEFAULT_DB_PATH,
    speakerLabels = new Map(),
    learnSpeakers = false,
    noDbWrite = false,
    interactive = false,
    matchThreshold = 0.6,
    suggestThreshold = 0.5,
  } = options;
  const centroids = await computeSpeakerCentroids(wavPath, diarizationSegments);
  let matches = matchProfiles(centroids, dbPath, { matchThreshold, suggestThreshold });
  const labels =
    interactive && learnSpeakers && !speakerLabels.size
      ? await promptForSpeakerLabels(centroids, matches)
      : speakerLabels;
  const learned = {};

  if (learnSpeakers && labels.size) {
    for (const [speakerId, data] of Object.entries(centroids)) {
      const name = getLabelForSpeaker(labels, speakerId);
      if (!name) continue;
      if (!data?.embedding) continue;
      const profile = noDbWrite ? null : upsertSpeakerProfile(dbPath, name, data.embedding);
      learned[speakerId] = {
        displayName: name,
        profileId: profile?.id ?? null,
        sampleCount: profile?.sample_count ?? null,
        dbWrite: !noDbWrite,
      };
      matches[speakerId] = {
        state: "learned",
        profileId: profile?.id ?? null,
        displayName: name,
        confidence: 1,
      };
    }
  }

  return {
    dbPath,
    matchThreshold,
    suggestThreshold,
    speakers: Object.fromEntries(
      Object.entries(centroids).map(([speakerId, data]) => [
        speakerId,
        {
          samples: data.samples,
          embeddingDimensions: data.embedding.length,
          match: matches[speakerId] || { state: "unknown", confidence: 0 },
          learned: learned[speakerId] || null,
        },
      ])
    ),
    matches,
    learned,
  };
}
