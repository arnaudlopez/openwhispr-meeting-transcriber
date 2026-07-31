#!/usr/bin/env node
import fs from "node:fs";
import fsp from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import { diarizeWithFluidAudio } from "./fluidaudio-diarization.mjs";
import {
  applySpeakerLearning,
  getDefaultSpeakerDbPath,
} from "./speaker-profiles.mjs";

const SCRIPT_PATH = fileURLToPath(import.meta.url);
const ELECTRON_NODE = "/Applications/OpenWhispr.app/Contents/MacOS/OpenWhispr";
const SAMPLE_RATE = 16000;
const MAX_PREVIEW_SECONDS = 8;
const MIN_PREVIEW_SECONDS = 1.2;

function parseArgs(argv) {
  const args = {};
  for (let index = 0; index < argv.length; index += 1) {
    const argument = argv[index];
    if (!argument.startsWith("--")) continue;
    const key = argument.slice(2);
    const value = argv[index + 1];
    if (!value || value.startsWith("--")) {
      args[key] = true;
    } else {
      args[key] = value;
      index += 1;
    }
  }
  return args;
}

function run(command, args, options = {}) {
  return new Promise((resolve, reject) => {
    const process = spawn(command, args, {
      stdio: ["ignore", "pipe", "pipe"],
      ...options,
    });
    let stdout = "";
    let stderr = "";
    process.stdout.on("data", (data) => {
      stdout += data.toString();
    });
    process.stderr.on("data", (data) => {
      stderr += data.toString();
    });
    process.on("error", reject);
    process.on("close", (code) => {
      if (code !== 0) {
        reject(
          new Error(
            `${path.basename(command)} exited ${code}: ${(stderr || stdout).slice(-2000)}`
          )
        );
        return;
      }
      resolve({ stdout, stderr });
    });
  });
}

function ffmpegPath() {
  return [
    process.env.FFMPEG_PATH,
    "/opt/homebrew/bin/ffmpeg",
    "/usr/local/bin/ffmpeg",
    "/usr/bin/ffmpeg",
  ].find((candidate) => candidate && fs.existsSync(candidate)) || "ffmpeg";
}

function normalizeSpeakerId(value) {
  const match = String(value || "").match(/^speaker_0*(\d+)$/);
  return match ? `speaker_${Number(match[1])}` : String(value || "");
}

function speakerIndex(value) {
  const match = normalizeSpeakerId(value).match(/^speaker_(\d+)$/);
  return match ? Number(match[1]) : Number.MAX_SAFE_INTEGER;
}

function speakerLabel(index) {
  if (index < 26) return `Voix ${String.fromCharCode(65 + index)}`;
  return `Voix ${index + 1}`;
}

function overlapSeconds(left, right) {
  return Math.max(
    0,
    Math.min(Number(left.end), Number(right.end)) -
      Math.max(Number(left.start), Number(right.start))
  );
}

function previewCandidate(speakerId, segments) {
  const own = segments.filter((segment) => segment.speaker === speakerId);
  const competing = segments.filter((segment) => segment.speaker !== speakerId);
  const ranked = own
    .map((segment) => {
      const duration = Number(segment.end) - Number(segment.start);
      const overlap = competing.reduce(
        (total, other) => total + overlapSeconds(segment, other),
        0
      );
      return {
        start: Number(segment.start),
        end: Number(segment.end),
        duration,
        overlap,
        cleanRatio: duration > 0 ? Math.max(0, 1 - overlap / duration) : 0,
      };
    })
    .filter(
      (segment) =>
        Number.isFinite(segment.start) &&
        Number.isFinite(segment.end) &&
        segment.duration >= MIN_PREVIEW_SECONDS
    )
    .sort(
      (left, right) =>
        right.cleanRatio - left.cleanRatio ||
        Math.min(right.duration, MAX_PREVIEW_SECONDS) -
          Math.min(left.duration, MAX_PREVIEW_SECONDS) ||
        left.start - right.start
    );
  const selected = ranked[0] || own
    .map((segment) => ({
      start: Number(segment.start),
      end: Number(segment.end),
      duration: Number(segment.end) - Number(segment.start),
      overlap: 0,
      cleanRatio: 0,
    }))
    .filter((segment) => segment.duration > 0)
    .sort((left, right) => right.duration - left.duration)[0];
  if (!selected) return null;

  const duration = Math.min(MAX_PREVIEW_SECONDS, selected.duration);
  const start = selected.start + Math.max(0, (selected.duration - duration) / 2);
  return {
    start: Number(start.toFixed(3)),
    end: Number((start + duration).toFixed(3)),
    duration: Number(duration.toFixed(3)),
    cleanRatio: Number(selected.cleanRatio.toFixed(3)),
  };
}

async function convertToWav(inputPath, wavPath) {
  await run(ffmpegPath(), [
    "-hide_banner",
    "-loglevel",
    "error",
    "-i",
    inputPath,
    "-vn",
    "-ar",
    String(SAMPLE_RATE),
    "-ac",
    "1",
    "-c:a",
    "pcm_s16le",
    "-y",
    wavPath,
  ]);
}

async function createPreview(wavPath, outputPath, candidate) {
  await run(ffmpegPath(), [
    "-hide_banner",
    "-loglevel",
    "error",
    "-ss",
    String(candidate.start),
    "-t",
    String(candidate.duration),
    "-i",
    wavPath,
    "-ac",
    "1",
    "-ar",
    String(SAMPLE_RATE),
    "-c:a",
    "aac",
    "-b:a",
    "64k",
    "-y",
    outputPath,
  ]);
}

function totalSpeakerDuration(speakerId, segments) {
  return segments
    .filter((segment) => segment.speaker === speakerId)
    .reduce(
      (total, segment) =>
        total + Math.max(0, Number(segment.end) - Number(segment.start)),
      0
    );
}

function requireInput(filePath) {
  const resolved = path.resolve(String(filePath || ""));
  if (!fs.existsSync(resolved) || !fs.statSync(resolved).isFile()) {
    throw new Error(`Audio file not found: ${resolved}`);
  }
  return resolved;
}

function shouldReexecWithElectron() {
  return (
    !process.versions.electron &&
    !process.env.OPENWHISPR_NO_ELECTRON_REEXEC &&
    fs.existsSync(ELECTRON_NODE)
  );
}

async function reexecWithElectron() {
  const child = spawn(ELECTRON_NODE, [SCRIPT_PATH, ...process.argv.slice(2)], {
    stdio: "inherit",
    env: {
      ...process.env,
      ELECTRON_RUN_AS_NODE: "1",
      OPENWHISPR_NO_ELECTRON_REEXEC: "1",
    },
  });
  await new Promise((resolve, reject) => {
    child.on("error", reject);
    child.on("close", (code) => {
      process.exitCode = code ?? 1;
      resolve();
    });
  });
}

export async function analyzeMeetingSpeakers({
  audioPath,
  outputPath,
  participants = "auto",
}) {
  const input = requireInput(audioPath);
  const output = path.resolve(
    outputPath ||
      path.join(
        path.dirname(input),
        `${path.basename(input).replace(/\.[^.]+$/, "")}-speaker-analysis.json`
      )
  );
  const previewDir = output.replace(/\.json$/i, "-previews");
  const workDir = await fsp.mkdtemp(path.join(os.tmpdir(), "openwhispr-analysis-"));
  const wavPath = path.join(workDir, "meeting.wav");

  try {
    await fsp.mkdir(path.dirname(output), { recursive: true });
    await fsp.rm(previewDir, { recursive: true, force: true });
    await fsp.mkdir(previewDir, { recursive: true });
    await convertToWav(input, wavPath);

    const diarization = await diarizeWithFluidAudio({
      wavPath,
      workDir,
      expectedSpeakers: participants,
      minimumSegmentSeconds: 0.3,
    });
    const diarizationSegments = diarization.segments.map((segment) => ({
      start: Number(segment.start),
      end: Number(segment.end),
      speaker: normalizeSpeakerId(segment.speaker),
    }));
    const dbPath = getDefaultSpeakerDbPath();
    let speakerLearning = null;
    if (fs.existsSync(dbPath)) {
      speakerLearning = await applySpeakerLearning({
        wavPath,
        diarizationSegments,
        dbPath,
        learnSpeakers: false,
        noDbWrite: true,
        interactive: false,
        matchThreshold: 0.6,
        suggestThreshold: 0.5,
      });
    }

    const speakerIds = [
      ...new Set(diarizationSegments.map((segment) => segment.speaker)),
    ].sort((left, right) => speakerIndex(left) - speakerIndex(right));
    const speakers = [];
    for (const [index, speakerId] of speakerIds.entries()) {
      const candidate = previewCandidate(speakerId, diarizationSegments);
      let previewPath = null;
      if (candidate) {
        previewPath = path.join(
          previewDir,
          `${String(index + 1).padStart(2, "0")}-${speakerId}.m4a`
        );
        await createPreview(wavPath, previewPath, candidate);
      }
      const match =
        speakerLearning?.matches?.[speakerId] || {
          state: "unknown",
          confidence: 0,
        };
      speakers.push({
        id: speakerId,
        label: speakerLabel(index),
        speakingDurationSeconds: Number(
          totalSpeakerDuration(speakerId, diarizationSegments).toFixed(2)
        ),
        turnCount: diarizationSegments.filter(
          (segment) => segment.speaker === speakerId
        ).length,
        previewPath,
        previewStartSeconds: candidate?.start ?? null,
        previewEndSeconds: candidate?.end ?? null,
        previewCleanRatio: candidate?.cleanRatio ?? null,
        match,
      });
    }

    const result = {
      success: true,
      input,
      generated_at: new Date().toISOString(),
      participantsRequested: participants,
      detectedSpeakers: speakers.length,
      diarizationProvider: "fluidaudio",
      diarizationMetadata: diarization.metadata,
      speakerLearning,
      speakers,
      diarizationSegments,
    };
    await fsp.writeFile(output, `${JSON.stringify(result, null, 2)}\n`);
    return {
      success: true,
      analysis_path: output,
      preview_directory: previewDir,
      detected_speakers: speakers.length,
      speakers,
    };
  } finally {
    await fsp.rm(workDir, { recursive: true, force: true });
  }
}

async function main() {
  if (shouldReexecWithElectron()) {
    await reexecWithElectron();
    return;
  }
  const args = parseArgs(process.argv.slice(2));
  if (!args.input) throw new Error("Missing --input");
  const originalLog = console.log;
  console.log = (...values) => console.error(...values);
  try {
    const result = await analyzeMeetingSpeakers({
      audioPath: args.input,
      outputPath: args.output,
      participants: args.participants || "auto",
    });
    process.stdout.write(`${JSON.stringify(result)}\n`);
  } finally {
    console.log = originalLog;
  }
}

const isMain =
  process.argv[1] && path.resolve(process.argv[1]) === path.resolve(SCRIPT_PATH);

if (isMain) {
  main().catch((error) => {
    console.error(error instanceof Error ? error.stack || error.message : String(error));
    process.exit(1);
  });
}
