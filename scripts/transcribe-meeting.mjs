#!/usr/bin/env node
import fs from "node:fs";
import fsp from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { spawn } from "node:child_process";
import readline from "node:readline/promises";

const DEFAULT_DB_PATH = path.join(
  os.homedir(),
  "Library/Application Support/open-whispr/transcriptions.db"
);
const LOCAL_SCRIPT = path.join(
  path.dirname(new URL(import.meta.url).pathname),
  "local-diarized-markdown.mjs"
);
const DEFAULT_ASR_PROVIDER = "openai";
const DEFAULT_ASR_MODEL = "gpt-transcribe";
const DEFAULT_DIARIZATION_PROVIDER = "fluidaudio";
const DEFAULT_ALIGNMENT_ASR_MODEL = "nemotron-3.5-asr-streaming-0.6b";

function parseArgs(argv) {
  const args = { positional: [] };
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (!arg.startsWith("--")) {
      args.positional.push(arg);
      continue;
    }
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

function usage() {
  console.log(`Usage:
  node scripts/transcribe-meeting.mjs <audio-file> [options]

Options:
  --out-dir <dir>            Output directory. Default: same folder as the source audio
  --output <path>            Exact Markdown output path. Default: source filename with .md
  --participants <auto|N>    Expected speaker count. Default: 2
  --diarization-provider <provider>
                            sherpa or fluidaudio. Default: fluidaudio
  --fluidaudio-cli <path>    Override the FluidAudio CLI path
  --diarization-min-segment-seconds <N>
                            Minimum local speech segment. Default: 0.3
  --asr-provider <provider>  local, openrouter, or openai. Default: openai
  --asr-model <model>        ASR model. Default: gpt-transcribe
  --alignment-asr-model <model>
                            Local timed ASR used to align gpt-transcribe.
                            Default: nemotron-3.5-asr-streaming-0.6b
  --chunk-seconds <N>        ASR chunk length. Default: 15 offline, 600 streaming
  --language <code>          Cloud ASR language. Default: fr
  --concurrency <N>          Concurrent OpenRouter requests. Default: 3
  --turn-gap-seconds <N>     Merge same-speaker speech across short gaps. Default: 1.2
  --openrouter-url <url>     Override the OpenRouter transcription endpoint
  --asr-cache <path>         Resume cache for OpenRouter calls
  --openai-url <url>         Override the OpenAI transcription endpoint
  --openai-transcript-json <path>
                            Re-align a prior GPT transcript without another API call
  --openai-bitrate-kbps <N>  Speech compression bitrate for oversized files. Default: 40
  --openai-max-upload-mb <N> Safe upload ceiling. Default: 24
  --openai-segment-seconds <N>
                            Maximum gpt-transcribe part duration. Default: 900
  --openai-chunk-cache <path>
                            Resume cache written after every cloud part
  --known-speaker-references <map>
                            Reference files as JSON or Name=/path/ref.wav,...
  --openai-auto-speaker-references
                            Derive references from local diarization (less reliable)
  --diarization-json <path>  Reuse diarizationSegments from an existing JSON output
  --db-mode <real|copy>      real writes OpenWhispr speaker_profiles; copy uses a temporary DB copy. Default: real
  --speaker-db <path>        Override SQLite DB path
  --speaker-labels <map>     Non-interactive labels, e.g. speaker_00=Alice,speaker_01=Bob
  --no-db-write             Resolve labels without updating OpenWhispr speaker profiles
  --no-learn                Match only; do not ask names or write new profiles
  --verify-match             After learning, run a second match-only transcript using the same DB
  --open                     Reveal the final Markdown in Finder
  --yes                      Accepted for compatibility; real DB is already the default
  --help                     Show this help
`);
}

async function ask(question, fallback = "") {
  const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
  try {
    const suffix = fallback ? ` [${fallback}]` : "";
    const answer = (await rl.question(`${question}${suffix}: `)).trim();
    return answer || fallback;
  } finally {
    rl.close();
  }
}

function run(command, args, options = {}) {
  return new Promise((resolve, reject) => {
    const proc = spawn(command, args, { stdio: "inherit", ...options });
    proc.on("error", reject);
    proc.on("close", (code) => {
      if (code === 0) resolve();
      else reject(new Error(`${path.basename(command)} exited ${code}`));
    });
  });
}

async function copyDb(sourceDb) {
  const tmpDir = await fsp.mkdtemp(path.join(os.tmpdir(), "openwhispr-meeting-"));
  const targetDb = path.join(tmpDir, "transcriptions.db");
  await fsp.copyFile(sourceDb, targetDb);
  return { tmpDir, dbPath: targetDb };
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) {
    usage();
    return;
  }

  let audioPath = args.positional[0] || args.input;
  if (!audioPath) audioPath = await ask("Audio file path");
  audioPath = path.resolve(audioPath);
  if (!fs.existsSync(audioPath)) throw new Error(`Audio file not found: ${audioPath}`);

  const participants = args.participants || args["expected-speakers"] || "2";
  const outDir = path.resolve(args["out-dir"] || path.dirname(audioPath));
  const baseName = path.parse(path.basename(audioPath)).name || "meeting";
  const finalOutput = path.resolve(args.output || path.join(outDir, `${baseName}.md`));
  const learnedOutput = args["verify-match"]
    ? path.join(outDir, `${baseName}-learned.md`)
    : finalOutput;

  const dbMode = args["db-mode"] || (args["copy-db"] ? "copy" : "real");
  const sourceDb = path.resolve(args["speaker-db"] || DEFAULT_DB_PATH);
  if (!fs.existsSync(sourceDb)) throw new Error(`Speaker DB not found: ${sourceDb}`);

  let speakerDb = sourceDb;
  let dbNote = "real OpenWhispr DB";
  if (dbMode === "copy") {
    const copied = await copyDb(sourceDb);
    speakerDb = copied.dbPath;
    dbNote = `temporary DB copy (${speakerDb})`;
  } else if (dbMode !== "real") {
    throw new Error(`Unsupported --db-mode: ${dbMode}. Use copy or real.`);
  }

  await fsp.mkdir(outDir, { recursive: true });

  const shouldLearn = !args["no-learn"];
  const asrProvider = args["asr-provider"] || DEFAULT_ASR_PROVIDER;
  const diarizationProvider = args["diarization-provider"] || DEFAULT_DIARIZATION_PROVIDER;
  const asrModel =
    args["asr-model"] ||
    (asrProvider === "openrouter"
      ? "openai/gpt-4o-transcribe"
      : asrProvider === "openai"
        ? DEFAULT_ASR_MODEL
        : "parakeet-tdt-0.6b-v3");
  const alignmentAsrModel = args["alignment-asr-model"] || DEFAULT_ALIGNMENT_ASR_MODEL;
  const effectiveArgs = {
    ...args,
    "asr-provider": asrProvider,
    "asr-model": asrModel,
    "diarization-provider": diarizationProvider,
    ...(asrProvider === "openai" && asrModel !== "gpt-4o-transcribe-diarize"
      ? { "alignment-asr-model": alignmentAsrModel }
      : {}),
  };
  const learnArgs = [
    LOCAL_SCRIPT,
    "--input",
    audioPath,
    "--output",
    learnedOutput,
    "--expected-speakers",
    String(participants),
    "--speaker-db",
    speakerDb,
  ];
  if (shouldLearn) learnArgs.push("--learn-speakers");
  if (args["speaker-labels"]) learnArgs.push("--speaker-labels", args["speaker-labels"]);
  if (args["no-db-write"]) learnArgs.push("--no-db-write");
  for (const key of [
    "asr-provider",
    "asr-model",
    "alignment-asr-model",
    "chunk-seconds",
    "language",
    "concurrency",
    "turn-gap-seconds",
    "openrouter-url",
    "asr-cache",
    "openai-url",
    "openai-transcript-json",
    "openai-bitrate-kbps",
    "openai-max-upload-mb",
    "openai-segment-seconds",
    "openai-chunk-cache",
    "known-speaker-references",
    "diarization-json",
    "diarization-provider",
    "fluidaudio-cli",
    "diarization-min-segment-seconds",
  ]) {
    if (effectiveArgs[key]) {
      learnArgs.push(`--${key}`, String(effectiveArgs[key]));
    }
  }
  if (args["openai-auto-speaker-references"]) {
    learnArgs.push("--openai-auto-speaker-references");
  }

  console.log("");
  console.log("OpenWhispr meeting transcript");
  console.log(`Audio: ${audioPath}`);
  console.log(`Output: ${learnedOutput}`);
  console.log(`Participants: ${participants}`);
  console.log(`Diarization provider: ${diarizationProvider}`);
  console.log(`ASR provider: ${asrProvider}`);
  console.log(`ASR model: ${asrModel}`);
  console.log(`Speaker DB: ${dbNote}`);
  console.log("");

  await run(process.execPath, learnArgs);

  let output = learnedOutput;
  if (args["verify-match"]) {
    console.log("");
    console.log("Running match-only verification transcript...");
    await run(process.execPath, [
      LOCAL_SCRIPT,
      "--input",
      audioPath,
      "--output",
      finalOutput,
      "--expected-speakers",
      String(participants),
      "--speaker-db",
      speakerDb,
      "--asr-provider",
      asrProvider,
      "--asr-model",
      asrModel,
      ...(effectiveArgs["alignment-asr-model"]
        ? ["--alignment-asr-model", String(effectiveArgs["alignment-asr-model"])]
        : []),
      ...(args["chunk-seconds"] ? ["--chunk-seconds", String(args["chunk-seconds"])] : []),
      ...(args.language ? ["--language", String(args.language)] : []),
      ...(args.concurrency ? ["--concurrency", String(args.concurrency)] : []),
      ...(args["turn-gap-seconds"] ? ["--turn-gap-seconds", String(args["turn-gap-seconds"])] : []),
      ...(args["openrouter-url"] ? ["--openrouter-url", String(args["openrouter-url"])] : []),
      ...(args["asr-cache"] ? ["--asr-cache", String(args["asr-cache"])] : []),
      ...(args["openai-url"] ? ["--openai-url", String(args["openai-url"])] : []),
      ...(args["openai-transcript-json"]
        ? ["--openai-transcript-json", String(args["openai-transcript-json"])]
        : []),
      ...(args["openai-bitrate-kbps"]
        ? ["--openai-bitrate-kbps", String(args["openai-bitrate-kbps"])]
        : []),
      ...(args["openai-max-upload-mb"]
        ? ["--openai-max-upload-mb", String(args["openai-max-upload-mb"])]
        : []),
      ...(args["openai-segment-seconds"]
        ? ["--openai-segment-seconds", String(args["openai-segment-seconds"])]
        : []),
      ...(args["openai-chunk-cache"]
        ? ["--openai-chunk-cache", String(args["openai-chunk-cache"])]
        : []),
      ...(args["known-speaker-references"]
        ? ["--known-speaker-references", String(args["known-speaker-references"])]
        : []),
      ...(args["openai-auto-speaker-references"] ? ["--openai-auto-speaker-references"] : []),
      ...(args["diarization-json"] ? ["--diarization-json", String(args["diarization-json"])] : []),
      "--diarization-provider",
      diarizationProvider,
      ...(args["fluidaudio-cli"] ? ["--fluidaudio-cli", String(args["fluidaudio-cli"])] : []),
      ...(args["diarization-min-segment-seconds"]
        ? ["--diarization-min-segment-seconds", String(args["diarization-min-segment-seconds"])]
        : []),
    ]);
    output = finalOutput;
  }

  const jsonOutput = output.replace(/\.md$/i, ".json");
  console.log("");
  console.log("Done.");
  console.log(`Markdown: ${output}`);
  console.log(`JSON: ${jsonOutput}`);
  console.log(`Speaker DB used: ${speakerDb}`);
  if (dbMode === "copy") {
    console.log(
      "Note: this was a copied DB. Omit --db-mode copy when you want to persist profiles in OpenWhispr."
    );
  }

  if (args.open) await run("open", ["-R", output]);
}

main().catch((err) => {
  console.error(err.stack || err.message);
  process.exit(1);
});
