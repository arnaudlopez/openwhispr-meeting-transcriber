#!/usr/bin/env node
import fs from "node:fs";
import fsp from "node:fs/promises";
import path from "node:path";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";

const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(SCRIPT_DIR, "..");
const TRANSCRIBE_SCRIPT = path.join(SCRIPT_DIR, "transcribe-meeting.mjs");
const ANALYZE_SPEAKERS_SCRIPT = path.join(SCRIPT_DIR, "analyze-meeting-speakers.mjs");
const MAX_LOG_CHARS = 100_000;

export const DEFAULT_PIPELINE = Object.freeze({
  asrProvider: "openai",
  asrModel: "gpt-transcribe",
  diarizationProvider: "fluidaudio",
  alignmentAsrModel: "nemotron-3.5-asr-streaming-0.6b",
  participants: 2,
  language: "fr",
});

function sourceBaseName(filePath) {
  return path.parse(path.basename(filePath)).name || "meeting";
}

function appendLog(current, chunk) {
  const combined = current + chunk.toString();
  return combined.length > MAX_LOG_CHARS
    ? combined.slice(combined.length - MAX_LOG_CHARS)
    : combined;
}

function requireLocalFile(filePath, label) {
  const resolved = path.resolve(filePath);
  if (!fs.existsSync(resolved)) {
    throw new Error(`${label} not found: ${resolved}`);
  }
  if (!fs.statSync(resolved).isFile()) {
    throw new Error(`${label} is not a file: ${resolved}`);
  }
  return resolved;
}

function serializeMap(value) {
  if (!value || !Object.keys(value).length) return null;
  return JSON.stringify(value);
}

export function normalizeTranscriptionRequest(request) {
  const audioPath = requireLocalFile(request.audio_path, "Audio file");
  const outputDir = path.resolve(request.output_dir || path.dirname(audioPath));
  const defaultName = `${sourceBaseName(audioPath)}.md`;
  const outputPath = path.resolve(request.output_path || path.join(outputDir, defaultName));
  const speakerLabels = serializeMap(request.speaker_labels);
  const speakerReferences = serializeMap(request.speaker_references);
  const persistSpeakerProfiles = request.persist_speaker_profiles === true;

  if (persistSpeakerProfiles && !speakerLabels) {
    throw new Error(
      "persist_speaker_profiles requires speaker_labels so the MCP call never becomes interactive"
    );
  }

  return {
    audioPath,
    outputDir,
    outputPath,
    participants: request.participants ?? DEFAULT_PIPELINE.participants,
    language: request.language || DEFAULT_PIPELINE.language,
    asrProvider: request.asr_provider || DEFAULT_PIPELINE.asrProvider,
    asrModel: request.asr_model || DEFAULT_PIPELINE.asrModel,
    diarizationProvider: request.diarization_provider || DEFAULT_PIPELINE.diarizationProvider,
    alignmentAsrModel: request.alignment_asr_model || DEFAULT_PIPELINE.alignmentAsrModel,
    speakerLabels,
    speakerReferences,
    persistSpeakerProfiles,
    transcriptJson: request.transcript_json
      ? requireLocalFile(request.transcript_json, "Transcript replay JSON")
      : null,
    diarizationJson: request.diarization_json
      ? requireLocalFile(request.diarization_json, "Diarization replay JSON")
      : null,
    chunkCache: request.openai_chunk_cache ? path.resolve(request.openai_chunk_cache) : null,
  };
}

export function buildTranscriptionCommand(request) {
  const normalized = normalizeTranscriptionRequest(request);
  const args = [
    TRANSCRIBE_SCRIPT,
    normalized.audioPath,
    "--output",
    normalized.outputPath,
    "--participants",
    String(normalized.participants),
    "--language",
    normalized.language,
    "--asr-provider",
    normalized.asrProvider,
    "--asr-model",
    normalized.asrModel,
    "--diarization-provider",
    normalized.diarizationProvider,
  ];

  if (normalized.asrProvider === "openai" && normalized.asrModel !== "gpt-4o-transcribe-diarize") {
    args.push("--alignment-asr-model", normalized.alignmentAsrModel);
  }
  if (normalized.speakerLabels) {
    args.push("--speaker-labels", normalized.speakerLabels);
  }
  if (normalized.speakerReferences) {
    args.push("--known-speaker-references", normalized.speakerReferences);
  }
  if (normalized.transcriptJson) {
    args.push("--openai-transcript-json", normalized.transcriptJson);
  }
  if (normalized.diarizationJson) {
    args.push("--diarization-json", normalized.diarizationJson);
  }
  if (normalized.chunkCache) {
    args.push("--openai-chunk-cache", normalized.chunkCache);
  }

  if (!normalized.persistSpeakerProfiles) {
    args.push("--no-db-write");
    if (!normalized.speakerLabels) args.push("--no-learn");
  }

  return {
    command: process.execPath,
    args,
    cwd: REPO_ROOT,
    normalized,
  };
}

function runCommand(command, args, options) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      ...options,
      stdio: ["ignore", "pipe", "pipe"],
      env: { ...process.env, OPENWHISPR_MCP: "1" },
    });
    let stdout = "";
    let stderr = "";

    child.stdout.on("data", (chunk) => {
      stdout = appendLog(stdout, chunk);
    });
    child.stderr.on("data", (chunk) => {
      stderr = appendLog(stderr, chunk);
    });
    child.on("error", reject);
    child.on("close", (code, signal) => {
      if (code === 0) {
        resolve({ stdout, stderr });
        return;
      }
      const details = [stderr.trim(), stdout.trim()].filter(Boolean).join("\n");
      reject(
        new Error(
          `Transcription process failed (${signal || `exit ${code}`}): ${details.slice(-4000)}`
        )
      );
    });
  });
}

function speakerNames(sidecar) {
  return [
    ...new Set(
      Object.values(sidecar.speakerLearning?.matches || {})
        .filter((match) => match?.state === "matched" || match?.state === "learned")
        .map((match) => match.displayName)
        .filter(Boolean)
    ),
  ];
}

export async function runTranscription(request) {
  const spec = buildTranscriptionCommand(request);
  await fsp.mkdir(path.dirname(spec.normalized.outputPath), { recursive: true });
  await runCommand(spec.command, spec.args, { cwd: spec.cwd });

  const jsonPath = spec.normalized.outputPath.replace(/\.md$/i, ".json");
  const sidecar = JSON.parse(await fsp.readFile(jsonPath, "utf8"));
  const alignment = sidecar.asrMetadata?.alignment || {};
  const usage = sidecar.asrMetadata?.usage || {};

  return {
    success: true,
    markdown_path: spec.normalized.outputPath,
    json_path: jsonPath,
    chunk_cache_path:
      sidecar.asrMetadata?.transcriptCachePath ||
      spec.normalized.chunkCache ||
      spec.normalized.outputPath.replace(/\.md$/i, "-openai-chunks.json"),
    asr_provider: sidecar.asrProvider || spec.normalized.asrProvider,
    asr_model: sidecar.asrModel || spec.normalized.asrModel,
    diarization_provider: sidecar.diarizationProvider || spec.normalized.diarizationProvider,
    diarization_pipeline: sidecar.diarizationMetadata?.pipeline || null,
    alignment_model: alignment.alignmentModel || spec.normalized.alignmentAsrModel || null,
    alignment_ratio: Number.isFinite(Number(alignment.matchedRatio))
      ? Number(alignment.matchedRatio)
      : null,
    transcript_segments: Array.isArray(sidecar.transcriptSegments)
      ? sidecar.transcriptSegments.length
      : 0,
    diarization_segments: Array.isArray(sidecar.diarizationSegments)
      ? sidecar.diarizationSegments.length
      : 0,
    duration_seconds: Number.isFinite(Number(sidecar.asrMetadata?.duration))
      ? Number(sidecar.asrMetadata.duration)
      : null,
    billed_seconds: Number.isFinite(Number(usage.seconds)) ? Number(usage.seconds) : null,
    estimated_cost_usd: Number.isFinite(Number(usage.cost)) ? Number(usage.cost) : null,
    speaker_names: speakerNames(sidecar),
    degraded: sidecar.degraded || null,
  };
}

export async function runSpeakerAnalysis(request) {
  const audioPath = requireLocalFile(request.audio_path, "Audio file");
  const participants = request.participants ?? "auto";
  const outputPath = path.resolve(
    request.output_path ||
      path.join(
        request.output_dir ? path.resolve(request.output_dir) : path.dirname(audioPath),
        `${sourceBaseName(audioPath)}-speaker-analysis.json`
      )
  );
  await fsp.mkdir(path.dirname(outputPath), { recursive: true });
  const { stdout } = await runCommand(
    process.execPath,
    [
      ANALYZE_SPEAKERS_SCRIPT,
      "--input",
      audioPath,
      "--output",
      outputPath,
      "--participants",
      String(participants),
    ],
    { cwd: REPO_ROOT }
  );
  const resultLine = stdout
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .at(-1);
  if (!resultLine) throw new Error("Speaker analysis returned no result");
  return JSON.parse(resultLine);
}

export function createServer() {
  const server = new McpServer({
    name: "openwhispr-transcription",
    version: "1.1.0",
  });

  server.registerTool(
    "analyze_meeting_speakers",
    {
      title: "Analyser les intervenants d’un meeting",
      description:
        "Première étape locale et gratuite : diarise le fichier avec FluidAudio, compare les voix aux profils OpenWhispr existants et crée un extrait écoutable par groupe. Le JSON produit peut ensuite être passé à transcribe_meeting via diarization_json après validation des noms.",
      inputSchema: {
        audio_path: z.string().min(1).describe("Chemin absolu du fichier audio ou vidéo local"),
        output_path: z
          .string()
          .optional()
          .describe("Chemin du JSON d’analyse ; par défaut, à côté du fichier source"),
        output_dir: z
          .string()
          .optional()
          .describe("Dossier de sortie utilisé quand output_path est absent"),
        participants: z
          .union([z.number().int().min(1).max(20), z.literal("auto")])
          .optional()
          .describe("Nombre exact de voix ou détection automatique entre 1 et 8 ; défaut auto"),
      },
      outputSchema: {
        success: z.boolean(),
        analysis_path: z.string(),
        preview_directory: z.string(),
        detected_speakers: z.number().int(),
        speakers: z.array(
          z.object({
            id: z.string(),
            label: z.string(),
            speakingDurationSeconds: z.number(),
            turnCount: z.number().int(),
            previewPath: z.string().nullable(),
            previewStartSeconds: z.number().nullable(),
            previewEndSeconds: z.number().nullable(),
            previewCleanRatio: z.number().nullable(),
            match: z.object({
              state: z.string(),
              profileId: z.number().int().optional(),
              displayName: z.string().optional(),
              confidence: z.number(),
            }),
          })
        ),
      },
      annotations: {
        readOnlyHint: false,
        destructiveHint: false,
        idempotentHint: true,
        openWorldHint: false,
      },
    },
    async (request) => {
      try {
        const result = await runSpeakerAnalysis(request);
        return {
          content: [
            {
              type: "text",
              text: [
                `Analyse locale terminée : ${result.detected_speakers} groupe(s) de voix.`,
                `JSON réutilisable : ${result.analysis_path}`,
                `Extraits : ${result.preview_directory}`,
                "Écoutez les extraits, attribuez les noms, puis appelez transcribe_meeting avec diarization_json et speaker_labels.",
              ].join("\n"),
            },
          ],
          structuredContent: result,
        };
      } catch (error) {
        return {
          isError: true,
          content: [
            {
              type: "text",
              text: error instanceof Error ? error.message : String(error),
            },
          ],
        };
      }
    }
  );

  server.registerTool(
    "transcribe_meeting",
    {
      title: "Transcrire et diariser un meeting",
      description:
        "Transcrit un fichier audio local avec GPT-Transcribe, diarise localement avec FluidAudio, puis aligne les mots avec Nemotron. Par défaut, la commande est non interactive, n'écrit pas dans les profils vocaux, découpe les réunions longues près des silences et réutilise son cache OpenAI. Cette opération peut appeler l'API OpenAI et entraîner un coût.",
      inputSchema: {
        audio_path: z.string().min(1).describe("Chemin absolu du fichier audio ou vidéo local"),
        output_path: z
          .string()
          .optional()
          .describe("Chemin Markdown exact ; par défaut, à côté du fichier source"),
        output_dir: z
          .string()
          .optional()
          .describe("Dossier de sortie utilisé quand output_path est absent"),
        participants: z
          .union([z.number().int().min(1).max(20), z.literal("auto")])
          .optional()
          .describe("Nombre attendu de locuteurs ; défaut 2"),
        language: z.string().min(2).max(16).optional().describe("Code langue ; défaut fr"),
        speaker_labels: z
          .record(z.string(), z.string())
          .optional()
          .describe("Noms explicites, par exemple speaker_00 → Samuel"),
        speaker_references: z
          .record(z.string(), z.string())
          .optional()
          .describe("Références vocales nom → chemin audio"),
        persist_speaker_profiles: z
          .boolean()
          .optional()
          .describe("Enregistrer les profils nommés dans OpenWhispr ; défaut false"),
        transcript_json: z
          .string()
          .optional()
          .describe("Sidecar/cache GPT existant à réaligner sans nouvel appel API"),
        diarization_json: z
          .string()
          .optional()
          .describe("Sidecar existant dont réutiliser diarizationSegments"),
        openai_chunk_cache: z
          .string()
          .optional()
          .describe("Chemin du cache reprenable des parties OpenAI"),
        asr_provider: z
          .enum(["openai", "openrouter", "local"])
          .optional()
          .describe("Avancé ; défaut openai"),
        asr_model: z.string().optional().describe("Avancé ; défaut gpt-transcribe"),
        diarization_provider: z
          .enum(["fluidaudio", "sherpa"])
          .optional()
          .describe("Avancé ; défaut fluidaudio"),
        alignment_asr_model: z
          .string()
          .optional()
          .describe("Avancé ; défaut nemotron-3.5-asr-streaming-0.6b"),
      },
      outputSchema: {
        success: z.boolean(),
        markdown_path: z.string(),
        json_path: z.string(),
        chunk_cache_path: z.string(),
        asr_provider: z.string(),
        asr_model: z.string(),
        diarization_provider: z.string(),
        diarization_pipeline: z.string().nullable(),
        alignment_model: z.string().nullable(),
        alignment_ratio: z.number().nullable(),
        transcript_segments: z.number(),
        diarization_segments: z.number(),
        duration_seconds: z.number().nullable(),
        billed_seconds: z.number().nullable(),
        estimated_cost_usd: z.number().nullable(),
        speaker_names: z.array(z.string()),
        degraded: z.string().nullable(),
      },
      annotations: {
        readOnlyHint: false,
        destructiveHint: false,
        idempotentHint: false,
        openWorldHint: true,
      },
    },
    async (request) => {
      try {
        const result = await runTranscription(request);
        return {
          content: [
            {
              type: "text",
              text: [
                "Transcription terminée.",
                `Markdown: ${result.markdown_path}`,
                `JSON: ${result.json_path}`,
                `Pipeline: ${result.asr_model} + ${result.diarization_pipeline || result.diarization_provider}`,
                `Alignement: ${result.alignment_ratio ?? "n/a"}`,
                result.degraded
                  ? `Dégradation: ${result.degraded}`
                  : "Aucune dégradation signalée.",
              ].join("\n"),
            },
          ],
          structuredContent: result,
        };
      } catch (error) {
        return {
          isError: true,
          content: [
            {
              type: "text",
              text: error instanceof Error ? error.message : String(error),
            },
          ],
        };
      }
    }
  );

  return server;
}

async function main() {
  const server = createServer();
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error("OpenWhispr transcription MCP ready");
}

const isMain =
  process.argv[1] && path.resolve(process.argv[1]) === path.resolve(fileURLToPath(import.meta.url));

if (isMain) {
  main().catch((error) => {
    console.error(error instanceof Error ? error.stack || error.message : String(error));
    process.exit(1);
  });
}
