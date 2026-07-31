import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import { buildTranscriptionCommand, DEFAULT_PIPELINE } from "./openwhispr-transcription-mcp.mjs";

const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url));
const SERVER_SCRIPT = path.join(SCRIPT_DIR, "openwhispr-transcription-mcp.mjs");

function valueAfter(args, flag) {
  const index = args.indexOf(flag);
  return index === -1 ? null : args[index + 1];
}

test("MCP transcription uses the hybrid pipeline by default", () => {
  const audioPath = path.join(os.tmpdir(), "openwhispr-mcp-test.wav");
  fs.writeFileSync(audioPath, "placeholder");
  try {
    const spec = buildTranscriptionCommand({ audio_path: audioPath });
    assert.equal(valueAfter(spec.args, "--asr-provider"), DEFAULT_PIPELINE.asrProvider);
    assert.equal(valueAfter(spec.args, "--asr-model"), DEFAULT_PIPELINE.asrModel);
    assert.equal(
      valueAfter(spec.args, "--diarization-provider"),
      DEFAULT_PIPELINE.diarizationProvider
    );
    assert.equal(
      valueAfter(spec.args, "--alignment-asr-model"),
      DEFAULT_PIPELINE.alignmentAsrModel
    );
    assert.ok(spec.args.includes("--no-db-write"));
    assert.ok(spec.args.includes("--no-learn"));
  } finally {
    fs.rmSync(audioPath, { force: true });
  }
});

test("automatic output keeps the exact source filename in the selected directory", () => {
  const audioPath = path.join(os.tmpdir(), "Réunion Client 01.audio.mp4");
  const outputDir = path.join(os.tmpdir(), "openwhispr named output");
  fs.writeFileSync(audioPath, "placeholder");
  try {
    const spec = buildTranscriptionCommand({
      audio_path: audioPath,
      output_dir: outputDir,
    });
    assert.equal(
      valueAfter(spec.args, "--output"),
      path.join(outputDir, "Réunion Client 01.audio.md")
    );
  } finally {
    fs.rmSync(audioPath, { force: true });
  }
});

test("speaker labels are applied without persistence and never trigger a prompt", () => {
  const audioPath = path.join(os.tmpdir(), "openwhispr-mcp-label-test.wav");
  fs.writeFileSync(audioPath, "placeholder");
  try {
    const spec = buildTranscriptionCommand({
      audio_path: audioPath,
      speaker_labels: { speaker_00: "Samuel", speaker_01: "Arnaud Lopez" },
    });
    assert.deepEqual(JSON.parse(valueAfter(spec.args, "--speaker-labels")), {
      speaker_00: "Samuel",
      speaker_01: "Arnaud Lopez",
    });
    assert.ok(spec.args.includes("--no-db-write"));
    assert.ok(!spec.args.includes("--no-learn"));
  } finally {
    fs.rmSync(audioPath, { force: true });
  }
});

test("alternative OpenAI text models retain local Nemotron alignment", () => {
  const audioPath = path.join(os.tmpdir(), "openwhispr-mcp-model-test.wav");
  fs.writeFileSync(audioPath, "placeholder");
  try {
    const spec = buildTranscriptionCommand({
      audio_path: audioPath,
      asr_model: "gpt-4o-transcribe",
    });
    assert.equal(valueAfter(spec.args, "--asr-model"), "gpt-4o-transcribe");
    assert.equal(
      valueAfter(spec.args, "--alignment-asr-model"),
      DEFAULT_PIPELINE.alignmentAsrModel
    );
  } finally {
    fs.rmSync(audioPath, { force: true });
  }
});

test("profile persistence requires explicit labels", () => {
  const audioPath = path.join(os.tmpdir(), "openwhispr-mcp-persist-test.wav");
  fs.writeFileSync(audioPath, "placeholder");
  try {
    assert.throws(
      () =>
        buildTranscriptionCommand({
          audio_path: audioPath,
          persist_speaker_profiles: true,
        }),
      /requires speaker_labels/
    );
  } finally {
    fs.rmSync(audioPath, { force: true });
  }
});

test("a real MCP client discovers the staged speaker workflow", { timeout: 10_000 }, async () => {
  const transport = new StdioClientTransport({
    command: process.execPath,
    args: [SERVER_SCRIPT],
    cwd: path.resolve(SCRIPT_DIR, ".."),
    stderr: "pipe",
  });
  const client = new Client({
    name: "openwhispr-transcription-test",
    version: "1.0.0",
  });
  try {
    await client.connect(transport);
    const listed = await client.listTools();
    const transcriptionTool = listed.tools.find(
      (candidate) => candidate.name === "transcribe_meeting"
    );
    const analysisTool = listed.tools.find(
      (candidate) => candidate.name === "analyze_meeting_speakers"
    );
    assert.ok(transcriptionTool);
    assert.ok(analysisTool);
    assert.equal(
      transcriptionTool.inputSchema.properties.asr_model.description,
      "Avancé ; défaut gpt-transcribe"
    );
    assert.match(
      analysisTool.inputSchema.properties.participants.description,
      /détection automatique/
    );
  } finally {
    await client.close();
  }
});
