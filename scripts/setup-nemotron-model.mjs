#!/usr/bin/env node
import { createHash } from "node:crypto";
import { createReadStream, createWriteStream } from "node:fs";
import fsp from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";

const MODEL_NAME = "nemotron-3.5-asr-streaming-0.6b";
const REPOSITORY = "csukuangfj2/sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11";
const REVISION = "ab43d895f5985b1bbab8b6eac8607fcdc05343f3";
const DESTINATION = path.join(os.homedir(), ".cache/openwhispr/parakeet-models", MODEL_NAME);
const FILES = [
  {
    name: "tokens.txt",
    sha256: "729cc103155bafa785f9cd45746cd41cabe97eab7182fc04d594129587958f8a",
  },
  {
    name: "encoder.int8.onnx",
    sha256: "012e9321373af99021415e0b0eb3ec827b4be3153be6f30d9b448fe65e896e68",
  },
  {
    name: "decoder.int8.onnx",
    sha256: "19f9c98fc6d0a2c33a65a43b36fdb2e914c26c0aa9764be3aebc502a1e982fb0",
  },
  {
    name: "joiner.int8.onnx",
    sha256: "4101c7c679a0bc30483794b27a059e34e79232aa2068d78d51231a22c8b0d7ce",
  },
];

function sha256(filePath) {
  return new Promise((resolve, reject) => {
    const hash = createHash("sha256");
    const stream = createReadStream(filePath);
    stream.on("error", reject);
    stream.on("data", (chunk) => hash.update(chunk));
    stream.on("end", () => resolve(hash.digest("hex")));
  });
}

async function isValid(filePath, expected) {
  try {
    return (await sha256(filePath)) === expected;
  } catch (error) {
    if (error?.code === "ENOENT") return false;
    throw error;
  }
}

async function downloadFile(file) {
  const destination = path.join(DESTINATION, file.name);
  if (await isValid(destination, file.sha256)) {
    console.log(`Nemotron ready: ${file.name}`);
    return;
  }

  const temporary = `${destination}.part-${process.pid}`;
  const url = `https://huggingface.co/${REPOSITORY}/resolve/${REVISION}/${file.name}?download=true`;
  console.log(`Downloading ${file.name}...`);
  const response = await fetch(url, { redirect: "follow" });
  if (!response.ok || !response.body) {
    throw new Error(`Nemotron download failed for ${file.name}: HTTP ${response.status}`);
  }
  try {
    await pipeline(Readable.fromWeb(response.body), createWriteStream(temporary));
    const actual = await sha256(temporary);
    if (actual !== file.sha256) {
      throw new Error(
        `Checksum mismatch for ${file.name}: expected ${file.sha256}, received ${actual}`
      );
    }
    await fsp.rename(temporary, destination);
  } catch (error) {
    await fsp.rm(temporary, { force: true }).catch(() => {});
    throw error;
  }
}

async function main() {
  await fsp.mkdir(DESTINATION, { recursive: true });
  for (const file of FILES) await downloadFile(file);
  console.log(`Nemotron model ready: ${DESTINATION}`);
}

main().catch((error) => {
  console.error(error.stack || error.message);
  process.exit(1);
});
