#!/usr/bin/env node
import fs from "node:fs";
import path from "node:path";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const FLUIDAUDIO_DIR = path.join(REPO_ROOT, ".cache/FluidAudio");
const FLUIDAUDIO_REPOSITORY = "https://github.com/FluidInference/FluidAudio.git";
const VERIFIED_COMMIT = "88d6d8166880dee1ac7c32c80f8e10cd782f8ca8";
const BINARY = path.join(FLUIDAUDIO_DIR, ".build/release/fluidaudiocli");
const SORTFORMER_SOURCE = path.join(
  FLUIDAUDIO_DIR,
  "Sources/FluidAudioCLI/Commands/SortformerCommand.swift"
);
const SORTFORMER_PATCH = path.join(
  REPO_ROOT,
  "patches/fluidaudio-sortformer-enrollment.patch"
);

function hasSortformerEnrollmentPatch() {
  return (
    fs.existsSync(SORTFORMER_SOURCE) &&
    fs.readFileSync(SORTFORMER_SOURCE, "utf8").includes('case "--enroll"')
  );
}

function run(command, args, cwd = REPO_ROOT) {
  return new Promise((resolve, reject) => {
    const proc = spawn(command, args, { cwd, stdio: "inherit" });
    proc.on("error", reject);
    proc.on("close", (code) => {
      if (code === 0) resolve();
      else reject(new Error(`${path.basename(command)} exited ${code}`));
    });
  });
}

async function main() {
  if (fs.existsSync(BINARY) && hasSortformerEnrollmentPatch()) {
    console.log(`FluidAudio is ready: ${BINARY}`);
    return;
  }
  if (!fs.existsSync(path.join(FLUIDAUDIO_DIR, ".git"))) {
    await run("git", [
      "clone",
      "--filter=blob:none",
      FLUIDAUDIO_REPOSITORY,
      FLUIDAUDIO_DIR,
    ]);
  }
  if (!fs.existsSync(SORTFORMER_SOURCE)) {
    await run("git", ["fetch", "--depth", "1", "origin", VERIFIED_COMMIT], FLUIDAUDIO_DIR);
    await run("git", ["checkout", "--detach", VERIFIED_COMMIT], FLUIDAUDIO_DIR);
  }
  if (!hasSortformerEnrollmentPatch()) {
    if (!fs.existsSync(SORTFORMER_PATCH)) {
      throw new Error(`Missing FluidAudio patch: ${SORTFORMER_PATCH}`);
    }
    await run("git", ["apply", SORTFORMER_PATCH], FLUIDAUDIO_DIR);
  }
  await run("swift", ["build", "-c", "release", "--product", "fluidaudiocli"], FLUIDAUDIO_DIR);
  if (!fs.existsSync(BINARY)) throw new Error(`FluidAudio build produced no CLI at ${BINARY}`);
  console.log(`FluidAudio is ready: ${BINARY}`);
}

main().catch((error) => {
  console.error(error.stack || error.message);
  process.exit(1);
});
