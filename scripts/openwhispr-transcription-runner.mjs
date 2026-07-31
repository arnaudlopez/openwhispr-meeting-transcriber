#!/usr/bin/env node
import { runTranscription } from "./openwhispr-transcription-mcp.mjs";

async function readStdin() {
  let input = "";
  for await (const chunk of process.stdin) input += chunk.toString();
  return input;
}

async function main() {
  const raw = await readStdin();
  if (!raw.trim()) throw new Error("Missing transcription request on stdin");
  const request = JSON.parse(raw);
  const result = await runTranscription(request);
  process.stdout.write(`${JSON.stringify(result)}\n`);
}

main().catch((error) => {
  console.error(error instanceof Error ? error.stack || error.message : String(error));
  process.exit(1);
});
