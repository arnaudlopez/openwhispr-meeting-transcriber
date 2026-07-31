#!/usr/bin/env node
import fs from "node:fs";

function parseArgs(argv) {
  const args = { requireNames: [] };
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === "--require-name") {
      const name = argv[i + 1];
      if (!name) throw new Error("--require-name requires a value");
      args.requireNames.push(name);
      i += 1;
      continue;
    }
    if (!arg.startsWith("--") && !args.filePath) args.filePath = arg;
  }
  return args;
}

const args = parseArgs(process.argv.slice(2));
const filePath = args.filePath;
if (!filePath) {
  console.error(
    "Usage: node scripts/verify-local-diarized-markdown.mjs <markdown-file> [--require-name <name>]"
  );
  process.exit(1);
}

if (!fs.existsSync(filePath)) {
  console.error(`Markdown artifact not found: ${filePath}`);
  process.exit(1);
}

const text = fs.readFileSync(filePath, "utf8");
const timestampMatches = text.match(/`\d{2}:\d{2}:\d{2}`/g) || [];
const speakerLineMatches = text.match(/\*\*[^*\n]+\*\*\s+`\d{2}:\d{2}:\d{2}`/g) || [];
const hasTitle = /^#\s+\S+/m.test(text);
const hasBodyText = text.split(/\s+/).filter(Boolean).length > 40;

const failures = [];
if (!hasTitle) failures.push("missing title");
if (timestampMatches.length < 2) failures.push("fewer than two timestamps");
if (speakerLineMatches.length < 1) failures.push("missing speaker label");
if (!hasBodyText) failures.push("artifact has too little body text");
for (const name of args.requireNames) {
  if (!text.includes(`**${name}**`)) failures.push(`missing required speaker name: ${name}`);
}

if (failures.length) {
  console.error(`Verification failed: ${failures.join(", ")}`);
  process.exit(1);
}

console.log(
  JSON.stringify(
    {
      success: true,
      filePath,
      timestamps: timestampMatches.length,
      speakerLabels: new Set(speakerLineMatches.map((line) => line.replace(/\s+`.*/, ""))).size,
      requiredNames: args.requireNames,
      bytes: fs.statSync(filePath).size,
    },
    null,
    2
  )
);
