import fs from "node:fs";
import fsp from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const DEFAULT_FLUIDAUDIO_CANDIDATES = [
  process.env.FLUIDAUDIO_CLI_PATH,
  path.join(REPO_ROOT, ".cache/FluidAudio/.build/release/fluidaudiocli"),
  path.join(os.homedir(), ".cache/openwhispr/FluidAudio/.build/release/fluidaudiocli"),
].filter(Boolean);

function run(command, args) {
  return new Promise((resolve, reject) => {
    const proc = spawn(command, args, { stdio: ["ignore", "pipe", "pipe"] });
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
        reject(new Error(`FluidAudio exited ${code}: ${(stderr || stdout).slice(-1600)}`));
        return;
      }
      resolve({ stdout, stderr });
    });
  });
}

function findFluidAudioCli(explicitPath) {
  const candidates = explicitPath
    ? [path.resolve(explicitPath), ...DEFAULT_FLUIDAUDIO_CANDIDATES]
    : DEFAULT_FLUIDAUDIO_CANDIDATES;
  const found = candidates.find((candidate) => fs.existsSync(candidate));
  if (!found) {
    throw new Error(
      "FluidAudio CLI not found. Build it once or set FLUIDAUDIO_CLI_PATH."
    );
  }
  return found;
}

function parseReferencePaths(value) {
  if (!value || value === true) return [];
  const text = String(value).trim();
  if (!text) return [];
  const entries = text.startsWith("{")
    ? Object.entries(JSON.parse(text))
    : text.split(",").map((entry) => {
        const [name, ...fileParts] = entry.split("=");
        return [name, fileParts.join("=")];
      });
  return entries
    .map(([name, filePath]) => ({
      name: String(name || "").trim(),
      path: path.resolve(String(filePath || "").trim()),
    }))
    .filter((reference) => reference.name && reference.path);
}

function speakerIdForName(speakerLabels, name) {
  for (const [speakerId, displayName] of speakerLabels || []) {
    if (displayName !== name || !/^speaker_0*\d+$/.test(speakerId)) continue;
    const match = speakerId.match(/^speaker_0*(\d+)$/);
    return `speaker_${Number(match[1])}`;
  }
  return null;
}

function vectorCentroid(segments) {
  const valid = (segments || []).filter(
    (segment) =>
      Array.isArray(segment.embedding) &&
      segment.embedding.length &&
      segment.embedding.every(Number.isFinite)
  );
  if (!valid.length) return null;
  const result = new Float64Array(valid[0].embedding.length);
  let totalWeight = 0;
  for (const segment of valid) {
    const duration = Number(segment.endTimeSeconds) - Number(segment.startTimeSeconds);
    const weight = Math.max(0.05, Number.isFinite(duration) ? duration : 1);
    totalWeight += weight;
    for (let index = 0; index < result.length; index += 1) {
      result[index] += weight * segment.embedding[index];
    }
  }
  for (let index = 0; index < result.length; index += 1) result[index] /= totalWeight;
  return result;
}

function cosineSimilarity(left, right) {
  if (!left || !right || left.length !== right.length) return -1;
  let dot = 0;
  let leftNorm = 0;
  let rightNorm = 0;
  for (let index = 0; index < left.length; index += 1) {
    dot += left[index] * right[index];
    leftNorm += left[index] * left[index];
    rightNorm += right[index] * right[index];
  }
  const denominator = Math.sqrt(leftNorm) * Math.sqrt(rightNorm);
  return denominator ? dot / denominator : -1;
}

function bestOneToOneAssignment(clusters, references, scores) {
  let best = { score: Number.NEGATIVE_INFINITY, pairs: [] };
  const visit = (referenceIndex, usedClusters, total, pairs) => {
    if (referenceIndex >= references.length) {
      if (total > best.score) best = { score: total, pairs: [...pairs] };
      return;
    }
    for (const cluster of clusters) {
      if (usedClusters.has(cluster)) continue;
      usedClusters.add(cluster);
      pairs.push([cluster, references[referenceIndex].name]);
      visit(
        referenceIndex + 1,
        usedClusters,
        total + scores[cluster][references[referenceIndex].name],
        pairs
      );
      pairs.pop();
      usedClusters.delete(cluster);
    }
  };
  visit(0, new Set(), 0, []);
  return best;
}

async function processWithFluidAudio({
  cliPath,
  audioPath,
  outputPath,
  speakers,
  minimumSpeakers = 1,
  maximumSpeakers = 8,
  minimumSegmentSeconds,
}) {
  const args = [
    "process",
    audioPath,
    "--mode",
    "offline",
    "--min-segment-duration",
    String(minimumSegmentSeconds),
    "--output",
    outputPath,
  ];
  if (speakers === "auto") {
    args.push(
      "--min-speakers",
      String(minimumSpeakers),
      "--max-speakers",
      String(maximumSpeakers)
    );
  } else {
    args.push("--num-speakers", String(speakers));
  }
  await run(cliPath, args);
  return JSON.parse(await fsp.readFile(outputPath, "utf8"));
}

async function createSortformerEnrollmentClip({
  wavPath,
  outputPath,
  segments,
  targetSeconds = 38,
}) {
  const candidates = (segments || [])
    .map((segment) => ({
      start: Number(segment.start),
      end: Number(segment.end),
      duration: Number(segment.end) - Number(segment.start),
    }))
    .filter(
      (segment) =>
        Number.isFinite(segment.start) &&
        Number.isFinite(segment.end) &&
        segment.duration >= 4
    )
    .sort((left, right) => right.duration - left.duration);
  const selected = [];
  let remaining = targetSeconds;
  for (const candidate of candidates) {
    if (remaining <= 0) break;
    const available = Math.max(0, candidate.duration - 0.5);
    const duration = Math.min(available, remaining);
    if (duration < 3) continue;
    const start = candidate.start + Math.max(0.25, (candidate.duration - duration) / 2);
    selected.push({ start, end: start + duration, duration });
    remaining -= duration;
  }
  const selectedDuration = selected.reduce(
    (total, segment) => total + segment.duration,
    0
  );
  if (selectedDuration < 31) {
    throw new Error(
      `Sortformer enrollment needs 31 seconds of clean speech; only ${selectedDuration.toFixed(1)} seconds were available`
    );
  }

  if (selected.length === 1) {
    await run("ffmpeg", [
      "-hide_banner",
      "-loglevel",
      "error",
      "-i",
      wavPath,
      "-ss",
      selected[0].start.toFixed(3),
      "-t",
      selected[0].duration.toFixed(3),
      "-ar",
      "16000",
      "-ac",
      "1",
      "-c:a",
      "pcm_s16le",
      "-y",
      outputPath,
    ]);
  } else {
    const filters = selected.map(
      (segment, index) =>
        `[0:a]atrim=start=${segment.start.toFixed(3)}:end=${segment.end.toFixed(3)},asetpts=PTS-STARTPTS[a${index}]`
    );
    filters.push(
      `${selected.map((_segment, index) => `[a${index}]`).join("")}concat=n=${selected.length}:v=0:a=1[out]`
    );
    await run("ffmpeg", [
      "-hide_banner",
      "-loglevel",
      "error",
      "-i",
      wavPath,
      "-filter_complex",
      filters.join(";"),
      "-map",
      "[out]",
      "-ar",
      "16000",
      "-ac",
      "1",
      "-c:a",
      "pcm_s16le",
      "-y",
      outputPath,
    ]);
  }
  return {
    path: outputPath,
    durationSeconds: selectedDuration,
    sourceSegments: selected,
  };
}

function mappedCommunitySegments(meetingSegments, speakerMap) {
  return meetingSegments.map((segment) => ({
    start: Number(segment.startTimeSeconds),
    end: Number(segment.endTimeSeconds),
    speaker: speakerMap.get(segment.speakerId),
  }));
}

function resolveSortformerUnknownSpeaker(
  communitySegments,
  knownSortformerSegments,
  referenceSpeakerIds,
  start,
  end
) {
  const overlappingKnown = new Map();
  for (const segment of knownSortformerSegments) {
    const overlap = Math.max(
      0,
      Math.min(end, segment.end) - Math.max(start, segment.start)
    );
    if (overlap > 0) {
      overlappingKnown.set(
        segment.speaker,
        (overlappingKnown.get(segment.speaker) || 0) + overlap
      );
    }
  }
  if (overlappingKnown.size === 1 && referenceSpeakerIds.length === 2) {
    const activeSpeaker = [...overlappingKnown.keys()][0];
    const otherSpeaker = referenceSpeakerIds.find(
      (speakerId) => speakerId !== activeSpeaker
    );
    if (otherSpeaker) return otherSpeaker;
  }

  let bestSpeaker = null;
  let bestOverlap = 0;
  for (const segment of communitySegments) {
    const overlap = Math.max(
      0,
      Math.min(end, segment.end) - Math.max(start, segment.start)
    );
    if (overlap > bestOverlap) {
      bestOverlap = overlap;
      bestSpeaker = segment.speaker;
    }
  }
  if (bestSpeaker) return bestSpeaker;
  const midpoint = start + (end - start) / 2;
  return communitySegments
    .map((segment) => ({
      speaker: segment.speaker,
      distance:
        midpoint < segment.start
          ? segment.start - midpoint
          : midpoint > segment.end
            ? midpoint - segment.end
            : 0,
    }))
    .sort((left, right) => left.distance - right.distance)[0]?.speaker;
}

async function processSortformerEnrolled({
  cliPath,
  wavPath,
  outputPath,
  references,
  communitySegments,
  minimumSegmentSeconds,
}) {
  const enrollmentClips = [];
  for (let index = 0; index < references.length; index += 1) {
    const reference = references[index];
    const clip = await createSortformerEnrollmentClip({
      wavPath,
      outputPath: path.join(
        path.dirname(outputPath),
        `sortformer-enrollment-${index}.wav`
      ),
      segments: communitySegments.filter(
        (segment) => segment.speaker === reference.speakerId
      ),
    });
    enrollmentClips.push({ ...clip, name: reference.name, speakerId: reference.speakerId });
  }

  const args = ["sortformer", wavPath, "--config", "high"];
  for (const clip of enrollmentClips) {
    args.push("--enroll", `${clip.name}=${clip.path}`);
  }
  args.push("--output", outputPath);
  await run(cliPath, args);
  const result = JSON.parse(await fsp.readFile(outputPath, "utf8"));
  const speakerIdsByName = new Map(
    references.map((reference) => [reference.name, reference.speakerId])
  );
  const rawSegments = result.segments || [];
  const knownSortformerSegments = rawSegments
    .map((segment) => ({
      start: Number(segment.startTimeSeconds),
      end: Number(segment.endTimeSeconds),
      speaker: speakerIdsByName.get(String(segment.speaker || "")),
    }))
    .filter((segment) => segment.speaker);
  const referenceSpeakerIds = [
    ...new Set(references.map((reference) => reference.speakerId)),
  ];
  const segments = rawSegments
    .map((segment) => {
      const start = Number(segment.startTimeSeconds);
      const end = Number(segment.endTimeSeconds);
      const knownSpeaker = speakerIdsByName.get(String(segment.speaker || ""));
      return {
        start,
        end,
        speaker:
          knownSpeaker ||
          resolveSortformerUnknownSpeaker(
            communitySegments,
            knownSortformerSegments,
            referenceSpeakerIds,
            start,
            end
          ),
      };
    })
    .filter(
      (segment) =>
        Number.isFinite(segment.start) &&
        Number.isFinite(segment.end) &&
        segment.end - segment.start >= minimumSegmentSeconds &&
        segment.speaker
    )
    .map((segment) => ({
      start: Number(segment.start.toFixed(3)),
      end: Number(segment.end.toFixed(3)),
      speaker: segment.speaker,
    }));
  return {
    segments,
    metadata: {
      processingTimeSeconds: result.processingTimeSeconds,
      realTimeFactor: result.rtfx,
      rawSegments: result.segmentCount,
      retainedSegments: segments.length,
      enrollmentClips: enrollmentClips.map((clip) => ({
        name: clip.name,
        speakerId: clip.speakerId,
        durationSeconds: clip.durationSeconds,
        sourceSegments: clip.sourceSegments,
      })),
    },
  };
}

export async function diarizeWithFluidAudio({
  wavPath,
  workDir,
  expectedSpeakers,
  knownSpeakerReferencePaths,
  speakerLabels = new Map(),
  cliPath,
  minimumSegmentSeconds = 0.3,
}) {
  const binary = findFluidAudioCli(cliPath);
  const parsedSpeakers = Number.parseInt(expectedSpeakers, 10);
  const automaticSpeakerCount = String(expectedSpeakers).toLowerCase() === "auto";
  const speakerCount =
    !automaticSpeakerCount && Number.isFinite(parsedSpeakers) && parsedSpeakers > 0
      ? parsedSpeakers
      : "auto";
  const maximumSpeakers = 8;
  const meetingOutput = path.join(workDir, "fluidaudio-meeting.json");
  console.log(
    `[Diarization FluidAudio] Community-1/VBx, ${
      speakerCount === "auto" ? `automatic count (1-${maximumSpeakers})` : `${speakerCount} speakers`
    }`
  );
  const meeting = await processWithFluidAudio({
    cliPath: binary,
    audioPath: wavPath,
    outputPath: meetingOutput,
    speakers: speakerCount,
    minimumSpeakers: 1,
    maximumSpeakers,
    minimumSegmentSeconds,
  });

  const references = parseReferencePaths(knownSpeakerReferencePaths).slice(
    0,
    speakerCount === "auto" ? maximumSpeakers : speakerCount
  );
  for (const reference of references) {
    if (!fs.existsSync(reference.path)) {
      throw new Error(`Known speaker reference not found: ${reference.path}`);
    }
    const outputPath = path.join(
      workDir,
      `fluidaudio-reference-${references.indexOf(reference)}.json`
    );
    reference.result = await processWithFluidAudio({
      cliPath: binary,
      audioPath: reference.path,
      outputPath,
      speakers: 1,
      minimumSegmentSeconds,
    });
    reference.embedding = vectorCentroid(reference.result.segments);
    reference.speakerId = speakerIdForName(speakerLabels, reference.name);
  }

  const clusters = [...new Set(meeting.segments.map((segment) => segment.speakerId))];
  const clusterEmbeddings = Object.fromEntries(
    clusters.map((cluster) => [
      cluster,
      vectorCentroid(
        meeting.segments.filter(
          (segment) =>
            segment.speakerId === cluster &&
            Number(segment.endTimeSeconds) - Number(segment.startTimeSeconds) >= 0.8
        )
      ) ||
        vectorCentroid(meeting.segments.filter((segment) => segment.speakerId === cluster)),
    ])
  );
  const scores = Object.fromEntries(
    clusters.map((cluster) => [
      cluster,
      Object.fromEntries(
        references.map((reference) => [
          reference.name,
          cosineSimilarity(clusterEmbeddings[cluster], reference.embedding),
        ])
      ),
    ])
  );

  const assignment =
    references.length && clusters.length >= references.length
      ? bestOneToOneAssignment(clusters, references, scores)
      : { score: 0, pairs: [] };
  const speakerMap = new Map();
  for (const [cluster, name] of assignment.pairs) {
    const reference = references.find((candidate) => candidate.name === name);
    speakerMap.set(
      cluster,
      reference?.speakerId || `speaker_${references.indexOf(reference)}`
    );
  }
  let fallbackIndex = 0;
  for (const cluster of clusters) {
    if (speakerMap.has(cluster)) continue;
    while ([...speakerMap.values()].includes(`speaker_${fallbackIndex}`)) fallbackIndex += 1;
    speakerMap.set(cluster, `speaker_${fallbackIndex}`);
    fallbackIndex += 1;
  }

  const communitySegments = mappedCommunitySegments(meeting.segments, speakerMap);
  let finalSegments = communitySegments;
  let pipeline = "community-1-wespeaker-vbx";
  let sortformerMetadata = null;
  if (
    speakerCount !== "auto" &&
    references.length === speakerCount &&
    references.every((reference) => reference.speakerId)
  ) {
    try {
      console.log(
        `[Diarization FluidAudio] Sortformer high-context enrollment, ${references.length} known speakers`
      );
      const sortformer = await processSortformerEnrolled({
        cliPath: binary,
        wavPath,
        outputPath: path.join(workDir, "fluidaudio-sortformer-enrolled.json"),
        references,
        communitySegments,
        minimumSegmentSeconds,
      });
      finalSegments = sortformer.segments;
      pipeline = "community-1-vbx+sortformer-high-enrolled";
      sortformerMetadata = sortformer.metadata;
    } catch (error) {
      console.warn(
        `[Diarization FluidAudio] Sortformer enrollment unavailable; keeping Community-1 (${error.message})`
      );
    }
  }

  return {
    segments: finalSegments.map((segment) => ({
      start: Number(Number(segment.start).toFixed(3)),
      end: Number(Number(segment.end).toFixed(3)),
      speaker: segment.speaker,
    })),
    metadata: {
      provider: "fluidaudio",
      pipeline,
      binary,
      processingTimeSeconds: meeting.processingTimeSeconds,
      realTimeFactor: meeting.realTimeFactor,
      detectedSpeakers: meeting.speakerCount,
      requestedSpeakers: speakerCount,
      automaticSpeakerRange:
        speakerCount === "auto" ? { minimum: 1, maximum: maximumSpeakers } : null,
      rawSpeakerMap: Object.fromEntries(speakerMap),
      referenceScores: scores,
      referenceAssignmentScore: assignment.score,
      sortformer: sortformerMetadata,
      references: references.map((reference) => ({
        name: reference.name,
        speakerId: reference.speakerId,
        path: reference.path,
      })),
    },
  };
}
