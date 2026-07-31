const WORD_PATTERN = /[\p{L}\p{N}]+(?:['’][\p{L}\p{N}]+)*/gu;
const DEFAULT_ALIGNMENT_BAND = 180;

function tokenize(text) {
  const tokens = [];
  for (const match of String(text || "").matchAll(WORD_PATTERN)) {
    tokens.push({
      raw: match[0],
      normalized: normalizeWord(match[0]),
      start: match.index,
      end: match.index + match[0].length,
    });
  }
  return tokens;
}

function normalizeWord(word) {
  return String(word || "")
    .normalize("NFKD")
    .replace(/\p{M}/gu, "")
    .replace(/[’']/g, "")
    .toLocaleLowerCase("fr");
}

function editDistance(left, right) {
  if (left === right) return 0;
  if (!left.length) return right.length;
  if (!right.length) return left.length;
  let previous = new Uint16Array(right.length + 1);
  let current = new Uint16Array(right.length + 1);
  for (let j = 0; j <= right.length; j += 1) previous[j] = j;
  for (let i = 1; i <= left.length; i += 1) {
    current[0] = i;
    for (let j = 1; j <= right.length; j += 1) {
      current[j] = Math.min(
        previous[j] + 1,
        current[j - 1] + 1,
        previous[j - 1] + (left[i - 1] === right[j - 1] ? 0 : 1)
      );
    }
    [previous, current] = [current, previous];
  }
  return previous[right.length];
}

function wordSimilarity(left, right) {
  if (!left || !right) return 0;
  if (left === right) return 1;
  const longest = Math.max(left.length, right.length);
  return 1 - editDistance(left, right) / longest;
}

function substitutionCost(left, right) {
  const similarity = wordSimilarity(left.normalized, right.normalized);
  if (similarity === 1) return 0;
  if (similarity >= 0.72) return 0.35 + (1 - similarity);
  if (similarity >= 0.5 && Math.max(left.normalized.length, right.normalized.length) >= 5) {
    return 0.85;
  }
  return 1.25;
}

function buildRoughTokens(segments) {
  const result = [];
  for (const segment of segments || []) {
    const words = tokenize(segment.text);
    if (!words.length) continue;
    const start = Number(segment.timestamp);
    const fallbackEnd = start + Math.max(0.6, words.length * 0.32);
    let end = Number.isFinite(Number(segment.endTimestamp))
      ? Number(segment.endTimestamp)
      : fallbackEnd;
    if (String(segment.asrModel || "").startsWith("nemotron")) {
      end = Math.max(start + 0.1, end - 0.6);
    }
    for (let index = 0; index < words.length; index += 1) {
      result.push({
        ...words[index],
        time: start + ((index + 0.5) / words.length) * Math.max(0.1, end - start),
        segmentId: segment.id || null,
      });
    }
  }
  return result;
}

function bandedAlignment(roughTokens, finalTokens, requestedBand = DEFAULT_ALIGNMENT_BAND) {
  const roughCount = roughTokens.length;
  const finalCount = finalTokens.length;
  if (!roughCount || !finalCount) return new Map();

  const band = Math.max(
    requestedBand,
    Math.ceil(Math.abs(finalCount - roughCount) * 0.08) + 40
  );
  const rows = new Array(roughCount + 1);
  const rowZeroMax = Math.min(finalCount, band);
  let previous = new Float64Array(rowZeroMax + 1);
  const rowZeroDirections = new Uint8Array(rowZeroMax + 1);
  for (let j = 0; j <= rowZeroMax; j += 1) {
    previous[j] = j;
    if (j > 0) rowZeroDirections[j] = 3;
  }
  rows[0] = { min: 0, directions: rowZeroDirections };
  let previousMin = 0;

  for (let i = 1; i <= roughCount; i += 1) {
    const center = Math.round((i * finalCount) / roughCount);
    const min = Math.max(0, center - band);
    const max = Math.min(finalCount, center + band);
    const current = new Float64Array(max - min + 1);
    current.fill(Number.POSITIVE_INFINITY);
    const directions = new Uint8Array(current.length);

    for (let j = min; j <= max; j += 1) {
      const index = j - min;
      let best = Number.POSITIVE_INFINITY;
      let direction = 0;

      if (j >= previousMin && j < previousMin + previous.length) {
        const cost = previous[j - previousMin] + 1;
        if (cost < best) {
          best = cost;
          direction = 2;
        }
      }
      if (j > min && current[index - 1] + 1 < best) {
        best = current[index - 1] + 1;
        direction = 3;
      }
      if (
        j > 0 &&
        j - 1 >= previousMin &&
        j - 1 < previousMin + previous.length
      ) {
        const cost =
          previous[j - 1 - previousMin] +
          substitutionCost(roughTokens[i - 1], finalTokens[j - 1]);
        if (cost <= best) {
          best = cost;
          direction = 1;
        }
      }
      current[index] = best;
      directions[index] = direction;
    }

    rows[i] = { min, directions };
    previous = current;
    previousMin = min;
  }

  const mapping = new Map();
  let i = roughCount;
  let j = finalCount;
  while (i > 0 || j > 0) {
    const row = rows[i];
    const direction =
      row && j >= row.min && j < row.min + row.directions.length
        ? row.directions[j - row.min]
        : i > 0
          ? 2
          : 3;
    if (direction === 1) {
      const similarity = wordSimilarity(
        roughTokens[i - 1].normalized,
        finalTokens[j - 1].normalized
      );
      if (
        similarity >= 0.5 ||
        roughTokens[i - 1].normalized === finalTokens[j - 1].normalized
      ) {
        mapping.set(j - 1, i - 1);
      }
      i -= 1;
      j -= 1;
    } else if (direction === 2 && i > 0) {
      i -= 1;
    } else if (j > 0) {
      j -= 1;
    } else {
      break;
    }
  }
  return mapping;
}

function nearestSpeaker(diarization, time) {
  let active = null;
  let activeDuration = Number.POSITIVE_INFINITY;
  let nearest = null;
  let nearestDistance = Number.POSITIVE_INFINITY;
  for (const segment of diarization || []) {
    const start = Number(segment.start);
    const end = Number(segment.end);
    if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) continue;
    if (time >= start && time <= end) {
      const duration = end - start;
      if (duration < activeDuration) {
        active = segment.speaker;
        activeDuration = duration;
      }
      continue;
    }
    const distance = time < start ? start - time : time - end;
    if (distance < nearestDistance) {
      nearest = segment.speaker;
      nearestDistance = distance;
    }
  }
  return active || nearest || "speaker_0";
}

function speakerByIntervalOverlap(diarization, start, end) {
  const scores = new Map();
  for (const segment of diarization || []) {
    const segmentStart = Number(segment.start);
    const segmentEnd = Number(segment.end);
    if (
      !Number.isFinite(segmentStart) ||
      !Number.isFinite(segmentEnd) ||
      segmentEnd <= segmentStart
    ) {
      continue;
    }
    const overlap = Math.max(
      0,
      Math.min(end, segmentEnd) - Math.max(start, segmentStart)
    );
    if (overlap > 0) {
      scores.set(segment.speaker, (scores.get(segment.speaker) || 0) + overlap);
    }
  }
  const ranked = [...scores].sort((left, right) => right[1] - left[1]);
  if (
    ranked.length > 1 &&
    Math.abs(ranked[0][1] - ranked[1][1]) <= 0.02
  ) {
    return nearestSpeaker(diarization, start + (end - start) / 2);
  }
  return ranked[0]?.[0] || nearestSpeaker(diarization, start + (end - start) / 2);
}

function interpolateTimes(times, startTime, endTime) {
  const known = [];
  for (let index = 0; index < times.length; index += 1) {
    if (Number.isFinite(times[index])) known.push(index);
  }
  if (!known.length) {
    for (let index = 0; index < times.length; index += 1) {
      times[index] = startTime + ((index + 0.5) / Math.max(1, times.length)) * (endTime - startTime);
    }
    return;
  }

  const first = known[0];
  for (let index = 0; index < first; index += 1) {
    times[index] =
      startTime + ((index + 1) / (first + 1)) * Math.max(0, times[first] - startTime);
  }
  for (let anchor = 0; anchor < known.length - 1; anchor += 1) {
    const left = known[anchor];
    const right = known[anchor + 1];
    for (let index = left + 1; index < right; index += 1) {
      times[index] =
        times[left] +
        ((index - left) / (right - left)) * Math.max(0, times[right] - times[left]);
    }
  }
  const last = known.at(-1);
  for (let index = last + 1; index < times.length; index += 1) {
    times[index] =
      times[last] +
      ((index - last) / (times.length - last)) * Math.max(0, endTime - times[last]);
  }
  for (let index = 1; index < times.length; index += 1) {
    times[index] = Math.max(times[index], times[index - 1]);
  }
}

function sentenceRanges(text) {
  const ranges = [];
  let start = 0;
  let forceSpeakerChange = false;
  const boundary =
    /[.!?…]+(?:["»”')\]]+)?(?=\s+\p{Lu})|\s+—\s+| {2,}/gu;
  for (const match of text.matchAll(boundary)) {
    let end = match.index + match[0].length;
    while (end < text.length && /\s/u.test(text[end])) end += 1;
    ranges.push({ start, end, forceSpeakerChange });
    forceSpeakerChange = /—| {2,}/u.test(match[0]);
    start = end;
  }
  if (start < text.length) {
    ranges.push({ start, end: text.length, forceSpeakerChange });
  }
  return ranges;
}

export function alignTranscriptToDiarization({
  text,
  roughSegments,
  diarization,
  asrModel = "gpt-transcribe",
  alignmentModel = "local",
  band = DEFAULT_ALIGNMENT_BAND,
  windowStart,
  windowEnd,
}) {
  const finalText = String(text || "").trim();
  const finalTokens = tokenize(finalText);
  const roughTokens = buildRoughTokens(roughSegments);
  if (!finalTokens.length) {
    return {
      segments: [],
      metadata: { finalWords: 0, roughWords: roughTokens.length, matchedWords: 0 },
    };
  }

  const mapping = bandedAlignment(roughTokens, finalTokens, band);
  const times = new Array(finalTokens.length).fill(null);
  for (const [finalIndex, roughIndex] of mapping) {
    times[finalIndex] = roughTokens[roughIndex].time;
  }
  const diarizationStarts = (diarization || []).map((segment) => Number(segment.start)).filter(Number.isFinite);
  const diarizationEnds = (diarization || []).map((segment) => Number(segment.end)).filter(Number.isFinite);
  const roughStarts = (roughSegments || []).map((segment) => Number(segment.timestamp)).filter(Number.isFinite);
  const roughEnds = (roughSegments || []).map((segment) => Number(segment.endTimestamp)).filter(Number.isFinite);
  const availableStarts = [...diarizationStarts, ...roughStarts];
  const availableEnds = [...diarizationEnds, ...roughEnds];
  const requestedStart = Number(windowStart);
  const requestedEnd = Number(windowEnd);
  const startTime = Number.isFinite(requestedStart)
    ? requestedStart
    : availableStarts.length
      ? Math.min(...availableStarts)
      : 0;
  const endTime =
    Number.isFinite(requestedEnd) && requestedEnd > startTime
      ? requestedEnd
      : availableEnds.length
        ? Math.max(...availableEnds)
        : startTime + 1;
  interpolateTimes(times, startTime, endTime);

  const speakers = times.map((time) => nearestSpeaker(diarization, time));
  const diarizationSpeakers = [
    ...new Set((diarization || []).map((segment) => segment.speaker).filter(Boolean)),
  ];
  let previousSentenceSpeaker = null;
  for (const range of sentenceRanges(finalText)) {
    const indexes = finalTokens
      .map((token, index) => ({ token, index }))
      .filter(({ token }) => token.start >= range.start && token.start < range.end)
      .map(({ index }) => index);
    if (!indexes.length) continue;
    const firstTime = times[indexes[0]];
    const lastTime = times[indexes.at(-1)];
    let sentenceSpeaker = speakerByIntervalOverlap(
      diarization,
      Math.max(startTime, firstTime - 0.08),
      Math.min(endTime, Math.max(firstTime + 0.08, lastTime + 0.08))
    );
    if (
      range.forceSpeakerChange &&
      previousSentenceSpeaker &&
      diarizationSpeakers.length === 2
    ) {
      sentenceSpeaker =
        diarizationSpeakers.find(
          (speaker) => speaker !== previousSentenceSpeaker
        ) || sentenceSpeaker;
    }
    for (const index of indexes) speakers[index] = sentenceSpeaker;
    previousSentenceSpeaker = sentenceSpeaker;
  }
  const groups = [];
  let groupStart = 0;
  for (let index = 1; index <= finalTokens.length; index += 1) {
    if (index < finalTokens.length && speakers[index] === speakers[groupStart]) continue;
    const nextCharacter =
      index < finalTokens.length ? finalTokens[index].start : finalText.length;
    const characterStart = groupStart === 0 ? 0 : finalTokens[groupStart].start;
    const groupText = finalText.slice(characterStart, nextCharacter).trim();
    if (groupText) {
      groups.push({
        id: `segment-${groups.length}`,
        timestamp: Number(times[groupStart].toFixed(2)),
        endTimestamp: Number(times[index - 1].toFixed(2)),
        text: groupText,
        source: "openai-aligned",
        speaker: speakers[groupStart],
        asrModel,
        alignmentModel,
      });
    }
    groupStart = index;
  }

  return {
    segments: groups,
    metadata: {
      finalWords: finalTokens.length,
      roughWords: roughTokens.length,
      matchedWords: mapping.size,
      matchedRatio: Number((mapping.size / finalTokens.length).toFixed(4)),
      band,
      alignmentModel,
      strategy: "global",
      ...(process.env.OPENWHISPR_ALIGNMENT_DEBUG
        ? {
            wordAssignments: finalTokens.map((token, index) => ({
              word: token.raw,
              time: Number(times[index].toFixed(3)),
              speaker: speakers[index],
              matched: mapping.has(index),
            })),
          }
        : {}),
    },
  };
}

function overlapsWindow(segment, start, end) {
  const segmentStart = Number(segment.timestamp);
  const segmentEnd = Number(segment.endTimestamp);
  if (!Number.isFinite(segmentStart)) return false;
  const effectiveEnd =
    Number.isFinite(segmentEnd) && segmentEnd > segmentStart
      ? segmentEnd
      : segmentStart + 0.1;
  return effectiveEnd > start && segmentStart < end;
}

function clipRoughSegment(segment, start, end) {
  const segmentStart = Number(segment.timestamp);
  const segmentEnd = Number(segment.endTimestamp);
  return {
    ...segment,
    timestamp: Math.max(start, segmentStart),
    endTimestamp: Math.min(
      end,
      Number.isFinite(segmentEnd) && segmentEnd > segmentStart
        ? segmentEnd
        : segmentStart + 0.1
    ),
  };
}

function clipDiarizationSegment(segment, start, end) {
  return {
    ...segment,
    start: Math.max(start, Number(segment.start)),
    end: Math.min(end, Number(segment.end)),
  };
}

export function alignChunkedTranscriptToDiarization({
  chunks,
  roughSegments,
  diarization,
  asrModel = "gpt-transcribe",
  alignmentModel = "local",
  band = DEFAULT_ALIGNMENT_BAND,
}) {
  const usableChunks = (chunks || [])
    .map((chunk, originalIndex) => ({
      ...chunk,
      originalIndex,
      start: Number(chunk.start),
      end: Number(chunk.end),
      text: String(chunk.text || "").trim(),
    }))
    .filter(
      (chunk) =>
        Number.isFinite(chunk.start) &&
        Number.isFinite(chunk.end) &&
        chunk.end > chunk.start &&
        chunk.text
    )
    .sort((left, right) => left.start - right.start);

  if (!usableChunks.length) {
    return alignTranscriptToDiarization({
      text: "",
      roughSegments,
      diarization,
      asrModel,
      alignmentModel,
      band,
    });
  }

  const alignedChunks = usableChunks.map((chunk, chunkIndex) => {
    const chunkRoughSegments = (roughSegments || [])
      .filter((segment) => overlapsWindow(segment, chunk.start, chunk.end))
      .map((segment) => clipRoughSegment(segment, chunk.start, chunk.end));
    const chunkDiarization = (diarization || [])
      .filter(
        (segment) =>
          Number(segment.end) > chunk.start &&
          Number(segment.start) < chunk.end
      )
      .map((segment) =>
        clipDiarizationSegment(segment, chunk.start, chunk.end)
      )
      .filter((segment) => segment.end > segment.start);
    const aligned = alignTranscriptToDiarization({
      text: chunk.text,
      roughSegments: chunkRoughSegments,
      diarization: chunkDiarization,
      asrModel,
      alignmentModel,
      band,
      windowStart: chunk.start,
      windowEnd: chunk.end,
    });
    return {
      ...aligned,
      chunk: {
        index: chunkIndex,
        originalIndex: chunk.originalIndex,
        start: chunk.start,
        end: chunk.end,
      },
    };
  });

  const segments = alignedChunks.flatMap(({ segments: chunkSegments }) =>
    chunkSegments.map((segment) => ({
      ...segment,
      id: "",
    }))
  );
  for (let index = 0; index < segments.length; index += 1) {
    segments[index].id = `segment-${index}`;
  }

  const totals = alignedChunks.reduce(
    (result, aligned) => {
      result.finalWords += Number(aligned.metadata.finalWords) || 0;
      result.roughWords += Number(aligned.metadata.roughWords) || 0;
      result.matchedWords += Number(aligned.metadata.matchedWords) || 0;
      return result;
    },
    { finalWords: 0, roughWords: 0, matchedWords: 0 }
  );
  const matchedRatio = totals.finalWords
    ? Number((totals.matchedWords / totals.finalWords).toFixed(4))
    : 0;

  return {
    segments,
    metadata: {
      ...totals,
      matchedRatio,
      band,
      alignmentModel,
      strategy: "per-cloud-chunk",
      chunks: alignedChunks.map((aligned) => ({
        ...aligned.chunk,
        finalWords: aligned.metadata.finalWords,
        roughWords: aligned.metadata.roughWords,
        matchedWords: aligned.metadata.matchedWords,
        matchedRatio: aligned.metadata.matchedRatio,
      })),
    },
  };
}
