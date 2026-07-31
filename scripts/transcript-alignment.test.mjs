import assert from "node:assert/strict";
import test from "node:test";
import {
  alignChunkedTranscriptToDiarization,
  alignTranscriptToDiarization,
} from "./transcript-alignment.mjs";

test("aligns corrected text to alternating local speakers", () => {
  const result = alignTranscriptToDiarization({
    text: "Bonjour. Ça marche. Très bien.",
    roughSegments: [
      {
        id: "rough-0",
        timestamp: 1,
        endTimestamp: 2,
        text: "Bonjour",
        asrModel: "nemotron-test",
      },
      {
        id: "rough-1",
        timestamp: 3,
        endTimestamp: 4,
        text: "Sa marche",
        asrModel: "nemotron-test",
      },
      {
        id: "rough-2",
        timestamp: 5,
        endTimestamp: 6,
        text: "Très bien",
        asrModel: "nemotron-test",
      },
    ],
    diarization: [
      { start: 0.5, end: 2.5, speaker: "speaker_0" },
      { start: 2.5, end: 4.5, speaker: "speaker_1" },
      { start: 4.5, end: 6.5, speaker: "speaker_0" },
    ],
  });

  assert.deepEqual(
    result.segments.map((segment) => [segment.speaker, segment.text]),
    [
      ["speaker_0", "Bonjour."],
      ["speaker_1", "Ça marche."],
      ["speaker_0", "Très bien."],
    ]
  );
  assert.ok(result.metadata.matchedRatio >= 0.8);
});

test("interpolates final words omitted by the rough local ASR", () => {
  const result = alignTranscriptToDiarization({
    text: "Bonjour Samuel. Merci pour votre temps.",
    roughSegments: [
      {
        id: "rough-0",
        timestamp: 4,
        endTimestamp: 6,
        text: "Merci pour votre temps",
        asrModel: "nemotron-test",
      },
    ],
    diarization: [
      { start: 1, end: 2.5, speaker: "speaker_0" },
      { start: 3.5, end: 6.5, speaker: "speaker_1" },
    ],
  });

  assert.equal(result.segments[0].text, "Bonjour Samuel.");
  assert.equal(result.segments.at(-1).text, "Merci pour votre temps.");
  assert.ok(result.segments[0].timestamp >= 1);
  assert.ok(result.segments.at(-1).endTimestamp <= 6.5);
});

for (const separator of ["   ", " — "]) {
  test(`treats ${JSON.stringify(separator)} as an explicit two-speaker turn boundary`, () => {
    const result = alignTranscriptToDiarization({
      text: `Début.${separator}OK.${separator}Suite.`,
      roughSegments: [
        {
          timestamp: 1,
          endTimestamp: 4,
          text: "Début OK Suite",
          asrModel: "nemotron-test",
        },
      ],
      diarization: [
        { start: 0.5, end: 4.5, speaker: "speaker_0" },
        { start: 5, end: 6, speaker: "speaker_1" },
      ],
    });

    assert.deepEqual(
      result.segments.map((segment) => segment.speaker),
      ["speaker_0", "speaker_1", "speaker_0"]
    );
  });
}

test("aligns independent cloud chunks without accumulating global drift", () => {
  const result = alignChunkedTranscriptToDiarization({
    chunks: [
      {
        start: 0,
        end: 10,
        text: "Bonjour Samuel. Présentation du premier projet.",
      },
      {
        start: 10,
        end: 20,
        text: "Deuxième sujet. Merci Arnaud.",
      },
    ],
    roughSegments: [
      {
        timestamp: 1,
        endTimestamp: 4,
        text: "Bonjour Samuel présentation premier projet",
        asrModel: "nemotron-test",
      },
      {
        timestamp: 12,
        endTimestamp: 16,
        text: "Deuxième sujet merci Arnaud",
        asrModel: "nemotron-test",
      },
    ],
    diarization: [
      { start: 0.5, end: 9, speaker: "speaker_0" },
      { start: 10.5, end: 19, speaker: "speaker_1" },
    ],
  });

  assert.equal(result.metadata.strategy, "per-cloud-chunk");
  assert.equal(result.metadata.chunks.length, 2);
  assert.ok(result.metadata.matchedRatio >= 0.8);
  assert.deepEqual(
    result.segments.map((segment) => segment.speaker),
    ["speaker_0", "speaker_1"]
  );
  assert.ok(result.segments[0].timestamp >= 0);
  assert.ok(result.segments.at(-1).endTimestamp <= 20);
});
