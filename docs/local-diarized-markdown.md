# Local Diarized Markdown

This local companion workflow turns an audio file into a Markdown transcript artifact without touching the OpenWhispr note database or requiring cloud transcription quota.

## Friendly wrapper

For daily use, prefer the guided wrapper:

```sh
node scripts/transcribe-meeting.mjs "/path/to/meeting.m4a"
```

Defaults:

- writes Markdown and JSON next to the source audio file;
- uses `--participants 2`;
- uses `gpt-transcribe` for the final French wording;
- uses FluidAudio locally for diarization and Nemotron locally for timed alignment;
- splits long cloud transcription near silences and resumes from the per-part cache;
- uses the real OpenWhispr `speaker_profiles` DB so learned speakers are available for future files.

Useful examples:

```sh
# Learn names interactively into the real OpenWhispr speaker_profiles DB
node scripts/transcribe-meeting.mjs "/path/to/meeting.m4a"

# Try safely with a temporary copied DB instead
node scripts/transcribe-meeting.mjs "/path/to/meeting.m4a" --db-mode copy

# Non-interactive labels
node scripts/transcribe-meeting.mjs "/path/to/meeting.m4a" \
  --speaker-labels "speaker_00=Alice,speaker_01=Bob"

# Match only; do not ask names or write new profiles
node scripts/transcribe-meeting.mjs "/path/to/meeting.m4a" --no-learn

# Run a second transcript pass to prove match-only reuse
node scripts/transcribe-meeting.mjs "/path/to/meeting.m4a" --verify-match
```

Use `--db-mode copy` when you want a sandbox run that does not persist speaker profiles.

## Recommended hybrid: local speakers and GPT text

For accurate French text with one paid transcription model pass, use FluidAudio
locally for speaker turns, Nemotron locally for rough timestamps, and
`gpt-transcribe` for the final wording:

```sh
node scripts/setup-fluidaudio-diarizer.mjs

node scripts/transcribe-meeting.mjs "/path/to/meeting.mp4" \
  --participants 2 \
  --diarization-provider fluidaudio \
  --asr-provider openai \
  --asr-model gpt-transcribe \
  --alignment-asr-model nemotron-3.5-asr-streaming-0.6b \
  --speaker-labels "speaker_00=Alice,speaker_01=Bob" \
  --known-speaker-references \
  '{"Alice":"/path/to/alice.wav","Bob":"/path/to/bob.wav"}'
```

This mode:

- runs Community-1 segmentation, WeSpeaker embeddings, and VBx clustering
  locally to identify the supplied voice references;
- derives clean 38-second enrollment clips from those identified turns, then
  runs enrolled high-context Sortformer for finer turns and overlaps;
- runs local Nemotron only as a rough timed transcript;
- sends short recordings once and splits long meetings near silences into
  resumable parts of about 15 minutes for `gpt-transcribe`;
- aligns the accurate GPT text to the local timed words, then projects it onto
  the local speaker turns, including explicit dash-separated dialogue;
- stores the local alignment evidence, reference scores, and raw timed
  transcript in the JSON sidecar.

Use `--openai-transcript-json /path/to/previous-sidecar.json` to rerun local
diarization or alignment against an existing GPT transcript without making or
charging another API request.

Each successful cloud part is persisted immediately in
`<output>-openai-chunks.json`. A retry reuses those parts automatically.
`--openai-segment-seconds` changes the default 900-second ceiling and
`--openai-chunk-cache` selects another cache path.

The reference names must exactly match the names used in `--speaker-labels`.
FluidAudio is pinned by the setup script to the revision verified by this
workflow. Set `FLUIDAUDIO_CLI_PATH` or pass `--fluidaudio-cli` to use another
build.

Very short replies below roughly half a second remain the hardest case. Keep
the supplied identity reference clips clean and between 2 and 10 seconds; the
workflow creates the longer Sortformer enrollment audio itself. Inspect
greetings and fully overlapped interruptions when they are critical.

## Local MCP server

The local stdio MCP server exposes the staged hybrid workflow through
`analyze_meeting_speakers` and `transcribe_meeting`:

```sh
npm run mcp:transcription
```

Example Codex configuration:

```toml
[mcp_servers.openwhispr-transcribe]
command = "/opt/homebrew/bin/node"
args = ["/absolute/path/to/openwhispr/scripts/openwhispr-transcription-mcp.mjs"]
```

MCP defaults are deliberately non-interactive:

- `analyze_meeting_speakers` runs FluidAudio and profile matching locally,
  creates one preview per voice group, and accepts an exact count or `auto`;
- its `analysis_path` is passed back as `diarization_json` after the user maps
  detected speaker IDs to names in `speaker_labels`;
- `gpt-transcribe` + FluidAudio + Nemotron remains the final transcription
  pipeline, in French by default;
- Markdown and JSON next to the source unless an output path is supplied;
- speaker matching can read the existing OpenWhispr database;
- profiles are not changed unless `persist_speaker_profiles=true` and explicit
  `speaker_labels` are supplied;
- `transcript_json` and `diarization_json` provide a no-charge replay path for
  testing or realignment.

The project `.env.local` supplies `OPENAI_API_KEY`; it is never sent as an MCP
argument or returned by the tool.

## macOS micro app

A native SwiftUI frontend is available for users who prefer a launchable app:

```sh
npm run micro-app:install
```

It installs:

```text
~/Applications/OpenWhispr Transcribe.app
```

The app selects an audio/video file and first runs a free local speaker
analysis. It shows every detected group with an audio preview, duration,
profile suggestion and confidence. The user then confirms names before the
single paid transcription stage. A name can be assigned to several clusters
when automatic speaker counting over-segments one person.

The validated analysis is passed unchanged to the same `runTranscription`
implementation as the MCP through
`scripts/openwhispr-transcription-runner.mjs`. A replacement OpenAI key entered
in the app is stored in the macOS Keychain and can be removed to return to
`.env.local`.

The model picker exposes the current file-transcription alternatives supported
by this workflow. `gpt-transcribe` remains the recommended default. Known local
speaker profiles are listed in the interface. Voice A/B/C are clustering
labels local to one recording, not stable identities. Strong profile matches
are prefilled, weaker suggestions require validation, and any group may remain
anonymous. Speaker-profile persistence is disabled by default and only stores
explicitly confirmed names.

## OpenAI diarized transcription

Use the direct OpenAI diarization model as a fallback when cloud-provided
speaker labels matter more than the hybrid mode's lower cost:

```sh
node scripts/transcribe-meeting.mjs "/path/to/meeting.mp4" \
  --asr-provider openai \
  --asr-model gpt-4o-transcribe-diarize \
  --speaker-labels "speaker_00=Alice,speaker_01=Bob" \
  --known-speaker-references \
  '{"Alice":"/path/to/alice.wav","Bob":"/path/to/bob.wav"}'
```

The direct provider:

- loads `OPENAI_API_KEY` from the environment, `.env.local`, or `.env`;
- sends one continuous recording instead of isolated speaker turns;
- requests `diarized_json` with `chunking_strategy=auto`;
- accepts explicit 2–10 second reference clips for stable speaker naming;
- re-encodes oversized inputs as mono WebM/Opus, targeting 40 kbps and a safe
  24 MB upload ceiling;
- aligns OpenAI speaker labels back to the local diarization timeline so existing
  OpenWhispr speaker profiles and names still apply.

Useful OpenAI-specific flags:

- `--openai-bitrate-kbps 40` changes the initial speech compression bitrate.
- `--openai-max-upload-mb 24` changes the safe upload ceiling.
- `--openai-segment-seconds 900` controls the reliable `gpt-transcribe` part
  duration.
- `--openai-chunk-cache /path/cache.json` controls the per-part resume cache.
- `--known-speaker-references` accepts a JSON object or
  `Name=/path/reference.wav,...`.
- `--openai-auto-speaker-references` derives clips from local diarization when
  explicit references are unavailable; explicit files are safer when local
  diarization is uncertain.
- `--openai-url` overrides the transcription endpoint for testing.

If the compressed recording remains above the upload ceiling even at 16 kbps,
the command stops with an explicit split-required error rather than silently
dropping audio.

## Low-level command

```sh
node scripts/local-diarized-markdown.mjs \
  --input "/path/to/meeting.m4a" \
  --output "/path/to/meeting-transcript.md" \
  --expected-speakers auto
```

The script writes:

- the Markdown transcript at the requested output path;
- a sidecar JSON file next to it with diarization and transcript segments.

`--expected-speakers auto` uses a two-speaker meeting default, which fits the first local workflow target better than unconstrained clustering. Pass an explicit number for larger meetings.

## Speaker learning

The companion workflow can learn local speaker profiles from labels you provide. It writes only to the selected OpenWhispr SQLite database's `speaker_profiles` table. For testing, copy the database first:

```sh
TMPDIR="$(mktemp -d)"
cp "$HOME/Library/Application Support/open-whispr/transcriptions.db" "$TMPDIR/transcriptions.db"

node scripts/local-diarized-markdown.mjs \
  --input "/path/to/meeting.m4a" \
  --output "$TMPDIR/meeting-learned.md" \
  --expected-speakers 2 \
  --speaker-db "$TMPDIR/transcriptions.db" \
  --learn-speakers \
  --speaker-labels "speaker_00=Alice,speaker_01=Bob"
```

On a later run with the same database, omit `--learn-speakers` and `--speaker-labels`; matching profiles above the threshold are rendered as human names in Markdown:

```sh
node scripts/local-diarized-markdown.mjs \
  --input "/path/to/meeting.m4a" \
  --output "$TMPDIR/meeting-matched.md" \
  --expected-speakers 2 \
  --speaker-db "$TMPDIR/transcriptions.db"
```

Useful flags:

- `--speaker-labels` accepts `speaker_00=Alice,speaker_01=Bob` or a JSON object.
- `--match-threshold` defaults to `0.6`; matches above this are confirmed.
- `--suggest-threshold` defaults to `0.5`; weaker suggestions are kept in the sidecar JSON but not rendered as confirmed names.
- `--no-db-write` computes matches and learning output without writing profiles.

The sidecar JSON records `speakerLearning` evidence, including profile ids, confidence values, sample timestamps, and whether each speaker was learned, matched, suggested, or unknown.
