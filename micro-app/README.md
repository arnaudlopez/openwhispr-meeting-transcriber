# OpenWhispr Transcribe micro app

Native SwiftUI frontend for the local `openwhispr-transcribe` MCP pipeline.

Build and install it for the current macOS user:

```sh
./micro-app/build-app.sh --install
```

The installed app lives at:

```text
~/Applications/OpenWhispr Transcribe.app
```

The frontend first invokes `scripts/analyze-meeting-speakers.mjs` locally, then
passes the resulting diarization JSON to
`scripts/openwhispr-transcription-runner.mjs`. The second script calls the same
`runTranscription` implementation exported by the MCP server.

The build records the current repository root in the generated app bundle.
Set `OPENWHISPR_TRANSCRIBE_ROOT` to override it when developing against another
checkout.

The app can:

- detect an exact number of voices or estimate between one and eight;
- create and play a clean preview for every detected voice group;
- suggest existing OpenWhispr profiles with a confidence score;
- assign a name, leave a group anonymous, or assign one name to several groups
  when automatic diarization over-segments a person;
- reuse the validated local analysis during transcription, avoiding a second
  diarization pass;
- select `gpt-transcribe` (recommended), `gpt-4o-transcribe`,
  `gpt-4o-mini-transcribe`, `whisper-1`, or
  `gpt-4o-transcribe-diarize`;
- save a custom OpenAI key as a generic password in the macOS Keychain;
- remove that override and fall back to the project `.env.local`;
- list the speaker profiles currently present in OpenWhispr's local database.
- reveal the completed Markdown in Finder without opening it in the default app.

The speaker-analysis stage is fully local and does not consume OpenAI credits.
Voice A/B/C are file-specific diarization clusters rather than stable
identities. Existing profile matches above the strict threshold are prefilled;
weaker suggestions require explicit validation. Durable profile writes remain
opt-in and only apply to groups carrying a confirmed name.

Automatic final outputs preserve the source basename exactly. For example,
`Réunion Client.mp4` produces `Réunion Client.md` and `Réunion Client.json` in
the selected output directory. Internal speaker analysis uses the separate
`Réunion Client-speaker-analysis.json` name.

After a successful transcription, the app removes its speaker-analysis JSON,
voice-preview directory, and resumable cloud cache. The final JSON embeds the
GPT text chunks, rough timing transcript, and diarization evidence, so it stays
self-contained for later no-charge realignment. Failed runs keep their working
files and cache so they can resume without repeating successful cloud calls.
