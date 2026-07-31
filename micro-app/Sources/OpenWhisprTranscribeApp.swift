import AppKit
import AVFoundation
import Foundation
import Security
import SwiftUI
import UniformTypeIdentifiers

private let projectRootPath: String = {
    if let override = ProcessInfo.processInfo.environment["OPENWHISPR_TRANSCRIBE_ROOT"],
        !override.isEmpty
    {
        return override
    }
    if let bundled = Bundle.main.object(forInfoDictionaryKey: "OpenWhisprProjectRoot")
        as? String, !bundled.isEmpty
    {
        return bundled
    }
    return URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .path
}()
private let runnerPath = URL(fileURLWithPath: projectRootPath)
    .appendingPathComponent("scripts/openwhispr-transcription-runner.mjs").path
private let analysisScriptPath = URL(fileURLWithPath: projectRootPath)
    .appendingPathComponent("scripts/analyze-meeting-speakers.mjs").path
private let speakerDatabasePath = FileManager.default.homeDirectoryForCurrentUser
    .appendingPathComponent("Library/Application Support/open-whispr/transcriptions.db").path

private enum TranscriptionModel: String, CaseIterable, Identifiable, Sendable {
    case gptTranscribe = "gpt-transcribe"
    case gpt4oTranscribe = "gpt-4o-transcribe"
    case gpt4oMiniTranscribe = "gpt-4o-mini-transcribe"
    case whisper = "whisper-1"
    case gpt4oDiarize = "gpt-4o-transcribe-diarize"

    var id: String { rawValue }

    var title: String {
        switch self {
        case .gptTranscribe: "GPT‑Transcribe"
        case .gpt4oTranscribe: "GPT‑4o Transcribe"
        case .gpt4oMiniTranscribe: "GPT‑4o Mini Transcribe"
        case .whisper: "Whisper‑1"
        case .gpt4oDiarize: "GPT‑4o Transcribe Diarize"
        }
    }

    var detail: String {
        switch self {
        case .gptTranscribe:
            "Recommandé pour les enregistrements dans leur langue d’origine."
        case .gpt4oTranscribe:
            "Alternative GPT‑4o haute qualité, alignée localement."
        case .gpt4oMiniTranscribe:
            "Alternative plus économique, alignée localement."
        case .whisper:
            "Ancien modèle ; utile surtout pour comparer ou produire des sous-titres."
        case .gpt4oDiarize:
            "Diarisation OpenAI spécialisée, confrontée aux preuves locales."
        }
    }

    var usesCloudDiarization: Bool { self == .gpt4oDiarize }
}

private enum KeychainStore {
    private static let service = "com.arnaud.openwhispr-transcribe"
    private static let account = "openai-api-key"

    private static var baseQuery: [String: Any] {
        [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
        ]
    }

    static func load() throws -> String? {
        var query = baseQuery
        query[kSecReturnData as String] = true
        query[kSecMatchLimit as String] = kSecMatchLimitOne
        var item: CFTypeRef?
        let status = SecItemCopyMatching(query as CFDictionary, &item)
        if status == errSecItemNotFound { return nil }
        guard status == errSecSuccess, let data = item as? Data,
            let value = String(data: data, encoding: .utf8)
        else {
            throw RunnerFailure(message: "Lecture du Trousseau impossible (\(status)).")
        }
        return value
    }

    static func save(_ value: String) throws {
        let data = Data(value.utf8)
        let updateStatus = SecItemUpdate(
            baseQuery as CFDictionary,
            [kSecValueData as String: data] as CFDictionary
        )
        if updateStatus == errSecSuccess { return }
        if updateStatus != errSecItemNotFound {
            throw RunnerFailure(message: "Écriture du Trousseau impossible (\(updateStatus)).")
        }
        var item = baseQuery
        item[kSecValueData as String] = data
        let addStatus = SecItemAdd(item as CFDictionary, nil)
        guard addStatus == errSecSuccess else {
            throw RunnerFailure(message: "Écriture du Trousseau impossible (\(addStatus)).")
        }
    }

    static func delete() throws {
        let status = SecItemDelete(baseQuery as CFDictionary)
        guard status == errSecSuccess || status == errSecItemNotFound else {
            throw RunnerFailure(message: "Suppression du Trousseau impossible (\(status)).")
        }
    }
}

private struct TranscriptionRequest: Encodable, Sendable {
    let audioPath: String
    let outputPath: String
    let participants: String
    let language: String
    let asrModel: String
    let speakerLabels: [String: String]?
    let persistSpeakerProfiles: Bool
    let diarizationJSON: String

    enum CodingKeys: String, CodingKey {
        case audioPath = "audio_path"
        case outputPath = "output_path"
        case participants
        case language
        case asrModel = "asr_model"
        case speakerLabels = "speaker_labels"
        case persistSpeakerProfiles = "persist_speaker_profiles"
        case diarizationJSON = "diarization_json"
    }
}

private struct SpeakerProfile: Decodable, Identifiable, Sendable {
    let id: Int
    let displayName: String
    let sampleCount: Int

    enum CodingKeys: String, CodingKey {
        case id
        case displayName = "display_name"
        case sampleCount = "sample_count"
    }
}

private struct SpeakerMatch: Decodable, Sendable {
    let state: String
    let displayName: String?
    let confidence: Double
}

private struct AnalyzedSpeaker: Decodable, Sendable {
    let id: String
    let label: String
    let speakingDurationSeconds: Double
    let turnCount: Int
    let previewPath: String?
    let previewStartSeconds: Double?
    let previewEndSeconds: Double?
    let previewCleanRatio: Double?
    let match: SpeakerMatch
}

private struct SpeakerAnalysisResult: Decodable, Sendable {
    let success: Bool
    let analysisPath: String
    let previewDirectory: String
    let detectedSpeakers: Int
    let speakers: [AnalyzedSpeaker]

    enum CodingKeys: String, CodingKey {
        case success
        case analysisPath = "analysis_path"
        case previewDirectory = "preview_directory"
        case detectedSpeakers = "detected_speakers"
        case speakers
    }
}

private struct SpeakerAssignment: Identifiable {
    let id: String
    let label: String
    let speakingDurationSeconds: Double
    let turnCount: Int
    let previewPath: String?
    let previewStartSeconds: Double?
    let matchState: String
    let suggestedName: String?
    let confidence: Double
    var assignedName: String
}

private struct TranscriptionResult: Decodable, Sendable {
    let success: Bool
    let markdownPath: String
    let jsonPath: String
    let asrModel: String
    let diarizationPipeline: String?
    let alignmentRatio: Double?
    let transcriptSegments: Int
    let speakerNames: [String]
    let degraded: String?

    enum CodingKeys: String, CodingKey {
        case success
        case markdownPath = "markdown_path"
        case jsonPath = "json_path"
        case asrModel = "asr_model"
        case diarizationPipeline = "diarization_pipeline"
        case alignmentRatio = "alignment_ratio"
        case transcriptSegments = "transcript_segments"
        case speakerNames = "speaker_names"
        case degraded
    }
}

private struct RunnerFailure: LocalizedError {
    let message: String
    var errorDescription: String? { message }
}

@main
struct OpenWhisprTranscribeApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
        .windowStyle(.hiddenTitleBar)
        .defaultSize(width: 820, height: 900)
    }
}

@MainActor
private struct ContentView: View {
    @State private var audioPath = ""
    @State private var outputDirectory = ""
    @State private var participantMode = "auto"
    @State private var selectedModel: TranscriptionModel = .gptTranscribe
    @State private var persistProfiles = false
    @State private var revealWhenFinished = true
    @State private var apiKeyDraft = ""
    @State private var keychainAPIKey: String?
    @State private var keyStatus = "Clé du projet utilisée par défaut"
    @State private var keyStatusIsError = false
    @State private var savedProfiles: [SpeakerProfile] = []
    @State private var assignments: [SpeakerAssignment] = []
    @State private var analysisPath = ""
    @State private var analysisPreviewDirectory = ""
    @State private var isAnalyzing = false
    @State private var isTranscribing = false
    @State private var statusTitle = "Prêt à analyser"
    @State private var statusDetail = "Choisissez un enregistrement audio ou vidéo."
    @State private var statusIsError = false
    @State private var lastMarkdownPath = ""
    @State private var audioPlayer: AVAudioPlayer?
    @State private var playingSpeakerID: String?
    @State private var playbackResetTask: Task<Void, Never>?

    private var isBusy: Bool { isAnalyzing || isTranscribing }
    private var hasAnalysis: Bool { !analysisPath.isEmpty && !assignments.isEmpty }
    private var hasAssignedNames: Bool {
        assignments.contains {
            !$0.assignedName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        }
    }

    private var outputPath: String {
        guard !audioPath.isEmpty else { return "" }
        return outputBaseURL.appendingPathExtension("md").path
    }

    private var analysisOutputPath: String {
        guard !audioPath.isEmpty else { return "" }
        return outputBaseURL
            .deletingLastPathComponent()
            .appendingPathComponent(
                "\(outputBaseURL.lastPathComponent)-speaker-analysis"
            )
            .appendingPathExtension("json")
            .path
    }

    private var outputBaseURL: URL {
        let source = URL(fileURLWithPath: audioPath)
        let directory =
            outputDirectory.isEmpty
            ? source.deletingLastPathComponent()
            : URL(fileURLWithPath: outputDirectory)
        let base = source.deletingPathExtension().lastPathComponent
        return directory.appendingPathComponent(base)
    }

    var body: some View {
        ZStack {
            LinearGradient(
                colors: [
                    Color(red: 0.035, green: 0.055, blue: 0.14),
                    Color(red: 0.11, green: 0.055, blue: 0.24),
                ],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
            .ignoresSafeArea()

            ScrollView {
                VStack(spacing: 14) {
                    header
                    sourceCard
                    configurationCard
                    speakersCard
                    statusCard
                    footer
                }
                .padding(22)
            }
        }
        .preferredColorScheme(.dark)
        .frame(minWidth: 720, minHeight: 720)
        .onAppear {
            refreshKeyStatus()
            loadSpeakerProfiles()
        }
        .onDisappear {
            stopPreview()
        }
    }

    private var header: some View {
        HStack(spacing: 14) {
            if let icon = NSImage(named: "AppIcon") {
                Image(nsImage: icon)
                    .resizable()
                    .frame(width: 66, height: 66)
                    .clipShape(RoundedRectangle(cornerRadius: 15, style: .continuous))
                    .shadow(color: .purple.opacity(0.35), radius: 16, y: 7)
            } else {
                Image(systemName: "waveform.and.badge.magnifyingglass")
                    .font(.system(size: 38, weight: .semibold))
                    .frame(width: 66, height: 66)
            }

            VStack(alignment: .leading, spacing: 4) {
                Text("OpenWhispr Transcribe")
                    .font(.system(size: 25, weight: .bold, design: .rounded))
                Text("Analyse locale, identification humaine, transcription précise")
                    .foregroundStyle(.secondary)
            }
            Spacer()
        }
    }

    private var sourceCard: some View {
        card {
            VStack(alignment: .leading, spacing: 13) {
                label("1 · ENREGISTREMENT")
                HStack(spacing: 10) {
                    Image(systemName: audioPath.isEmpty ? "waveform" : "checkmark.circle.fill")
                        .foregroundStyle(audioPath.isEmpty ? Color.secondary : Color.green)
                    Text(audioPath.isEmpty ? "Aucun fichier sélectionné" : audioPath)
                        .lineLimit(1)
                        .truncationMode(.middle)
                        .foregroundStyle(audioPath.isEmpty ? .secondary : .primary)
                    Spacer()
                    Button("Choisir…", action: chooseAudio)
                        .buttonStyle(.borderedProminent)
                        .disabled(isBusy)
                }

                Divider().opacity(0.3)

                HStack(spacing: 10) {
                    Image(systemName: "folder")
                        .foregroundStyle(.secondary)
                    Text(
                        outputDirectory.isEmpty
                            ? "Sortie : à côté de l’enregistrement"
                            : outputDirectory
                    )
                    .lineLimit(1)
                    .truncationMode(.middle)
                    .foregroundStyle(.secondary)
                    Spacer()
                    Button("Dossier…", action: chooseOutputDirectory)
                        .buttonStyle(.bordered)
                        .disabled(isBusy)
                }
            }
        }
    }

    private var configurationCard: some View {
        card {
            VStack(alignment: .leading, spacing: 12) {
                label("TRANSCRIPTION OPENAI")
                HStack(spacing: 12) {
                    VStack(alignment: .leading, spacing: 3) {
                        Text("Modèle de transcription")
                            .font(.caption.bold())
                        Text(selectedModel.detail)
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                            .lineLimit(2)
                    }
                    Spacer()
                    Picker("Modèle", selection: $selectedModel) {
                        ForEach(TranscriptionModel.allCases) { model in
                            Text(model.title).tag(model)
                        }
                    }
                    .labelsHidden()
                    .frame(width: 240)
                    .disabled(isBusy)
                }

                Divider().opacity(0.3)

                HStack(spacing: 10) {
                    SecureField("Nouvelle clé OpenAI (sk-…)", text: $apiKeyDraft)
                        .textFieldStyle(.roundedBorder)
                    Button("Enregistrer", action: saveAPIKey)
                        .buttonStyle(.borderedProminent)
                        .disabled(
                            isBusy
                                || apiKeyDraft.trimmingCharacters(
                                    in: .whitespacesAndNewlines
                                ).isEmpty
                        )
                    if keychainAPIKey != nil {
                        Button("Clé projet", action: useProjectAPIKey)
                            .buttonStyle(.bordered)
                            .disabled(isBusy)
                    }
                }
                Label(
                    keyStatus,
                    systemImage: keyStatusIsError ? "exclamationmark.triangle" : "key.fill"
                )
                .foregroundStyle(keyStatusIsError ? .orange : .secondary)
                .font(.caption)

                HStack(spacing: 10) {
                    pipelinePill(
                        icon: "person.2.wave.2",
                        title: "FluidAudio local",
                        subtitle: "étape 1 · voix"
                    )
                    pipelinePill(
                        icon: "person.crop.circle.badge.checkmark",
                        title: "Validation humaine",
                        subtitle: "étape 2 · noms"
                    )
                    pipelinePill(
                        icon: "text.quote",
                        title: selectedModel.title,
                        subtitle: "étape 3 · texte"
                    )
                }
                Text(
                    "L’analyse des voix ne consomme aucun crédit OpenAI. "
                        + "La clé personnalisée reste dans le Trousseau macOS."
                )
                .font(.caption)
                .foregroundStyle(.secondary)
            }
        }
    }

    private var speakersCard: some View {
        card {
            VStack(alignment: .leading, spacing: 12) {
                HStack {
                    label("2 · ANALYSE ET IDENTIFICATION DES VOIX")
                    Spacer()
                    Picker("Voix attendues", selection: $participantMode) {
                        Text("Automatique (1–8)").tag("auto")
                        Divider()
                        ForEach(1 ... 8, id: \.self) { count in
                            Text("\(count) voix").tag(String(count))
                        }
                    }
                    .frame(width: 190)
                    .disabled(isBusy)
                    .onChange(of: participantMode) { _ in resetAnalysis() }
                }

                if !hasAnalysis {
                    Text(
                        "L’analyse locale détecte les groupes de voix et prépare un extrait par groupe. "
                            + "Le mode automatique peut parfois séparer une personne en deux groupes : "
                            + "vous pourrez donner le même nom aux deux."
                    )
                    .font(.caption)
                    .foregroundStyle(.secondary)

                    HStack {
                        profileSummary
                        Spacer()
                        Button(action: startAnalysis) {
                            Label(
                                isAnalyzing ? "Analyse en cours…" : "Analyser les voix",
                                systemImage: "waveform.badge.magnifyingglass"
                            )
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(audioPath.isEmpty || isBusy)
                    }
                } else {
                    HStack {
                        Label(
                            "\(assignments.count) groupe\(assignments.count > 1 ? "s" : "") de voix détecté\(assignments.count > 1 ? "s" : "")",
                            systemImage: "checkmark.seal.fill"
                        )
                        .foregroundStyle(.green)
                        .font(.caption.bold())
                        Spacer()
                        Button("Ré-analyser") {
                            resetAnalysis()
                            startAnalysis()
                        }
                        .buttonStyle(.bordered)
                        .disabled(isBusy)
                    }

                    Text(
                        "Écoutez chaque extrait, puis confirmez un nom ou laissez la voix anonyme. "
                            + "Un même nom peut être utilisé plusieurs fois."
                    )
                    .font(.caption)
                    .foregroundStyle(.secondary)

                    ForEach($assignments) { $assignment in
                        speakerAssignmentRow($assignment)
                    }

                    Toggle(
                        "Mémoriser les identités confirmées dans OpenWhispr",
                        isOn: $persistProfiles
                    )
                    .disabled(!hasAssignedNames || isBusy)
                    .help(
                        "Seules les lignes portant un nom seront ajoutées ou renforcées dans les profils vocaux."
                    )

                    profileSummary
                }
            }
        }
    }

    private var profileSummary: some View {
        Group {
            if savedProfiles.isEmpty {
                Text("Aucun profil vocal enregistré.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else {
                Text(
                    "Profils : "
                        + savedProfiles
                            .map { "\($0.displayName) (\($0.sampleCount))" }
                            .joined(separator: " • ")
                )
                .font(.caption)
                .foregroundStyle(.secondary)
                .lineLimit(2)
            }
        }
    }

    private func speakerAssignmentRow(
        _ assignment: Binding<SpeakerAssignment>
    ) -> some View {
        let value = assignment.wrappedValue
        return VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 10) {
                Button {
                    togglePreview(value)
                } label: {
                    Image(
                        systemName:
                            playingSpeakerID == value.id
                            ? "stop.fill"
                            : "play.fill"
                    )
                    .frame(width: 18)
                }
                .buttonStyle(.borderedProminent)
                .tint(playingSpeakerID == value.id ? .orange : .cyan)
                .disabled(value.previewPath == nil || isBusy)
                .help("Écouter l’extrait de \(value.label)")

                VStack(alignment: .leading, spacing: 2) {
                    Text(value.label)
                        .font(.caption.bold())
                    Text(
                        "\(formatDuration(value.speakingDurationSeconds)) de parole • "
                            + "\(value.turnCount) prises • extrait à "
                            + formatTimestamp(value.previewStartSeconds)
                    )
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                }
                .frame(width: 190, alignment: .leading)

                TextField(namePlaceholder(value), text: assignment.assignedName)
                    .textFieldStyle(.roundedBorder)
                    .disabled(isBusy)

                Menu {
                    Button("Laisser anonyme") {
                        assignment.wrappedValue.assignedName = ""
                    }
                    Divider()
                    ForEach(savedProfiles) { profile in
                        Button(profile.displayName) {
                            assignment.wrappedValue.assignedName = profile.displayName
                        }
                    }
                } label: {
                    Image(systemName: "person.crop.circle.badge.plus")
                }
                .menuStyle(.borderlessButton)
                .fixedSize()
                .disabled(savedProfiles.isEmpty || isBusy)
            }

            if let suggestion = value.suggestedName {
                HStack(spacing: 5) {
                    Image(
                        systemName:
                            value.matchState == "matched"
                            ? "person.crop.circle.badge.checkmark"
                            : "questionmark.circle"
                    )
                    Text(
                        value.matchState == "matched"
                            ? "Profil reconnu : \(suggestion)"
                            : "Profil possible : \(suggestion), à vérifier"
                    )
                    Text("• \(Int((value.confidence * 100).rounded())) %")
                }
                .font(.caption2)
                .foregroundStyle(value.matchState == "matched" ? .green : .orange)
            } else {
                Text("Aucun profil enregistré reconnu avec assez de confiance.")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(11)
        .background(.white.opacity(0.045), in: RoundedRectangle(cornerRadius: 11))
    }

    private var statusCard: some View {
        card {
            HStack(spacing: 13) {
                if isBusy {
                    ProgressView()
                        .controlSize(.small)
                } else {
                    Image(
                        systemName:
                            statusIsError
                            ? "exclamationmark.triangle.fill"
                            : (lastMarkdownPath.isEmpty ? "sparkles" : "checkmark.seal.fill")
                    )
                    .foregroundStyle(statusIsError ? .orange : .cyan)
                    .font(.title3)
                }

                VStack(alignment: .leading, spacing: 3) {
                    Text(statusTitle).fontWeight(.semibold)
                    Text(statusDetail)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(3)
                }
                Spacer()
                if !lastMarkdownPath.isEmpty && !isBusy {
                    Button("Afficher dans le Finder", action: revealLastTranscript)
                        .buttonStyle(.bordered)
                }
            }
        }
    }

    private var footer: some View {
        HStack {
            Toggle("Afficher dans le Finder à la fin", isOn: $revealWhenFinished)
                .toggleStyle(.checkbox)
            Spacer()
            if hasAnalysis {
                Button(action: startTranscription) {
                    Label(
                        isTranscribing
                            ? "Transcription en cours…"
                            : "Transcrire avec ces identités",
                        systemImage: "text.badge.checkmark"
                    )
                    .frame(minWidth: 220)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
                .disabled(isBusy)
            } else {
                Text("Commencez par l’analyse locale des voix.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
    }

    private func pipelinePill(icon: String, title: String, subtitle: String) -> some View {
        HStack(spacing: 8) {
            Image(systemName: icon)
                .foregroundStyle(.cyan)
            VStack(alignment: .leading, spacing: 1) {
                Text(title).font(.caption.bold())
                Text(subtitle).font(.caption2).foregroundStyle(.secondary)
            }
            Spacer(minLength: 0)
        }
        .padding(10)
        .frame(maxWidth: .infinity)
        .background(.white.opacity(0.055), in: RoundedRectangle(cornerRadius: 10))
    }

    private func card<Content: View>(
        @ViewBuilder content: () -> Content
    ) -> some View {
        content()
            .padding(16)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 16))
            .overlay(
                RoundedRectangle(cornerRadius: 16)
                    .stroke(.white.opacity(0.08), lineWidth: 1)
            )
    }

    private func label(_ text: String) -> some View {
        Text(text)
            .font(.system(size: 11, weight: .bold, design: .rounded))
            .tracking(1.1)
            .foregroundStyle(.secondary)
    }

    private func chooseAudio() {
        let panel = NSOpenPanel()
        panel.title = "Choisir un enregistrement"
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        panel.allowedContentTypes = [.audio, .movie, .mpeg4Movie]
        guard panel.runModal() == .OK, let url = panel.url else { return }
        audioPath = url.path
        if outputDirectory.isEmpty {
            outputDirectory = url.deletingLastPathComponent().path
        }
        resetAnalysis()
        statusTitle = "Fichier prêt pour l’analyse"
        statusDetail = url.lastPathComponent
        statusIsError = false
        lastMarkdownPath = ""
    }

    private func chooseOutputDirectory() {
        let panel = NSOpenPanel()
        panel.title = "Choisir le dossier de sortie"
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.canCreateDirectories = true
        panel.allowsMultipleSelection = false
        guard panel.runModal() == .OK, let url = panel.url else { return }
        outputDirectory = url.path
        resetAnalysis()
        statusTitle = "Dossier de sortie modifié"
        statusDetail = "Relancez l’analyse locale pour créer les fichiers de travail ici."
    }

    private func startAnalysis() {
        guard !audioPath.isEmpty else { return }
        stopPreview()
        isAnalyzing = true
        statusIsError = false
        lastMarkdownPath = ""
        statusTitle = "Analyse locale des voix"
        statusDetail =
            "FluidAudio détecte les prises de parole et prépare les extraits. Aucun appel OpenAI."
        let source = audioPath
        let destination = analysisOutputPath
        let participants = participantMode

        Task {
            do {
                let result = try await Task.detached(priority: .userInitiated) {
                    try Self.runAnalysis(
                        audioPath: source,
                        outputPath: destination,
                        participants: participants
                    )
                }.value
                assignments = result.speakers.map { speaker in
                    SpeakerAssignment(
                        id: speaker.id,
                        label: speaker.label,
                        speakingDurationSeconds: speaker.speakingDurationSeconds,
                        turnCount: speaker.turnCount,
                        previewPath: speaker.previewPath,
                        previewStartSeconds: speaker.previewStartSeconds,
                        matchState: speaker.match.state,
                        suggestedName: speaker.match.displayName,
                        confidence: speaker.match.confidence,
                        assignedName:
                            speaker.match.state == "matched"
                            ? speaker.match.displayName ?? ""
                            : ""
                    )
                }
                analysisPath = result.analysisPath
                analysisPreviewDirectory = result.previewDirectory
                isAnalyzing = false
                statusTitle = "Analyse terminée"
                statusDetail =
                    "\(result.detectedSpeakers) groupe(s) détecté(s). "
                    + "Écoutez les extraits et validez les noms avant la transcription."
            } catch {
                isAnalyzing = false
                statusIsError = true
                statusTitle = "Échec de l’analyse locale"
                statusDetail = error.localizedDescription
            }
        }
    }

    private func startTranscription() {
        guard hasAnalysis else { return }
        stopPreview()
        let labels = Dictionary(
            uniqueKeysWithValues: assignments.compactMap { assignment in
                let name = assignment.assignedName.trimmingCharacters(
                    in: .whitespacesAndNewlines
                )
                return name.isEmpty ? nil : (assignment.id, name)
            }
        )
        let request = TranscriptionRequest(
            audioPath: audioPath,
            outputPath: outputPath,
            participants: participantMode,
            language: "fr",
            asrModel: selectedModel.rawValue,
            speakerLabels: labels.isEmpty ? nil : labels,
            persistSpeakerProfiles: persistProfiles,
            diarizationJSON: analysisPath
        )

        isTranscribing = true
        statusIsError = false
        lastMarkdownPath = ""
        statusTitle = "Transcription en cours"
        statusDetail =
            "\(selectedModel.title) produit le texte ; l’analyse locale validée est réutilisée."
        let customAPIKey = keychainAPIKey

        Task {
            do {
                let result = try await Task.detached(priority: .userInitiated) {
                    try Self.runTranscription(request, apiKey: customAPIKey)
                }.value
                let cleanupFailures = cleanupGeneratedTemporaryFiles(
                    markdownPath: result.markdownPath
                )
                isTranscribing = false
                lastMarkdownPath = result.markdownPath
                statusIsError = result.degraded != nil || !cleanupFailures.isEmpty
                statusTitle =
                    result.degraded == nil && cleanupFailures.isEmpty
                    ? "Transcription terminée"
                    : "Transcription terminée avec avertissement"
                let ratio =
                    result.alignmentRatio.map { String(format: "%.0f %%", $0 * 100) }
                    ?? "n/a"
                let names =
                    result.speakerNames.isEmpty
                    ? "intervenants anonymes"
                    : result.speakerNames.joined(separator: ", ")
                statusDetail =
                    "\(result.asrModel) • \(result.transcriptSegments) segments • "
                    + "alignement \(ratio) • \(names)"
                    + (cleanupFailures.isEmpty
                        ? " • fichiers temporaires supprimés"
                        : " • nettoyage incomplet : \(cleanupFailures.joined(separator: ", "))")
                resetAnalysis()
                loadSpeakerProfiles()
                if revealWhenFinished {
                    revealInFinder(result.markdownPath)
                }
            } catch {
                isTranscribing = false
                statusIsError = true
                statusTitle = "Échec de la transcription"
                statusDetail = error.localizedDescription
            }
        }
    }

    private func togglePreview(_ speaker: SpeakerAssignment) {
        if playingSpeakerID == speaker.id {
            stopPreview()
            return
        }
        guard let previewPath = speaker.previewPath else { return }
        do {
            stopPreview()
            let player = try AVAudioPlayer(
                contentsOf: URL(fileURLWithPath: previewPath)
            )
            player.prepareToPlay()
            player.play()
            audioPlayer = player
            playingSpeakerID = speaker.id
            let expectedID = speaker.id
            playbackResetTask = Task {
                try? await Task.sleep(
                    nanoseconds: UInt64((player.duration + 0.15) * 1_000_000_000)
                )
                if !Task.isCancelled && playingSpeakerID == expectedID {
                    playingSpeakerID = nil
                    audioPlayer = nil
                }
            }
        } catch {
            statusIsError = true
            statusTitle = "Lecture de l’extrait impossible"
            statusDetail = error.localizedDescription
        }
    }

    private func stopPreview() {
        playbackResetTask?.cancel()
        playbackResetTask = nil
        audioPlayer?.stop()
        audioPlayer = nil
        playingSpeakerID = nil
    }

    private func resetAnalysis() {
        stopPreview()
        analysisPath = ""
        analysisPreviewDirectory = ""
        assignments = []
        persistProfiles = false
    }

    private func cleanupGeneratedTemporaryFiles(markdownPath: String) -> [String] {
        let markdownURL = URL(fileURLWithPath: markdownPath)
        let baseURL = markdownURL.deletingPathExtension()
        let directory = baseURL.deletingLastPathComponent()
        let baseName = baseURL.lastPathComponent
        let paths = Set([
            analysisPath,
            analysisPreviewDirectory,
            directory.appendingPathComponent("\(baseName)-openai-chunks.json").path,
            directory.appendingPathComponent("\(baseName)-openrouter-cache.json").path,
        ].filter { !$0.isEmpty })
        var failures: [String] = []
        for path in paths where FileManager.default.fileExists(atPath: path) {
            do {
                try FileManager.default.removeItem(atPath: path)
            } catch {
                failures.append(URL(fileURLWithPath: path).lastPathComponent)
            }
        }
        return failures.sorted()
    }

    private func namePlaceholder(_ speaker: SpeakerAssignment) -> String {
        if speaker.matchState == "suggested", let name = speaker.suggestedName {
            return "Suggestion : \(name)"
        }
        return "Nom réel ou laisser anonyme"
    }

    private func formatDuration(_ seconds: Double) -> String {
        let total = max(0, Int(seconds.rounded()))
        if total < 60 { return "\(total) s" }
        return "\(total / 60) min \(total % 60) s"
    }

    private func formatTimestamp(_ seconds: Double?) -> String {
        guard let seconds else { return "n/a" }
        let total = max(0, Int(seconds.rounded()))
        return String(format: "%02d:%02d", total / 60, total % 60)
    }

    private func refreshKeyStatus() {
        do {
            keychainAPIKey = try KeychainStore.load()
            keyStatusIsError = false
            if let key = keychainAPIKey {
                keyStatus = "Clé personnalisée active dans le Trousseau ••••\(key.suffix(4))"
            } else {
                keyStatus = "Clé du projet (.env.local) utilisée par défaut"
            }
        } catch {
            keyStatusIsError = true
            keyStatus = error.localizedDescription
        }
    }

    private func saveAPIKey() {
        let value = apiKeyDraft.trimmingCharacters(in: .whitespacesAndNewlines)
        guard value.hasPrefix("sk-"), value.count >= 20 else {
            keyStatusIsError = true
            keyStatus = "La clé doit commencer par sk- et être complète."
            return
        }
        do {
            try KeychainStore.save(value)
            apiKeyDraft = ""
            keychainAPIKey = value
            keyStatusIsError = false
            keyStatus = "Nouvelle clé enregistrée dans le Trousseau ••••\(value.suffix(4))"
        } catch {
            keyStatusIsError = true
            keyStatus = error.localizedDescription
        }
    }

    private func useProjectAPIKey() {
        do {
            try KeychainStore.delete()
            keychainAPIKey = nil
            apiKeyDraft = ""
            keyStatusIsError = false
            keyStatus = "Clé personnalisée supprimée ; retour à la clé du projet (.env.local)"
        } catch {
            keyStatusIsError = true
            keyStatus = error.localizedDescription
        }
    }

    private func loadSpeakerProfiles() {
        Task {
            do {
                savedProfiles = try await Task.detached(priority: .utility) {
                    try Self.readSpeakerProfiles()
                }.value
            } catch {
                savedProfiles = []
            }
        }
    }

    nonisolated private static func runAnalysis(
        audioPath: String,
        outputPath: String,
        participants: String
    ) throws -> SpeakerAnalysisResult {
        guard FileManager.default.fileExists(atPath: analysisScriptPath) else {
            throw RunnerFailure(message: "L’analyseur OpenWhispr est introuvable.")
        }
        let output = try runNodeProcess(
            script: analysisScriptPath,
            arguments: [
                "--input", audioPath,
                "--output", outputPath,
                "--participants", participants,
            ],
            input: nil,
            apiKey: nil
        )
        return try JSONDecoder().decode(SpeakerAnalysisResult.self, from: output)
    }

    nonisolated private static func runTranscription(
        _ request: TranscriptionRequest,
        apiKey: String?
    ) throws -> TranscriptionResult {
        guard FileManager.default.fileExists(atPath: runnerPath) else {
            throw RunnerFailure(message: "Le runner OpenWhispr est introuvable.")
        }
        let requestData = try JSONEncoder().encode(request)
        let output = try runNodeProcess(
            script: runnerPath,
            arguments: [],
            input: requestData,
            apiKey: apiKey
        )
        return try JSONDecoder().decode(TranscriptionResult.self, from: output)
    }

    nonisolated private static func runNodeProcess(
        script: String,
        arguments: [String],
        input: Data?,
        apiKey: String?
    ) throws -> Data {
        let nodeCandidates = [
            "/opt/homebrew/bin/node",
            "/usr/local/bin/node",
        ]
        guard let nodePath = nodeCandidates.first(where: FileManager.default.fileExists) else {
            throw RunnerFailure(message: "Node.js est introuvable sur cette machine.")
        }

        let process = Process()
        process.executableURL = URL(fileURLWithPath: nodePath)
        process.arguments = [script] + arguments
        var environment = ProcessInfo.processInfo.environment
        environment["PATH"] =
            "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
        if let apiKey, !apiKey.isEmpty {
            environment["OPENAI_API_KEY"] = apiKey
        }
        process.environment = environment

        let inputPipe = Pipe()
        let outputPipe = Pipe()
        if input != nil { process.standardInput = inputPipe }
        process.standardOutput = outputPipe
        process.standardError = outputPipe

        try process.run()
        if let input {
            inputPipe.fileHandleForWriting.write(input)
            try inputPipe.fileHandleForWriting.close()
        }
        let combinedOutput = outputPipe.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()

        guard process.terminationStatus == 0 else {
            let details =
                String(data: combinedOutput, encoding: .utf8)?
                .trimmingCharacters(in: .whitespacesAndNewlines)
            throw RunnerFailure(
                message: details?.isEmpty == false
                    ? details!
                    : "Le processus OpenWhispr s’est arrêté."
            )
        }
        let lines =
            String(data: combinedOutput, encoding: .utf8)?
            .split(whereSeparator: \.isNewline)
            .map(String.init) ?? []
        guard let jsonLine = lines.last(where: { $0.trimmingCharacters(
            in: .whitespacesAndNewlines
        ).hasPrefix("{") }) else {
            throw RunnerFailure(message: "Le processus OpenWhispr n’a retourné aucun résultat JSON.")
        }
        return Data(jsonLine.utf8)
    }

    nonisolated private static func readSpeakerProfiles() throws -> [SpeakerProfile] {
        guard FileManager.default.fileExists(atPath: speakerDatabasePath) else { return [] }
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/sqlite3")
        process.arguments = [
            "-json",
            speakerDatabasePath,
            """
            SELECT id, display_name, COALESCE(sample_count, 0) AS sample_count
            FROM speaker_profiles
            ORDER BY display_name
            """,
        ]
        let outputPipe = Pipe()
        let errorPipe = Pipe()
        process.standardOutput = outputPipe
        process.standardError = errorPipe
        try process.run()
        let output = outputPipe.fileHandleForReading.readDataToEndOfFile()
        let errorOutput = errorPipe.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        guard process.terminationStatus == 0 else {
            let details = String(data: errorOutput, encoding: .utf8) ?? ""
            throw RunnerFailure(message: "Lecture des profils impossible : \(details)")
        }
        if output.isEmpty { return [] }
        return try JSONDecoder().decode([SpeakerProfile].self, from: output)
    }

    private func revealLastTranscript() {
        guard !lastMarkdownPath.isEmpty else { return }
        revealInFinder(lastMarkdownPath)
    }

    private func revealInFinder(_ path: String) {
        NSWorkspace.shared.activateFileViewerSelecting([
            URL(fileURLWithPath: path)
        ])
    }

}
