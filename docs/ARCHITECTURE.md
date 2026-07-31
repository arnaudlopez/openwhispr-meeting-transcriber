# Architecture du pipeline

## Principes

Le texte, les timestamps et l’identité des locuteurs proviennent de sources différentes.
Le pipeline garde ces responsabilités séparées afin qu’une faiblesse locale n’oblige pas à
refaire un appel cloud déjà réussi.

| Responsabilité               | Composant                                | Exécution             |
| ---------------------------- | ---------------------------------------- | --------------------- |
| Détection des voix           | FluidAudio Community‑1 + WeSpeaker + VBx | locale                |
| Tours fins et chevauchements | Sortformer avec enrôlement               | locale                |
| Validation des identités     | micro‑app SwiftUI                        | humaine               |
| Texte final                  | GPT‑Transcribe ou modèle sélectionné     | cloud                 |
| Brouillon horodaté           | Nemotron 3.5 ASR Streaming ONNX INT8     | locale                |
| Alignement et attribution    | `transcript-alignment.mjs`               | locale                |
| Profils vocaux               | SQLite OpenWhispr                        | locale et optionnelle |

## Réunions longues

Les limites d’upload sont gérées en découpant près des silences, autour de 15 minutes.
Chaque réponse cloud est écrite immédiatement dans un cache associé à la source. Après un
échec, seuls les morceaux absents sont envoyés de nouveau.

L’alignement global d’une réunion longue accumule les omissions et insertions du brouillon
local. Le pipeline aligne donc chaque morceau GPT avec les segments Nemotron couvrant la
même fenêtre, puis réunit les résultats. Le score final est pondéré par le nombre de mots.

## Fichiers produits

Une réussite conserve uniquement :

- `<nom-source>.md` : transcription lisible ;
- `<nom-source>.json` : transcription, diarisation, morceaux GPT, brouillon local,
  scores d’alignement, usage et preuves d’identification.

Pendant le traitement, les éléments suivants peuvent exister :

- `*-speaker-analysis.json` ;
- `*-speaker-analysis-previews/` ;
- `*-openai-chunks.json` ou `*-openrouter-cache.json`.

La micro‑app les supprime après une réussite. Elle les conserve après un échec pour rendre
la reprise possible. Le dossier de conversion WAV/PCM est toujours placé dans le dossier
temporaire du système et supprimé dans un bloc `finally`.

## Profils et sécurité

Une identité n’est jamais déduite silencieusement. Elle doit provenir d’un libellé confirmé
par l’utilisateur ou d’un profil local dépassant le seuil strict. L’écriture dans la base
de profils reste désactivée par défaut et nécessite des noms explicites.

La clé OpenAI peut être lue depuis `.env.local` pour les scripts. La micro‑app permet une
surcharge dans le Trousseau macOS. Aucune clé ne doit entrer dans le JSON final, les logs ou
le dépôt Git.
