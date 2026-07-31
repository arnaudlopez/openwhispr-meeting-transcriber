# Contribuer

Merci de votre intérêt pour OpenWhispr Meeting Transcriber.

## Avant de commencer

- Consultez les issues existantes avant d’en ouvrir une nouvelle.
- Pour une évolution importante, ouvrez d’abord une discussion ou une issue afin de valider
  le besoin et le périmètre.
- Ne publiez aucune clé API, réunion, transcription confidentielle, base d’identités ou
  empreinte vocale réelle.
- Signalez les vulnérabilités par la procédure privée de [SECURITY.md](SECURITY.md).

## Installation de développement

Le projet cible macOS 13 ou supérieur sur Apple Silicon et Node.js 24.

```bash
git clone git@github.com:arnaudlopez/openwhispr-meeting-transcriber.git
cd openwhispr-meeting-transcriber
npm ci
cp .env.example .env.local
```

Installez aussi `ffmpeg` et les outils de ligne de commande Xcode. Les scripts de setup
téléchargent FluidAudio et Nemotron à partir de versions épinglées ; leurs poids ne sont
pas inclus dans le dépôt.

## Proposer un changement

1. Créez une branche dédiée depuis `main`.
2. Limitez le changement à un objectif clair et documentez les choix non évidents.
3. Ajoutez ou adaptez les tests couvrant le comportement modifié.
4. Vérifiez qu’aucune donnée privée ni fichier généré n’est suivi par Git.
5. Ouvrez une pull request en décrivant le problème, la solution et les validations réalisées.

## Validation minimale

```bash
npm run transcription:test
npm run micro-app:build
```

Pour un changement documentaire, vérifiez au minimum les liens, les commandes et le rendu
Markdown concernés.

## Licence

En proposant une contribution, vous acceptez qu’elle soit distribuée sous la licence MIT
du dépôt. Conservez les avis d’attribution existants pour le code dérivé d’OpenWhispr et
des autres projets amont.
