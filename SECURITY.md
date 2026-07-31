# Politique de sécurité

## Signaler une vulnérabilité

N’ouvrez pas d’issue publique pour une vulnérabilité ou une fuite de données.

Utilisez le bouton **Report a vulnerability** dans l’onglet
[Security](https://github.com/arnaudlopez/openwhispr-meeting-transcriber/security) du dépôt.
GitHub transmettra le rapport et les échanges de manière privée au mainteneur.

Indiquez si possible :

- la version ou le commit concerné ;
- les conditions et étapes de reproduction ;
- l’impact observé ou potentiel ;
- une piste de correction, si vous en avez une.

## Périmètre prioritaire

- exposition de clés API ou de données de réunion ;
- exécution de code via un fichier audio, vidéo ou JSON préparé ;
- écriture ou lecture inattendue en dehors des chemins choisis ;
- contournement du Trousseau macOS ou persistance involontaire d’un secret ;
- dépendance ou script d’installation compromis.

## Modèle de données

La diarisation, les profils vocaux, Nemotron et l’alignement sont locaux. Les morceaux
audio à transcrire sont toutefois envoyés au fournisseur d’API choisi. Les utilisateurs
doivent donc vérifier que cet envoi est compatible avec leurs obligations de confidentialité.

Les clés API ne doivent jamais être enregistrées dans Git. La micro‑app utilise le
Trousseau macOS ; le pipeline en ligne de commande peut charger une clé depuis le fichier
local ignoré `.env.local`.

## Divulgation coordonnée

Merci de laisser au mainteneur un délai raisonnable pour analyser et corriger le problème
avant toute publication. Le crédit sera attribué au chercheur qui le souhaite.
