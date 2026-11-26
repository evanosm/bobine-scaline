#!/bin/sh
# Télécharge les modèles convertis (safetensors f16) depuis la release `modeles-1` du dépôt et
# vérifie leurs sommes SHA-256 (models/SHA256SUMS). Pour les refaire depuis les poids d'origine :
# scripts/convert_models.py (adresses dans le README, « Modèles »).
set -eu
cd "$(dirname "$0")/../models"
base="https://github.com/evanosm/bobine-scaline/releases/download/modeles-1"
while read -r sum name; do
  if [ -f "$name" ] && echo "$sum  $name" | shasum -a 256 -c - >/dev/null 2>&1; then
    continue
  fi
  echo "téléchargement de $name"
  curl -fsSL --retry 3 -o "$name.part" "$base/$name"
  mv "$name.part" "$name"
done < SHA256SUMS
shasum -a 256 -c SHA256SUMS
