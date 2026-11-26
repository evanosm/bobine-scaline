#!/bin/sh
# Télécharge les modèles convertis (safetensors f16) depuis la release `modeles-1` du dépôt et
# vérifie leurs sommes SHA-256 (models/SHA256SUMS). Pour les refaire depuis les poids d'origine :
# scripts/convert_models.py (adresses dans le README, « Modèles »).
set -eu
cd "$(dirname "$0")/../models"
base="https://github.com/evanosm/bobine-scaline/releases/download/modeles-1"

# sha256sum (Linux, Git Bash sous Windows) ou shasum (macOS).
sha256() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1"; else shasum -a 256 "$1"; fi | cut -d ' ' -f 1
}

# `tr -d '\r'` : un SHA256SUMS en CRLF (extraction Windows) se lit pareil.
tr -d '\r' < SHA256SUMS | while read -r sum name; do
  [ -n "$name" ] || continue
  if [ -f "$name" ] && [ "$(sha256 "$name")" = "$sum" ]; then
    continue
  fi
  echo "téléchargement de $name"
  curl -fsSL --retry 3 -o "$name.part" "$base/$name"
  mv "$name.part" "$name"
  if [ "$(sha256 "$name")" != "$sum" ]; then
    echo "$name : somme SHA-256 différente" >&2
    exit 1
  fi
  echo "$name : OK"
done
