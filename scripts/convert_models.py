#!/usr/bin/env python3
"""Convertit les poids d'origine (.pth de Real-ESRGAN, flownet.pkl de Practical-RIFE) en
safetensors f16, le format que lit le programme natif de Scaline.

Les poids d'origine ne vont pas dans git : les télécharger d'abord (adresses et sommes SHA-256
dans models/SOURCES.md), puis, dans un environnement Python avec torch et numpy :

    python scripts/convert_models.py <dossier des poids d'origine> models/

Le dossier d'entrée contient realesr-general-x4v3.pth, realesr-animevideov3.pth et
rife-4.25/flownet.pkl (le dossier train_log de l'archive de Practical-RIFE 4.25, renommé).
Les noms des tenseurs restent ceux d'origine (préfixe `module.` de RIFE retiré), seuls les
tenseurs servant à l'inférence sont gardés (RIFE : ni `teacher` ni `caltime`).
"""

import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch

MODELS = [
    {
        "out": "realesr-general-x4v3.safetensors",
        "src": "realesr-general-x4v3.pth",
        "meta": {
            "model": "SRVGGNetCompact num_feat=64 num_conv=32 upscale=4",
            "origin": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
            "license": "BSD-3-Clause (Real-ESRGAN, Xintao Wang)",
        },
    },
    {
        "out": "realesr-animevideov3.safetensors",
        "src": "realesr-animevideov3.pth",
        "meta": {
            "model": "SRVGGNetCompact num_feat=64 num_conv=16 upscale=4",
            "origin": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth",
            "license": "BSD-3-Clause (Real-ESRGAN, Xintao Wang)",
        },
    },
    {
        "out": "rife-4.25.safetensors",
        "src": "rife-4.25/flownet.pkl",
        "meta": {
            "model": "RIFE IFNet_HDv3 4.25 (Practical-RIFE)",
            "origin": "https://github.com/hzwer/Practical-RIFE (Google Drive, v4.25, 2024-09-19)",
            "license": "MIT (Practical-RIFE, hzwer)",
        },
    },
]


def load(path: Path) -> dict:
    data = torch.load(path, map_location="cpu", weights_only=True)
    if "params" in data:  # Real-ESRGAN
        data = data["params"]
    out = {}
    for name, tensor in data.items():
        name = name.removeprefix("module.")
        if name.startswith(("teacher.", "caltime.")):
            continue  # entraînement seulement
        out[name] = tensor.detach().float().numpy()
    return out


def write_safetensors(tensors: dict, meta: dict, path: Path) -> None:
    header = {"__metadata__": meta}
    blobs = []
    offset = 0
    for name in sorted(tensors):
        blob = tensors[name].astype("<f2").tobytes()
        header[name] = {
            "dtype": "F16",
            "shape": list(tensors[name].shape),
            "data_offsets": [offset, offset + len(blob)],
        }
        blobs.append(blob)
        offset += len(blob)
    text = json.dumps(header, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 8)  # données alignées sur 8 octets
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(text)))
        f.write(text)
        for blob in blobs:
            f.write(blob)


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    dst.mkdir(parents=True, exist_ok=True)
    for model in MODELS:
        origin = src / model["src"]
        digest = hashlib.sha256(origin.read_bytes()).hexdigest()
        tensors = load(origin)
        biggest = max(float(np.abs(t).max()) for t in tensors.values())
        if biggest > 60000:
            sys.exit(f"{origin}: a weight does not fit in f16 ({biggest})")
        meta = {**model["meta"], "origin_sha256": digest, "format": "f16"}
        write_safetensors(tensors, meta, dst / model["out"])
        print(f"{model['out']}: {len(tensors)} tensors, from {model['src']} ({digest[:12]}…)")


if __name__ == "__main__":
    main()
