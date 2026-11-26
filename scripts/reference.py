#!/usr/bin/env python3
"""Sorties de référence PyTorch pour vérifier les shaders de Scaline (voir README, « Vérification »).

    python scripts/reference.py <poids d'origine> models/ <dossier de sortie>

Écrit dans le dossier de sortie deux images d'essai (`a.rgba`, `b.rgba`, RGBA 8 bits, b = a
décalée de quelques pixels) et, pour chaque réseau, la sortie de PyTorch (en f32, sur le
processeur) avec deux jeux de poids : `*-f16w.rgba` (les poids f16 du paquet : écart dû à la
seule implémentation) et `*-orig.rgba` (les poids f32 d'origine : écart total). Le programme
`cargo run --release --example compare -- <dossier de sortie>` compare ensuite.

Les réseaux sont réécrits ici d'après le code d'origine (Real-ESRGAN `SRVGGNetCompact`,
Practical-RIFE 4.25 `IFNet_HDv3`) plutôt qu'importés : aucun code téléchargé n'est exécuté.
"""

import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Tailles volontairement pas multiples de 128 (RIFE complète l'image de zéros).
ESRGAN_SIZE = (96, 64)
RIFE_SIZE = (200, 120)


def test_image(w: int, h: int, seed: int) -> np.ndarray:
    """Image d'essai : ondulations, disques et bruit doux (des bords et des aplats)."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.zeros((h, w, 3), np.float32)
    for c in range(3):
        img[..., c] = 0.5 + 0.25 * np.sin(x / (5 + 3 * c) + c) * np.cos(y / (7 + c))
    for _ in range(12):
        cx, cy, r = rng.uniform(0, w), rng.uniform(0, h), rng.uniform(4, 16)
        color = rng.uniform(0, 1, 3)
        disk = ((x - cx) ** 2 + (y - cy) ** 2) < r * r
        img[disk] = color
    img += rng.normal(0, 0.02, img.shape).astype(np.float32)
    return (np.clip(img, 0, 1) * 255).round().astype(np.uint8)


def save_rgba(path: Path, rgb: np.ndarray) -> None:
    h, w, _ = rgb.shape
    rgba = np.concatenate([rgb, np.full((h, w, 1), 255, np.uint8)], axis=2)
    path.write_bytes(rgba.tobytes())


def to_tensor(rgb: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)[None]


def to_u8(t: torch.Tensor) -> np.ndarray:
    # Comme le shader `pack` : bornée à [0, 1], arrondie au plus proche.
    return (t[0].clamp(0, 1) * 255 + 0.5).floor().byte().permute(1, 2, 0).numpy()


def read_safetensors(path: Path) -> dict:
    data = path.read_bytes()
    (n,) = struct.unpack("<Q", data[:8])
    header = json.loads(data[8 : 8 + n])
    body = data[8 + n :]
    out = {}
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        a, b = entry["data_offsets"]
        arr = np.frombuffer(body[a:b], "<f2").astype(np.float32).reshape(entry["shape"])
        out[name] = torch.from_numpy(arr.copy())
    return out


def read_original(path: Path) -> dict:
    data = torch.load(path, map_location="cpu", weights_only=True)
    data = data.get("params", data)
    return {k.removeprefix("module."): v.float() for k, v in data.items()}


# --- Real-ESRGAN compact (SRVGGNetCompact) ------------------------------------------------


def srvgg(w: dict, x: torch.Tensor, num_conv: int) -> torch.Tensor:
    out = x
    for i in range(num_conv + 2):
        out = F.conv2d(out, w[f"body.{2 * i}.weight"], w[f"body.{2 * i}.bias"], padding=1)
        if i < num_conv + 1:
            out = F.prelu(out, w[f"body.{2 * i + 1}.weight"])
    out = F.pixel_shuffle(out, 4)
    return out + F.interpolate(x, scale_factor=4, mode="nearest")


# --- RIFE 4.25 (IFNet_HDv3) ---------------------------------------------------------------


def warp(x: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    _, _, h, w = flow.shape
    hor = torch.linspace(-1.0, 1.0, w).view(1, 1, 1, w).expand(1, -1, h, -1)
    ver = torch.linspace(-1.0, 1.0, h).view(1, 1, h, 1).expand(1, -1, -1, w)
    grid = torch.cat([hor, ver], 1)
    flow = torch.cat([flow[:, 0:1] / ((x.shape[3] - 1.0) / 2.0), flow[:, 1:2] / ((x.shape[2] - 1.0) / 2.0)], 1)
    g = (grid + flow).permute(0, 2, 3, 1)
    return F.grid_sample(x, g, mode="bilinear", padding_mode="border", align_corners=True)


def lrelu(x):
    return F.leaky_relu(x, 0.2)


def head(w, x):
    x = lrelu(F.conv2d(x, w["encode.cnn0.weight"], w["encode.cnn0.bias"], stride=2, padding=1))
    x = lrelu(F.conv2d(x, w["encode.cnn1.weight"], w["encode.cnn1.bias"], padding=1))
    x = lrelu(F.conv2d(x, w["encode.cnn2.weight"], w["encode.cnn2.bias"], padding=1))
    return F.conv_transpose2d(x, w["encode.cnn3.weight"], w["encode.cnn3.bias"], stride=2, padding=1)


def ifblock(w, i, x, flow, scale):
    p = f"block{i}"
    x = F.interpolate(x, scale_factor=1.0 / scale, mode="bilinear", align_corners=False)
    if flow is not None:
        flow = F.interpolate(flow, scale_factor=1.0 / scale, mode="bilinear", align_corners=False) * 1.0 / scale
        x = torch.cat((x, flow), 1)
    feat = lrelu(F.conv2d(x, w[f"{p}.conv0.0.0.weight"], w[f"{p}.conv0.0.0.bias"], stride=2, padding=1))
    feat = lrelu(F.conv2d(feat, w[f"{p}.conv0.1.0.weight"], w[f"{p}.conv0.1.0.bias"], stride=2, padding=1))
    for k in range(8):
        q = f"{p}.convblock.{k}"
        conv = F.conv2d(feat, w[f"{q}.conv.weight"], w[f"{q}.conv.bias"], padding=1)
        feat = lrelu(conv * w[f"{q}.beta"] + feat)
    tmp = F.conv_transpose2d(feat, w[f"{p}.lastconv.0.weight"], w[f"{p}.lastconv.0.bias"], stride=2, padding=1)
    tmp = F.pixel_shuffle(tmp, 2)
    tmp = F.interpolate(tmp, scale_factor=scale, mode="bilinear", align_corners=False)
    return tmp[:, :4] * scale, tmp[:, 4:5], tmp[:, 5:]


def ifnet(w, img0, img1, scale_list=(16, 8, 4, 2, 1)):
    timestep = (img0[:, :1].clone() * 0 + 1) * 0.5
    f0, f1 = head(w, img0), head(w, img1)
    warped_img0, warped_img1 = img0, img1
    flow = mask = feat = None
    for i in range(5):
        if flow is None:
            flow, mask, feat = ifblock(w, i, torch.cat((img0, img1, f0, f1, timestep), 1), None, scale_list[i])
        else:
            wf0 = warp(f0, flow[:, :2])
            wf1 = warp(f1, flow[:, 2:4])
            fd, mask, feat = ifblock(
                w, i, torch.cat((warped_img0, warped_img1, wf0, wf1, timestep, mask, feat), 1), flow, scale_list[i]
            )
            flow = flow + fd
        warped_img0 = warp(img0, flow[:, :2])
        warped_img1 = warp(img1, flow[:, 2:4])
    mask = torch.sigmoid(mask)
    return warped_img0 * mask + warped_img1 * (1 - mask)


def rife_middle(w, a, b):
    _, _, h, wd = a.shape
    ph, pw = ((h - 1) // 128 + 1) * 128, ((wd - 1) // 128 + 1) * 128
    pad = (0, pw - wd, 0, ph - h)
    out = ifnet(w, F.pad(a, pad), F.pad(b, pad))
    return out[:, :, :h, :wd]


def main() -> None:
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    orig, models, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)

    ew, eh = ESRGAN_SIZE
    rw, rh = RIFE_SIZE
    small = test_image(ew, eh, 1)
    a = test_image(rw, rh, 2)
    b = np.roll(a, (2, 3), axis=(0, 1))
    save_rgba(out / "small.rgba", small)
    save_rgba(out / "a.rgba", a)
    save_rgba(out / "b.rgba", b)

    for name, file, num_conv in [
        ("photo", "realesr-general-x4v3", 32),
        ("anime", "realesr-animevideov3", 16),
    ]:
        for kind, weights in [
            ("f16w", read_safetensors(models / f"{file}.safetensors")),
            ("orig", read_original(orig / f"{file}.pth")),
        ]:
            save_rgba(out / f"{name}-{kind}.rgba", to_u8(srvgg(weights, to_tensor(small), num_conv)))
    for kind, weights in [
        ("f16w", read_safetensors(models / "rife-4.25.safetensors")),
        ("orig", read_original(orig / "rife-4.25/flownet.pkl")),
    ]:
        save_rgba(out / f"rife-{kind}.rgba", to_u8(rife_middle(weights, to_tensor(a), to_tensor(b))))
    print(f"references written to {out} (esrgan {ew}x{eh}, rife {rw}x{rh})")


if __name__ == "__main__":
    main()
