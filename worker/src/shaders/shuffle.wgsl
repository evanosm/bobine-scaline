// Pixel shuffle de facteur r (`nn.PixelShuffle(r)`) : c × r² canaux à (w, h) deviennent c canaux
// à (w r, h r). Option : ajoute `base` agrandi au plus proche (`F.interpolate(mode="nearest")`),
// l'image d'entrée de Real-ESRGAN compact.

struct P {
    in_c4: u32,
    out_c: u32,
    out_c4: u32,
    in_w: u32,
    in_h: u32,
    r: u32,
    // 1 : ajoute `base` (même taille que l'entrée, out_c4 groupes)
    flags: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> src: array<vec4<S>>;
@group(0) @binding(2) var<storage, read> base: array<vec4<S>>;
@group(0) @binding(3) var<storage, read_write> dst: array<vec4<S>>;

/// Canal de sortie `c` d'un pixel : canal `c × r² + sub` de l'entrée (zéro au-delà de `out_c`).
fn channel(c: u32, sub: u32, in_px: u32, in_plane: u32) -> f32 {
    if (c >= p.out_c) {
        return 0.0;
    }
    let ic = c * p.r * p.r + sub;
    return f32(src[(ic / 4u) * in_plane + in_px][ic % 4u]);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let out_w = p.in_w * p.r;
    let out_h = p.in_h * p.r;
    let x = gid.x;
    let y = gid.y;
    if (x >= out_w || y >= out_h) {
        return;
    }
    let ix = x / p.r;
    let iy = y / p.r;
    let sub = (y % p.r) * p.r + (x % p.r);
    let in_px = iy * p.in_w + ix;
    let in_plane = p.in_w * p.in_h;
    for (var c4 = 0u; c4 < p.out_c4; c4++) {
        // Quatre lectures séparées : pas d'écriture à un indice variable dans un vecteur
        // (le compilateur FXC de DirectX déroulerait la boucle et refuserait le shader).
        var o = vec4<f32>(
            channel(c4 * 4u, sub, in_px, in_plane),
            channel(c4 * 4u + 1u, sub, in_px, in_plane),
            channel(c4 * 4u + 2u, sub, in_px, in_plane),
            channel(c4 * 4u + 3u, sub, in_px, in_plane),
        );
        if ((p.flags & 1u) != 0u) {
            o += vec4<f32>(base[c4 * in_plane + in_px]);
        }
        dst[(c4 * out_h + y) * out_w + x] = vec4<S>(o);
    }
}
