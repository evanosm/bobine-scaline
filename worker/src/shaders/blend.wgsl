// Mélange final de RIFE : a × sigmoïde(m) + b × (1 − sigmoïde(m)), m lu dans un canal de `mask`.

struct P {
    c4: u32,
    w: u32,
    h: u32,
    mask_c4: u32,
    mask_ch: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> a: array<vec4<S>>;
@group(0) @binding(2) var<storage, read> b: array<vec4<S>>;
@group(0) @binding(3) var<storage, read> mask: array<vec4<S>>;
@group(0) @binding(4) var<storage, read_write> dst: array<vec4<S>>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= p.w || y >= p.h) {
        return;
    }
    let px = y * p.w + x;
    let plane = p.w * p.h;
    let m = f32(mask[(p.mask_ch / 4u) * plane + px][p.mask_ch % 4u]);
    let k = 1.0 / (1.0 + exp(-m));
    for (var c = 0u; c < p.c4; c++) {
        let i = c * plane + px;
        dst[i] = vec4<S>(vec4<f32>(a[i]) * k + vec4<f32>(b[i]) * (1.0 - k));
    }
}
