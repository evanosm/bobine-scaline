// Redimensionnement bilinéaire, `F.interpolate(mode="bilinear", align_corners=False)` de PyTorch :
// source = (dst + 0,5) × ratio − 0,5, ramenée à 0 si négative. Les groupes de canaux
// [mul_from, mul_to) sont multipliés par `mul` (un flux optique change d'échelle avec l'image).

struct P {
    c4: u32,
    in_w: u32,
    in_h: u32,
    out_w: u32,
    out_h: u32,
    mul_from: u32,
    mul_to: u32,
    mul: f32,
    ratio_x: f32,
    ratio_y: f32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> src: array<vec4<S>>;
@group(0) @binding(2) var<storage, read_write> dst: array<vec4<S>>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= p.out_w || y >= p.out_h) {
        return;
    }
    let sx = max((f32(x) + 0.5) * p.ratio_x - 0.5, 0.0);
    let sy = max((f32(y) + 0.5) * p.ratio_y - 0.5, 0.0);
    let x0 = min(u32(sx), p.in_w - 1u);
    let y0 = min(u32(sy), p.in_h - 1u);
    let x1 = min(x0 + 1u, p.in_w - 1u);
    let y1 = min(y0 + 1u, p.in_h - 1u);
    let lx = sx - f32(x0);
    let ly = sy - f32(y0);
    for (var c = 0u; c < p.c4; c++) {
        let a = vec4<f32>(src[(c * p.in_h + y0) * p.in_w + x0]);
        let b = vec4<f32>(src[(c * p.in_h + y0) * p.in_w + x1]);
        let d = vec4<f32>(src[(c * p.in_h + y1) * p.in_w + x0]);
        let e = vec4<f32>(src[(c * p.in_h + y1) * p.in_w + x1]);
        var o = (a * (1.0 - lx) + b * lx) * (1.0 - ly) + (d * (1.0 - lx) + e * lx) * ly;
        if (c >= p.mul_from && c < p.mul_to) {
            o *= p.mul;
        }
        dst[(c * p.out_h + y) * p.out_w + x] = vec4<S>(o);
    }
}
