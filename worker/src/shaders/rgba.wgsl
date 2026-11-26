// Passage entre les images RGBA 8 bits (un u32 par pixel, octets R, G, B, A) et les tenseurs de
// 3 canaux entre 0 et 1 (un groupe de 4, le 4ᵉ à zéro).
// `unpack` : le tenseur (tw × th) lit l'image à partir de (ox, oy) ; hors de l'image, zéro.
// `pack` : l'image (iw × ih) lit le tenseur à partir de (ox, oy), bornée à [0, 1], arrondie.

struct P {
    iw: u32,
    ih: u32,
    tw: u32,
    th: u32,
    ox: u32,
    oy: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> image: array<u32>;
@group(0) @binding(2) var<storage, read_write> tensor: array<vec4<S>>;

@compute @workgroup_size(8, 8, 1)
fn unpack(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= p.tw || y >= p.th) {
        return;
    }
    let sx = p.ox + x;
    let sy = p.oy + y;
    var v = vec4<f32>(0.0);
    if (sx < p.iw && sy < p.ih) {
        v = unpack4x8unorm(image[sy * p.iw + sx]);
        v.w = 0.0;
    }
    tensor[y * p.tw + x] = vec4<S>(v);
}

@compute @workgroup_size(8, 8, 1)
fn pack(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= p.iw || y >= p.ih) {
        return;
    }
    var v = clamp(vec4<f32>(tensor[(p.oy + y) * p.tw + p.ox + x]), vec4<f32>(0.0), vec4<f32>(1.0));
    v.w = 1.0;
    // pack4x8unorm arrondit au plus proche (round(v × 255)).
    image[y * p.iw + x] = pack4x8unorm(v);
}
