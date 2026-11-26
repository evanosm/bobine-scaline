// Addition : les `n4` premiers groupes de 4 canaux de `a` et de `b`, dans un nouveau tenseur.

struct P {
    w: u32,
    h: u32,
    a_c4: u32,
    b_c4: u32,
    n4: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> a: array<vec4<S>>;
@group(0) @binding(2) var<storage, read> b: array<vec4<S>>;
@group(0) @binding(3) var<storage, read_write> dst: array<vec4<S>>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= p.w || y >= p.h) {
        return;
    }
    let px = y * p.w + x;
    for (var c = 0u; c < p.n4; c++) {
        let i = c * p.w * p.h + px;
        dst[i] = vec4<S>(vec4<f32>(a[i]) + vec4<f32>(b[i]));
    }
}
