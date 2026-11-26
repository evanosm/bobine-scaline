// Concaténation de canaux : copie `count` canaux de `src` (à partir de `src_off`) dans `dst` (à
// partir de `dst_off`), ou les remplit d'une constante (`fill` = 1). Un fil par pixel : un
// seul fil écrit chaque groupe de 4 canaux.

struct P {
    w: u32,
    h: u32,
    src_c4: u32,
    dst_c4: u32,
    src_off: u32,
    dst_off: u32,
    count: u32,
    fill: u32,
    value: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> src: array<vec4<S>>;
@group(0) @binding(2) var<storage, read_write> dst: array<vec4<S>>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= p.w || y >= p.h) {
        return;
    }
    let px = y * p.w + x;
    let plane = p.w * p.h;
    for (var i = 0u; i < p.count; i++) {
        var v = S(p.value);
        if (p.fill == 0u) {
            let s = p.src_off + i;
            v = src[(s / 4u) * plane + px][s % 4u];
        }
        let d = p.dst_off + i;
        dst[(d / 4u) * plane + px][d % 4u] = v;
    }
}
