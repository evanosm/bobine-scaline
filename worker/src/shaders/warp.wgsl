// Déformation par un flux optique (`warp` de RIFE : `grid_sample` bilinéaire, align_corners=True,
// bord répété) : la sortie en (x, y) lit l'entrée en (x + fx, y + fy), ramené dans l'image.

struct P {
    c4: u32,
    w: u32,
    h: u32,
    flow_c4: u32,
    // canal de fx dans le tenseur du flux (fy est le suivant)
    flow_off: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> src: array<vec4<S>>;
@group(0) @binding(2) var<storage, read> flow: array<vec4<S>>;
@group(0) @binding(3) var<storage, read_write> dst: array<vec4<S>>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= p.w || y >= p.h) {
        return;
    }
    let px = y * p.w + x;
    let plane = p.w * p.h;
    let fx = f32(flow[(p.flow_off / 4u) * plane + px][p.flow_off % 4u]);
    let fy = f32(flow[((p.flow_off + 1u) / 4u) * plane + px][(p.flow_off + 1u) % 4u]);
    let sx = clamp(f32(x) + fx, 0.0, f32(p.w - 1u));
    let sy = clamp(f32(y) + fy, 0.0, f32(p.h - 1u));
    let x0 = u32(floor(sx));
    let y0 = u32(floor(sy));
    let x1 = min(x0 + 1u, p.w - 1u);
    let y1 = min(y0 + 1u, p.h - 1u);
    let lx = sx - f32(x0);
    let ly = sy - f32(y0);
    for (var c = 0u; c < p.c4; c++) {
        let a = vec4<f32>(src[(c * p.h + y0) * p.w + x0]);
        let b = vec4<f32>(src[(c * p.h + y0) * p.w + x1]);
        let d = vec4<f32>(src[(c * p.h + y1) * p.w + x0]);
        let e = vec4<f32>(src[(c * p.h + y1) * p.w + x1]);
        let o = (a * (1.0 - lx) + b * lx) * (1.0 - ly) + (d * (1.0 - lx) + e * lx) * ly;
        dst[(c * p.h + y) * p.w + x] = vec4<S>(o);
    }
}
