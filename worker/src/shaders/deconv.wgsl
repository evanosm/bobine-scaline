// Déconvolution 4×4, pas 2, marge 1 (`ConvTranspose2d(c_in, c_out, 4, 2, 1)` de PyTorch) : la
// sortie fait deux fois la taille de l'entrée. Chaque pixel de sortie reçoit 2 × 2 prises,
// choisies par la parité de ses coordonnées. Un fil = un pixel × CO groupes de sortie.

struct P {
    cin4: u32,
    cout4: u32,
    in_w: u32,
    in_h: u32,
    out_w: u32,
    out_h: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> src: array<vec4<S>>;
// [groupe de sortie][prise ky * 4 + kx][groupe d'entrée][canal d'entrée 0..3] → vec4 des sorties
@group(0) @binding(2) var<storage, read> wts: array<vec4<S>>;
@group(0) @binding(3) var<storage, read> bias: array<vec4<S>>;
@group(0) @binding(4) var<storage, read_write> dst: array<vec4<S>>;

const CO: u32 = 4u;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    let co0 = gid.z * CO;
    if (x >= p.out_w || y >= p.out_h) {
        return;
    }
    var acc: array<vec4<f32>, CO>;
    // Entrée iy = (y + 1 - ky) / 2 quand y + 1 - ky est pair : ky = 1 - y % 2 + 2 j.
    let ky0 = 1u - (y & 1u);
    let kx0 = 1u - (x & 1u);
    for (var j = 0u; j < 2u; j++) {
        let ky = ky0 + 2u * j;
        let ny = i32(y) + 1 - i32(ky);
        if (ny < 0 || ny / 2 >= i32(p.in_h)) {
            continue;
        }
        let iy = u32(ny / 2);
        for (var k = 0u; k < 2u; k++) {
            let kx = kx0 + 2u * k;
            let nx = i32(x) + 1 - i32(kx);
            if (nx < 0 || nx / 2 >= i32(p.in_w)) {
                continue;
            }
            let ix = u32(nx / 2);
            let tap = ky * 4u + kx;
            for (var ci = 0u; ci < p.cin4; ci++) {
                let v = vec4<f32>(src[(ci * p.in_h + iy) * p.in_w + ix]);
                for (var c = 0u; c < CO; c++) {
                    let co = co0 + c;
                    if (co >= p.cout4) {
                        break;
                    }
                    let base = ((co * 16u + tap) * p.cin4 + ci) * 4u;
                    acc[c] += vec4<f32>(wts[base]) * v.x + vec4<f32>(wts[base + 1u]) * v.y
                        + vec4<f32>(wts[base + 2u]) * v.z + vec4<f32>(wts[base + 3u]) * v.w;
                }
            }
        }
    }
    for (var c = 0u; c < CO; c++) {
        let co = co0 + c;
        if (co >= p.cout4) {
            break;
        }
        dst[(co * p.out_h + y) * p.out_w + x] = vec4<S>(acc[c] + vec4<f32>(bias[co]));
    }
}
