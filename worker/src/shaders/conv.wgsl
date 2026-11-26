// Convolution 3×3 (pas 1 ou 2, bord à zéro) suivie, au choix, d'une addition (bloc résiduel) et
// d'une activation PReLU (LeakyReLU = PReLU de pente fixe).
//
// Tenseurs rangés par plans de 4 canaux : `données[(groupe × h + y) × w + x]` (`vec4`). Un groupe
// de 32 × 2 fils calcule 128 × 2 pixels et 4 groupes de 4 canaux de sortie. Chaque fil fait les
// pixels x, x + 32, x + 64, x + 96 d'une ligne × 4 groupes : un poids lu sert 4 fois, une
// entrée lue 16 fois, et 32 fils voisins lisent 32 entrées voisines (lectures groupées).
//
// Les 16 accumulateurs sont des variables séparées et la boucle interne est écrite à la main :
// naga traduit toute boucle en `while` gardé, que le compilateur du GPU ne déroule pas, et un
// tableau indexé dans une boucle finit en mémoire (dix fois plus lent).
// Les poids, biais et pentes sont complétés de zéros jusqu'à un multiple de 4 groupes de sortie.
// `S` (type de stockage, f16 ou f32) et `A` (type des calculs : f16, deux fois plus rapide, ou
// f32, plus exact) sont définis par le préambule ajouté au chargement.

struct P {
    cin4: u32,
    cout4: u32,
    in_w: u32,
    in_h: u32,
    out_w: u32,
    out_h: u32,
    stride: u32,
    // bit 0 : activation, bit 1 : addition de `res`
    flags: u32,
}

@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> src: array<vec4<S>>;
// [groupe de sortie][prise 0..8][groupe d'entrée][canal d'entrée 0..3] → vec4 des 4 sorties
@group(0) @binding(2) var<storage, read> wts: array<vec4<S>>;
@group(0) @binding(3) var<storage, read> bias: array<vec4<S>>;
@group(0) @binding(4) var<storage, read> slope: array<vec4<S>>;
@group(0) @binding(5) var<storage, read> res: array<vec4<S>>;
@group(0) @binding(6) var<storage, read_write> dst: array<vec4<S>>;

/// Les 4 poids (4 sorties chacun) du groupe de sortie `co`, de la prise `tap`, du groupe d'entrée `ci`.
fn weights(co: u32, tap: u32, ci: u32) -> mat4x4<A> {
    let base = ((co * 9u + tap) * p.cin4 + ci) * 4u;
    return mat4x4<A>(
        vec4<A>(wts[base]),
        vec4<A>(wts[base + 1u]),
        vec4<A>(wts[base + 2u]),
        vec4<A>(wts[base + 3u]),
    );
}

/// Biais, résidu, activation, puis écriture d'un pixel d'un groupe de sortie.
fn store(co: u32, x: u32, y: u32, acc_a: vec4<A>) {
    let acc = vec4<f32>(acc_a);
    if (co >= p.cout4 || x >= p.out_w) {
        return;
    }
    let idx = (co * p.out_h + y) * p.out_w + x;
    var o = acc + vec4<f32>(bias[co]);
    if ((p.flags & 2u) != 0u) {
        o += vec4<f32>(res[idx]);
    }
    if ((p.flags & 1u) != 0u) {
        o = select(o * vec4<f32>(slope[co]), o, o >= vec4<f32>(0.0));
    }
    dst[idx] = vec4<S>(o);
}

@compute @workgroup_size(32, 2, 1)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let xs = vec4<u32>(0u, 32u, 64u, 96u) + wid.x * 128u + lid.x;
    let y = wid.y * 2u + lid.y;
    let co0 = wid.z * 4u;
    if (xs.x >= p.out_w || y >= p.out_h) {
        return;
    }
    // aCI : groupe de sortie co0 + C, pixel xs[I].
    var a00 = vec4<A>(0.0);
    var a01 = vec4<A>(0.0);
    var a02 = vec4<A>(0.0);
    var a03 = vec4<A>(0.0);
    var a10 = vec4<A>(0.0);
    var a11 = vec4<A>(0.0);
    var a12 = vec4<A>(0.0);
    var a13 = vec4<A>(0.0);
    var a20 = vec4<A>(0.0);
    var a21 = vec4<A>(0.0);
    var a22 = vec4<A>(0.0);
    var a23 = vec4<A>(0.0);
    var a30 = vec4<A>(0.0);
    var a31 = vec4<A>(0.0);
    var a32 = vec4<A>(0.0);
    var a33 = vec4<A>(0.0);
    let last_x = i32(p.in_w) - 1;
    let plane = p.in_w * p.in_h;
    for (var tap = 0u; tap < 9u; tap++) {
        let ky = tap / 3u;
        let kx = tap % 3u;
        let iy = i32(y * p.stride + ky) - 1;
        if (iy < 0 || iy >= i32(p.in_h)) {
            continue;
        }
        let row = u32(iy) * p.in_w;
        // Colonnes lues par les 4 pixels (bornées ; hors de l'image, valeur nulle).
        let ix = vec4<i32>(xs * p.stride + kx) - vec4<i32>(1);
        let inside = (ix >= vec4<i32>(0)) & (ix <= vec4<i32>(last_x));
        let cx = vec4<u32>(clamp(ix, vec4<i32>(0), vec4<i32>(last_x))) + row;
        for (var ci = 0u; ci < p.cin4; ci++) {
            let o = ci * plane;
            let v0 = select(vec4<A>(0.0), vec4<A>(src[o + cx.x]), inside.x);
            let v1 = select(vec4<A>(0.0), vec4<A>(src[o + cx.y]), inside.y);
            let v2 = select(vec4<A>(0.0), vec4<A>(src[o + cx.z]), inside.z);
            let v3 = select(vec4<A>(0.0), vec4<A>(src[o + cx.w]), inside.w);
            // Une matrice 4 × 4 par groupe de sortie : colonne j = poids du canal d'entrée j.
            let w0 = weights(co0, tap, ci);
            a00 += w0 * v0;
            a01 += w0 * v1;
            a02 += w0 * v2;
            a03 += w0 * v3;
            let w1 = weights(co0 + 1u, tap, ci);
            a10 += w1 * v0;
            a11 += w1 * v1;
            a12 += w1 * v2;
            a13 += w1 * v3;
            let w2 = weights(co0 + 2u, tap, ci);
            a20 += w2 * v0;
            a21 += w2 * v1;
            a22 += w2 * v2;
            a23 += w2 * v3;
            let w3 = weights(co0 + 3u, tap, ci);
            a30 += w3 * v0;
            a31 += w3 * v1;
            a32 += w3 * v2;
            a33 += w3 * v3;
        }
    }
    store(co0, xs.x, y, a00);
    store(co0, xs.y, y, a01);
    store(co0, xs.z, y, a02);
    store(co0, xs.w, y, a03);
    store(co0 + 1u, xs.x, y, a10);
    store(co0 + 1u, xs.y, y, a11);
    store(co0 + 1u, xs.z, y, a12);
    store(co0 + 1u, xs.w, y, a13);
    store(co0 + 2u, xs.x, y, a20);
    store(co0 + 2u, xs.y, y, a21);
    store(co0 + 2u, xs.z, y, a22);
    store(co0 + 2u, xs.w, y, a23);
    store(co0 + 3u, xs.x, y, a30);
    store(co0 + 3u, xs.y, y, a31);
    store(co0 + 3u, xs.z, y, a32);
    store(co0 + 3u, xs.w, y, a33);
}
