//! Chaque opération WGSL comparée à un calcul de référence sur le processeur (mêmes formules que
//! PyTorch), sur de petits tenseurs aléatoires. Sans GPU (ni logiciel), les tests sont sautés,
//! sauf si `SCALINE_REQUIRE_GPU=1` (la CI installe lavapipe, le Vulkan logiciel de Mesa).

use scaline_worker::gpu::Gpu;

/// Générateur pseudo-aléatoire déterministe (xorshift), valeurs dans [-1, 1].
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
    fn vec(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.next()).collect()
    }
}

fn gpu(f16: bool) -> Option<Gpu> {
    match Gpu::new(f16) {
        Ok(gpu) => Some(gpu),
        Err(e) if std::env::var_os("SCALINE_REQUIRE_GPU").is_some() => {
            panic!("GPU exigé : {e:?}")
        }
        Err(e) => {
            eprintln!("aucun GPU, test sauté : {e:?}");
            None
        }
    }
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

/// Tenseur `[c][h][w]` lu avec zéro hors de l'image.
fn at(t: &[f32], w: usize, h: usize, ch: usize, y: i64, x: i64) -> f32 {
    if x < 0 || y < 0 || x >= w as i64 || y >= h as i64 {
        return 0.0;
    }
    t[(ch * h + y as usize) * w + x as usize]
}

#[allow(clippy::too_many_arguments)]
fn conv_ref(
    x: &[f32],
    cin: usize,
    w: usize,
    h: usize,
    wt: &[f32],
    b: &[f32],
    cout: usize,
    stride: usize,
    slope: Option<&[f32]>,
    res: Option<&[f32]>,
) -> Vec<f32> {
    let (ow, oh) = (w.div_ceil(stride), h.div_ceil(stride));
    let mut out = vec![0f32; cout * ow * oh];
    for co in 0..cout {
        for y in 0..oh {
            for xx in 0..ow {
                let mut acc = b[co];
                for ci in 0..cin {
                    for ky in 0..3 {
                        for kx in 0..3 {
                            let iy = (y * stride + ky) as i64 - 1;
                            let ix = (xx * stride + kx) as i64 - 1;
                            acc +=
                                wt[((co * cin + ci) * 3 + ky) * 3 + kx] * at(x, w, h, ci, iy, ix);
                        }
                    }
                }
                let i = (co * oh + y) * ow + xx;
                if let Some(r) = res {
                    acc += r[i];
                }
                if let Some(s) = slope
                    && acc < 0.0
                {
                    acc *= s[co];
                }
                out[i] = acc;
            }
        }
    }
    out
}

#[test]
fn conv3x3_matches_the_reference_with_stride_residual_and_prelu() {
    let Some(gpu) = gpu(false) else { return };
    let mut rng = Rng(7);
    for (cin, cout, w, h, stride) in [
        (3, 16, 13, 9, 1),
        (15, 8, 12, 10, 2),
        (64, 48, 9, 7, 1),
        (5, 6, 11, 5, 2),
    ] {
        let x = rng.vec(cin * w * h);
        let wt = rng.vec(cout * cin * 9);
        let b = rng.vec(cout);
        let slope = rng.vec(cout);
        let xt = gpu.upload(&x, cin as u32, w as u32, h as u32);
        let weights = gpu.weights(&wt, &b, &slope, cin as u32, cout as u32, 3, false);
        let (ow, oh) = (w.div_ceil(stride), h.div_ceil(stride));
        let res = rng.vec(cout * ow * oh);
        let rt = gpu.upload(&res, cout as u32, ow as u32, oh as u32);
        let mut batch = gpu.batch();
        let plain = batch.conv(&xt, &weights, stride as u32, false, None);
        let full = batch.conv(&xt, &weights, stride as u32, true, Some(&rt));
        batch.submit();
        let want = conv_ref(&x, cin, w, h, &wt, &b, cout, stride, None, None);
        let d = max_diff(&gpu.download(&plain).expect("relu"), &want);
        assert!(d < 1e-4, "conv {cin}->{cout} s{stride}: {d}");
        let want = conv_ref(
            &x,
            cin,
            w,
            h,
            &wt,
            &b,
            cout,
            stride,
            Some(&slope),
            Some(&res),
        );
        let d = max_diff(&gpu.download(&full).expect("relu"), &want);
        assert!(d < 1e-4, "conv+res+prelu {cin}->{cout} s{stride}: {d}");
    }
}

#[test]
fn deconv4x4_stride2_matches_conv_transpose2d() {
    let Some(gpu) = gpu(false) else { return };
    let mut rng = Rng(11);
    for (cin, cout, w, h) in [(16, 4, 5, 4), (32, 52, 6, 3)] {
        let x = rng.vec(cin * w * h);
        let wt = rng.vec(cin * cout * 16); // [cin][cout][4][4]
        let b = rng.vec(cout);
        let xt = gpu.upload(&x, cin as u32, w as u32, h as u32);
        let weights = gpu.weights(&wt, &b, &[], cin as u32, cout as u32, 4, true);
        let mut batch = gpu.batch();
        let out = batch.deconv(&xt, &weights);
        batch.submit();
        // Définition de ConvTranspose2d : chaque entrée (iy, ix) s'étale sur 4 × 4 sorties.
        let (ow, oh) = (2 * w, 2 * h);
        let mut want = vec![0f32; cout * ow * oh];
        for co in 0..cout {
            for i in 0..ow * oh {
                want[co * ow * oh + i] = b[co];
            }
        }
        for ci in 0..cin {
            for iy in 0..h {
                for ix in 0..w {
                    for ky in 0..4 {
                        for kx in 0..4 {
                            let oy = (iy * 2 + ky) as i64 - 1;
                            let ox = (ix * 2 + kx) as i64 - 1;
                            if oy < 0 || ox < 0 || oy >= oh as i64 || ox >= ow as i64 {
                                continue;
                            }
                            for co in 0..cout {
                                want[(co * oh + oy as usize) * ow + ox as usize] += x
                                    [(ci * h + iy) * w + ix]
                                    * wt[((ci * cout + co) * 4 + ky) * 4 + kx];
                            }
                        }
                    }
                }
            }
        }
        let d = max_diff(&gpu.download(&out).expect("relu"), &want);
        assert!(d < 1e-4, "deconv {cin}->{cout}: {d}");
    }
}

#[test]
fn pixel_shuffle_moves_channels_to_pixels_and_adds_the_nearest_base() {
    let Some(gpu) = gpu(false) else { return };
    let mut rng = Rng(3);
    let (c, r, w, h) = (3usize, 4usize, 5usize, 3usize);
    let x = rng.vec(c * r * r * w * h);
    let base = rng.vec(c * w * h);
    let xt = gpu.upload(&x, (c * r * r) as u32, w as u32, h as u32);
    let bt = gpu.upload(&base, c as u32, w as u32, h as u32);
    let mut batch = gpu.batch();
    let out = batch.shuffle(&xt, r as u32, Some(&bt));
    let x52 = rng.vec(52 * w * h);
    let plain = batch.shuffle(&gpu.upload(&x52, 52, w as u32, h as u32), 2, None);
    batch.submit();
    let (ow, oh) = (w * r, h * r);
    let mut want = vec![0f32; c * ow * oh];
    for ch in 0..c {
        for y in 0..oh {
            for xx in 0..ow {
                let ic = ch * r * r + (y % r) * r + xx % r;
                want[(ch * oh + y) * ow + xx] =
                    x[(ic * h + y / r) * w + xx / r] + base[(ch * h + y / r) * w + xx / r];
            }
        }
    }
    assert!(max_diff(&gpu.download(&out).expect("relu"), &want) < 1e-6);
    let got = gpu.download(&plain).expect("relu");
    assert_eq!(got.len(), 13 * 4 * w * h);
    // r = 2, 13 canaux : la sortie (1, 3) du canal 5 vient du canal 5 × 4 + 1 × 2 + 1.
    assert_eq!(
        got[(5 * 2 * h + 3) * 2 * w + 1],
        x52[((5 * 4 + 3) * h + 1) * w]
    );
}

#[test]
fn bilinear_resize_matches_interpolate_without_aligned_corners() {
    let Some(gpu) = gpu(false) else { return };
    let mut rng = Rng(5);
    let (c, w, h) = (6usize, 16usize, 8usize);
    let x = rng.vec(c * w * h);
    let xt = gpu.upload(&x, c as u32, w as u32, h as u32);
    for (ow, oh, mul) in [(4usize, 2usize, 0.25f32), (64, 32, 4.0)] {
        let mut batch = gpu.batch();
        let out = batch.resize(&xt, ow as u32, oh as u32, mul, (0, 1));
        batch.submit();
        let (rx, ry) = (w as f32 / ow as f32, h as f32 / oh as f32);
        let mut want = vec![0f32; c * ow * oh];
        for ch in 0..c {
            for y in 0..oh {
                for xx in 0..ow {
                    let sx = ((xx as f32 + 0.5) * rx - 0.5).max(0.0);
                    let sy = ((y as f32 + 0.5) * ry - 0.5).max(0.0);
                    let (x0, y0) = ((sx as usize).min(w - 1), (sy as usize).min(h - 1));
                    let (x1, y1) = ((x0 + 1).min(w - 1), (y0 + 1).min(h - 1));
                    let (lx, ly) = (sx - x0 as f32, sy - y0 as f32);
                    let v = |yy: usize, xx: usize| x[(ch * h + yy) * w + xx];
                    let mut o = (v(y0, x0) * (1.0 - lx) + v(y0, x1) * lx) * (1.0 - ly)
                        + (v(y1, x0) * (1.0 - lx) + v(y1, x1) * lx) * ly;
                    if ch < 4 {
                        o *= mul;
                    }
                    want[(ch * oh + y) * ow + xx] = o;
                }
            }
        }
        let d = max_diff(&gpu.download(&out).expect("relu"), &want);
        assert!(d < 1e-5, "resize {ow}x{oh}: {d}");
    }
}

#[test]
fn warp_samples_at_the_flow_with_border_clamping() {
    let Some(gpu) = gpu(false) else { return };
    let mut rng = Rng(9);
    let (c, w, h) = (4usize, 9usize, 7usize);
    let x = rng.vec(c * w * h);
    let flow: Vec<f32> = rng.vec(4 * w * h).iter().map(|v| v * 4.0).collect();
    let xt = gpu.upload(&x, c as u32, w as u32, h as u32);
    let ft = gpu.upload(&flow, 4, w as u32, h as u32);
    let mut batch = gpu.batch();
    let out = batch.warp(&xt, &ft, 2);
    batch.submit();
    let mut want = vec![0f32; c * w * h];
    for y in 0..h {
        for xx in 0..w {
            let fx = flow[(2 * h + y) * w + xx];
            let fy = flow[(3 * h + y) * w + xx];
            let sx = (xx as f32 + fx).clamp(0.0, (w - 1) as f32);
            let sy = (y as f32 + fy).clamp(0.0, (h - 1) as f32);
            let (x0, y0) = (sx.floor() as usize, sy.floor() as usize);
            let (x1, y1) = ((x0 + 1).min(w - 1), (y0 + 1).min(h - 1));
            let (lx, ly) = (sx - x0 as f32, sy - y0 as f32);
            for ch in 0..c {
                let v = |yy: usize, xx: usize| x[(ch * h + yy) * w + xx];
                want[(ch * h + y) * w + xx] = (v(y0, x0) * (1.0 - lx) + v(y0, x1) * lx)
                    * (1.0 - ly)
                    + (v(y1, x0) * (1.0 - lx) + v(y1, x1) * lx) * ly;
            }
        }
    }
    let d = max_diff(&gpu.download(&out).expect("relu"), &want);
    assert!(d < 1e-5, "warp: {d}");
}

#[test]
fn concat_add_and_sigmoid_blend_match_the_reference() {
    let Some(gpu) = gpu(false) else { return };
    let mut rng = Rng(13);
    let (w, h) = (5usize, 4usize);
    let a = rng.vec(3 * w * h);
    let b = rng.vec(3 * w * h);
    let m = rng.vec(13 * w * h);
    let (at_, bt, mt) = (
        gpu.upload(&a, 3, w as u32, h as u32),
        gpu.upload(&b, 3, w as u32, h as u32),
        gpu.upload(&m, 13, w as u32, h as u32),
    );
    let cat = gpu.tensor(15, w as u32, h as u32);
    let mut batch = gpu.batch();
    batch.copy(&at_, 0, &cat, 0, 3);
    batch.copy(&bt, 0, &cat, 3, 3);
    batch.copy(&mt, 4, &cat, 6, 8);
    batch.fill(&cat, 14, 1, 0.5);
    let sum = batch.add(&mt, &mt, 4);
    let mix = batch.blend(&at_, &bt, &mt, 4);
    batch.submit();
    let got = gpu.download(&cat).expect("relu");
    let n = w * h;
    assert_eq!(&got[..3 * n], &a[..]);
    assert_eq!(&got[3 * n..6 * n], &b[..]);
    assert_eq!(&got[6 * n..14 * n], &m[4 * n..12 * n]);
    assert!(got[14 * n..].iter().all(|&v| v == 0.5));
    let s = gpu.download(&sum).expect("relu");
    assert!(max_diff(&s, &m[..4 * n].iter().map(|v| 2.0 * v).collect::<Vec<_>>()) < 1e-6);
    let mixed = gpu.download(&mix).expect("relu");
    for i in 0..3 * n {
        let k = 1.0 / (1.0 + (-m[4 * n + i % n]).exp());
        assert!((mixed[i] - (a[i] * k + b[i] * (1.0 - k))).abs() < 1e-5);
    }
}

#[test]
fn rgba_images_round_trip_through_tensors() {
    let Some(gpu) = gpu(true) else { return };
    let (w, h) = (6u32, 5u32);
    let pixels: Vec<u8> = (0..w * h * 4).map(|i| (i * 37 % 256) as u8).collect();
    let image = gpu.image_buffer(w, h);
    gpu.queue.write_buffer(&image, 0, &pixels);
    let out = gpu.image_buffer(4, 2);
    let mut batch = gpu.batch();
    // Tenseur 8 × 8 lu depuis (1, 1) : ce qui dépasse l'image vaut zéro.
    let t = batch.unpack(&image, w, h, 1, 1, 8, 8);
    batch.pack(&t, &out, 4, 2, 1, 2);
    batch.submit();
    let values = gpu.download(&t).expect("lecture");
    assert_eq!(values[7], 0.0); // (x = 7) hors de l'image
    let got = gpu.read_buffer(&out, 4 * 2 * 4).expect("lecture");
    for y in 0..2 {
        for x in 0..4 {
            let src = (((y + 3) * w + x + 2) * 4) as usize;
            let dst = ((y * 4 + x) * 4) as usize;
            assert_eq!(&got[dst..dst + 3], &pixels[src..src + 3], "({x}, {y})");
            assert_eq!(got[dst + 3], 255);
        }
    }
}

#[test]
fn f16_storage_and_math_stay_close_to_f32() {
    let (Some(g32), Some(g16)) = (gpu(false), gpu(true)) else {
        return;
    };
    if !g16.f16 {
        eprintln!("pas de f16 sur ce GPU : comparaison sautée");
        return;
    }
    let mut rng = Rng(21);
    let (cin, cout, w, h) = (64usize, 64usize, 10usize, 8usize);
    let x = rng.vec(cin * w * h);
    let wt: Vec<f32> = rng.vec(cout * cin * 9).iter().map(|v| v * 0.05).collect();
    let b = rng.vec(cout);
    let run = |gpu: &Gpu, exact: bool| {
        let xt = gpu.upload(&x, cin as u32, w as u32, h as u32);
        let mut weights = gpu.weights(&wt, &b, &[0.1; 64], cin as u32, cout as u32, 3, false);
        weights.exact = exact;
        let mut batch = gpu.batch();
        let out = batch.conv(&xt, &weights, 1, true, None);
        batch.submit();
        gpu.download(&out).expect("lecture")
    };
    // Stockage f16, calculs f32 (RIFE) : seul l'arrondi du stockage compte.
    let d = max_diff(&run(&g32, true), &run(&g16, true));
    assert!(d < 5e-3, "f16 exact vs f32: {d}");
    // Calculs en f16 (Real-ESRGAN) : somme de 576 produits en f16.
    let d = max_diff(&run(&g32, false), &run(&g16, false));
    assert!(d < 2e-2, "f16 vs f32: {d}");
}
