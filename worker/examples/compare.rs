//! Compare les sorties des shaders aux références PyTorch de `scripts/reference.py`, puis mesure
//! la vitesse en 1080p sur ce GPU.
//!
//!     cargo run --release --example compare -- <dossier des références> [models/]
//!
//! Écart en niveaux (0 à 255) sur les images 8 bits : maximum, moyenne, part des pixels exacts.

use scaline_worker::esrgan::Esrgan;
use scaline_worker::gpu::Gpu;
use scaline_worker::resize::resize_rgba;
use scaline_worker::rife::Rife;
use scaline_worker::safetensors::SafeTensors;
use std::path::{Path, PathBuf};
use std::time::Instant;

fn read(dir: &Path, name: &str) -> Vec<u8> {
    std::fs::read(dir.join(name)).unwrap_or_else(|e| panic!("{name}: {e}"))
}

/// (maximum, moyenne, part exacte) des écarts sur R, G, B.
fn diff(a: &[u8], b: &[u8]) -> (u8, f64, f64) {
    assert_eq!(a.len(), b.len(), "tailles différentes");
    let (mut max, mut sum, mut exact, mut n) = (0u8, 0u64, 0u64, 0u64);
    for (pa, pb) in a.chunks(4).zip(b.chunks(4)) {
        for c in 0..3 {
            let d = pa[c].abs_diff(pb[c]);
            max = max.max(d);
            sum += u64::from(d);
            exact += u64::from(d == 0);
            n += 1;
        }
    }
    (max, sum as f64 / n as f64, exact as f64 / n as f64)
}

fn report(label: &str, got: &[u8], dir: &Path, reference: &str) {
    let want = read(dir, reference);
    let (max, mean, exact) = diff(got, &want);
    println!(
        "{label:<28} vs {reference:<16} max {max:>3}  moyenne {mean:.4}  exacts {:.2} %",
        exact * 100.0
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = PathBuf::from(args.get(1).expect("dossier des références"));
    let models = PathBuf::from(args.get(2).map(String::as_str).unwrap_or("../models"));
    let small = read(&dir, "small.rgba");
    let a = read(&dir, "a.rgba");
    let b = read(&dir, "b.rgba");
    let load = |name: &str| SafeTensors::load(&models.join(name)).expect("poids");

    for allow_f16 in [false, true] {
        let gpu = Gpu::new(allow_f16).expect("GPU");
        let store = if gpu.f16 { "f16" } else { "f32" };
        println!("— {} , stockage {store}", gpu.adapter);
        for (name, file, n) in [
            ("photo", "realesr-general-x4v3.safetensors", 32),
            ("anime", "realesr-animevideov3.safetensors", 16),
        ] {
            let net = Esrgan::load(&gpu, &load(file), n).expect("modèle");
            let out = net
                .upscale(&gpu, &small, 96, 64, &mut |_| {})
                .expect("calcul");
            report(
                &format!("{name} ({store})"),
                &out,
                &dir,
                &format!("{name}-f16w.rgba"),
            );
            report(
                &format!("{name} ({store})"),
                &out,
                &dir,
                &format!("{name}-orig.rgba"),
            );
        }
        let rife = Rife::load(&gpu, &load("rife-4.25.safetensors")).expect("modèle");
        let out = rife.between(&gpu, &a, &b, 200, 120, 0.5).expect("calcul");
        report(&format!("rife ({store})"), &out, &dir, "rife-f16w.rgba");
        report(&format!("rife ({store})"), &out, &dir, "rife-orig.rgba");
    }

    // Vitesse en 1080p (stockage par défaut, f16 si possible).
    let gpu = Gpu::new(true).expect("GPU");
    let frame: Vec<u8> = (0..1920 * 1080)
        .flat_map(|i| [(i % 251) as u8, (i % 241) as u8, (i % 239) as u8, 255])
        .collect();
    for (name, file, n) in [
        ("photo", "realesr-general-x4v3.safetensors", 32),
        ("anime", "realesr-animevideov3.safetensors", 16),
    ] {
        let net = Esrgan::load(&gpu, &load(file), n).expect("modèle");
        net.upscale(&gpu, &frame[..256 * 256 * 4], 256, 256, &mut |_| {})
            .expect("chauffe");
        let start = Instant::now();
        let big = net
            .upscale(&gpu, &frame, 1920, 1080, &mut |_| {})
            .expect("calcul");
        let gpu_s = start.elapsed().as_secs_f64();
        let start = Instant::now();
        let out = resize_rgba(&big, 7680, 4320, 3840, 2160);
        let cpu_s = start.elapsed().as_secs_f64();
        println!(
            "{name} 1920x1080 -> 3840x2160 : {:.2} s par image ({gpu_s:.2} s GPU + {cpu_s:.2} s réduction), {} octets",
            gpu_s + cpu_s,
            out.len()
        );
    }
    let rife = Rife::load(&gpu, &load("rife-4.25.safetensors")).expect("modèle");
    rife.between(&gpu, &frame, &frame, 1920, 1080, 0.5)
        .expect("chauffe");
    let start = Instant::now();
    for _ in 0..3 {
        rife.between(&gpu, &frame, &frame, 1920, 1080, 0.5)
            .expect("calcul");
    }
    println!(
        "rife 1920x1080 : {:.3} s par image intermédiaire",
        start.elapsed().as_secs_f64() / 3.0
    );
}
