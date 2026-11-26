//! Le protocole 1 de bout en bout, comme Bobine l'utilise : lots d'images écrits dans un dossier
//! de travail (`media.frames`, format `rgba`), requêtes JSON, images rendues lues dans le
//! fichier `raw` de la réponse. Les calculs demandent un GPU (sautés sans GPU, sauf si
//! `SCALINE_REQUIRE_GPU=1`) et les modèles (`models/`, voir `scripts/fetch-models.sh`).

use scaline_worker::gpu::Gpu;
use scaline_worker::protocol::{State, handle_line};
use serde_json::{Value, json};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

fn models() -> PathBuf {
    std::env::var_os("SCALINE_MODELS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("../models"))
}

fn has_gpu() -> bool {
    let ok = Gpu::new(true).is_ok();
    if !ok && std::env::var_os("SCALINE_REQUIRE_GPU").is_some() {
        panic!("GPU exigé");
    }
    if !ok {
        eprintln!("aucun GPU : test sauté");
    }
    ok
}

/// Dossier de travail propre à un test.
fn work_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("scaline-test-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("dossier de test");
    dir
}

/// Image RGBA unie de la couleur `v`, avec un carré clair qui se déplace de `shift` pixels.
fn frame(w: usize, h: usize, v: u8, shift: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(w * h * 4);
    for y in 0..h {
        for x in 0..w {
            let inside = (10 + shift..26 + shift).contains(&x) && (8..24).contains(&y);
            let c = if inside { 230 } else { v };
            out.extend_from_slice(&[c, c / 2, 255 - c, 255]);
        }
    }
    out
}

/// Écrit un lot comme Bobine : `frames-<n>.json` + `frames-<n>.raw`, 25 im./s à partir de `t0`.
fn write_set(dir: &Path, n: u32, frames: &[Vec<u8>], w: usize, h: usize, t0: f64) -> String {
    let raw: Vec<u8> = frames.concat();
    std::fs::write(dir.join(format!("frames-{n}.raw")), raw).expect("raw");
    let header = json!({
        "raw": format!("frames-{n}.raw"),
        "count": frames.len(),
        "width": w,
        "height": h,
        "format": "rgba",
        "fps": 25,
        "every_frame": true,
        "bytes_per_frame": w * h * 4,
        "timestamps": (0..frames.len()).map(|i| t0 + i as f64 / 25.0).collect::<Vec<_>>(),
    });
    let name = format!("frames-{n}.json");
    std::fs::write(dir.join(&name), header.to_string()).expect("json");
    name
}

/// Envoie une requête et rend (lignes de progression, réponse).
fn send(state: &mut State, request: Value) -> (Vec<Value>, Value) {
    let mut lines = Vec::new();
    handle_line(state, &request.to_string(), &mut |v| lines.push(v));
    let reply = lines.pop().expect("une réponse");
    (lines, reply)
}

#[test]
fn ping_answers_pong_and_an_unknown_method_is_an_error() {
    let dir = work_dir("ping");
    let mut state = State::new(dir, models());
    let (_, reply) = send(&mut state, json!({ "id": 1, "method": "ping" }));
    assert_eq!(reply, json!({ "id": 1, "result": "pong" }));
    let (_, reply) = send(&mut state, json!({ "id": 2, "method": "nope" }));
    assert_eq!(reply["id"], 2);
    assert_eq!(reply["error"]["code"], "unknown_method");
}

#[test]
fn frames_outside_the_work_folder_are_refused() {
    let dir = work_dir("path");
    let mut state = State::new(dir, models());
    for frames in ["../frames-1.json", "/etc/passwd"] {
        let (_, reply) = send(
            &mut state,
            json!({ "id": 3, "method": "interpolate", "params": { "frames": frames, "count": 1 } }),
        );
        assert_eq!(reply["error"]["code"], "bad_path", "{frames}");
    }
}

#[test]
fn a_bad_output_size_is_refused_before_any_work() {
    let dir = work_dir("size");
    let header = write_set(&dir, 1, &[frame(40, 32, 20, 0)], 40, 32, 0.0);
    let mut state = State::new(dir, models());
    let (_, reply) = send(
        &mut state,
        json!({ "id": 4, "method": "upscale", "params": {
            "frames": header, "count": 1, "model": "anime", "width": 161, "height": 128 } }),
    );
    assert_eq!(reply["error"]["code"], "bad_params");
}

#[test]
fn upscaling_writes_one_frame_of_the_asked_size_per_new_frame() {
    if !has_gpu() {
        return;
    }
    let dir = work_dir("upscale");
    let (w, h) = (40, 32);
    let frames: Vec<Vec<u8>> = (0..3)
        .map(|i| frame(w, h, 40 + i * 10, i as usize))
        .collect();
    let header = write_set(&dir, 1, &frames, w, h, 2.0);
    let mut state = State::new(dir.clone(), models());
    // `after` : la première image (2,00 s) a déjà été rendue par le lot précédent.
    let (progress, reply) = send(
        &mut state,
        json!({ "id": 5, "method": "upscale", "params": {
            "frames": header, "first": 0, "count": 3, "after": 2.0,
            "model": "anime", "width": 120, "height": 96, "progress": [0.5, 0.5] } }),
    );
    let result = &reply["result"];
    assert_eq!(result["count"], 2, "{reply}");
    assert_eq!(
        (result["width"].as_u64(), result["height"].as_u64()),
        (Some(120), Some(96))
    );
    assert_eq!(result["first_time"], 2.04);
    let raw = std::fs::read(dir.join(result["raw"].as_str().expect("raw"))).expect("sortie");
    assert_eq!(raw.len(), 2 * 120 * 96 * 4);
    // Une couleur unie reste à peu près la même une fois agrandie (coin haut gauche).
    let (r, g, b) = (raw[0], raw[1], raw[2]);
    assert!(
        r.abs_diff(50) <= 12 && g.abs_diff(25) <= 12 && b.abs_diff(205) <= 12,
        "{r} {g} {b}"
    );
    assert!(raw.chunks(4).all(|p| p[3] == 255));
    // La progression va de 0,5 à 1 (part donnée par le plugin).
    let values: Vec<f64> = progress
        .iter()
        .map(|p| p["progress"]["value"].as_f64().expect("valeur"))
        .collect();
    assert!(values.first().is_some_and(|v| *v >= 0.5) && values.last() == Some(&1.0));
}

#[test]
fn interpolation_doubles_the_frames_across_two_overlapping_sets() {
    if !has_gpu() {
        return;
    }
    let dir = work_dir("interpolate");
    let (w, h) = (48, 32);
    let all: Vec<Vec<u8>> = (0..5).map(|i| frame(w, h, 30, 2 * i)).collect();
    let mut state = State::new(dir.clone(), models());
    // Lot 1 : images 0 à 2. Lot 2 : images 2 à 4 (la 2 en recouvrement, déjà rendue).
    let first = write_set(&dir, 1, &all[0..3], w, h, 0.0);
    let second = write_set(&dir, 2, &all[2..5], w, h, 2.0 / 25.0);
    let (_, a) = send(
        &mut state,
        json!({ "id": 6, "method": "interpolate", "params": { "frames": first, "count": 3 } }),
    );
    let (_, b) = send(
        &mut state,
        json!({ "id": 7, "method": "interpolate", "params": {
            "frames": second, "count": 3, "after": a["result"]["last_time"], "last": true } }),
    );
    // Lot 1 : 0, ½, 1, 1½, 2. Lot 2 : 2½, 3, 3½, 4, puis 4 répétée : 10 images pour 5.
    assert_eq!(a["result"]["count"], 5, "{a}");
    assert_eq!(b["result"]["count"], 5, "{b}");
    let read = |reply: &Value| {
        std::fs::read(dir.join(reply["result"]["raw"].as_str().expect("raw"))).expect("sortie")
    };
    let (ra, rb) = (read(&a), read(&b));
    let size = w * h * 4;
    assert_eq!(ra.len(), 5 * size);
    assert_eq!(rb.len(), 5 * size);
    // Les images d'origine sont recopiées telles quelles, aux positions paires.
    assert_eq!(&ra[0..size], &all[0][..]);
    assert_eq!(&ra[2 * size..3 * size], &all[1][..]);
    assert_eq!(&rb[size..2 * size], &all[3][..]);
    assert_eq!(&rb[4 * size..], &all[4][..]);
    // L'image intermédiaire entre 0 et 1 est différente des deux.
    let middle = &ra[size..2 * size];
    assert_ne!(middle, &all[0][..]);
    assert_ne!(middle, &all[1][..]);
}

#[test]
fn the_program_answers_on_stdout_with_an_emptied_environment() {
    let dir = work_dir("process");
    let exe = env!("CARGO_BIN_EXE_scaline-worker");
    let mut child = Command::new(exe)
        .args(["--bobine-worker", "1"])
        .env_clear()
        .env("BOBINE_WORK_DIR", &dir)
        .env("SCALINE_MODELS_DIR", models())
        .current_dir(&dir)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("lancement");
    let mut stdin = child.stdin.take().expect("stdin");
    writeln!(stdin, "{}", json!({ "id": 1, "method": "ping" })).expect("écriture");
    writeln!(stdin, "pas du json").expect("écriture");
    writeln!(stdin, "{}", json!({ "id": 2, "method": "ping" })).expect("écriture");
    drop(stdin); // stdin fermé : le programme s'arrête
    let lines: Vec<Value> = BufReader::new(child.stdout.take().expect("stdout"))
        .lines()
        .map(|l| serde_json::from_str(&l.expect("ligne")).expect("json"))
        .collect();
    assert_eq!(
        lines,
        vec![
            json!({ "id": 1, "result": "pong" }),
            json!({ "id": 2, "result": "pong" })
        ]
    );
    assert!(child.wait().expect("fin").success());
}

#[test]
fn the_program_refuses_to_start_without_the_worker_flag() {
    let status = Command::new(env!("CARGO_BIN_EXE_scaline-worker"))
        .env_clear()
        .stderr(Stdio::null())
        .status()
        .expect("lancement");
    assert_eq!(status.code(), Some(2));
}
