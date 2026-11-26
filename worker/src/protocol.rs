//! Protocole 1 des programmes natifs de Bobine : une requête JSON par ligne sur stdin
//! (`{id, method, params}`), une réponse par ligne sur stdout (`{id, result}` ou
//! `{id, error: {code, message}}`), et des lignes de progression `{progress: {id, value}}`.
//!
//! Méthodes :
//! - `ping` → `"pong"` ;
//! - `info` → GPU choisi (nom, f16, logiciel) ;
//! - `estimate` `{task, model?, width, height}` → secondes par image, mesurées sur ce GPU ;
//! - `upscale` `{frames, first, count, after?, model, width, height, progress?}` → images
//!   agrandies (Real-ESRGAN ×4 puis ramenées à `width` × `height`) ;
//! - `interpolate` `{frames, first, count, after?, last?, progress?}` → cadence doublée (RIFE) :
//!   pour chaque image, l'image intermédiaire avec la précédente puis l'image elle-même.
//!
//! `frames` est le chemin relatif de l'en-tête rendu par `media.frames` (champ `header`) ; `first`
//! et `count` choisissent les images du lot ; les images d'instant ≤ `after` (secondes, temps du
//! média) ne sont pas rendues (recouvrement entre deux lots) mais servent d'image précédente.
//! La réponse donne `raw`, fichier du dossier de travail à passer à `media.encode.append`.

use crate::esrgan::{self, Esrgan};
use crate::frames::{Failure, FrameSet, work_file};
use crate::gpu::{Gpu, GpuError};
use crate::resize::resize_rgba;
use crate::rife::Rife;
use crate::safetensors::SafeTensors;
use serde::Deserialize;
use serde_json::{Value, json};
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

/// Plus grande sortie : celle acceptée par `media.encode.begin` (8192 de côté, 8192 × 4320).
const MAX_OUT_SIDE: u32 = 8192;
const MAX_OUT_AREA: u64 = 8192 * 4320;
/// Plus grand nombre d'images rendues par appel (borne le fichier écrit).
const MAX_PER_CALL: usize = 240;
/// Écart de temps sous lequel deux images sont la même (secondes).
const SAME_TIME: f64 = 1e-4;

impl From<GpuError> for Failure {
    fn from(e: GpuError) -> Self {
        Failure::new(e.code, e.message)
    }
}

/// Les deux modèles d'agrandissement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Model {
    /// `realesr-general-x4v3` : photo réelle.
    Photo,
    /// `realesr-animevideov3` : dessin animé, animation.
    Anime,
}

impl Model {
    fn file(self) -> &'static str {
        match self {
            Model::Photo => "realesr-general-x4v3.safetensors",
            Model::Anime => "realesr-animevideov3.safetensors",
        }
    }
    fn num_conv(self) -> usize {
        match self {
            Model::Photo => 32,
            Model::Anime => 16,
        }
    }
}

const RIFE_FILE: &str = "rife-4.25.safetensors";

/// GPU et modèles, créés au premier calcul puis gardés tant que le programme vit.
struct Engine {
    gpu: Gpu,
    photo: Option<Esrgan>,
    anime: Option<Esrgan>,
    rife: Option<Rife>,
}

/// État du programme entre deux requêtes.
pub struct State {
    work_dir: PathBuf,
    models_dir: PathBuf,
    allow_f16: bool,
    engine: Option<Engine>,
    next_output: u32,
}

fn model_error(e: String) -> Failure {
    Failure::new("bad_model", e)
}

impl State {
    pub fn new(work_dir: PathBuf, models_dir: PathBuf) -> State {
        State {
            work_dir,
            models_dir,
            allow_f16: true,
            engine: None,
            next_output: 1,
        }
    }

    /// Stockage en f32 même si le GPU sait le f16 (comparaisons numériques).
    pub fn force_f32(mut self) -> State {
        self.allow_f16 = false;
        self
    }

    fn engine(&mut self) -> Result<&mut Engine, Failure> {
        if self.engine.is_none() {
            let gpu = Gpu::new(self.allow_f16)?;
            eprintln!(
                "scaline : GPU {} ({}{})",
                gpu.adapter,
                if gpu.f16 { "f16" } else { "f32" },
                if gpu.software { ", logiciel" } else { "" }
            );
            self.engine = Some(Engine {
                gpu,
                photo: None,
                anime: None,
                rife: None,
            });
        }
        self.engine
            .as_mut()
            .ok_or_else(|| Failure::new("no_gpu", "GPU unavailable"))
    }

    fn esrgan(&mut self, model: Model) -> Result<(&Gpu, &Esrgan), Failure> {
        let path = self.models_dir.join(model.file());
        let engine = self.engine()?;
        let slot = match model {
            Model::Photo => &mut engine.photo,
            Model::Anime => &mut engine.anime,
        };
        if slot.is_none() {
            let st = SafeTensors::load(&path).map_err(model_error)?;
            *slot = Some(Esrgan::load(&engine.gpu, &st, model.num_conv()).map_err(model_error)?);
        }
        let net = slot
            .as_ref()
            .ok_or_else(|| Failure::new("bad_model", "model not loaded"))?;
        Ok((&engine.gpu, net))
    }

    fn rife(&mut self) -> Result<(&Gpu, &Rife), Failure> {
        let path = self.models_dir.join(RIFE_FILE);
        let engine = self.engine()?;
        if engine.rife.is_none() {
            let st = SafeTensors::load(&path).map_err(model_error)?;
            engine.rife = Some(Rife::load(&engine.gpu, &st).map_err(model_error)?);
        }
        let net = engine
            .rife
            .as_ref()
            .ok_or_else(|| Failure::new("bad_model", "model not loaded"))?;
        Ok((&engine.gpu, net))
    }

    /// Nouveau fichier de sortie dans le dossier de travail (nom relatif, chemin complet).
    fn output(&mut self) -> Result<(String, std::io::BufWriter<std::fs::File>), Failure> {
        let name = format!("scaline-{}.raw", self.next_output);
        self.next_output += 1;
        let path = work_file(&self.work_dir, &name)?;
        let file =
            std::fs::File::create(path).map_err(|e| Failure::new("write_failed", e.to_string()))?;
        Ok((name, std::io::BufWriter::new(file)))
    }
}

/// Traite une ligne reçue. `emit` écrit une ligne JSON sur stdout (progression puis réponse).
/// Une ligne illisible ou sans `id` est ignorée (rien à quoi répondre).
pub fn handle_line(state: &mut State, line: &str, emit: &mut dyn FnMut(Value)) {
    let Ok(request) = serde_json::from_str::<Value>(line) else {
        eprintln!("scaline : requête illisible ignorée");
        return;
    };
    let id = request["id"].clone();
    if id.is_null() {
        eprintln!("scaline : requête sans id ignorée");
        return;
    }
    let method = request["method"].as_str().unwrap_or("");
    let params = &request["params"];
    let mut last_percent = -1i64;
    let mut progress = |value: f64| {
        // Une ligne par pour cent au plus : Bobine n'en affiche que quelques-unes par seconde.
        let percent = (value.clamp(0.0, 1.0) * 100.0).floor() as i64;
        if percent != last_percent {
            last_percent = percent;
            emit(json!({ "progress": { "id": id, "value": percent as f64 / 100.0 } }));
        }
    };
    let outcome = match method {
        "ping" => Ok(json!("pong")),
        "info" => info(state),
        "estimate" => estimate(state, params),
        "upscale" => upscale(state, params, &mut progress),
        "interpolate" => interpolate(state, params, &mut progress),
        other => Err(Failure::new(
            "unknown_method",
            other.chars().take(64).collect::<String>(),
        )),
    };
    emit(match outcome {
        Ok(result) => json!({ "id": id, "result": result }),
        Err(f) => json!({ "id": id, "error": { "code": f.code, "message": f.message } }),
    });
}

fn bad_params(error: serde_json::Error) -> Failure {
    Failure::new("bad_params", error.to_string())
}

fn info(state: &mut State) -> Result<Value, Failure> {
    let gpu = &state.engine()?.gpu;
    Ok(json!({
        "adapter": gpu.adapter,
        "f16": gpu.f16,
        "software": gpu.software,
        "version": env!("CARGO_PKG_VERSION"),
    }))
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Task {
    Upscale,
    Interpolate,
}

#[derive(Debug, Deserialize)]
struct EstimateParams {
    task: Task,
    #[serde(default)]
    model: Option<Model>,
    width: u32,
    height: u32,
}

/// Image d'essai : un dégradé (les calculs ne dépendent pas du contenu).
fn test_image(w: u32, h: u32, shift: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            out.extend_from_slice(&[(x + shift) as u8, y as u8, (x ^ y) as u8, 255]);
        }
    }
    out
}

/// Mesure la durée d'une image sur ce GPU (une fois pour charger, une fois pour mesurer).
fn estimate(state: &mut State, params: &Value) -> Result<Value, Failure> {
    let p: EstimateParams = serde_json::from_value(params.clone()).map_err(bad_params)?;
    check_input_size(p.width, p.height)?;
    let seconds = match p.task {
        Task::Upscale => {
            let (gpu, net) = state.esrgan(p.model.unwrap_or(Model::Photo))?;
            // Une tuile de la taille réelle, puis multipliée par le nombre de tuiles.
            let list = esrgan::tiles(p.width, p.height, esrgan::TILE, net.margin());
            let first = list
                .first()
                .ok_or_else(|| Failure::new("bad_params", "empty image"))?;
            let [_, _, w, h] = first.read;
            let image = test_image(w, h, 0);
            net.upscale(gpu, &image, w, h, &mut |_| {})?;
            let start = Instant::now();
            net.upscale(gpu, &image, w, h, &mut |_| {})?;
            start.elapsed().as_secs_f64() * list.len() as f64
        }
        Task::Interpolate => {
            let (gpu, net) = state.rife()?;
            let (a, b) = (
                test_image(p.width, p.height, 0),
                test_image(p.width, p.height, 7),
            );
            net.middle(gpu, &a, &b, p.width, p.height)?;
            let start = Instant::now();
            net.middle(gpu, &a, &b, p.width, p.height)?;
            start.elapsed().as_secs_f64()
        }
    };
    let gpu = &state.engine()?.gpu;
    Ok(json!({
        "seconds_per_frame": seconds,
        "adapter": gpu.adapter,
        "software": gpu.software,
    }))
}

fn check_input_size(w: u32, h: u32) -> Result<(), Failure> {
    if !(2..=4096).contains(&w) || !(2..=4096).contains(&h) {
        return Err(Failure::new("bad_params", "frame size out of range"));
    }
    Ok(())
}

/// Part du travail d'un appel dans la barre de progression de toute la tâche : [début, part].
#[derive(Debug, Clone, Copy, Deserialize)]
struct Span(f64, f64);

impl Span {
    fn at(&self, local: f64) -> f64 {
        self.0 + self.1 * local.clamp(0.0, 1.0)
    }
}

#[derive(Debug, Deserialize)]
struct UpscaleParams {
    frames: String,
    #[serde(default)]
    first: usize,
    count: usize,
    #[serde(default)]
    after: Option<f64>,
    model: Model,
    width: u32,
    height: u32,
    #[serde(default)]
    progress: Option<Span>,
}

/// Images `first..first + count` du lot, bornées au lot et à `MAX_PER_CALL`.
fn range(set: &FrameSet, first: usize, count: usize) -> Result<std::ops::Range<usize>, Failure> {
    if first >= set.count || count == 0 || count > MAX_PER_CALL {
        return Err(Failure::new("bad_params", "first or count out of range"));
    }
    Ok(first..(first + count).min(set.count))
}

fn is_new(t: f64, after: Option<f64>) -> bool {
    after.is_none_or(|a| t > a + SAME_TIME)
}

fn write_failed(e: std::io::Error) -> Failure {
    Failure::new("write_failed", e.to_string())
}

fn upscale(
    state: &mut State,
    params: &Value,
    progress: &mut dyn FnMut(f64),
) -> Result<Value, Failure> {
    let p: UpscaleParams = serde_json::from_value(params.clone()).map_err(bad_params)?;
    if !p.width.is_multiple_of(2)
        || !p.height.is_multiple_of(2)
        || p.width == 0
        || p.height == 0
        || p.width > MAX_OUT_SIDE
        || p.height > MAX_OUT_SIDE
        || u64::from(p.width) * u64::from(p.height) > MAX_OUT_AREA
    {
        return Err(Failure::new(
            "bad_params",
            "output size must be even, at most 8192 a side and 8192 x 4320 in area",
        ));
    }
    let mut set = FrameSet::open(&state.work_dir, &p.frames)?;
    let range = range(&set, p.first, p.count)?;
    let span = p.progress.unwrap_or(Span(0.0, 1.0));
    let (name, mut out) = state.output()?;
    let (gpu, net) = state.esrgan(p.model)?;
    let (w, h) = (set.width as u32, set.height as u32);
    let total = range.len() as f64;
    let mut written = 0usize;
    let mut times = Vec::new();
    for (n, index) in range.enumerate() {
        let t = set.timestamps[index];
        if !is_new(t, p.after) {
            continue;
        }
        let frame = set.frame(index)?;
        let big = net.upscale(gpu, &frame, w, h, &mut |part| {
            progress(span.at((n as f64 + part * 0.95) / total));
        })?;
        let sized = resize_rgba(
            &big,
            4 * w as usize,
            4 * h as usize,
            p.width as usize,
            p.height as usize,
        );
        out.write_all(&sized).map_err(write_failed)?;
        written += 1;
        times.push(t);
        progress(span.at((n + 1) as f64 / total));
    }
    out.flush().map_err(write_failed)?;
    Ok(result(name, written, p.width, p.height, &times))
}

fn result(raw: String, count: usize, width: u32, height: u32, times: &[f64]) -> Value {
    json!({
        "raw": raw,
        "count": count,
        "width": width,
        "height": height,
        "first_time": times.first(),
        "last_time": times.last(),
    })
}

#[derive(Debug, Deserialize)]
struct InterpolateParams {
    frames: String,
    #[serde(default)]
    first: usize,
    count: usize,
    #[serde(default)]
    after: Option<f64>,
    /// Dernier appel : l'image finale est répétée, la sortie compte alors 2 × N images.
    #[serde(default)]
    last: bool,
    #[serde(default)]
    progress: Option<Span>,
}

fn interpolate(
    state: &mut State,
    params: &Value,
    progress: &mut dyn FnMut(f64),
) -> Result<Value, Failure> {
    let p: InterpolateParams = serde_json::from_value(params.clone()).map_err(bad_params)?;
    let mut set = FrameSet::open(&state.work_dir, &p.frames)?;
    let range = range(&set, p.first, p.count)?;
    let span = p.progress.unwrap_or(Span(0.0, 1.0));
    let (name, mut out) = state.output()?;
    let (gpu, net) = state.rife()?;
    let (w, h) = (set.width as u32, set.height as u32);
    let total = range.len() as f64;
    let mut written = 0usize;
    let mut times = Vec::new();
    // L'image précédente du lot (même hors de `range`), lue une fois.
    let mut previous: Option<Vec<u8>> = if range.start > 0 {
        Some(set.frame(range.start - 1)?)
    } else {
        None
    };
    for (n, index) in range.enumerate() {
        let frame = set.frame(index)?;
        let t = set.timestamps[index];
        if is_new(t, p.after) {
            if let Some(before) = &previous {
                let middle = net.middle(gpu, before, &frame, w, h)?;
                out.write_all(&middle).map_err(write_failed)?;
                written += 1;
            }
            out.write_all(&frame).map_err(write_failed)?;
            written += 1;
            times.push(t);
        }
        progress(span.at((n + 1) as f64 / total));
        previous = Some(frame);
    }
    if p.last
        && let Some(frame) = &previous
    {
        out.write_all(frame).map_err(write_failed)?;
        written += 1;
    }
    out.flush().map_err(write_failed)?;
    Ok(result(name, written, w, h, &times))
}

/// Dossier des modèles : `<dossier du plugin>/models`, le programme étant dans `<plugin>/bin/`.
/// `SCALINE_MODELS_DIR` le remplace (essais hors de Bobine ; Bobine vide l'environnement).
pub fn models_dir() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("SCALINE_MODELS_DIR") {
        return Some(PathBuf::from(dir));
    }
    let exe = std::env::current_exe().ok()?;
    Some(exe.parent()?.parent()?.join("models"))
}
