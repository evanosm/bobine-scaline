//! Lecture des images écrites par Bobine (`media.frames`, format `rgba`) dans le dossier de
//! travail : `frames-<n>.json` (en-tête) et `frames-<n>.raw` (images bout à bout, ligne par
//! ligne, 4 octets par pixel), et écriture des images rendues (`scaline-<n>.raw`, même
//! disposition, lues par `media.encode.append`).
//!
//! Le chemin vient du plugin : il est HOSTILE. Seuls les chemins relatifs, sans `..`, à
//! l'intérieur du dossier de travail sont acceptés (`work_file`).

use serde::Deserialize;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Component, Path, PathBuf};

/// Plus grand côté accepté en entrée (Bobine n'écrit jamais plus de 1920 px).
const MAX_SIDE: usize = 4096;
/// Plus grand nombre d'images d'un lot (Bobine en écrit 20 000 au plus).
const MAX_FRAMES: usize = 20_000;
/// Taille maximale d'un en-tête.
const MAX_HEADER_BYTES: u64 = 4 * 1024 * 1024;

/// Erreur rendue à Bobine : un code court (`snake_case`) et un message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Failure {
    pub code: &'static str,
    pub message: String,
}

impl Failure {
    pub fn new(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }
}

/// Un fichier du dossier de travail : chemin relatif, sans `..`, sans racine ni lecteur.
pub fn work_file(work_dir: &Path, relative: &str) -> Result<PathBuf, Failure> {
    let refused = || {
        Failure::new(
            "bad_path",
            "expected a relative path inside the work folder",
        )
    };
    if relative.is_empty() || relative.len() > 200 || relative.contains('\0') {
        return Err(refused());
    }
    let path = Path::new(relative);
    // `Component::Normal` seulement : `..`, `/`, `C:\` et `.` sont refusés.
    let plain = path
        .components()
        .all(|part| matches!(part, Component::Normal(_)));
    // Sous Unix, `Path` ne voit pas `\` comme séparateur : on le refuse aussi (chemin Windows).
    if !plain || path.is_absolute() || relative.contains('\\') || relative.contains(':') {
        return Err(refused());
    }
    Ok(work_dir.join(path))
}

/// En-tête `frames-<n>.json` écrit par Bobine.
#[derive(Debug, Clone, Deserialize)]
struct Header {
    raw: String,
    count: usize,
    width: usize,
    height: usize,
    format: String,
    #[serde(default)]
    timestamps: Vec<f64>,
}

/// Lot d'images RGBA, lues à la demande (la mémoire ne dépend pas de la taille du lot).
pub struct FrameSet {
    pub width: usize,
    pub height: usize,
    pub count: usize,
    /// Instant de chaque image, en secondes (temps du média).
    pub timestamps: Vec<f64>,
    file: File,
}

impl FrameSet {
    /// Ouvre le lot dont `header` est l'en-tête (chemin relatif au dossier de travail).
    pub fn open(work_dir: &Path, header: &str) -> Result<FrameSet, Failure> {
        let header_path = work_file(work_dir, header)?;
        let io = |e: std::io::Error| Failure::new("io", e.to_string());
        let size = std::fs::metadata(&header_path).map_err(io)?.len();
        if size > MAX_HEADER_BYTES {
            return Err(Failure::new("bad_header", "header file is too big"));
        }
        let text = std::fs::read_to_string(&header_path).map_err(io)?;
        let header: Header =
            serde_json::from_str(&text).map_err(|e| Failure::new("bad_header", e.to_string()))?;
        if header.format != "rgba" {
            return Err(Failure::new(
                "bad_format",
                "frames must be in the rgba format",
            ));
        }
        if header.count == 0 || header.count > MAX_FRAMES {
            return Err(Failure::new("bad_header", "no frames, or too many"));
        }
        if header.width < 2
            || header.height < 2
            || header.width > MAX_SIDE
            || header.height > MAX_SIDE
        {
            return Err(Failure::new("bad_header", "frame size out of range"));
        }
        let raw_path = work_file(work_dir, &header.raw)?;
        let per_frame = (header.width * header.height * 4) as u64;
        let raw_size = std::fs::metadata(&raw_path).map_err(io)?.len();
        if raw_size < per_frame * header.count as u64 {
            return Err(Failure::new(
                "bad_header",
                "raw file is shorter than announced",
            ));
        }
        // Un en-tête sans instants (ou incomplet) : on numérote les images à 1 s d'écart.
        let mut timestamps = header.timestamps;
        if timestamps.len() != header.count || timestamps.iter().any(|t| !t.is_finite()) {
            timestamps = (0..header.count).map(|i| i as f64).collect();
        }
        Ok(FrameSet {
            width: header.width,
            height: header.height,
            count: header.count,
            timestamps,
            file: File::open(raw_path).map_err(io)?,
        })
    }

    /// Octets d'une image.
    pub fn frame_bytes(&self) -> usize {
        self.width * self.height * 4
    }

    /// L'image `index` (0 = la première du lot).
    pub fn frame(&mut self, index: usize) -> Result<Vec<u8>, Failure> {
        if index >= self.count {
            return Err(Failure::new("bad_params", "frame index out of range"));
        }
        let size = self.frame_bytes();
        let io = |e: std::io::Error| Failure::new("io", e.to_string());
        self.file
            .seek(SeekFrom::Start((index * size) as u64))
            .map_err(io)?;
        let mut data = vec![0u8; size];
        self.file.read_exact(&mut data).map_err(io)?;
        Ok(data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_plain_relative_name_is_accepted() {
        let dir = Path::new("/work");
        assert_eq!(
            work_file(dir, "frames-1.json"),
            Ok(PathBuf::from("/work/frames-1.json"))
        );
    }

    #[test]
    fn paths_leaving_the_work_folder_are_refused() {
        let dir = Path::new("/work");
        for bad in [
            "",
            "/etc/passwd",
            "../secret",
            "lots/../../secret",
            "./frames-1.json",
            "C:\\Windows\\win.ini",
            "C:frames.raw",
            "..\\secret",
            "\\\\server\\share",
        ] {
            assert_eq!(
                work_file(dir, bad).map_err(|e| e.code),
                Err("bad_path"),
                "{bad}"
            );
        }
    }
}
