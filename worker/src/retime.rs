//! Changement de cadence : quelles images sources encadrent chaque image de sortie, et à quel
//! instant entre les deux. Calcul sur le processeur, sans GPU.
//!
//! - Instants réels : les images de sortie tombent à `origin + k / cadence`, et chaque image est
//!   placée d'après l'horodatage des images sources, pas leur numéro (cadence variable d'OBS ou
//!   de ShadowPlay, rapports non entiers comme 24 → 60).
//! - Doublons : une image identique à la précédente (jeu à 45 im./s enregistré à 60, image
//!   répétée par l'enregistreur) n'est pas une image source. Les images de sortie qui tombent
//!   dessus sont recalculées entre ses voisines : le mouvement devient régulier. Au-delà de
//!   `MAX_DUPLICATES` répétitions de suite, l'image est gardée : la scène est vraiment immobile.
//! - Coupes : entre les deux images d'un changement de plan, rien n'est inventé ; la sortie
//!   reprend l'image source la plus proche.

/// Répétitions de suite au plus traitées comme des doublons.
pub const MAX_DUPLICATES: usize = 3;
/// Images relues avant le début d'un appel pour décider des doublons comme l'appel précédent.
pub const LOOKBACK: usize = 2 * MAX_DUPLICATES + 2;
/// Écart de temps sous lequel deux instants sont le même (secondes).
pub const SAME_TIME: f64 = 1e-4;

/// Côté des cases moyennées de la vignette (pixels).
const CELL: usize = 8;
/// Cases de l'histogramme de luminance.
const BINS: usize = 32;
/// Doublon : 99 % des cases changent de 4 niveaux au plus, et 0,5 en moyenne. Le 99e centile
/// plutôt que le maximum : un doublon compressé en H.264 garde quelques cases bruitées (jusqu'à 13
/// niveaux mesurés), alors que du vrai mouvement dépasse 20 niveaux sur bien plus de 1 % des cases.
/// Un faux doublon (objet minuscule qui bouge) est sans gravité : il est recalculé à son instant.
const DUPLICATE_P99: f32 = 4.0;
const DUPLICATE_MEAN: f32 = 0.5;
/// Coupe : histogrammes très différents (écart L1, de 0 à 2) ET image très changée (niveaux).
/// Un panoramique rapide change l'image mais garde ses couleurs : ce n'est pas une coupe.
const CUT_HISTOGRAM: f32 = 0.6;
const CUT_MEAN: f32 = 12.0;

/// Vignette d'une image : luminance moyenne par case de `CELL` × `CELL`, et son histogramme.
#[derive(Debug, Clone)]
pub struct Thumb {
    luma: Vec<f32>,
    histogram: [f32; BINS],
}

/// Vignette d'une image RGBA 8 bits de `width` × `height`.
pub fn thumb(rgba: &[u8], width: usize, height: usize) -> Thumb {
    let (cw, ch) = (width.div_ceil(CELL), height.div_ceil(CELL));
    let mut sum = vec![0f32; cw * ch];
    let mut count = vec![0u32; cw * ch];
    for y in 0..height {
        let row = &rgba[y * width * 4..(y + 1) * width * 4];
        for x in 0..width {
            let p = &row[x * 4..x * 4 + 4];
            // Luminance BT.601.
            let l = 0.299 * f32::from(p[0]) + 0.587 * f32::from(p[1]) + 0.114 * f32::from(p[2]);
            let cell = (y / CELL) * cw + x / CELL;
            sum[cell] += l;
            count[cell] += 1;
        }
    }
    let luma: Vec<f32> = sum.iter().zip(&count).map(|(s, n)| s / *n as f32).collect();
    let mut histogram = [0f32; BINS];
    for l in &luma {
        let bin = ((l / 256.0) * BINS as f32) as usize;
        histogram[bin.min(BINS - 1)] += 1.0;
    }
    for h in &mut histogram {
        *h /= luma.len() as f32;
    }
    Thumb { luma, histogram }
}

/// Différence entre deux vignettes de même taille.
#[derive(Debug, Clone, Copy)]
pub struct Change {
    /// Écart moyen des cases (niveaux, 0 à 255).
    pub mean: f32,
    /// Écart sous lequel tombent 99 % des cases.
    pub p99: f32,
    /// Écart L1 des histogrammes (0 : mêmes tons ; 2 : aucun ton commun).
    pub histogram: f32,
}

pub fn change(a: &Thumb, b: &Thumb) -> Change {
    let mut diffs: Vec<f32> = a
        .luma
        .iter()
        .zip(&b.luma)
        .map(|(x, y)| (x - y).abs())
        .collect();
    let mean = diffs.iter().sum::<f32>() / diffs.len().max(1) as f32;
    diffs.sort_by(f32::total_cmp);
    let p99 = diffs
        .get(diffs.len() * 99 / 100)
        .copied()
        .unwrap_or_default();
    let histogram = a
        .histogram
        .iter()
        .zip(&b.histogram)
        .map(|(x, y)| (x - y).abs())
        .sum();
    Change {
        mean,
        p99,
        histogram,
    }
}

pub fn is_duplicate(c: &Change) -> bool {
    c.p99 <= DUPLICATE_P99 && c.mean <= DUPLICATE_MEAN
}

pub fn is_cut(c: &Change) -> bool {
    c.histogram >= CUT_HISTOGRAM && c.mean >= CUT_MEAN
}

/// Images gardées comme images sources : `duplicate[i]` dit que l'image i répète la i − 1.
/// Une répétition est écartée, sauf au-delà de `MAX_DUPLICATES` de suite (scène immobile).
pub fn kept(duplicate: &[bool]) -> Vec<bool> {
    let mut run = 0;
    duplicate
        .iter()
        .map(|&d| {
            run = if d { run + 1 } else { 0 };
            run == 0 || run > MAX_DUPLICATES
        })
        .collect()
}

/// D'où vient une image de sortie : recopie d'une image source, ou calcul entre deux images à
/// l'instant `t` (0 : la première, 1 : la seconde). Indices dans la liste donnée à `Timeline`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Source {
    Copy(usize),
    Between { a: usize, b: usize, t: f32 },
}

/// Images sources gardées, dans l'ordre : leur indice, leur instant, et s'il y a une coupe entre
/// chacune et la suivante.
#[derive(Debug, Clone)]
pub struct Timeline {
    pub frames: Vec<usize>,
    pub times: Vec<f64>,
    pub cut_after: Vec<bool>,
}

impl Timeline {
    /// L'image source de l'instant `time`. Après la dernière image : rien (il faut la suivante,
    /// dans le lot d'après), sauf si `hold` (fin du clip : la dernière image est répétée).
    pub fn source_at(&self, time: f64, hold: bool) -> Option<Source> {
        let first = *self.frames.first()?;
        // Dernière image d'instant ≤ `time`.
        let Some(j) = self.times.iter().rposition(|t| *t <= time + SAME_TIME) else {
            return Some(Source::Copy(first));
        };
        let (a, ta) = (self.frames[j], self.times[j]);
        if (time - ta).abs() <= SAME_TIME {
            return Some(Source::Copy(a));
        }
        let Some((&b, &tb)) = self.frames.get(j + 1).zip(self.times.get(j + 1)) else {
            return hold.then_some(Source::Copy(a));
        };
        let t = (time - ta) / (tb - ta);
        if self.cut_after[j] {
            // À égalité, l'ancien plan reste affiché jusqu'à la coupe (marge pour l'arrondi).
            return Some(Source::Copy(if t <= 0.5 + 1e-6 { a } else { b }));
        }
        Some(Source::Between { a, b, t: t as f32 })
    }
}

/// Instants de sortie `origin + k / rate` (k ≥ 0) strictement après `after` et au plus `until`.
pub fn output_times(origin: f64, rate: f64, after: Option<f64>, until: f64) -> Vec<f64> {
    let step = 1.0 / rate;
    let mut k = match after {
        // Premier k dont l'instant dépasse `after` (instants recalculés depuis k : pas de dérive).
        Some(a) => (((a - origin) / step + 1e-6).floor() + 1.0).max(0.0) as u64,
        None => 0,
    };
    let mut out = Vec::new();
    loop {
        let t = origin + k as f64 * step;
        if t > until {
            return out;
        }
        if after.is_none_or(|a| t > a + SAME_TIME) {
            out.push(t);
        }
        k += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plain(width: usize, height: usize, v: u8) -> Vec<u8> {
        [v, v, v, 255].repeat(width * height)
    }

    fn timeline(times: &[f64], cuts: &[bool]) -> Timeline {
        Timeline {
            frames: (0..times.len()).collect(),
            times: times.to_vec(),
            cut_after: cuts.to_vec(),
        }
    }

    #[test]
    fn a_repeated_frame_with_compression_noise_is_a_duplicate() {
        let a = plain(64, 32, 100);
        let mut b = a.clone();
        b[0] = 104; // un pixel bruité, noyé dans sa case
        assert!(is_duplicate(&change(
            &thumb(&a, 64, 32),
            &thumb(&b, 64, 32)
        )));
    }

    #[test]
    fn a_compressed_duplicate_with_a_few_noisy_blocks_is_still_a_duplicate() {
        // 200 cases ; une seule (0,5 %) bruitée de 13 niveaux, comme un doublon H.264.
        let a = plain(160, 80, 100);
        let mut b = a.clone();
        for y in 0..8 {
            for x in 0..8 {
                b[(y * 160 + x) * 4..(y * 160 + x) * 4 + 3].copy_from_slice(&[113, 113, 113]);
            }
        }
        assert!(is_duplicate(&change(
            &thumb(&a, 160, 80),
            &thumb(&b, 160, 80)
        )));
    }

    #[test]
    fn a_small_moving_object_is_not_a_duplicate() {
        let a = plain(64, 32, 100);
        let mut b = a.clone();
        for y in 8..16 {
            for x in 8..16 {
                b[(y * 64 + x) * 4..(y * 64 + x) * 4 + 3].copy_from_slice(&[200, 200, 200]);
            }
        }
        assert!(!is_duplicate(&change(
            &thumb(&a, 64, 32),
            &thumb(&b, 64, 32)
        )));
    }

    #[test]
    fn a_change_of_shot_is_a_cut_but_a_fast_pan_is_not() {
        // Pan : mêmes tons (moitié sombre, moitié claire), décalés.
        let half = |dark_left: bool| {
            let mut img = Vec::new();
            for _ in 0..32 {
                for x in 0..64 {
                    let v = if (x < 32) == dark_left { 20 } else { 220 };
                    img.extend_from_slice(&[v, v, v, 255]);
                }
            }
            img
        };
        let pan = change(&thumb(&half(true), 64, 32), &thumb(&half(false), 64, 32));
        assert!(pan.mean > CUT_MEAN && !is_cut(&pan), "{pan:?}");
        let shot = change(
            &thumb(&plain(64, 32, 30), 64, 32),
            &thumb(&plain(64, 32, 200), 64, 32),
        );
        assert!(is_cut(&shot), "{shot:?}");
    }

    #[test]
    fn duplicates_are_dropped_unless_the_scene_stays_still() {
        // Jeu à 30 im./s enregistré à 60 : une image sur deux répète la précédente.
        assert_eq!(
            kept(&[false, true, false, true, false]),
            [true, false, true, false, true]
        );
        // Immobile : les trois premières répétitions sont écartées, les suivantes gardées.
        assert_eq!(
            kept(&[false, true, true, true, true, true]),
            [true, false, false, false, true, true]
        );
    }

    #[test]
    fn output_frames_fall_between_sources_at_their_real_instant() {
        // 24 → 60 : la 2e image de sortie (1/30 s) est aux 4/5 entre les sources 0 et 1.
        let line = timeline(&[0.0, 1.0 / 24.0, 2.0 / 24.0], &[false, false]);
        let times = output_times(0.0, 60.0, None, 2.0 / 24.0 + SAME_TIME);
        assert_eq!(times.len(), 6); // 0, 1/60 … 5/60 = 2/24
        assert_eq!(line.source_at(times[0], false), Some(Source::Copy(0)));
        match line.source_at(times[2], false) {
            Some(Source::Between { a: 0, b: 1, t }) => assert!((t - 0.8).abs() < 1e-5, "{t}"),
            other => panic!("{other:?}"),
        }
        assert_eq!(line.source_at(times[5], false), Some(Source::Copy(2)));
    }

    #[test]
    fn a_dropped_duplicate_is_replaced_by_an_in_between_frame() {
        // Sources 0, 1 (doublon écarté), 2 : l'instant de 1 tombe à mi-chemin entre 0 et 2.
        let line = Timeline {
            frames: vec![0, 2],
            times: vec![0.0, 2.0 / 60.0],
            cut_after: vec![false],
        };
        assert_eq!(
            line.source_at(1.0 / 60.0, false),
            Some(Source::Between { a: 0, b: 2, t: 0.5 })
        );
    }

    #[test]
    fn nothing_is_invented_across_a_cut() {
        let line = timeline(&[0.0, 0.1], &[true]);
        assert_eq!(line.source_at(0.03, false), Some(Source::Copy(0)));
        assert_eq!(line.source_at(0.07, false), Some(Source::Copy(1)));
    }

    #[test]
    fn after_the_last_source_frame_it_waits_for_the_next_set_unless_the_clip_ends() {
        let line = timeline(&[0.0, 0.1], &[false]);
        assert_eq!(line.source_at(0.15, false), None);
        assert_eq!(line.source_at(0.15, true), Some(Source::Copy(1)));
    }

    #[test]
    fn output_times_resume_after_the_previous_call_without_repeating_or_skipping() {
        let all = output_times(0.0, 60.0, None, 1.0);
        let head = output_times(0.0, 60.0, None, 0.5);
        let tail = output_times(0.0, 60.0, head.last().copied(), 1.0);
        assert_eq!(head.len() + tail.len(), all.len());
        assert_eq!(head.len(), 31); // 0 … 30/60
        assert!((tail[0] - 31.0 / 60.0).abs() < 1e-9);
    }
}
