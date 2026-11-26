//! Redimensionnement d'une image RGBA 8 bits sur le processeur : filtre triangle élargi à
//! l'échelle de la réduction (moyenne sans crénelage quand on réduit, bilinéaire quand on
//! agrandit), en deux passes (lignes puis colonnes). Sert à ramener la sortie ×4 de
//! Real-ESRGAN à la taille demandée (par exemple 7680 → 3840 de large).

use rayon::prelude::*;

/// Pour chaque pixel de sortie : premier pixel source et poids des suivants (somme = 1).
fn taps(src: usize, dst: usize) -> Vec<(usize, Vec<f32>)> {
    let scale = src as f64 / dst as f64;
    // Rayon du filtre en pixels source : 1 pour agrandir, `scale` / 2 pour réduire.
    let support = if scale > 1.0 { scale / 2.0 } else { 1.0 };
    (0..dst)
        .map(|i| {
            let center = (i as f64 + 0.5) * scale;
            let lo = ((center - support).floor().max(0.0)) as usize;
            let hi = ((center + support).ceil() as usize).min(src);
            let mut weights: Vec<f32> = (lo..hi)
                .map(|j| {
                    let j = j as f64;
                    if scale > 1.0 {
                        // Réduction : part du pixel source couverte par le pixel de sortie
                        // (moyenne de surface, exacte pour une réduction de moitié).
                        let overlap = (j + 1.0).min(center + support) - j.max(center - support);
                        overlap.max(0.0) as f32
                    } else {
                        // Agrandissement : bilinéaire.
                        (1.0 - (j + 0.5 - center).abs()).max(0.0) as f32
                    }
                })
                .collect();
            let sum: f32 = weights.iter().sum();
            if sum > 0.0 {
                weights.iter_mut().for_each(|w| *w /= sum);
            } else {
                // Cas limite (image d'un pixel) : on copie le pixel le plus proche.
                weights = vec![1.0];
            }
            (lo.min(src - 1), weights)
        })
        .collect()
}

/// Redimensionne `rgba` (sw × sh) en (dw × dh). L'alpha de la sortie vaut 255.
pub fn resize_rgba(rgba: &[u8], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<u8> {
    if sw == dw && sh == dh {
        return rgba.to_vec();
    }
    let xt = taps(sw, dw);
    let yt = taps(sh, dh);
    // Passe 1 : chaque ligne source à la nouvelle largeur (3 canaux en f32).
    let rows: Vec<Vec<f32>> = (0..sh)
        .into_par_iter()
        .map(|y| {
            let line = &rgba[y * sw * 4..(y + 1) * sw * 4];
            let mut out = vec![0f32; dw * 3];
            for (x, (lo, weights)) in xt.iter().enumerate() {
                let mut acc = [0f32; 3];
                for (k, w) in weights.iter().enumerate() {
                    let p = (lo + k) * 4;
                    for c in 0..3 {
                        acc[c] += w * f32::from(line[p + c]);
                    }
                }
                out[x * 3..x * 3 + 3].copy_from_slice(&acc);
            }
            out
        })
        .collect();
    // Passe 2 : chaque ligne de sortie mélange des lignes de la passe 1.
    let mut out = vec![0u8; dw * dh * 4];
    out.par_chunks_mut(dw * 4)
        .zip(yt.par_iter())
        .for_each(|(line, (lo, weights))| {
            for x in 0..dw {
                let mut acc = [0f32; 3];
                for (k, w) in weights.iter().enumerate() {
                    let row = &rows[lo + k];
                    for c in 0..3 {
                        acc[c] += w * row[x * 3 + c];
                    }
                }
                for c in 0..3 {
                    line[x * 4 + c] = acc[c].round().clamp(0.0, 255.0) as u8;
                }
                line[x * 4 + 3] = 255;
            }
        });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn halving_averages_each_two_by_two_block() {
        // 4 × 2 → 2 × 1 : chaque pixel de sortie est la moyenne de son bloc 2 × 2.
        let mut src = Vec::new();
        for v in [0u8, 100, 200, 40, 20, 60, 120, 255] {
            src.extend_from_slice(&[v, v, v, 255]);
        }
        let out = resize_rgba(&src, 4, 2, 2, 1);
        assert_eq!(out[0], 45); // (0 + 100 + 20 + 60) / 4
        assert_eq!(out[4], 154); // (200 + 40 + 120 + 255) / 4 = 153,75
        assert_eq!(out[3], 255);
    }

    #[test]
    fn a_flat_image_stays_flat_at_any_size() {
        let src: Vec<u8> = (0..7 * 5).flat_map(|_| [10u8, 128, 250, 255]).collect();
        for (w, h) in [(3, 2), (14, 10), (11, 3)] {
            let out = resize_rgba(&src, 7, 5, w, h);
            assert_eq!(out.len(), w * h * 4);
            assert!(out.chunks(4).all(|p| p == [10, 128, 250, 255]));
        }
    }

    #[test]
    fn same_size_is_a_copy() {
        let src: Vec<u8> = (0..16).collect();
        assert_eq!(resize_rgba(&src, 2, 2, 2, 2), src);
    }
}
