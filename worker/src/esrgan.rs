//! Real-ESRGAN compact (SRVGGNetCompact) : agrandissement ×4 d'une image.
//!
//! Réseau : convolution 3 → 64 canaux, `num_conv` convolutions 64 → 64, chacune suivie d'une
//! PReLU, puis une convolution 64 → 48 et un pixel shuffle ×4 (48 = 3 × 4²), plus l'image
//! d'entrée agrandie au plus proche. `realesr-general-x4v3` : 32 convolutions (photo réelle) ;
//! `realesr-animevideov3` : 16 (animation).
//!
//! L'image est traitée par tuiles pour borner la mémoire du GPU. Chaque tuile est lue avec une
//! marge d'autant de pixels que de convolutions (le champ de vision du réseau) : son centre est
//! alors identique au calcul sur l'image entière, sans couture.

use crate::gpu::{Gpu, GpuError, Tensor, Weights};
use crate::safetensors::SafeTensors;

/// Côté du centre d'une tuile, au plus (en pixels d'entrée).
pub const TILE: u32 = 512;
const FEAT: u32 = 64;

pub struct Esrgan {
    /// Couches dans l'ordre : `num_conv + 1` avec PReLU, la dernière sans.
    layers: Vec<Weights>,
}

/// Une tuile : centre (x, y, w, h) et zone lue avec la marge (x, y, w, h), en pixels d'entrée.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Tile {
    pub core: [u32; 4],
    pub read: [u32; 4],
}

/// Découpe une image (w × h) en tuiles de centre `tile` au plus, de tailles égales, lues avec
/// `margin` pixels autour (bornés à l'image).
pub fn tiles(w: u32, h: u32, tile: u32, margin: u32) -> Vec<Tile> {
    let (nx, ny) = (w.div_ceil(tile), h.div_ceil(tile));
    let (cw, ch) = (w.div_ceil(nx), h.div_ceil(ny));
    let mut out = Vec::new();
    for ty in 0..ny {
        for tx in 0..nx {
            let (x0, y0) = (tx * cw, ty * ch);
            let (x1, y1) = ((x0 + cw).min(w), (y0 + ch).min(h));
            if x0 >= x1 || y0 >= y1 {
                continue;
            }
            let (rx0, ry0) = (x0.saturating_sub(margin), y0.saturating_sub(margin));
            let (rx1, ry1) = ((x1 + margin).min(w), (y1 + margin).min(h));
            out.push(Tile {
                core: [x0, y0, x1 - x0, y1 - y0],
                read: [rx0, ry0, rx1 - rx0, ry1 - ry0],
            });
        }
    }
    out
}

impl Esrgan {
    pub fn load(gpu: &Gpu, st: &SafeTensors, num_conv: usize) -> Result<Esrgan, String> {
        let mut layers = Vec::new();
        for i in 0..=num_conv + 1 {
            let index = 2 * i;
            let last = i == num_conv + 1;
            let cin = if i == 0 { 3 } else { FEAT as usize };
            let cout = if last { 48 } else { FEAT as usize };
            let w = st.get(&format!("body.{index}.weight"), &[cout, cin, 3, 3])?;
            let b = st.get(&format!("body.{index}.bias"), &[cout])?;
            let slope = if last {
                Vec::new()
            } else {
                st.get(&format!("body.{}.weight", index + 1), &[cout])?
                    .data
                    .clone()
            };
            layers.push(gpu.weights(&w.data, &b.data, &slope, cin as u32, cout as u32, 3, false));
        }
        Ok(Esrgan { layers })
    }

    /// Marge de lecture d'une tuile : une par convolution.
    pub fn margin(&self) -> u32 {
        self.layers.len() as u32
    }

    /// Le réseau sur un tenseur `x` (3 canaux) ; `a` et `b` : tenseurs de travail de 64 canaux
    /// au moins aussi grands, `last` : 48 canaux. Rend 3 canaux, 4 fois plus grands.
    fn forward(&self, gpu: &Gpu, x: &Tensor, a: &Tensor, b: &Tensor, last: &Tensor) -> Tensor {
        let (w, h) = (x.w, x.h);
        let (a, b) = (a.view(FEAT, w, h), b.view(FEAT, w, h));
        let last = last.view(48, w, h);
        let mut batch = gpu.batch();
        let n = self.layers.len();
        for (i, layer) in self.layers.iter().enumerate() {
            let input = if i == 0 {
                x
            } else if i % 2 == 1 {
                &a
            } else {
                &b
            };
            let output = if i == n - 1 {
                &last
            } else if i % 2 == 0 {
                &a
            } else {
                &b
            };
            batch.conv_into(input, layer, 1, i < n - 1, None, output);
        }
        let out = batch.shuffle(&last, 4, Some(x));
        batch.submit();
        out
    }

    /// Agrandit ×4 une image RGBA 8 bits (w × h) ; rend l'image RGBA (4w × 4h).
    /// `progress` reçoit la part faite (0 à 1) après chaque tuile.
    pub fn upscale(
        &self,
        gpu: &Gpu,
        rgba: &[u8],
        w: u32,
        h: u32,
        progress: &mut dyn FnMut(f64),
    ) -> Result<Vec<u8>, GpuError> {
        let list = tiles(w, h, TILE, self.margin());
        let (mut max_w, mut max_h) = (1, 1);
        for t in &list {
            max_w = max_w.max(t.read[2]);
            max_h = max_h.max(t.read[3]);
        }
        let image = gpu.image_buffer(w, h);
        gpu.queue.write_buffer(&image, 0, rgba);
        let a = gpu.tensor(FEAT, max_w, max_h);
        let b = gpu.tensor(FEAT, max_w, max_h);
        let last = gpu.tensor(48, max_w, max_h);
        let out_buffer = gpu.image_buffer(4 * max_w, 4 * max_h);
        let (ow, oh) = (4 * w as usize, 4 * h as usize);
        let mut out = vec![0u8; ow * oh * 4];
        for (n, t) in list.iter().enumerate() {
            let [rx, ry, rw, rh] = t.read;
            let [cx, cy, cw, ch] = t.core;
            let mut batch = gpu.batch();
            let x = batch.unpack(&image, w, h, rx, ry, rw, rh);
            batch.submit();
            let up = self.forward(gpu, &x, &a, &b, &last);
            let mut batch = gpu.batch();
            batch.pack(
                &up,
                &out_buffer,
                4 * cw,
                4 * ch,
                4 * (cx - rx),
                4 * (cy - ry),
            );
            batch.submit();
            let bytes = gpu.read_buffer(&out_buffer, u64::from(16 * cw * ch * 4))?;
            let row = 4 * cw as usize * 4;
            for (j, line) in bytes.chunks_exact(row).enumerate() {
                let y = 4 * cy as usize + j;
                let start = (y * ow + 4 * cx as usize) * 4;
                out[start..start + row].copy_from_slice(line);
            }
            progress((n + 1) as f64 / list.len() as f64);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiles_cover_the_image_once_and_read_a_margin() {
        let list = tiles(1920, 1080, 512, 34);
        assert_eq!(list.len(), 4 * 3);
        let area: u32 = list.iter().map(|t| t.core[2] * t.core[3]).sum();
        assert_eq!(area, 1920 * 1080);
        let first = list[0];
        assert_eq!(first.core, [0, 0, 480, 360]);
        assert_eq!(first.read, [0, 0, 514, 394]);
        let middle = list[5];
        assert_eq!(middle.core, [480, 360, 480, 360]);
        assert_eq!(middle.read, [446, 326, 548, 428]);
    }

    #[test]
    fn a_small_image_is_one_tile() {
        assert_eq!(
            tiles(100, 60, 512, 34),
            vec![Tile {
                core: [0, 0, 100, 60],
                read: [0, 0, 100, 60]
            }]
        );
    }
}
