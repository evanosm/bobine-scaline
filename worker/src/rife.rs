//! RIFE 4.25 (Practical-RIFE, `IFNet_HDv3`) : image intermédiaire entre deux images.
//!
//! Un encodeur (`Head`) tire 4 canaux de traits de chaque image. Cinq blocs (`IFBlock`), de la
//! plus grossière (1/16) à la pleine résolution, affinent un flux optique dans les deux sens, un
//! masque de mélange et 8 canaux de traits. Les deux images, déformées par le flux, sont
//! mélangées par la sigmoïde du masque. Même calcul que `IFNet.forward` avec `timestep` 0,5,
//! `scale` 1, sans `ensemble` ; l'image est complétée de zéros jusqu'à un multiple de 128 comme
//! le fait `inference_video.py`.

use crate::gpu::{Gpu, GpuError, Tensor, Weights};
use crate::safetensors::SafeTensors;

/// Côtés complétés à un multiple de 128 (la plus petite échelle est 1/16, puis deux pas de 2).
pub const ALIGN: u32 = 128;
/// Échelles des cinq blocs.
const SCALES: [u32; 5] = [16, 8, 4, 2, 1];
/// Largeur (canaux) de chaque bloc.
const WIDTHS: [usize; 5] = [192, 128, 96, 64, 32];
/// Pente de la LeakyReLU(0,2) de RIFE.
const LEAKY: f32 = 0.2;

struct Block {
    conv0a: Weights,
    conv0b: Weights,
    res: Vec<Weights>,
    last: Weights,
}

pub struct Rife {
    head: [Weights; 4],
    blocks: Vec<Block>,
}

/// Les convolutions de RIFE calculent en f32 : en f16, le flux optique s'écarte de quelques
/// niveaux par endroits (13 au pire sur l'image d'essai), pour un gain de temps faible.
fn exact(mut w: Weights) -> Weights {
    w.exact = true;
    w
}

fn leaky(c: usize) -> Vec<f32> {
    vec![LEAKY; c]
}

fn conv(
    gpu: &Gpu,
    st: &SafeTensors,
    name: &str,
    cin: usize,
    cout: usize,
    act: bool,
) -> Result<Weights, String> {
    let w = st.get(&format!("{name}.weight"), &[cout, cin, 3, 3])?;
    let b = st.get(&format!("{name}.bias"), &[cout])?;
    let slope = if act { leaky(cout) } else { Vec::new() };
    Ok(exact(gpu.weights(
        &w.data,
        &b.data,
        &slope,
        cin as u32,
        cout as u32,
        3,
        false,
    )))
}

fn deconv(
    gpu: &Gpu,
    st: &SafeTensors,
    name: &str,
    cin: usize,
    cout: usize,
) -> Result<Weights, String> {
    let w = st.get(&format!("{name}.weight"), &[cin, cout, 4, 4])?;
    let b = st.get(&format!("{name}.bias"), &[cout])?;
    Ok(gpu.weights(&w.data, &b.data, &[], cin as u32, cout as u32, 4, true))
}

impl Rife {
    pub fn load(gpu: &Gpu, st: &SafeTensors) -> Result<Rife, String> {
        let head = [
            conv(gpu, st, "encode.cnn0", 3, 16, true)?,
            conv(gpu, st, "encode.cnn1", 16, 16, true)?,
            conv(gpu, st, "encode.cnn2", 16, 16, true)?,
            deconv(gpu, st, "encode.cnn3", 16, 4)?,
        ];
        let mut blocks = Vec::new();
        for (i, &c) in WIDTHS.iter().enumerate() {
            let p = format!("block{i}");
            let cin = if i == 0 { 15 } else { 28 };
            let mut res = Vec::new();
            for k in 0..8 {
                // ResConv : relu(conv(x) × beta + x). beta est replié dans les poids et le biais.
                let name = format!("{p}.convblock.{k}");
                let w = st.get(&format!("{name}.conv.weight"), &[c, c, 3, 3])?;
                let b = st.get(&format!("{name}.conv.bias"), &[c])?;
                let beta = &st.get(&format!("{name}.beta"), &[1, c, 1, 1])?.data;
                let per_out = c * 9;
                let w: Vec<f32> = w
                    .data
                    .iter()
                    .enumerate()
                    .map(|(j, v)| v * beta[j / per_out])
                    .collect();
                let b: Vec<f32> = b.data.iter().zip(beta).map(|(v, k)| v * k).collect();
                res.push(exact(gpu.weights(
                    &w,
                    &b,
                    &leaky(c),
                    c as u32,
                    c as u32,
                    3,
                    false,
                )));
            }
            blocks.push(Block {
                conv0a: conv(gpu, st, &format!("{p}.conv0.0.0"), cin, c / 2, true)?,
                conv0b: conv(gpu, st, &format!("{p}.conv0.1.0"), c / 2, c, true)?,
                res,
                last: deconv(gpu, st, &format!("{p}.lastconv.0"), c, 52)?,
            });
        }
        Ok(Rife { head, blocks })
    }

    /// Traits d'une image (4 canaux, même taille).
    fn encode(&self, batch: &mut crate::gpu::Batch, img: &Tensor) -> Tensor {
        let x = batch.conv(img, &self.head[0], 2, true, None);
        let x = batch.conv(&x, &self.head[1], 1, true, None);
        let x = batch.conv(&x, &self.head[2], 1, true, None);
        batch.deconv(&x, &self.head[3])
    }

    /// Un bloc sur `cat` (pleine résolution) à l'échelle 1/`s` ; rend 13 canaux à pleine
    /// résolution : flux (4, déjà multiplié par `s`), masque (1), traits (8).
    fn block(&self, batch: &mut crate::gpu::Batch, i: usize, cat: &Tensor) -> Tensor {
        let s = SCALES[i];
        let blk = &self.blocks[i];
        // Le flux (canaux 24 à 27, groupe 6) change d'échelle avec l'image.
        let flow_group = if i == 0 { (0, 0) } else { (6, 7) };
        let small;
        let x = if s == 1 {
            cat
        } else {
            small = batch.resize(cat, cat.w / s, cat.h / s, 1.0 / s as f32, flow_group);
            &small
        };
        let f = batch.conv(x, &blk.conv0a, 2, true, None);
        let mut f = batch.conv(&f, &blk.conv0b, 2, true, None);
        for r in &blk.res {
            f = batch.conv(&f, r, 1, true, Some(&f));
        }
        let tmp = batch.deconv(&f, &blk.last);
        let tmp = batch.shuffle(&tmp, 2, None);
        if s == 1 {
            tmp
        } else {
            batch.resize(&tmp, cat.w, cat.h, s as f32, (0, 1))
        }
    }

    /// Image intermédiaire (à mi-chemin) de deux images RGBA 8 bits de `w` × `h`.
    pub fn middle(
        &self,
        gpu: &Gpu,
        a: &[u8],
        b: &[u8],
        w: u32,
        h: u32,
    ) -> Result<Vec<u8>, GpuError> {
        let (pw, ph) = (w.div_ceil(ALIGN) * ALIGN, h.div_ceil(ALIGN) * ALIGN);
        let ia = gpu.image_buffer(w, h);
        let ib = gpu.image_buffer(w, h);
        gpu.queue.write_buffer(&ia, 0, a);
        gpu.queue.write_buffer(&ib, 0, b);
        let mut batch = gpu.batch();
        let img0 = batch.unpack(&ia, w, h, 0, 0, pw, ph);
        let img1 = batch.unpack(&ib, w, h, 0, 0, pw, ph);
        let f0 = self.encode(&mut batch, &img0);
        let f1 = self.encode(&mut batch, &img1);

        // Bloc 0 : [img0, img1, f0, f1, timestep] (15 canaux).
        let cat = gpu.tensor(15, pw, ph);
        batch.copy(&img0, 0, &cat, 0, 3);
        batch.copy(&img1, 0, &cat, 3, 3);
        batch.copy(&f0, 0, &cat, 6, 4);
        batch.copy(&f1, 0, &cat, 10, 4);
        batch.fill(&cat, 14, 1, 0.5);
        let mut tmp = self.block(&mut batch, 0, &cat);
        let mut flow = gpu.tensor(4, pw, ph);
        batch.copy(&tmp, 0, &flow, 0, 4);
        batch.submit();

        for i in 1..SCALES.len() {
            let mut batch = gpu.batch();
            let w0 = batch.warp(&img0, &flow, 0);
            let w1 = batch.warp(&img1, &flow, 2);
            let wf0 = batch.warp(&f0, &flow, 0);
            let wf1 = batch.warp(&f1, &flow, 2);
            // [img0 déformée, img1 déformée, traits déformés ×2, timestep, masque, traits, flux].
            let cat = gpu.tensor(28, pw, ph);
            batch.copy(&w0, 0, &cat, 0, 3);
            batch.copy(&w1, 0, &cat, 3, 3);
            batch.copy(&wf0, 0, &cat, 6, 4);
            batch.copy(&wf1, 0, &cat, 10, 4);
            batch.fill(&cat, 14, 1, 0.5);
            batch.copy(&tmp, 4, &cat, 15, 9);
            batch.copy(&flow, 0, &cat, 24, 4);
            tmp = self.block(&mut batch, i, &cat);
            flow = batch.add(&flow, &tmp, 4);
            batch.submit();
        }

        let mut batch = gpu.batch();
        let w0 = batch.warp(&img0, &flow, 0);
        let w1 = batch.warp(&img1, &flow, 2);
        let merged = batch.blend(&w0, &w1, &tmp, 4);
        let out = gpu.image_buffer(w, h);
        batch.pack(&merged, &out, w, h, 0, 0);
        batch.submit();
        gpu.read_buffer(&out, u64::from(w * h * 4))
    }
}
