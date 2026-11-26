//! Calcul sur le GPU avec wgpu (Metal sous macOS, DX12 sous Windows, Vulkan sous Linux) : tenseurs,
//! poids et opérations des deux réseaux, écrites en WGSL (`shaders/`).
//!
//! Un tenseur est rangé par plans de 4 canaux (`vec4`, le dernier groupe complété de zéros) :
//! `données[(g × hauteur + y) × largeur + x]` pour le groupe g. Il est stocké en f16 si le GPU
//! le permet (moitié moins de mémoire), sinon en f32 ; les calculs se font toujours en f32.

use half::f16;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use wgpu::util::DeviceExt;

/// Erreur de calcul, rendue au plugin avec un code court.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuError {
    pub code: &'static str,
    pub message: String,
}

impl GpuError {
    fn new(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }
}

/// Un tenseur sur le GPU : `c` canaux de `w` × `h` pixels.
pub struct Tensor {
    pub buffer: wgpu::Buffer,
    pub c: u32,
    pub w: u32,
    pub h: u32,
}

impl Tensor {
    /// Nombre de groupes de 4 canaux.
    pub fn c4(&self) -> u32 {
        self.c.div_ceil(4)
    }

    /// Le même tampon vu avec une autre forme (au plus aussi grande) : réutilise la mémoire.
    pub fn view(&self, c: u32, w: u32, h: u32) -> Tensor {
        Tensor {
            buffer: self.buffer.clone(),
            c,
            w,
            h,
        }
    }
}

/// Poids d'une convolution 3×3 ou d'une déconvolution 4×4, rangés comme les lit le shader.
pub struct Weights {
    pub w: wgpu::Buffer,
    pub bias: wgpu::Buffer,
    pub slope: wgpu::Buffer,
    pub cin: u32,
    pub cout: u32,
    /// Convolution calculée en f32 même si le stockage est en f16 (plus lente, plus exacte).
    pub exact: bool,
}

/// Les shaders : nom, source, point d'entrée, calculs en f32 même si le stockage est en f16
/// (`conv` calcule en f16 si possible, deux fois plus vite ; `conv_exact` en f32).
const SHADERS: [(&str, &str, &str, bool); 11] = [
    ("conv", include_str!("shaders/conv.wgsl"), "main", false),
    (
        "conv_exact",
        include_str!("shaders/conv.wgsl"),
        "main",
        true,
    ),
    ("deconv", include_str!("shaders/deconv.wgsl"), "main", true),
    (
        "shuffle",
        include_str!("shaders/shuffle.wgsl"),
        "main",
        true,
    ),
    ("resize", include_str!("shaders/resize.wgsl"), "main", true),
    ("warp", include_str!("shaders/warp.wgsl"), "main", true),
    ("blend", include_str!("shaders/blend.wgsl"), "main", true),
    ("copy", include_str!("shaders/copy.wgsl"), "main", true),
    ("add", include_str!("shaders/add.wgsl"), "main", true),
    ("unpack", include_str!("shaders/rgba.wgsl"), "unpack", true),
    ("pack", include_str!("shaders/rgba.wgsl"), "pack", true),
];

pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    /// Tenseurs en f16 (sinon f32).
    pub f16: bool,
    /// Nom du GPU et moteur (« Apple M1 Pro (Metal) »).
    pub adapter: String,
    /// GPU logiciel (aucun GPU matériel trouvé) : beaucoup plus lent.
    pub software: bool,
    pipelines: HashMap<&'static str, wgpu::ComputePipeline>,
    /// Tampon minuscule lié aux entrées inutilisées d'un shader.
    dummy: wgpu::Buffer,
    /// Première erreur du GPU non attrapée (shader refusé, tampon trop grand…).
    errors: Arc<Mutex<Option<String>>>,
}

/// Paramètres d'un shader : 12 mots de 32 bits (entiers, ou flottants par `f32::to_bits`).
pub type Params = [u32; 12];

impl Gpu {
    /// Choisit un GPU matériel, sinon le GPU logiciel du système (WARP sous Windows, lavapipe sous
    /// Linux s'il est installé). `f16` à `false` force le stockage en f32 (comparaisons).
    pub fn new(allow_f16: bool) -> Result<Gpu, GpuError> {
        let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
        // Sous Windows, DX12 (comme Bobine) ; Vulkan ailleurs hors Apple, Metal sous macOS.
        if cfg!(windows) {
            descriptor.backends = wgpu::Backends::DX12;
        }
        let instance = wgpu::Instance::new(descriptor);
        let mut options = wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
            apply_limit_buckets: false,
        };
        // `pollster::block_on` exécute une fonction asynchrone jusqu'au bout, de façon bloquante.
        let adapter = match pollster::block_on(instance.request_adapter(&options)) {
            Ok(adapter) => adapter,
            Err(_) => {
                options.force_fallback_adapter = true;
                pollster::block_on(instance.request_adapter(&options)).map_err(|_| {
                    GpuError::new(
                        "no_gpu",
                        "no usable GPU (Metal, DirectX 12 or Vulkan) and no software fallback",
                    )
                })?
            }
        };
        let info = adapter.get_info();
        let software = info.device_type == wgpu::DeviceType::Cpu;
        let f16 = allow_f16 && adapter.features().contains(wgpu::Features::SHADER_F16);
        let required_features = if f16 {
            wgpu::Features::SHADER_F16
        } else {
            wgpu::Features::empty()
        };
        // Les plus gros tenseurs (RIFE en 1080p) dépassent les limites par défaut (128 Mo) : on
        // prend celles du GPU.
        let limits = adapter.limits();
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("scaline"),
            required_features,
            required_limits: limits,
            ..Default::default()
        }))
        .map_err(|e| GpuError::new("no_gpu", e.to_string()))?;
        // Par défaut, wgpu panique sur une erreur de validation ; avec `panic = "abort"` le
        // programme se fermerait sans rien dire. On l'écrit au journal de Bobine et on la garde :
        // le calcul suivant échoue avec un message clair au lieu de rendre une image fausse
        // (sous DirectX 12, une erreur de compilation d'un shader arrive par ce chemin).
        let errors: Arc<Mutex<Option<String>>> = Arc::default();
        let sink = errors.clone();
        device.on_uncaptured_error(Arc::new(move |error: wgpu::Error| {
            eprintln!("scaline : erreur GPU : {error}");
            if let Ok(mut first) = sink.lock() {
                first.get_or_insert_with(|| error.to_string().chars().take(300).collect());
            }
        }));
        // Une erreur de compilation d'un shader est attrapée ici (sinon seulement journalisée, et
        // les calculs rendraient des zéros sans rien dire).
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let mut pipelines = HashMap::new();
        let mut modules: HashMap<(&str, bool), wgpu::ShaderModule> = HashMap::new();
        for (name, source, entry, exact) in SHADERS {
            let prelude = match (f16, exact) {
                (true, false) => "enable f16;\nalias S = f16;\nalias A = f16;\n",
                (true, true) => "enable f16;\nalias S = f16;\nalias A = f32;\n",
                (false, _) => "alias S = f32;\nalias A = f32;\n",
            };
            // Un module par fichier et par précision (`rgba.wgsl` sert deux points d'entrée).
            let module = modules
                .entry((source, exact))
                .or_insert_with(|| {
                    device.create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some(name),
                        source: wgpu::ShaderSource::Wgsl(format!("{prelude}{source}").into()),
                    })
                })
                .clone();
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(name),
                layout: None,
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            });
            pipelines.insert(name, pipeline);
        }
        if let Some(error) = pollster::block_on(scope.pop()) {
            return Err(GpuError::new(
                "gpu_failed",
                format!("shader error: {error}"),
            ));
        }
        let dummy = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dummy"),
            size: 64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let gpu = Gpu {
            device,
            queue,
            f16,
            adapter: format!("{} ({:?})", info.name, info.backend),
            software,
            pipelines,
            dummy,
            errors,
        };
        gpu.check()?;
        Ok(gpu)
    }

    /// Échoue si le GPU a signalé une erreur depuis sa création.
    pub fn check(&self) -> Result<(), GpuError> {
        match self.errors.lock().ok().and_then(|e| e.clone()) {
            Some(error) => Err(GpuError::new("gpu_failed", error)),
            None => Ok(()),
        }
    }

    /// Octets d'une valeur stockée.
    fn elem(&self) -> u64 {
        if self.f16 { 2 } else { 4 }
    }

    /// Tenseur de zéros (wgpu remet tout tampon neuf à zéro).
    pub fn tensor(&self, c: u32, w: u32, h: u32) -> Tensor {
        let size = u64::from(w) * u64::from(h) * u64::from(c.div_ceil(4)) * 4 * self.elem();
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: size.max(16),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Tensor { buffer, c, w, h }
    }

    /// Valeurs f32 vers les octets du type de stockage.
    fn bytes(&self, values: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(values.len() * self.elem() as usize);
        for &v in values {
            if self.f16 {
                out.extend_from_slice(&f16::from_f32(v).to_le_bytes());
            } else {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        out
    }

    fn storage(&self, values: &[f32]) -> wgpu::Buffer {
        let mut contents = self.bytes(values);
        contents.resize(contents.len().max(16), 0);
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: &contents,
                usage: wgpu::BufferUsages::STORAGE,
            })
    }

    /// Envoie les poids d'une couche. `weight[co][ci][k][k]` (convolution, `k` = 3) ou
    /// `weight[ci][co][k][k]` (déconvolution, `k` = 4, `transposed`), comme PyTorch.
    /// `slope` : pente de la partie négative par canal de sortie (PReLU), ou vide.
    // Les arguments suivent la forme des poids de PyTorch ; un struct de plus n'aiderait pas.
    #[allow(clippy::too_many_arguments)]
    pub fn weights(
        &self,
        weight: &[f32],
        bias: &[f32],
        slope: &[f32],
        cin: u32,
        cout: u32,
        k: u32,
        transposed: bool,
    ) -> Weights {
        // Groupes de sortie complétés jusqu'à un multiple de 4 (bornes fixes dans les shaders).
        let (cin4, cout4, taps) = (cin.div_ceil(4), cout.div_ceil(16) * 4, k * k);
        let mut packed = vec![0f32; (cout4 * taps * cin4 * 16) as usize];
        for co in 0..cout {
            for ci in 0..cin {
                for tap in 0..taps {
                    let src = if transposed {
                        (ci * cout + co) * taps + tap
                    } else {
                        (co * cin + ci) * taps + tap
                    };
                    let dst = (((co / 4 * taps + tap) * cin4 + ci / 4) * 4 + ci % 4) * 4 + co % 4;
                    packed[dst as usize] = weight[src as usize];
                }
            }
        }
        let mut b = bias.to_vec();
        b.resize((cout4 * 4) as usize, 0.0);
        let mut s = slope.to_vec();
        s.resize((cout4 * 4) as usize, 0.0);
        Weights {
            w: self.storage(&packed),
            bias: self.storage(&b),
            slope: self.storage(&s),
            cin,
            cout,
            exact: false,
        }
    }

    /// Envoie un tenseur depuis des valeurs `[c][h][w]` (ordre de PyTorch). Pour les tests.
    pub fn upload(&self, values: &[f32], c: u32, w: u32, h: u32) -> Tensor {
        let mut packed = vec![0f32; (w * h * c.div_ceil(4) * 4) as usize];
        for ch in 0..c {
            for y in 0..h {
                for x in 0..w {
                    packed[(((ch / 4 * h + y) * w + x) * 4 + ch % 4) as usize] =
                        values[((ch * h + y) * w + x) as usize];
                }
            }
        }
        let t = self.tensor(c, w, h);
        self.queue.write_buffer(&t.buffer, 0, &self.bytes(&packed));
        t
    }

    /// Relit un tenseur en valeurs `[c][h][w]`. Pour les tests.
    pub fn download(&self, t: &Tensor) -> Result<Vec<f32>, GpuError> {
        let bytes = self.read_buffer(&t.buffer, t.buffer.size())?;
        let value = |i: usize| -> f32 {
            if self.f16 {
                f16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]]).to_f32()
            } else {
                f32::from_le_bytes([
                    bytes[4 * i],
                    bytes[4 * i + 1],
                    bytes[4 * i + 2],
                    bytes[4 * i + 3],
                ])
            }
        };
        let mut out = vec![0f32; (t.c * t.w * t.h) as usize];
        for ch in 0..t.c {
            for y in 0..t.h {
                for x in 0..t.w {
                    out[((ch * t.h + y) * t.w + x) as usize] =
                        value((((ch / 4 * t.h + y) * t.w + x) * 4 + ch % 4) as usize);
                }
            }
        }
        Ok(out)
    }

    /// Tampon d'image RGBA 8 bits (un u32 par pixel).
    pub fn image_buffer(&self, w: u32, h: u32) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("image"),
            size: (u64::from(w) * u64::from(h) * 4).max(16),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Copie les `size` premiers octets d'un tampon vers le processeur (attend la fin du GPU).
    pub fn read_buffer(&self, buffer: &wgpu::Buffer, size: u64) -> Result<Vec<u8>, GpuError> {
        let fail = |e: String| GpuError::new("gpu_failed", e);
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
        self.queue.submit([encoder.finish()]);
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| fail(e.to_string()))?;
        rx.recv()
            .map_err(|e| fail(e.to_string()))?
            .map_err(|e| fail(e.to_string()))?;
        let data = slice
            .get_mapped_range()
            .map_err(|e| fail(e.to_string()))?
            .to_vec();
        staging.unmap();
        // Une erreur de validation pendant le calcul : l'image lue serait fausse.
        self.check()?;
        Ok(data)
    }

    /// Nouvelle suite d'opérations, envoyée au GPU par `submit`.
    pub fn batch(&self) -> Batch<'_> {
        Batch {
            gpu: self,
            encoder: self.device.create_command_encoder(&Default::default()),
        }
    }
}

/// Opérations enregistrées puis envoyées d'un coup au GPU (dans l'ordre).
pub struct Batch<'a> {
    gpu: &'a Gpu,
    encoder: wgpu::CommandEncoder,
}

/// Groupes de travail de 8 × 8 fils pour couvrir `n` éléments.
fn groups(n: u32, per_group: u32) -> u32 {
    n.div_ceil(per_group).max(1)
}

impl Batch<'_> {
    pub fn submit(self) {
        self.gpu.queue.submit([self.encoder.finish()]);
    }

    /// Lance le shader `name` : paramètres en binding 0, tampons ensuite dans l'ordre.
    fn dispatch(&mut self, name: &str, params: Params, buffers: &[&wgpu::Buffer], size: [u32; 3]) {
        let gpu = self.gpu;
        let pipeline = &gpu.pipelines[name];
        let mut bytes = Vec::with_capacity(48);
        for word in params {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        let uniform = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: &bytes,
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let mut entries = vec![wgpu::BindGroupEntry {
            binding: 0,
            resource: uniform.as_entire_binding(),
        }];
        for (i, buffer) in buffers.iter().enumerate() {
            entries.push(wgpu::BindGroupEntry {
                binding: i as u32 + 1,
                resource: buffer.as_entire_binding(),
            });
        }
        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        });
        let mut pass = self.encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(size[0], size[1], size[2]);
    }

    /// Convolution 3×3 (pas 1 ou 2), plus `residual` et activation (si `act`).
    pub fn conv(
        &mut self,
        x: &Tensor,
        w: &Weights,
        stride: u32,
        act: bool,
        residual: Option<&Tensor>,
    ) -> Tensor {
        let out = self
            .gpu
            .tensor(w.cout, x.w.div_ceil(stride), x.h.div_ceil(stride));
        self.conv_into(x, w, stride, act, residual, &out);
        out
    }

    /// `conv` dans un tenseur déjà alloué (`out` : `w.cout` canaux, taille de sortie).
    pub fn conv_into(
        &mut self,
        x: &Tensor,
        w: &Weights,
        stride: u32,
        act: bool,
        residual: Option<&Tensor>,
        out: &Tensor,
    ) {
        let (ow, oh) = (x.w.div_ceil(stride), x.h.div_ceil(stride));
        debug_assert!(out.w == ow && out.h == oh && out.c == w.cout);
        let flags = u32::from(act) | (u32::from(residual.is_some()) << 1);
        let res = residual.map(|r| &r.buffer).unwrap_or(&self.gpu.dummy);
        let cout4 = w.cout.div_ceil(4);
        self.dispatch(
            if w.exact { "conv_exact" } else { "conv" },
            [
                w.cin.div_ceil(4),
                cout4,
                x.w,
                x.h,
                ow,
                oh,
                stride,
                flags,
                0,
                0,
                0,
                0,
            ],
            &[&x.buffer, &w.w, &w.bias, &w.slope, res, &out.buffer],
            [groups(ow, 128), groups(oh, 2), cout4.div_ceil(4)],
        );
    }

    /// Déconvolution 4×4 pas 2 : sortie deux fois plus grande.
    pub fn deconv(&mut self, x: &Tensor, w: &Weights) -> Tensor {
        let (ow, oh) = (x.w * 2, x.h * 2);
        let out = self.gpu.tensor(w.cout, ow, oh);
        let cout4 = w.cout.div_ceil(4);
        self.dispatch(
            "deconv",
            [w.cin.div_ceil(4), cout4, x.w, x.h, ow, oh, 0, 0, 0, 0, 0, 0],
            &[&x.buffer, &w.w, &w.bias, &out.buffer],
            [groups(ow, 8), groups(oh, 8), cout4.div_ceil(4)],
        );
        out
    }

    /// Pixel shuffle de facteur `r`, plus `base` agrandi au plus proche si donné.
    pub fn shuffle(&mut self, x: &Tensor, r: u32, base: Option<&Tensor>) -> Tensor {
        let c = x.c / (r * r);
        let out = self.gpu.tensor(c, x.w * r, x.h * r);
        let b = base.map(|b| &b.buffer).unwrap_or(&self.gpu.dummy);
        self.dispatch(
            "shuffle",
            [
                x.c4(),
                c,
                out.c4(),
                x.w,
                x.h,
                r,
                u32::from(base.is_some()),
                0,
                0,
                0,
                0,
                0,
            ],
            &[&x.buffer, b, &out.buffer],
            [groups(out.w, 8), groups(out.h, 8), 1],
        );
        out
    }

    /// Redimensionnement bilinéaire vers (w, h) ; les groupes [mul_from, mul_to) × `mul`.
    pub fn resize(&mut self, x: &Tensor, w: u32, h: u32, mul: f32, mul4: (u32, u32)) -> Tensor {
        let out = self.gpu.tensor(x.c, w, h);
        let rx = x.w as f32 / w as f32;
        let ry = x.h as f32 / h as f32;
        self.dispatch(
            "resize",
            [
                x.c4(),
                x.w,
                x.h,
                w,
                h,
                mul4.0,
                mul4.1,
                mul.to_bits(),
                rx.to_bits(),
                ry.to_bits(),
                0,
                0,
            ],
            &[&x.buffer, &out.buffer],
            [groups(w, 8), groups(h, 8), 1],
        );
        out
    }

    /// Déforme `x` par le flux `flow` (canaux `offset` et `offset + 1`).
    pub fn warp(&mut self, x: &Tensor, flow: &Tensor, offset: u32) -> Tensor {
        let out = self.gpu.tensor(x.c, x.w, x.h);
        self.dispatch(
            "warp",
            [x.c4(), x.w, x.h, flow.c4(), offset, 0, 0, 0, 0, 0, 0, 0],
            &[&x.buffer, &flow.buffer, &out.buffer],
            [groups(x.w, 8), groups(x.h, 8), 1],
        );
        out
    }

    /// a × sigmoïde(m) + b × (1 − sigmoïde(m)), m = canal `channel` de `mask`.
    pub fn blend(&mut self, a: &Tensor, b: &Tensor, mask: &Tensor, channel: u32) -> Tensor {
        let out = self.gpu.tensor(a.c, a.w, a.h);
        self.dispatch(
            "blend",
            [a.c4(), a.w, a.h, mask.c4(), channel, 0, 0, 0, 0, 0, 0, 0],
            &[&a.buffer, &b.buffer, &mask.buffer, &out.buffer],
            [groups(a.w, 8), groups(a.h, 8), 1],
        );
        out
    }

    /// Copie `count` canaux de `src` (depuis `src_off`) dans `dst` (depuis `dst_off`).
    pub fn copy(&mut self, src: &Tensor, src_off: u32, dst: &Tensor, dst_off: u32, count: u32) {
        self.dispatch(
            "copy",
            [
                dst.w,
                dst.h,
                src.c4(),
                dst.c4(),
                src_off,
                dst_off,
                count,
                0,
                0,
                0,
                0,
                0,
            ],
            &[&src.buffer, &dst.buffer],
            [groups(dst.w, 8), groups(dst.h, 8), 1],
        );
    }

    /// Remplit `count` canaux de `dst` (depuis `dst_off`) avec `value`.
    pub fn fill(&mut self, dst: &Tensor, dst_off: u32, count: u32, value: f32) {
        let dummy = &self.gpu.dummy;
        self.dispatch(
            "copy",
            [
                dst.w,
                dst.h,
                1,
                dst.c4(),
                0,
                dst_off,
                count,
                1,
                value.to_bits(),
                0,
                0,
                0,
            ],
            &[dummy, &dst.buffer],
            [groups(dst.w, 8), groups(dst.h, 8), 1],
        );
    }

    /// Somme des `channels` premiers canaux (multiple de 4, ou tous) de `a` et `b`.
    pub fn add(&mut self, a: &Tensor, b: &Tensor, channels: u32) -> Tensor {
        let out = self.gpu.tensor(channels, a.w, a.h);
        let n4 = channels.div_ceil(4);
        self.dispatch(
            "add",
            [a.w, a.h, a.c4(), b.c4(), n4, 0, 0, 0, 0, 0, 0, 0],
            &[&a.buffer, &b.buffer, &out.buffer],
            [groups(a.w, 8), groups(a.h, 8), 1],
        );
        out
    }

    /// Image RGBA (iw × ih) vers un tenseur de 3 canaux (tw × th) lu à partir de (ox, oy).
    #[allow(clippy::too_many_arguments)]
    pub fn unpack(
        &mut self,
        image: &wgpu::Buffer,
        iw: u32,
        ih: u32,
        ox: u32,
        oy: u32,
        tw: u32,
        th: u32,
    ) -> Tensor {
        let out = self.gpu.tensor(3, tw, th);
        self.dispatch(
            "unpack",
            [iw, ih, tw, th, ox, oy, 0, 0, 0, 0, 0, 0],
            &[image, &out.buffer],
            [groups(tw, 8), groups(th, 8), 1],
        );
        out
    }

    /// Tenseur de 3 canaux vers une image RGBA (iw × ih), lue à partir de (ox, oy) du tenseur.
    pub fn pack(&mut self, x: &Tensor, image: &wgpu::Buffer, iw: u32, ih: u32, ox: u32, oy: u32) {
        self.dispatch(
            "pack",
            [iw, ih, x.w, x.h, ox, oy, 0, 0, 0, 0, 0, 0],
            &[image, &x.buffer],
            [groups(iw, 8), groups(ih, 8), 1],
        );
    }
}
