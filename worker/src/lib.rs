//! Scaline : agrandissement (Real-ESRGAN compact) et images intermédiaires (RIFE 4.25) calculés
//! en shaders WGSL sur wgpu, et protocole 1 des programmes natifs de Bobine.
//!
//! Les poids (safetensors f16) sont dans le dossier `models/` du paquet du plugin.

pub mod esrgan;
pub mod frames;
pub mod gpu;
pub mod protocol;
pub mod resize;
pub mod rife;
pub mod safetensors;
