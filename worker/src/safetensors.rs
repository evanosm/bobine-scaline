//! Lecture d'un fichier safetensors (poids des modèles, en f16) : 8 octets de longueur d'en-tête
//! (petit-boutiste), l'en-tête JSON (nom → type, forme, position des octets), puis les données.

use half::f16;
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Deserialize)]
struct Entry {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

/// Un tenseur des poids, converti en f32.
#[derive(Debug, Clone)]
pub struct Array {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

pub struct SafeTensors {
    tensors: HashMap<String, Array>,
}

/// Plus gros fichier de poids accepté (le plus gros modèle pèse 11 Mo).
const MAX_FILE: u64 = 256 * 1024 * 1024;

impl SafeTensors {
    pub fn load(path: &Path) -> Result<SafeTensors, String> {
        let size = std::fs::metadata(path).map_err(|e| e.to_string())?.len();
        if size > MAX_FILE {
            return Err("weights file is too big".into());
        }
        let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
        Self::parse(&bytes)
    }

    pub fn parse(bytes: &[u8]) -> Result<SafeTensors, String> {
        let bad = |what: &str| format!("bad safetensors file: {what}");
        let len_bytes: [u8; 8] = bytes
            .get(..8)
            .and_then(|b| b.try_into().ok())
            .ok_or_else(|| bad("too short"))?;
        let header_len = u64::from_le_bytes(len_bytes) as usize;
        let header = bytes
            .get(8..8usize.saturating_add(header_len))
            .ok_or_else(|| bad("header"))?;
        let data = &bytes[8 + header_len..];
        let entries: HashMap<String, serde_json::Value> =
            serde_json::from_slice(header).map_err(|e| bad(&e.to_string()))?;
        let mut tensors = HashMap::new();
        for (name, value) in entries {
            if name == "__metadata__" {
                continue;
            }
            let entry: Entry = serde_json::from_value(value).map_err(|e| bad(&e.to_string()))?;
            if entry.dtype != "F16" {
                return Err(bad("only F16 tensors are read"));
            }
            let count: usize = entry.shape.iter().product();
            let [start, end] = entry.data_offsets;
            let raw = data.get(start..end).ok_or_else(|| bad("data offsets"))?;
            if raw.len() != count * 2 {
                return Err(bad("tensor size"));
            }
            let values = raw
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| f16::from_le_bytes(*b).to_f32())
                .collect();
            tensors.insert(
                name,
                Array {
                    shape: entry.shape,
                    data: values,
                },
            );
        }
        Ok(SafeTensors { tensors })
    }

    /// Le tenseur `name`, de la forme attendue.
    pub fn get(&self, name: &str, shape: &[usize]) -> Result<&Array, String> {
        let array = self
            .tensors
            .get(name)
            .ok_or_else(|| format!("missing weight {name}"))?;
        if array.shape != shape {
            return Err(format!(
                "weight {name}: shape {:?}, expected {shape:?}",
                array.shape
            ));
        }
        Ok(array)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn file(header: &str, data: &[u8]) -> Vec<u8> {
        let mut out = (header.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(data);
        out
    }

    #[test]
    fn f16_tensors_are_read_as_f32() {
        let one = f16::from_f32(1.0).to_le_bytes();
        let half = f16::from_f32(-0.5).to_le_bytes();
        let bytes = file(
            r#"{"__metadata__":{"a":"b"},"w":{"dtype":"F16","shape":[2],"data_offsets":[0,4]}}"#,
            &[one[0], one[1], half[0], half[1]],
        );
        let st = SafeTensors::parse(&bytes).expect("lisible");
        assert_eq!(st.get("w", &[2]).expect("présent").data, vec![1.0, -0.5]);
        assert!(st.get("w", &[1, 2]).is_err());
        assert!(st.get("absent", &[2]).is_err());
    }

    #[test]
    fn a_truncated_file_is_refused() {
        let bytes = file(
            r#"{"w":{"dtype":"F16","shape":[4],"data_offsets":[0,8]}}"#,
            &[0, 0],
        );
        assert!(SafeTensors::parse(&bytes).is_err());
        assert!(SafeTensors::parse(&[1, 2, 3]).is_err());
    }
}
