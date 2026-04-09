//! Voice embedding loading and management.

use std::collections::HashMap;
use std::path::Path;

use crate::error::{Result, VoxtralError};
use crate::tensor::{DType, Device, Tensor};

/// Preset voice names with OpenAI-compatible aliases.
pub const PRESET_VOICES: &[&str] = &[
    "casual_female",
    "casual_male",
    "cheerful_female",
    "neutral_female",
    "neutral_male",
    "pt_male",
    "pt_female",
    "nl_male",
    "nl_female",
    "it_male",
    "it_female",
    "fr_male",
    "fr_female",
    "es_male",
    "es_female",
    "de_male",
    "de_female",
    "ar_male",
    "hi_male",
    "hi_female",
];

/// Map OpenAI TTS API voice names to Voxtral preset names.
pub fn resolve_voice_alias(name: &str) -> String {
    match name.to_lowercase().as_str() {
        "alloy" => "neutral_female".to_string(),
        "echo" => "casual_male".to_string(),
        "fable" => "cheerful_female".to_string(),
        "onyx" => "neutral_male".to_string(),
        "nova" => "casual_female".to_string(),
        "shimmer" => "fr_female".to_string(),
        _ => name.to_string(),
    }
}

/// Voice embedding store.
pub struct VoiceStore {
    embeddings: HashMap<String, Tensor>,
    device: Device,
}

impl VoiceStore {
    /// Load all voice embeddings from a model directory.
    ///
    /// Loads preset voices from `<model_dir>/voice_embedding/<name>.safetensors` (or `.pt`).
    /// Also scans the directory for any additional `.safetensors` files (custom voices).
    pub fn from_dir(model_dir: &Path, device: Device) -> Result<Self> {
        let voice_dir = model_dir.join("voice_embedding");
        let mut embeddings = HashMap::new();

        if !voice_dir.exists() {
            tracing::warn!(
                "Voice embedding directory not found: {}",
                voice_dir.display()
            );
            return Ok(Self { embeddings, device });
        }

        // Keep BF16 on all backends — libtorch 2.7+ supports BF16 on CPU
        let need_f32 = false;

        // Load preset voices by name.
        for name in PRESET_VOICES {
            let pt_path = voice_dir.join(format!("{}.pt", name));
            let safetensors_path = voice_dir.join(format!("{}.safetensors", name));

            if safetensors_path.exists() {
                let tensors = Tensor::load_safetensors(&safetensors_path)?;
                if let Some((_, tensor)) = tensors.into_iter().next() {
                    let tensor = if need_f32 {
                        tensor.to_dtype(DType::Float32).to_device(device)
                    } else {
                        tensor.to_device(device)
                    };
                    embeddings.insert(name.to_string(), tensor);
                    tracing::debug!("Loaded voice embedding: {} (safetensors)", name);
                }
            } else if pt_path.exists() {
                match Tensor::load_pt(&pt_path) {
                    Ok(tensor) => {
                        let tensor = if need_f32 {
                            tensor.to_dtype(DType::Float32).to_device(device)
                        } else {
                            tensor.to_device(device)
                        };
                        embeddings.insert(name.to_string(), tensor);
                        tracing::debug!("Loaded voice embedding: {} (.pt)", name);
                    }
                    Err(e) => {
                        tracing::warn!("Failed to load {}: {}", pt_path.display(), e);
                    }
                }
            } else {
                tracing::debug!("Voice embedding not found: {}", name);
            }
        }

        // Scan for any extra .safetensors files (custom / user-created voices).
        if let Ok(entries) = std::fs::read_dir(&voice_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().and_then(|e| e.to_str()) != Some("safetensors") {
                    continue;
                }
                let stem = match path.file_stem().and_then(|s| s.to_str()) {
                    Some(s) => s.to_string(),
                    None => continue,
                };
                // Skip already-loaded presets.
                if embeddings.contains_key(&stem) {
                    continue;
                }
                match Tensor::load_safetensors(&path) {
                    Ok(tensors) => {
                        if let Some((_, tensor)) = tensors.into_iter().next() {
                            let tensor = tensor.to_device(device);
                            tracing::info!("Loaded custom voice: {}", stem);
                            embeddings.insert(stem, tensor);
                        }
                    }
                    Err(e) => {
                        tracing::warn!(
                            "Skipping {}: could not load safetensors: {}",
                            path.display(),
                            e
                        );
                    }
                }
            }
        }

        tracing::info!("Loaded {} voice embeddings", embeddings.len());
        Ok(Self { embeddings, device })
    }

    /// Load a voice embedding from a `.safetensors` file at runtime.
    ///
    /// The file must contain a single tensor with key `"embedding"` and
    /// shape `[N, 3072]`. Overwrites any existing voice with the same name.
    pub fn load_from_file(&mut self, name: &str, path: &Path) -> Result<()> {
        let tensors = Tensor::load_safetensors(path)
            .map_err(|e| VoxtralError::Inference(format!("Failed to load {}: {}", path.display(), e)))?;

        let tensor = tensors
            .into_iter()
            .next()
            .ok_or_else(|| {
                VoxtralError::Inference(format!(
                    "No tensors found in {}",
                    path.display()
                ))
            })?
            .1
            .to_device(self.device);

        let shape = tensor.size();
        if shape.len() != 2 || shape[1] != 3072 {
            return Err(VoxtralError::Inference(format!(
                "Expected shape [N, 3072], got {:?} in {}",
                shape,
                path.display()
            )));
        }

        tracing::info!(
            "Loaded custom voice '{}' from {} ({} frames)",
            name,
            path.display(),
            shape[0]
        );
        self.embeddings.insert(name.to_string(), tensor);
        Ok(())
    }

    /// Remove a custom voice by name. Preset voices cannot be removed.
    ///
    /// Returns `true` if the voice was found and removed.
    pub fn remove_voice(&mut self, name: &str) -> bool {
        if PRESET_VOICES.contains(&name) {
            tracing::warn!("Refusing to remove preset voice '{}'", name);
            return false;
        }
        let removed = self.embeddings.remove(name).is_some();
        if removed {
            tracing::info!("Removed custom voice '{}'", name);
        }
        removed
    }

    /// Get a voice embedding by name (resolves aliases).
    pub fn get(&self, name: &str) -> Result<&Tensor> {
        let resolved = resolve_voice_alias(name);
        self.embeddings.get(&resolved).ok_or_else(|| {
            VoxtralError::VoiceNotFound(format!(
                "Voice '{}' (resolved to '{}') not found. Available: {:?}",
                name,
                resolved,
                self.list_voices(),
            ))
        })
    }

    /// List all available voice names (presets + custom).
    pub fn list_voices(&self) -> Vec<&str> {
        self.embeddings.keys().map(|s| s.as_str()).collect()
    }

    /// Check if a voice name is available.
    pub fn has_voice(&self, name: &str) -> bool {
        let resolved = resolve_voice_alias(name);
        self.embeddings.contains_key(&resolved)
    }
}
