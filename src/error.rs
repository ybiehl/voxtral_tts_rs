//! Error types for Voxtral TTS.

/// Result type alias using VoxtralError.
pub type Result<T> = std::result::Result<T, VoxtralError>;

/// Errors that can occur during Voxtral TTS operations.
#[derive(Debug, thiserror::Error)]
pub enum VoxtralError {
    /// Failed to load model weights.
    #[error("Model load error: {0}")]
    ModelLoad(String),

    /// Failed to load or parse configuration.
    #[error("Config error: {0}")]
    Config(String),

    /// Failed to load or use the tokenizer.
    #[error("Tokenizer error: {0}")]
    Tokenizer(String),

    /// Failed to process audio.
    #[error("Audio error: {0}")]
    Audio(String),

    /// Failed during inference.
    #[error("Inference error: {0}")]
    Inference(String),

    /// Voice not found.
    #[error("Voice not found: {0}")]
    VoiceNotFound(String),

    /// I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON parsing error.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// Safetensors error.
    #[error("Safetensors error: {0}")]
    Safetensors(String),

    /// Backend-specific error.
    #[error("Backend error: {0}")]
    Backend(String),
}

#[cfg(feature = "tch-backend")]
impl From<tch::TchError> for VoxtralError {
    fn from(e: tch::TchError) -> Self {
        VoxtralError::Backend(e.to_string())
    }
}
