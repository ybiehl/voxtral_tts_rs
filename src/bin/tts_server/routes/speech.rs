//! Speech synthesis endpoint (OpenAI-compatible).
//!
//! `POST /v1/audio/speech` - Generate speech from text.

use axum::extract::State;
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response, Sse};
use axum::Json;
use serde::{Deserialize, Serialize};
use tokio_stream::wrappers::ReceiverStream;

use crate::state::AppState;

/// Maximum allowed input text length (in characters).
const MAX_INPUT_LENGTH: usize = 4096;

/// Minimum allowed speed multiplier.
const MIN_SPEED: f32 = 0.25;

/// Maximum allowed speed multiplier.
const MAX_SPEED: f32 = 4.0;

// ---------------------------------------------------------------------------
// Request / Response types
// ---------------------------------------------------------------------------

/// OpenAI-compatible speech request body.
#[derive(Debug, Deserialize)]
pub struct SpeechRequest {
    /// Model name (currently only "voxtral-4b-tts" is accepted).
    #[serde(default = "default_model")]
    pub model: String,

    /// Text to synthesize.
    pub input: String,

    /// Voice name (preset name or OpenAI alias).
    #[serde(default = "default_voice")]
    pub voice: String,

    /// Response audio format. Currently only "wav" is supported.
    #[serde(default = "default_format")]
    pub response_format: String,

    /// Speed multiplier (0.25 - 4.0). Reserved for future use.
    #[serde(default = "default_speed")]
    pub speed: f32,

    /// Whether to stream the response via SSE.
    #[serde(default)]
    pub stream: bool,
}

fn default_model() -> String {
    "voxtral-4b-tts".to_string()
}

fn default_voice() -> String {
    "alloy".to_string()
}

fn default_format() -> String {
    "wav".to_string()
}

fn default_speed() -> f32 {
    1.0
}

/// Error response body.
#[derive(Serialize)]
struct ErrorBody {
    error: ErrorDetail,
}

#[derive(Serialize)]
struct ErrorDetail {
    message: String,
    r#type: &'static str,
    code: &'static str,
}

/// SSE streaming chunk.
#[derive(Serialize)]
struct StreamChunk {
    /// Base64-encoded PCM audio bytes.
    audio: String,
    /// Whether this is the final chunk.
    done: bool,
}

// ---------------------------------------------------------------------------
// Handler
// ---------------------------------------------------------------------------

/// `POST /v1/audio/speech` - Generate speech from text.
pub async fn create_speech(
    State(state): State<AppState>,
    Json(req): Json<SpeechRequest>,
) -> Response {
    // ---- Input validation ----

    if req.input.is_empty() {
        return error_response(
            StatusCode::BAD_REQUEST,
            "Input text must not be empty.",
            "invalid_request_error",
            "invalid_input",
        );
    }

    if req.input.len() > MAX_INPUT_LENGTH {
        return error_response(
            StatusCode::BAD_REQUEST,
            &format!(
                "Input text too long ({} chars). Maximum is {} characters.",
                req.input.len(),
                MAX_INPUT_LENGTH,
            ),
            "invalid_request_error",
            "input_too_long",
        );
    }

    if req.speed < MIN_SPEED || req.speed > MAX_SPEED {
        return error_response(
            StatusCode::BAD_REQUEST,
            &format!(
                "Speed must be between {} and {}. Got {}.",
                MIN_SPEED, MAX_SPEED, req.speed
            ),
            "invalid_request_error",
            "invalid_speed",
        );
    }

    if req.response_format != "wav" && req.response_format != "pcm" {
        return error_response(
            StatusCode::BAD_REQUEST,
            &format!(
                "Unsupported response_format '{}'. Supported: wav, pcm.",
                req.response_format,
            ),
            "invalid_request_error",
            "invalid_format",
        );
    }

    // ---- Dispatch ----

    if req.stream {
        handle_streaming(state, req).await
    } else {
        handle_non_streaming(state, req).await
    }
}

// ---------------------------------------------------------------------------
// Non-streaming response
// ---------------------------------------------------------------------------

async fn handle_non_streaming(state: AppState, req: SpeechRequest) -> Response {
    let voice = req.voice.clone();
    let input = req.input.clone();
    let format = req.response_format.clone();

    // Run inference on a blocking thread to avoid starving the Tokio runtime.
    let result = tokio::task::spawn_blocking(move || {
        let tts = state.lock().map_err(|e| {
            VoxtralError::Inference(format!("Failed to acquire model lock: {}", e))
        })?;
        tts.generate(&input, &voice, 0.7, 4096)
    })
    .await;

    match result {
        Ok(Ok((samples, sample_rate))) => {
            let content_type = if format == "pcm" {
                "audio/pcm"
            } else {
                "audio/wav"
            };

            let body = if format == "pcm" {
                voxtral_tts::audio::encode_pcm_i16(&samples)
            } else {
                match voxtral_tts::audio::write_wav_bytes(&samples, sample_rate) {
                    Ok(bytes) => bytes,
                    Err(e) => {
                        return error_response(
                            StatusCode::INTERNAL_SERVER_ERROR,
                            &format!("Failed to encode WAV: {}", e),
                            "server_error",
                            "encoding_error",
                        );
                    }
                }
            };

            (
                StatusCode::OK,
                [(header::CONTENT_TYPE, content_type)],
                body,
            )
                .into_response()
        }
        Ok(Err(e)) => error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("Inference failed: {}", e),
            "server_error",
            "inference_error",
        ),
        Err(e) => error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("Task join error: {}", e),
            "server_error",
            "internal_error",
        ),
    }
}

// ---------------------------------------------------------------------------
// Streaming response (SSE with base64-encoded PCM chunks)
// ---------------------------------------------------------------------------

async fn handle_streaming(state: AppState, req: SpeechRequest) -> Response {
    let voice = req.voice.clone();
    let input = req.input.clone();

    let (tx, rx) = tokio::sync::mpsc::channel::<Result<axum::response::sse::Event, std::convert::Infallible>>(32);

    // Spawn inference on a blocking thread.
    tokio::task::spawn_blocking(move || {
        let tts = match state.lock() {
            Ok(guard) => guard,
            Err(e) => {
                tracing::error!("Failed to acquire model lock: {}", e);
                return;
            }
        };

        match tts.generate(&input, &voice, 0.7, 4096) {
            Ok((samples, _sample_rate)) => {
                // Chunk the PCM output into ~0.5s segments for streaming.
                let chunk_size = 12000; // 0.5s at 24kHz
                let chunks: Vec<&[f32]> = samples.chunks(chunk_size).collect();
                let total = chunks.len();

                for (i, chunk) in chunks.into_iter().enumerate() {
                    let pcm_bytes = voxtral_tts::audio::encode_pcm_i16(chunk);
                    let b64 = base64::engine::general_purpose::STANDARD
                        .encode(&pcm_bytes);
                    let is_last = i + 1 == total;
                    let payload = serde_json::json!({
                        "audio": b64,
                        "done": is_last,
                    });
                    let event = axum::response::sse::Event::default()
                        .data(payload.to_string());
                    if tx.blocking_send(Ok(event)).is_err() {
                        break; // Client disconnected.
                    }
                }
            }
            Err(e) => {
                tracing::error!("Streaming inference failed: {}", e);
                let payload = serde_json::json!({
                    "error": e.to_string(),
                    "done": true,
                });
                let event = axum::response::sse::Event::default()
                    .data(payload.to_string());
                let _ = tx.blocking_send(Ok(event));
            }
        }
    });

    let stream = ReceiverStream::new(rx);
    Sse::new(stream).into_response()
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

use base64::Engine;
use voxtral_tts::error::VoxtralError;

fn error_response(
    status: StatusCode,
    message: &str,
    error_type: &'static str,
    code: &'static str,
) -> Response {
    let body = ErrorBody {
        error: ErrorDetail {
            message: message.to_string(),
            r#type: error_type,
            code,
        },
    };
    (status, Json(body)).into_response()
}
