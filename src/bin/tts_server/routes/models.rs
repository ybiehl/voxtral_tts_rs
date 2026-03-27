//! Model listing endpoint (OpenAI-compatible).

use axum::Json;
use serde::Serialize;

#[derive(Serialize)]
pub struct ModelObject {
    id: &'static str,
    object: &'static str,
    created: u64,
    owned_by: &'static str,
}

#[derive(Serialize)]
pub struct ModelsResponse {
    object: &'static str,
    data: Vec<ModelObject>,
}

/// `GET /v1/models` - List available models (OpenAI-compatible format).
pub async fn list_models() -> Json<ModelsResponse> {
    Json(ModelsResponse {
        object: "list",
        data: vec![ModelObject {
            id: "voxtral-4b-tts",
            object: "model",
            created: 1700000000,
            owned_by: "mistral",
        }],
    })
}
