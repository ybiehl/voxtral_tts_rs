//! Health check endpoint.

use axum::Json;
use serde::Serialize;

#[derive(Serialize)]
pub struct HealthResponse {
    status: &'static str,
}

/// `GET /health` - Returns a simple health check response.
pub async fn health_check() -> Json<HealthResponse> {
    Json(HealthResponse { status: "ok" })
}
