//! Computation backends for Voxtral TTS.
//!
//! Currently only the MLX backend is defined here. The tch (libtorch) backend
//! operates through the `tensor` module's tch-specific implementations.

#[cfg(feature = "mlx")]
pub mod mlx;
