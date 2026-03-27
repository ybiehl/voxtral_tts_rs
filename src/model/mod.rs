//! Model components for Voxtral TTS.
//!
//! The Voxtral TTS model consists of three main components:
//! - **Backbone**: 26-layer Mistral transformer decoder (3.4B params)
//! - **Flow-Matching**: 3-layer acoustic transformer (390M params)
//! - **Codec**: Neural audio codec encoder/decoder (300M params)

pub mod backbone;
pub mod codec;
pub mod flow_matching;
pub mod kv_cache;
pub mod layers;
pub mod sampling;
pub mod weights;
