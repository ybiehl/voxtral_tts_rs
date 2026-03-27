// Copyright 2026 Claude Code on behalf of Michael Yuan.
// SPDX-License-Identifier: Apache-2.0

//! Flow-matching acoustic transformer (390M parameters).
//!
//! For each audio frame, this component:
//! 1. Predicts a semantic code from the backbone hidden state
//! 2. Generates 36 acoustic codes via Euler ODE flow matching
//!
//! The transformer operates on a 3-token sequence: [acoustic, time, llm_hidden]
//! using bidirectional attention (no causal mask).

use std::collections::HashMap;

use crate::config::AcousticTransformerConfig;
use crate::error::{Result, VoxtralError};
use crate::tensor::{DType, Device, Tensor};

use super::layers::{Linear, RMSNorm, RotaryEmbedding, TransformerLayer};

/// Number of Euler ODE integration steps.
const NUM_EULER_STEPS: usize = 8;

/// Classifier-free guidance alpha.
const CFG_ALPHA: f64 = 1.2;

/// Noise scale for initial acoustic latent.
const NOISE_SCALE: f64 = 1.0;

// ---------------------------------------------------------------------------
// FlowMatchingTransformer
// ---------------------------------------------------------------------------

/// The 390M parameter flow-matching acoustic transformer.
///
/// For each semantic token position, predicts:
/// 1. A semantic code (argmax of logits over the semantic codebook)
/// 2. 36 acoustic codes (via Euler ODE integration with CFG)
pub struct FlowMatchingTransformer {
    /// 3 bidirectional transformer layers.
    layers: Vec<TransformerLayer>,
    /// Project 36-dim acoustic noise → model dim.
    input_projection: Linear,
    /// Project backbone hidden states → model dim.
    llm_projection: Linear,
    /// Project sinusoidal time embedding → model dim.
    time_projection: Linear,
    /// Output: model dim → 36 acoustic dimensions.
    acoustic_codebook_output: Linear,
    /// Output: model dim → semantic codebook logits.
    semantic_codebook_output: Linear,
    /// Final RMSNorm.
    norm: RMSNorm,
    /// Rotary embedding for the 3-token bidirectional attention.
    rotary_emb: RotaryEmbedding,
    /// Configuration.
    config: AcousticTransformerConfig,
}

impl FlowMatchingTransformer {
    /// Load from a weight map.
    ///
    /// All keys are prefixed with `acoustic_transformer.`.
    pub fn from_weights(
        weights: &HashMap<String, Tensor>,
        config: AcousticTransformerConfig,
        device: Device,
    ) -> Self {
        let prefix = "acoustic_transformer";

        let input_projection = Linear::from_weights(
            weights[&format!("{}.input_projection.weight", prefix)]
                .clone()
                .to_device(device),
        );

        let llm_projection = Linear::from_weights(
            weights[&format!("{}.llm_projection.weight", prefix)]
                .clone()
                .to_device(device),
        );

        let time_projection = Linear::from_weights(
            weights[&format!("{}.time_projection.weight", prefix)]
                .clone()
                .to_device(device),
        );

        let acoustic_codebook_output = Linear::from_weights(
            weights[&format!("{}.acoustic_codebook_output.weight", prefix)]
                .clone()
                .to_device(device),
        );

        let semantic_codebook_output = Linear::from_weights(
            weights[&format!("{}.semantic_codebook_output.weight", prefix)]
                .clone()
                .to_device(device),
        );

        let norm = RMSNorm::from_weights(
            weights[&format!("{}.norm.weight", prefix)]
                .clone()
                .to_device(device),
            1e-5,
        );

        let mut layers = Vec::with_capacity(config.n_layers);
        for i in 0..config.n_layers {
            let layer_prefix = format!("{}.layers.{}", prefix, i);
            let layer = TransformerLayer::from_weights(
                weights,
                &layer_prefix,
                config.n_heads,
                config.n_kv_heads,
                config.head_dim,
                1e-5,
            );
            layers.push(layer);
        }

        let rotary_emb = RotaryEmbedding::new(
            config.head_dim,
            64, // only need 3 positions max
            config.rope_theta,
            device,
        );

        Self {
            layers,
            input_projection,
            llm_projection,
            time_projection,
            acoustic_codebook_output,
            semantic_codebook_output,
            norm,
            rotary_emb,
            config,
        }
    }

    /// Generate one audio frame (37 codes) from a backbone hidden state.
    ///
    /// * `llm_hidden` – backbone hidden state at this frame position, shape `[dim]`.
    /// * `device` – target device.
    ///
    /// Returns `(semantic_code, acoustic_codes)` where:
    /// - `semantic_code` is in range [0, semantic_codebook_size)
    /// - `acoustic_codes` is a Vec of 36 values, each in [2, 22] (with AudioSpecialTokens offset)
    ///
    /// Returns `None` if end-of-audio is predicted.
    pub fn generate_frame(
        &self,
        llm_hidden: &Tensor,
        device: Device,
    ) -> Option<Vec<i64>> {
        llm_hidden.eval();

        // 1. Predict semantic code from LLM hidden state
        let llm_input = llm_hidden.unsqueeze(0); // [1, dim]
        let semantic_logits = self.semantic_codebook_output.forward(&llm_input);
        semantic_logits.eval();
        let semantic_logits = semantic_logits.squeeze_dim(0); // [codebook_size]

        // Mask: set end-of-audio and padding to -inf
        // Valid semantic codes are in [2, 2 + 8192) = [2, 8194)
        let mut logits_vec = semantic_logits.to_vec_f32();
        // Mask special tokens (0, 1) and padding (>= 8194)
        if logits_vec.len() > 0 {
            logits_vec[0] = f32::NEG_INFINITY; // end-of-audio
        }
        if logits_vec.len() > 1 {
            logits_vec[1] = f32::NEG_INFINITY; // padding
        }
        for i in (crate::NUM_AUDIO_SPECIAL_TOKENS + crate::SEMANTIC_CODEBOOK_SIZE)..logits_vec.len()
        {
            logits_vec[i] = f32::NEG_INFINITY; // padding beyond valid range
        }

        let semantic_logits = Tensor::from_slice_f32(&logits_vec).to_device(device);
        let semantic_code = semantic_logits.argmax(0, false).int64_value(&[]);

        // Check for end-of-audio
        if semantic_code == crate::END_AUDIO_TOKEN_ID {
            return None;
        }

        // 2. Generate 36 acoustic codes via flow matching ODE
        let acoustic_codes = self.decode_acoustic(llm_hidden, device);

        // 3. Combine: [semantic_code, acoustic_code_0, ..., acoustic_code_35]
        let mut codes = Vec::with_capacity(crate::TOKENS_PER_FRAME);
        codes.push(semantic_code);
        codes.extend_from_slice(&acoustic_codes);

        Some(codes)
    }

    /// Euler ODE integration with classifier-free guidance to produce 36 acoustic codes.
    fn decode_acoustic(&self, llm_hidden: &Tensor, device: Device) -> Vec<i64> {
        // Initialize from noise: x_0 ~ N(0, noise_scale^2)
        let mut x = Tensor::random_normal(&[1, 36], DType::Float32, device)
            * NOISE_SCALE;

        // Timesteps: linspace(0, 1, NUM_EULER_STEPS)
        let llm_zero = Tensor::zeros(&[self.config.dim as i64], DType::Float32, device);

        for i in 0..(NUM_EULER_STEPS - 1) {
            let t = i as f64 / (NUM_EULER_STEPS - 1) as f64;
            let dt = 1.0 / (NUM_EULER_STEPS - 1) as f64;

            let t_emb = sinusoidal_time_embedding(t, self.config.dim, device);
            let t_emb = self.time_projection.forward(&t_emb); // [1, dim]

            // Classifier-free guidance: run both conditional and unconditional
            let v_cond = self.predict_velocity(&x, llm_hidden, &t_emb, device);
            let v_uncond = self.predict_velocity(&x, &llm_zero, &t_emb, device);

            // CFG blend: v = alpha * v_cond + (1 - alpha) * v_uncond
            let v = &(&v_cond * CFG_ALPHA) + &(&v_uncond * (1.0 - CFG_ALPHA));

            // Euler step: x_{t+dt} = x_t + dt * v
            x = &x + &(&v * dt);
            x.eval(); // Prevent MLX lazy graph growth across Euler steps
        }

        // Quantize to FSQ codes
        let x = x.squeeze_dim(0).clamp(-1.0, 1.0); // [36]
        let levels = crate::FSQ_LEVELS as f64;
        let scaled = &(&x + 1.0) * (0.5 * (levels - 1.0)); // map [-1,1] to [0, 20]
        let codes_f32 = (scaled + 0.5).to_vec_f32(); // round

        // Convert to integers with AudioSpecialTokens offset (+2)
        codes_f32
            .iter()
            .map(|&c| {
                let c = (c as i64).clamp(0, crate::FSQ_LEVELS as i64 - 1);
                c + crate::NUM_AUDIO_SPECIAL_TOKENS as i64
            })
            .collect()
    }

    /// Predict velocity field v(x_t, t, condition).
    ///
    /// Operates on a 3-token sequence: [acoustic_proj, time_emb, llm_proj].
    fn predict_velocity(
        &self,
        x_t: &Tensor,
        llm_hidden: &Tensor,
        t_emb: &Tensor,
        _device: Device,
    ) -> Tensor {
        // Project each input to model dim
        let x_proj = self.input_projection.forward(x_t); // [1, dim]
        let llm_input = llm_hidden.unsqueeze(0); // [1, dim]
        let llm_proj = self.llm_projection.forward(&llm_input); // [1, dim]

        // Build 3-token sequence: [1, 3, dim]
        let h = Tensor::cat(
            &[
                x_proj.unsqueeze(1),   // [1, 1, dim]
                t_emb.unsqueeze(1),    // [1, 1, dim] (already [1, dim], add seq dim)
                llm_proj.unsqueeze(1), // [1, 1, dim]
            ],
            1,
        );

        // Forward through bidirectional transformer layers (no causal mask)
        let mut h = h;
        for layer in &self.layers {
            let (out, _k, _v) =
                layer.forward(&h, &self.rotary_emb, 0, None, false);
            out.eval();
            h = out;
        }

        // Norm and take output from position 0 (the acoustic token)
        let h = self.norm.forward(&h);
        let acoustic_hidden = h.select(1, 0); // [1, dim]
        self.acoustic_codebook_output.forward(&acoustic_hidden) // [1, 36]
    }

    /// Access the model configuration.
    pub fn config(&self) -> &AcousticTransformerConfig {
        &self.config
    }
}

/// Sinusoidal time embedding for a scalar `t`.
///
/// Returns a tensor of shape `[1, dim]`.
fn sinusoidal_time_embedding(t: f64, dim: usize, device: Device) -> Tensor {
    let half_dim = dim / 2;
    let mut emb = vec![0.0f32; dim];

    let log_10000 = (10000.0f64).ln();
    for i in 0..half_dim {
        let freq = (-log_10000 * i as f64 / half_dim as f64).exp();
        let angle = t * freq;
        emb[i] = angle.sin() as f32;
        emb[i + half_dim] = angle.cos() as f32;
    }

    Tensor::from_slice_f32(&emb)
        .reshape(&[1, dim as i64])
        .to_device(device)
}
