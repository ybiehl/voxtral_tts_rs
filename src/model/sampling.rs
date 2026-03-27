// Copyright 2026 Claude Code on behalf of Michael Yuan.
// SPDX-License-Identifier: Apache-2.0

//! Token sampling utilities for autoregressive generation.
//!
//! Supports temperature scaling, top-k filtering, top-p (nucleus) filtering,
//! and multinomial sampling from the resulting distribution.

use crate::tensor::{DType, Tensor};

/// Sample a single token index from a logits vector.
///
/// # Arguments
///
/// * `logits` – Raw (unnormalized) logits, shape `[vocab_size]` or `[1, vocab_size]`.
/// * `temperature` – Softmax temperature. Values < 1.0 sharpen the distribution;
///   values > 1.0 flatten it. A value of 0.0 triggers greedy (argmax) decoding.
/// * `top_k` – If provided, only the top-k highest-probability tokens are kept.
/// * `top_p` – If provided, tokens are selected from the smallest set whose
///   cumulative probability exceeds `top_p` (nucleus sampling).
///
/// # Returns
///
/// The sampled token index as `i64`.
pub fn sample_token(
    logits: &Tensor,
    temperature: f32,
    top_k: Option<usize>,
    top_p: Option<f32>,
) -> i64 {
    // Ensure logits is 1-D: [vocab_size]
    let logits = if logits.dim() > 1 {
        logits.reshape(&[-1])
    } else {
        logits.clone()
    };

    // Greedy decoding for temperature == 0
    if temperature <= f32::EPSILON {
        return logits.argmax(0, false).int64_value(&[]);
    }

    // Temperature scaling
    let scaled = &logits / (temperature as f64);

    // Apply top-k filtering
    let filtered = match top_k {
        Some(k) if k > 0 && k < scaled.size()[0] as usize => top_k_filter(&scaled, k),
        _ => scaled,
    };

    // Apply top-p (nucleus) filtering
    let filtered = match top_p {
        Some(p) if p > 0.0 && p < 1.0 => top_p_filter(&filtered, p),
        _ => filtered,
    };

    // Convert to probabilities
    let probs = filtered.softmax(0);

    // Sample from the distribution
    // multinomial expects a 2D input [batch, vocab] for tch compatibility
    let probs_2d = probs.unsqueeze(0); // [1, vocab]
    let sampled = probs_2d.multinomial(1, false); // [1, 1]
    sampled.int64_value(&[0, 0])
}

/// Apply top-k filtering: keep only the `k` highest logits; set all others to -inf.
fn top_k_filter(logits: &Tensor, k: usize) -> Tensor {
    let (top_values, _top_indices) = logits.topk(k as i64, 0, true, true);
    // The k-th value (last in the sorted top-k) is our threshold
    let threshold = top_values.select(0, k as i64 - 1);
    let threshold_val = threshold.f64_value(&[]);

    // Create mask for values below threshold
    let threshold_tensor = Tensor::full(
        &logits.size(),
        threshold_val,
        logits.kind(),
        logits.device(),
    );
    let mask = logits.lt_tensor(&threshold_tensor);

    logits.masked_fill(&mask, f64::NEG_INFINITY)
}

/// Apply top-p (nucleus) filtering: keep the smallest set of tokens whose
/// cumulative probability exceeds `p`, setting all others to -inf.
fn top_p_filter(logits: &Tensor, p: f32) -> Tensor {
    let vocab_size = logits.size()[0];

    // Sort logits in descending order
    let (sorted_values, sorted_indices) = logits.topk(vocab_size, 0, true, true);

    // Compute cumulative probabilities of the sorted values
    let sorted_probs = sorted_values.softmax(0);
    let cumulative_probs = cumulative_sum(&sorted_probs);

    // Create mask: true where cumulative probability exceeds p
    // We want to zero out tokens whose cumulative prob (exclusive of the current token)
    // is already above p. So shift by one.
    let p_tensor = Tensor::full(&[vocab_size], p as f64, DType::Float32, logits.device());

    // shifted_cumprobs[i] = cumprobs[i-1] (0 for i=0)
    // Tokens where shifted_cumprobs > p should be filtered out
    let shifted = shift_right(&cumulative_probs);
    let remove_mask = shifted.lt_tensor(&p_tensor);
    // remove_mask is true where we KEEP the token; we want the inverse
    // Actually: keep tokens where shifted_cumprobs < p (i.e. not yet saturated)
    // So: remove where shifted_cumprobs >= p

    // Apply mask to sorted values
    let ones_mask = Tensor::ones(&[vocab_size], DType::Bool, logits.device());
    let keep_mask = remove_mask; // true = keep
    let remove_mask = &ones_mask - &keep_mask.to_dtype(DType::Bool); // invert

    let filtered_sorted = sorted_values.masked_fill(&remove_mask, f64::NEG_INFINITY);

    // Scatter back to original order
    scatter_1d(&filtered_sorted, &sorted_indices, vocab_size)
}

/// Compute cumulative sum along a 1-D tensor.
fn cumulative_sum(x: &Tensor) -> Tensor {
    let n = x.size()[0];
    let mut result = Vec::with_capacity(n as usize);
    let mut running = 0.0f64;
    for i in 0..n {
        running += x.f64_value(&[i]);
        result.push(running as f32);
    }
    Tensor::from_slice_f32(&result).to_device(x.device())
}

/// Shift a 1-D tensor right by one position, inserting 0.0 at the front.
fn shift_right(x: &Tensor) -> Tensor {
    let n = x.size()[0];
    let zero = Tensor::zeros(&[1], DType::Float32, x.device());
    let prefix = x.narrow(0, 0, n - 1);
    Tensor::cat(&[zero, prefix], 0)
}

/// Scatter sorted values back to their original positions.
///
/// Given `values` in sorted order and `indices` mapping sorted -> original,
/// produce a tensor in original order.
fn scatter_1d(values: &Tensor, indices: &Tensor, size: i64) -> Tensor {
    // Build the result by indexing: result[indices[i]] = values[i]
    // For simplicity, do this on CPU via extraction and reconstruction.
    let n = size as usize;
    let mut result = vec![f64::NEG_INFINITY; n];
    for i in 0..n {
        let idx = indices.int64_value(&[i as i64]) as usize;
        let val = values.f64_value(&[i as i64]);
        if idx < n {
            result[idx] = val;
        }
    }
    let result_f32: Vec<f32> = result.iter().map(|&v| v as f32).collect();
    Tensor::from_slice_f32(&result_f32).to_device(values.device())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{DType, Device};

    fn setup() {
        #[cfg(feature = "mlx")]
        crate::backend::mlx::stream::init_mlx(false);
    }

    #[test]
    fn test_greedy_sampling() {
        setup();
        // Create logits where index 3 has the highest value
        let data = vec![0.1f32, 0.2, 0.3, 10.0, 0.5];
        let logits = Tensor::from_slice_f32(&data);
        let token = sample_token(&logits, 0.0, None, None);
        assert_eq!(token, 3);
    }

    #[test]
    fn test_temperature_sampling() {
        setup();
        // With high temperature, distribution is more uniform
        // With very low temperature, should approximate greedy
        let data = vec![0.1f32, 0.2, 0.3, 10.0, 0.5];
        let logits = Tensor::from_slice_f32(&data);
        let token = sample_token(&logits, 0.001, None, None);
        // Very low temperature should nearly always pick the max
        assert_eq!(token, 3);
    }

    #[test]
    fn test_top_k_filter() {
        setup();
        let data = vec![1.0f32, 5.0, 3.0, 2.0, 4.0];
        let logits = Tensor::from_slice_f32(&data);
        let filtered = top_k_filter(&logits, 2);
        // Only indices 1 (5.0) and 4 (4.0) should remain; others should be -inf
        let v0 = filtered.f64_value(&[0]);
        let v1 = filtered.f64_value(&[1]);
        let v4 = filtered.f64_value(&[4]);
        assert!(v0.is_infinite() && v0.is_sign_negative());
        assert!((v1 - 5.0).abs() < 1e-4);
        assert!((v4 - 4.0).abs() < 1e-4);
    }
}
