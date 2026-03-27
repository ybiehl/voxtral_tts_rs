// Copyright 2026 Claude Code on behalf of Michael Yuan.
// SPDX-License-Identifier: Apache-2.0

//! Key-Value cache for autoregressive transformer generation.
//!
//! During autoregressive decoding, the backbone transformer processes one
//! token at a time. To avoid re-computing attention over previous tokens,
//! we cache the Key and Value projections from each layer.

use crate::tensor::Tensor;

/// Per-layer KV cache for autoregressive generation.
///
/// Each layer stores its own key and value tensors. On the first forward
/// pass the tensors are created; on subsequent passes new KV pairs are
/// concatenated along the sequence dimension.
pub struct KVCache {
    /// Key caches, one per layer. `None` until the layer has been called.
    k_cache: Vec<Option<Tensor>>,
    /// Value caches, one per layer.
    v_cache: Vec<Option<Tensor>>,
}

impl KVCache {
    /// Create a new empty KV cache for `n_layers` transformer layers.
    pub fn new(n_layers: usize) -> Self {
        Self {
            k_cache: (0..n_layers).map(|_| None).collect(),
            v_cache: (0..n_layers).map(|_| None).collect(),
        }
    }

    /// Replace the cache for a given layer with updated K and V tensors.
    ///
    /// * `layer_idx` – the transformer layer index (0-based).
    /// * `new_k` – full key tensor including previous cache, shape `[batch, n_kv_heads, total_seq_len, head_dim]`.
    /// * `new_v` – full value tensor, same shape as `new_k`.
    ///
    /// The attention layer already concatenates old cache + new KV internally,
    /// so this method simply replaces the stored tensors.
    pub fn update(&mut self, layer_idx: usize, new_k: Tensor, new_v: Tensor) {
        self.k_cache[layer_idx] = Some(new_k);
        self.v_cache[layer_idx] = Some(new_v);
    }

    /// Retrieve the cached K and V for a given layer.
    ///
    /// Returns `None` if the layer has not yet been called (i.e. during the
    /// first forward pass of prefill when using incremental updates).
    pub fn get(&self, layer_idx: usize) -> Option<(&Tensor, &Tensor)> {
        match (&self.k_cache[layer_idx], &self.v_cache[layer_idx]) {
            (Some(k), Some(v)) => Some((k, v)),
            _ => None,
        }
    }

    /// Return the current sequence length stored in the cache.
    ///
    /// Returns 0 if the cache is empty.
    pub fn seq_len(&self) -> usize {
        // All layers should have the same sequence length; read from layer 0.
        self.k_cache
            .first()
            .and_then(|opt| opt.as_ref())
            .map(|k| k.size()[2] as usize) // dim 2 is the seq dimension
            .unwrap_or(0)
    }

    /// Reset the cache to empty state.
    pub fn clear(&mut self) {
        for slot in &mut self.k_cache {
            *slot = None;
        }
        for slot in &mut self.v_cache {
            *slot = None;
        }
    }

    /// Number of layers this cache was created for.
    pub fn n_layers(&self) -> usize {
        self.k_cache.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{DType, Device};

    #[test]
    fn test_kv_cache_empty() {
        let cache = KVCache::new(4);
        assert_eq!(cache.n_layers(), 4);
        assert_eq!(cache.seq_len(), 0);
        assert!(cache.get(0).is_none());
    }

    #[test]
    fn test_kv_cache_update_and_get() {
        let mut cache = KVCache::new(2);

        // First update: insert initial KV (from prefill)
        let k = Tensor::ones(&[1, 8, 3, 128], DType::Float32, Device::Cpu);
        let v = Tensor::ones(&[1, 8, 3, 128], DType::Float32, Device::Cpu);
        cache.update(0, k, v);

        assert_eq!(cache.seq_len(), 3);
        let (ck, cv) = cache.get(0).unwrap();
        assert_eq!(ck.size(), vec![1, 8, 3, 128]);
        assert_eq!(cv.size(), vec![1, 8, 3, 128]);

        // Second update: replace with full concatenated KV (attention does the cat)
        let k2 = Tensor::ones(&[1, 8, 4, 128], DType::Float32, Device::Cpu);
        let v2 = Tensor::ones(&[1, 8, 4, 128], DType::Float32, Device::Cpu);
        cache.update(0, k2, v2);

        assert_eq!(cache.seq_len(), 4);
        let (ck, cv) = cache.get(0).unwrap();
        assert_eq!(ck.size(), vec![1, 8, 4, 128]);
        assert_eq!(cv.size(), vec![1, 8, 4, 128]);
    }
}
