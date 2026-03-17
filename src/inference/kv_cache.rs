/// Paged KV cache for efficient memory management.
///
/// Instead of pre-allocating a contiguous KV cache for max_seq_len,
/// this allocates fixed-size pages on demand. Benefits:
/// - Memory usage proportional to actual sequence length
/// - Multiple sequences can share the page pool
/// - No wasted memory for short generations
///
/// Inspired by vLLM's PagedAttention.

use alloc::vec;
use alloc::vec::Vec;

/// Number of token positions per KV cache page.
const TOKENS_PER_PAGE: usize = 256;

/// A single page of KV cache for one layer.
/// Stores key and value vectors for TOKENS_PER_PAGE positions.
pub struct KvPage {
    /// Key vectors: [TOKENS_PER_PAGE, kv_dim]
    pub keys: Vec<f32>,
    /// Value vectors: [TOKENS_PER_PAGE, kv_dim]
    pub values: Vec<f32>,
    /// How many positions are filled in this page.
    pub used: usize,
}

impl KvPage {
    fn new(kv_dim: usize) -> Self {
        Self {
            keys: vec![0.0; TOKENS_PER_PAGE * kv_dim],
            values: vec![0.0; TOKENS_PER_PAGE * kv_dim],
            used: 0,
        }
    }

    fn is_full(&self) -> bool {
        self.used >= TOKENS_PER_PAGE
    }
}

/// Page table for one layer's KV cache.
/// Maps sequence positions to pages.
struct LayerPageTable {
    /// Ordered list of pages for this layer.
    pages: Vec<KvPage>,
    /// KV dimension (head_dim * n_kv_heads).
    kv_dim: usize,
}

impl LayerPageTable {
    fn new(kv_dim: usize) -> Self {
        Self {
            pages: Vec::new(),
            kv_dim,
        }
    }

    /// Ensure we have a page for the given position, allocate if needed.
    fn ensure_page(&mut self, pos: usize) -> &mut KvPage {
        let page_idx = pos / TOKENS_PER_PAGE;
        while self.pages.len() <= page_idx {
            self.pages.push(KvPage::new(self.kv_dim));
        }
        &mut self.pages[page_idx]
    }

    /// Write key/value at a position.
    fn write(&mut self, pos: usize, key: &[f32], value: &[f32]) {
        let slot = pos % TOKENS_PER_PAGE;
        let kv_dim = self.kv_dim;

        let page = self.ensure_page(pos);
        let offset = slot * kv_dim;
        page.keys[offset..offset + kv_dim].copy_from_slice(&key[..kv_dim]);
        page.values[offset..offset + kv_dim].copy_from_slice(&value[..kv_dim]);
        if slot >= page.used {
            page.used = slot + 1;
        }
    }

    /// Read key at a position.
    fn read_key(&self, pos: usize) -> Option<&[f32]> {
        let page_idx = pos / TOKENS_PER_PAGE;
        let slot = pos % TOKENS_PER_PAGE;
        let kv_dim = self.kv_dim;

        self.pages.get(page_idx).map(|page| {
            let offset = slot * kv_dim;
            &page.keys[offset..offset + kv_dim]
        })
    }

    /// Read value at a position.
    fn read_value(&self, pos: usize) -> Option<&[f32]> {
        let page_idx = pos / TOKENS_PER_PAGE;
        let slot = pos % TOKENS_PER_PAGE;
        let kv_dim = self.kv_dim;

        self.pages.get(page_idx).map(|page| {
            let offset = slot * kv_dim;
            &page.values[offset..offset + kv_dim]
        })
    }

    /// Total memory used by this layer's pages.
    fn memory_bytes(&self) -> usize {
        self.pages.len() * TOKENS_PER_PAGE * self.kv_dim * 4 * 2 // keys + values
    }
}

/// Paged KV cache for all layers.
pub struct PagedKvCache {
    layers: Vec<LayerPageTable>,
    n_layers: usize,
    kv_dim: usize,
}

impl PagedKvCache {
    /// Create a new paged KV cache (no memory allocated until first write).
    pub fn new(n_layers: usize, kv_dim: usize) -> Self {
        let mut layers = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            layers.push(LayerPageTable::new(kv_dim));
        }
        Self { layers, n_layers, kv_dim }
    }

    /// Write key/value for a specific layer and position.
    pub fn write(&mut self, layer: usize, pos: usize, key: &[f32], value: &[f32]) {
        if layer < self.n_layers {
            self.layers[layer].write(pos, key, value);
        }
    }

    /// Read key at a specific layer and position.
    pub fn read_key(&self, layer: usize, pos: usize) -> Option<&[f32]> {
        self.layers.get(layer)?.read_key(pos)
    }

    /// Read value at a specific layer and position.
    pub fn read_value(&self, layer: usize, pos: usize) -> Option<&[f32]> {
        self.layers.get(layer)?.read_value(pos)
    }

    /// Total memory used across all layers.
    pub fn memory_bytes(&self) -> usize {
        self.layers.iter().map(|l| l.memory_bytes()).sum()
    }

    /// Total pages allocated.
    pub fn total_pages(&self) -> usize {
        self.layers.iter().map(|l| l.pages.len()).sum()
    }

    /// Reset cache (free all pages).
    pub fn clear(&mut self) {
        for layer in &mut self.layers {
            layer.pages.clear();
        }
    }

    pub fn tokens_per_page() -> usize { TOKENS_PER_PAGE }
}
