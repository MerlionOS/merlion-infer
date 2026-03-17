/// Token sampling strategies for text generation.

use crate::inference::kernels::scalar;

/// Sample the next token from logits.
pub struct Sampler {
    pub temperature: f32,
    pub top_p: f32,
}

impl Sampler {
    pub fn new(temperature: f32, top_p: f32) -> Self {
        Self { temperature, top_p }
    }

    /// Greedy (argmax) sampling.
    pub fn greedy() -> Self {
        Self { temperature: 0.0, top_p: 1.0 }
    }

    /// Sample a token from logits.
    /// Returns token ID.
    pub fn sample(&self, logits: &mut [f32], rng_state: &mut u64) -> u32 {
        if self.temperature <= 0.0 || self.temperature < 1e-6 {
            // Greedy: just pick argmax
            return scalar::argmax(logits) as u32;
        }

        // Apply temperature
        let inv_temp = 1.0 / self.temperature;
        for v in logits.iter_mut() {
            *v *= inv_temp;
        }

        // Softmax
        scalar::softmax(logits);

        if self.top_p < 1.0 {
            self.sample_top_p(logits, rng_state)
        } else {
            self.sample_categorical(logits, rng_state)
        }
    }

    /// Categorical sampling from probability distribution.
    fn sample_categorical(&self, probs: &[f32], rng_state: &mut u64) -> u32 {
        let r = random_f32(rng_state);
        let mut cumsum = 0.0f32;
        for (i, &p) in probs.iter().enumerate() {
            cumsum += p;
            if cumsum > r {
                return i as u32;
            }
        }
        (probs.len() - 1) as u32
    }

    /// Top-p (nucleus) sampling.
    fn sample_top_p(&self, probs: &[f32], rng_state: &mut u64) -> u32 {
        // Build (index, prob) pairs and sort by prob descending
        let n = probs.len();
        // Use a simple selection: accumulate until we reach top_p
        // First find the cutoff
        let mut indices: alloc::vec::Vec<(usize, f32)> = probs.iter()
            .enumerate()
            .map(|(i, &p)| (i, p))
            .collect();

        // Sort descending by probability
        indices.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(core::cmp::Ordering::Equal));

        // Accumulate until we pass top_p
        let mut cumsum = 0.0f32;
        let mut cutoff_idx = n;
        for (i, &(_, p)) in indices.iter().enumerate() {
            cumsum += p;
            if cumsum > self.top_p {
                cutoff_idx = i + 1;
                break;
            }
        }

        // Re-normalize the kept tokens
        let kept = &indices[..cutoff_idx];
        let sum: f32 = kept.iter().map(|&(_, p)| p).sum();
        let r = random_f32(rng_state) * sum;

        let mut cumsum = 0.0f32;
        for &(idx, p) in kept {
            cumsum += p;
            if cumsum > r {
                return idx as u32;
            }
        }

        kept.last().map(|&(idx, _)| idx as u32).unwrap_or(0)
    }
}

/// Simple xorshift64 PRNG. Returns a value in [0, 1).
fn random_f32(state: &mut u64) -> f32 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state as f32) / (u64::MAX as f32)
}
