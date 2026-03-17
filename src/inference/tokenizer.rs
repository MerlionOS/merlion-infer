/// BPE tokenizer for LLM inference.
/// Loads vocabulary from GGUF metadata and performs greedy BPE encoding/decoding.

use alloc::string::String;
use alloc::vec::Vec;

pub struct Tokenizer {
    /// Token ID → string piece
    vocab: Vec<String>,
    /// Token scores (for BPE merge priority)
    scores: Vec<f32>,
    /// Vocabulary size
    vocab_size: usize,
}

impl Tokenizer {
    pub fn new() -> Self {
        Self {
            vocab: Vec::new(),
            scores: Vec::new(),
            vocab_size: 0,
        }
    }

    /// Load vocabulary from GGUF metadata arrays.
    pub fn load_from_gguf(
        tokens: &[crate::inference::gguf::GgufValue],
        scores: &[crate::inference::gguf::GgufValue],
    ) {
        let tok = TOKENIZER.lock();
        drop(tok);

        let mut t = TOKENIZER.lock();
        t.vocab.clear();
        t.scores.clear();

        for token in tokens {
            if let crate::inference::gguf::GgufValue::String(s) = token {
                t.vocab.push(s.clone());
            } else {
                t.vocab.push(String::new());
            }
        }

        for score in scores {
            match score {
                crate::inference::gguf::GgufValue::Float32(f) => t.scores.push(*f),
                _ => t.scores.push(0.0),
            }
        }

        t.vocab_size = t.vocab.len();
    }

    /// Encode a string into token IDs using greedy BPE.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        if self.vocab_size == 0 { return Vec::new(); }

        // Start with individual UTF-8 bytes/characters as initial tokens
        let mut tokens: Vec<u32> = Vec::new();

        // First pass: map each character to its vocab ID
        for ch in text.chars() {
            let mut buf = [0u8; 4];
            let s = ch.encode_utf8(&mut buf);
            if let Some(id) = self.find_token(s) {
                tokens.push(id);
            }
        }

        // BPE merge loop: greedily merge the highest-scoring pair
        loop {
            let mut best_score = f32::NEG_INFINITY;
            let mut best_idx = usize::MAX;
            let mut best_token = 0u32;

            // Find the best pair to merge
            for i in 0..tokens.len().saturating_sub(1) {
                let merged = alloc::format!("{}{}",
                    self.decode_token(tokens[i]),
                    self.decode_token(tokens[i + 1]));
                if let Some(id) = self.find_token(&merged) {
                    let score = if (id as usize) < self.scores.len() {
                        self.scores[id as usize]
                    } else {
                        0.0
                    };
                    if score > best_score {
                        best_score = score;
                        best_idx = i;
                        best_token = id;
                    }
                }
            }

            if best_idx == usize::MAX { break; }

            // Perform the merge
            tokens[best_idx] = best_token;
            tokens.remove(best_idx + 1);
        }

        tokens
    }

    /// Decode a single token ID to its string representation.
    pub fn decode_token(&self, id: u32) -> &str {
        if (id as usize) < self.vocab.len() {
            &self.vocab[id as usize]
        } else {
            ""
        }
    }

    /// Decode a sequence of token IDs to a string.
    pub fn decode(&self, tokens: &[u32]) -> String {
        let mut result = String::new();
        for &id in tokens {
            result.push_str(self.decode_token(id));
        }
        result
    }

    /// Find a token string in the vocabulary, return its ID.
    fn find_token(&self, s: &str) -> Option<u32> {
        for (i, v) in self.vocab.iter().enumerate() {
            if v == s {
                return Some(i as u32);
            }
        }
        None
    }

    pub fn vocab_size(&self) -> usize { self.vocab_size }
}

/// Global tokenizer instance.
static TOKENIZER: spin::Mutex<Tokenizer> = spin::Mutex::new(Tokenizer {
    vocab: Vec::new(),
    scores: Vec::new(),
    vocab_size: 0,
});

pub fn global() -> spin::MutexGuard<'static, Tokenizer> {
    TOKENIZER.lock()
}

pub fn load_from_gguf(
    tokens: &[crate::inference::gguf::GgufValue],
    scores: &[crate::inference::gguf::GgufValue],
) {
    let mut t = TOKENIZER.lock();
    t.vocab.clear();
    t.scores.clear();

    for token in tokens {
        if let crate::inference::gguf::GgufValue::String(s) = token {
            t.vocab.push(s.clone());
        } else {
            t.vocab.push(String::new());
        }
    }

    for score in scores {
        match score {
            crate::inference::gguf::GgufValue::Float32(f) => t.scores.push(*f),
            _ => t.scores.push(0.0),
        }
    }

    t.vocab_size = t.vocab.len();
}
