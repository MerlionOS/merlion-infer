/// Text generation loop.
/// Drives the LlamaEngine to generate tokens autoregressively.

use crate::inference::engine::LlamaEngine;
use crate::inference::sampler::Sampler;
use crate::inference::tokenizer;
use alloc::string::String;

/// Generate text from a prompt.
/// Prints tokens to serial as they are generated.
/// Returns (generated_text, num_tokens, elapsed_ticks).
pub fn generate(
    engine: &mut LlamaEngine,
    prompt: &str,
    max_tokens: usize,
    sampler: &Sampler,
) -> (String, usize, u64) {
    let start_ticks = crate::arch::x86_64::timer::ticks();

    let tok = tokenizer::global();
    let prompt_tokens = tok.encode(prompt);
    let n_prompt = prompt_tokens.len();
    drop(tok);

    if n_prompt == 0 {
        crate::serial_println!("[generate] empty prompt after tokenization");
        return (String::new(), 0, 0);
    }

    crate::serial_println!("[generate] prompt: {} tokens, max_new: {}", n_prompt, max_tokens);

    let mut output = String::new();
    let mut token = prompt_tokens[0];
    let mut rng_state: u64 = 42 ^ start_ticks;
    let mut n_generated = 0usize;

    let total_len = core::cmp::min(n_prompt + max_tokens, engine.config.max_seq_len);

    for pos in 0..total_len {
        // Forward pass
        let _logits = engine.forward(token, pos);

        // Determine next token
        let next_token = if pos + 1 < n_prompt {
            // Still processing prompt — teacher-force
            prompt_tokens[pos + 1]
        } else {
            // Generate — sample directly from logits (no clone)
            sampler.sample(&mut engine.state.logits, &mut rng_state)
        };

        // If we're past the prompt, output the token
        if pos >= n_prompt - 1 {
            let tok = tokenizer::global();
            let piece = tok.decode_token(next_token);
            crate::serial_print!("{}", piece);
            output.push_str(piece);
            drop(tok);
            n_generated += 1;
        }

        token = next_token;

        // Stop on EOS (token 2 is common EOS, but varies by model)
        if next_token == 2 || next_token == 0 {
            break;
        }
    }

    let elapsed = crate::arch::x86_64::timer::ticks() - start_ticks;
    crate::serial_println!();

    (output, n_generated, elapsed)
}
