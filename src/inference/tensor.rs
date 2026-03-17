/// Quantized tensor types for LLM inference.
/// Supports Q4_0 and Q8_0 block formats from GGML.

/// Q4_0 block: 32 elements packed into 18 bytes.
/// Layout: 2-byte f16 scale + 16 bytes of 4-bit quantized values.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct BlockQ4_0 {
    pub scale: u16, // f16 scale factor
    pub qs: [u8; 16], // 32 x 4-bit values packed into 16 bytes
}

impl BlockQ4_0 {
    pub const BLOCK_SIZE: usize = 32;
    pub const BYTE_SIZE: usize = 18; // 2 + 16

    /// Dequantize this block into f32 values.
    pub fn dequantize(&self, out: &mut [f32; 32]) {
        let scale = f16_to_f32(self.scale);
        for i in 0..16 {
            let byte = self.qs[i];
            let lo = (byte & 0x0F) as i8 - 8;
            let hi = ((byte >> 4) & 0x0F) as i8 - 8;
            out[i * 2] = lo as f32 * scale;
            out[i * 2 + 1] = hi as f32 * scale;
        }
    }
}

/// Q8_0 block: 32 elements in 34 bytes.
/// Layout: 2-byte f16 scale + 32 bytes of 8-bit quantized values.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct BlockQ8_0 {
    pub scale: u16, // f16 scale factor
    pub qs: [i8; 32],
}

impl BlockQ8_0 {
    pub const BLOCK_SIZE: usize = 32;
    pub const BYTE_SIZE: usize = 34; // 2 + 32

    pub fn dequantize(&self, out: &mut [f32; 32]) {
        let scale = f16_to_f32(self.scale);
        for i in 0..32 {
            out[i] = self.qs[i] as f32 * scale;
        }
    }
}

/// Convert f16 (IEEE 754 half-precision) to f32.
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x3FF) as u32;

    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign << 31);
        }
        // Subnormal
        let mut m = mant;
        let mut e = 0i32;
        while m & 0x400 == 0 {
            m <<= 1;
            e -= 1;
        }
        m &= 0x3FF;
        let f32_exp = (127 - 15 + 1 + e) as u32;
        return f32::from_bits((sign << 31) | (f32_exp << 23) | (m << 13));
    }

    if exp == 0x1F {
        // Inf or NaN
        return f32::from_bits((sign << 31) | (0xFF << 23) | (mant << 13));
    }

    let f32_exp = exp + (127 - 15);
    f32::from_bits((sign << 31) | (f32_exp << 23) | (mant << 13))
}

/// Convert f32 to f16.
pub fn f32_to_f16(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = ((bits >> 31) & 1) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = (bits & 0x7FFFFF) as u32;

    if exp == 0xFF {
        // Inf/NaN
        return (sign << 15) | (0x1F << 10) | ((mant >> 13) as u16);
    }

    let new_exp = exp - 127 + 15;
    if new_exp >= 0x1F {
        return (sign << 15) | (0x1F << 10); // overflow → inf
    }
    if new_exp <= 0 {
        return sign << 15; // underflow → zero
    }

    (sign << 15) | ((new_exp as u16) << 10) | ((mant >> 13) as u16)
}
