/// GGUF (GGML Unified Format) model file parser.
/// Parses header, metadata, and tensor info from raw bytes.
/// Reference: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md

use alloc::string::String;
use alloc::vec::Vec;

const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" as LE u32 (bytes: 47 47 55 46)

/// GGUF metadata value types.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(u32)]
pub enum GgufType {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
}

impl GgufType {
    fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::Uint8),
            1 => Some(Self::Int8),
            2 => Some(Self::Uint16),
            3 => Some(Self::Int16),
            4 => Some(Self::Uint32),
            5 => Some(Self::Int32),
            6 => Some(Self::Float32),
            7 => Some(Self::Bool),
            8 => Some(Self::String),
            9 => Some(Self::Array),
            10 => Some(Self::Uint64),
            11 => Some(Self::Int64),
            12 => Some(Self::Float64),
            _ => None,
        }
    }
}

/// GGML tensor types (quantization formats).
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(u32)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Q8K = 15,
    IQ2XXS = 16,
    IQ2XS = 17,
    IQ3XXS = 18,
    IQ1S = 19,
    IQ4NL = 20,
    IQ3S = 21,
    IQ2S = 22,
    IQ4XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1M = 29,
    BF16 = 30,
    Unknown(u32),
}

impl GgmlType {
    fn from_u32(v: u32) -> Self {
        match v {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            10 => Self::Q2K,
            11 => Self::Q3K,
            12 => Self::Q4K,
            13 => Self::Q5K,
            14 => Self::Q6K,
            15 => Self::Q8K,
            24 => Self::I8,
            25 => Self::I16,
            26 => Self::I32,
            27 => Self::I64,
            28 => Self::F64,
            30 => Self::BF16,
            other => Self::Unknown(other),
        }
    }

    /// Bytes per element (approximate, for non-block types).
    pub fn element_size(&self) -> f32 {
        match self {
            Self::F32 => 4.0,
            Self::F16 | Self::BF16 => 2.0,
            Self::Q4_0 => 0.5 + 2.0 / 32.0, // 18 bytes per 32 elements
            Self::Q4_1 => 0.5 + 4.0 / 32.0,
            Self::Q8_0 => 1.0 + 2.0 / 32.0,  // 34 bytes per 32 elements
            Self::Q8_1 => 1.0 + 4.0 / 32.0,
            Self::I8 => 1.0,
            Self::I16 => 2.0,
            Self::I32 => 4.0,
            Self::I64 | Self::F64 => 8.0,
            _ => 1.0, // conservative estimate
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::Q4_0 => "Q4_0",
            Self::Q4_1 => "Q4_1",
            Self::Q5_0 => "Q5_0",
            Self::Q5_1 => "Q5_1",
            Self::Q8_0 => "Q8_0",
            Self::Q8_1 => "Q8_1",
            Self::Q2K => "Q2_K",
            Self::Q3K => "Q3_K",
            Self::Q4K => "Q4_K",
            Self::Q5K => "Q5_K",
            Self::Q6K => "Q6_K",
            Self::Q8K => "Q8_K",
            Self::BF16 => "BF16",
            Self::I8 => "I8",
            Self::I16 => "I16",
            Self::I32 => "I32",
            Self::I64 => "I64",
            Self::F64 => "F64",
            _ => "?",
        }
    }
}

/// Parsed GGUF metadata key-value pair.
#[derive(Debug, Clone)]
pub struct GgufMetadata {
    pub key: String,
    pub value: GgufValue,
}

#[derive(Debug, Clone)]
pub enum GgufValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
}

impl GgufValue {
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            Self::Uint32(v) => Some(*v),
            Self::Int32(v) => Some(*v as u32),
            Self::Uint64(v) => Some(*v as u32),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Self::Uint64(v) => Some(*v),
            Self::Uint32(v) => Some(*v as u64),
            Self::Int64(v) => Some(*v as u64),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s.as_str()),
            _ => None,
        }
    }
}

/// Tensor descriptor from GGUF header.
#[derive(Debug, Clone)]
pub struct GgufTensorInfo {
    pub name: String,
    pub n_dims: u32,
    pub dims: [u64; 4],
    pub tensor_type: GgmlType,
    pub offset: u64,
}

impl GgufTensorInfo {
    pub fn n_elements(&self) -> u64 {
        let mut n = 1u64;
        for i in 0..self.n_dims as usize {
            n *= self.dims[i];
        }
        n
    }

    pub fn byte_size(&self) -> u64 {
        (self.n_elements() as f64 * self.tensor_type.element_size() as f64) as u64
    }
}

/// Parsed GGUF file.
pub struct GgufModel {
    pub version: u32,
    pub metadata: Vec<GgufMetadata>,
    pub tensors: Vec<GgufTensorInfo>,
    pub data_offset: usize,
}

impl GgufModel {
    /// Look up a metadata value by key.
    pub fn get_metadata(&self, key: &str) -> Option<&GgufValue> {
        self.metadata.iter().find(|m| m.key == key).map(|m| &m.value)
    }

    /// Total size of all tensor data in bytes.
    pub fn total_tensor_bytes(&self) -> u64 {
        self.tensors.iter().map(|t| t.byte_size()).sum()
    }
}

/// A simple cursor over a byte slice for parsing.
struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(data: &'a [u8]) -> Self { Self { data, pos: 0 } }

    fn remaining(&self) -> usize { self.data.len() - self.pos }

    fn read_u8(&mut self) -> Result<u8, &'static str> {
        if self.pos >= self.data.len() { return Err("gguf: unexpected EOF"); }
        let v = self.data[self.pos];
        self.pos += 1;
        Ok(v)
    }

    fn read_u16(&mut self) -> Result<u16, &'static str> {
        if self.pos + 2 > self.data.len() { return Err("gguf: unexpected EOF"); }
        let v = u16::from_le_bytes([self.data[self.pos], self.data[self.pos + 1]]);
        self.pos += 2;
        Ok(v)
    }

    fn read_u32(&mut self) -> Result<u32, &'static str> {
        if self.pos + 4 > self.data.len() { return Err("gguf: unexpected EOF"); }
        let v = u32::from_le_bytes(self.data[self.pos..self.pos + 4].try_into().unwrap());
        self.pos += 4;
        Ok(v)
    }

    fn read_u64(&mut self) -> Result<u64, &'static str> {
        if self.pos + 8 > self.data.len() { return Err("gguf: unexpected EOF"); }
        let v = u64::from_le_bytes(self.data[self.pos..self.pos + 8].try_into().unwrap());
        self.pos += 8;
        Ok(v)
    }

    fn read_i8(&mut self) -> Result<i8, &'static str> { self.read_u8().map(|v| v as i8) }
    fn read_i16(&mut self) -> Result<i16, &'static str> { self.read_u16().map(|v| v as i16) }
    fn read_i32(&mut self) -> Result<i32, &'static str> { self.read_u32().map(|v| v as i32) }
    fn read_i64(&mut self) -> Result<i64, &'static str> { self.read_u64().map(|v| v as i64) }

    fn read_f32(&mut self) -> Result<f32, &'static str> {
        self.read_u32().map(f32::from_bits)
    }

    fn read_f64(&mut self) -> Result<f64, &'static str> {
        self.read_u64().map(f64::from_bits)
    }

    fn read_bool(&mut self) -> Result<bool, &'static str> {
        self.read_u8().map(|v| v != 0)
    }

    fn read_string(&mut self) -> Result<String, &'static str> {
        let len = self.read_u64()? as usize;
        if self.pos + len > self.data.len() { return Err("gguf: string too long"); }
        let s = core::str::from_utf8(&self.data[self.pos..self.pos + len])
            .map_err(|_| "gguf: invalid UTF-8")?;
        self.pos += len;
        Ok(String::from(s))
    }

    fn read_value(&mut self, vtype: GgufType) -> Result<GgufValue, &'static str> {
        match vtype {
            GgufType::Uint8 => Ok(GgufValue::Uint8(self.read_u8()?)),
            GgufType::Int8 => Ok(GgufValue::Int8(self.read_i8()?)),
            GgufType::Uint16 => Ok(GgufValue::Uint16(self.read_u16()?)),
            GgufType::Int16 => Ok(GgufValue::Int16(self.read_i16()?)),
            GgufType::Uint32 => Ok(GgufValue::Uint32(self.read_u32()?)),
            GgufType::Int32 => Ok(GgufValue::Int32(self.read_i32()?)),
            GgufType::Float32 => Ok(GgufValue::Float32(self.read_f32()?)),
            GgufType::Uint64 => Ok(GgufValue::Uint64(self.read_u64()?)),
            GgufType::Int64 => Ok(GgufValue::Int64(self.read_i64()?)),
            GgufType::Float64 => Ok(GgufValue::Float64(self.read_f64()?)),
            GgufType::Bool => Ok(GgufValue::Bool(self.read_bool()?)),
            GgufType::String => Ok(GgufValue::String(self.read_string()?)),
            GgufType::Array => {
                let elem_type_raw = self.read_u32()?;
                let elem_type = GgufType::from_u32(elem_type_raw)
                    .ok_or("gguf: unknown array element type")?;
                let count = self.read_u64()? as usize;
                // Limit array size to prevent OOM
                if count > 1_000_000 { return Err("gguf: array too large"); }
                let mut arr = Vec::with_capacity(count);
                for _ in 0..count {
                    arr.push(self.read_value(elem_type)?);
                }
                Ok(GgufValue::Array(arr))
            }
        }
    }
}

/// Parse a GGUF file from raw bytes.
/// Returns the parsed model header (metadata + tensor info).
pub fn parse(data: &[u8]) -> Result<GgufModel, &'static str> {
    let mut r = Reader::new(data);

    // Magic
    let magic = r.read_u32()?;
    if magic != GGUF_MAGIC { return Err("gguf: bad magic"); }

    // Version
    let version = r.read_u32()?;
    if version < 2 || version > 3 { return Err("gguf: unsupported version"); }

    // Counts
    let n_tensors = r.read_u64()? as usize;
    let n_kv = r.read_u64()? as usize;

    if n_tensors > 100_000 { return Err("gguf: too many tensors"); }
    if n_kv > 100_000 { return Err("gguf: too many metadata entries"); }

    // Parse metadata
    let mut metadata = Vec::with_capacity(n_kv);
    for _ in 0..n_kv {
        let key = r.read_string()?;
        let vtype_raw = r.read_u32()?;
        let vtype = GgufType::from_u32(vtype_raw).ok_or("gguf: unknown value type")?;
        let value = r.read_value(vtype)?;
        metadata.push(GgufMetadata { key, value });
    }

    // Parse tensor info
    let mut tensors = Vec::with_capacity(n_tensors);
    for _ in 0..n_tensors {
        let name = r.read_string()?;
        let n_dims = r.read_u32()?;
        if n_dims > 4 { return Err("gguf: tensor has >4 dims"); }
        let mut dims = [1u64; 4];
        for d in 0..n_dims as usize {
            dims[d] = r.read_u64()?;
        }
        let tensor_type_raw = r.read_u32()?;
        let tensor_type = GgmlType::from_u32(tensor_type_raw);
        let offset = r.read_u64()?;
        tensors.push(GgufTensorInfo { name, n_dims, dims, tensor_type, offset });
    }

    // Data starts at next alignment boundary (32 bytes in GGUF v2+)
    let alignment = 32usize;
    let data_offset = (r.pos + alignment - 1) & !(alignment - 1);

    Ok(GgufModel { version, metadata, tensors, data_offset })
}
