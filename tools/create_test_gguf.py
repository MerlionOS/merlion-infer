#!/usr/bin/env python3
"""Create a tiny GGUF model file for testing MerlionOS Inference.

Model: dim=32, hidden_dim=64, 1 layer, 2 heads, vocab=256 (byte-level)
Format: GGUF v3 with F32 weights
Total size: ~120 KiB

Usage:
    python3 tools/create_test_gguf.py
    # Creates test-model.gguf and writes it to disk.img
"""

import struct
import random
import os

# Model config
DIM = 32
HIDDEN_DIM = 64
N_LAYERS = 1
N_HEADS = 2
KV_HEADS = 2
VOCAB = 256
MAX_SEQ = 128
HEAD_DIM = DIM // N_HEADS
KV_DIM = HEAD_DIM * KV_HEADS

# GGUF constants
GGUF_MAGIC = 0x46554747  # "GGUF" as LE u32
GGUF_VERSION = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_FLOAT32 = 6
GGML_TYPE_F32 = 0
ALIGNMENT = 32

class GGUFWriter:
    def __init__(self):
        self.kv = []  # (key, type, value_bytes)
        self.tensors = []  # (name, ndims, dims, type, data)

    def add_string(self, key, value):
        val = self._encode_string(value)
        self.kv.append((key, GGUF_TYPE_STRING, val))

    def add_uint32(self, key, value):
        self.kv.append((key, GGUF_TYPE_UINT32, struct.pack('<I', value)))

    def add_string_array(self, key, values):
        buf = struct.pack('<IQ', GGUF_TYPE_STRING, len(values))
        for v in values:
            buf += self._encode_string(v)
        self.kv.append((key, GGUF_TYPE_ARRAY, buf))

    def add_float_array(self, key, values):
        buf = struct.pack('<IQ', GGUF_TYPE_FLOAT32, len(values))
        for v in values:
            buf += struct.pack('<f', v)
        self.kv.append((key, GGUF_TYPE_ARRAY, buf))

    def add_tensor(self, name, dims, data_f32):
        """Add an F32 tensor."""
        data = struct.pack(f'<{len(data_f32)}f', *data_f32)
        self.tensors.append((name, len(dims), dims, GGML_TYPE_F32, data))

    def write(self, filename):
        # Serialize header
        header = b''
        # Magic + version
        header += struct.pack('<II', GGUF_MAGIC, GGUF_VERSION)
        # n_tensors, n_kv
        header += struct.pack('<QQ', len(self.tensors), len(self.kv))

        # KV pairs
        for key, vtype, vbytes in self.kv:
            header += self._encode_string(key)
            header += struct.pack('<I', vtype)
            header += vbytes

        # Tensor infos
        data_offset = 0
        tensor_data_parts = []
        for name, ndims, dims, ttype, data in self.tensors:
            header += self._encode_string(name)
            header += struct.pack('<I', ndims)
            for d in dims:
                header += struct.pack('<Q', d)
            header += struct.pack('<I', ttype)
            header += struct.pack('<Q', data_offset)
            tensor_data_parts.append(data)
            data_offset += len(data)

        # Pad header to alignment
        header_len = len(header)
        pad = (ALIGNMENT - (header_len % ALIGNMENT)) % ALIGNMENT
        header += b'\x00' * pad

        # Write file
        with open(filename, 'wb') as f:
            f.write(header)
            for part in tensor_data_parts:
                f.write(part)

        total = len(header) + sum(len(p) for p in tensor_data_parts)
        print(f"Created {filename}: {total} bytes ({total/1024:.1f} KiB)")
        print(f"  Header: {len(header)} bytes, Data: {total - len(header)} bytes")
        return total

    def _encode_string(self, s):
        b = s.encode('utf-8')
        return struct.pack('<Q', len(b)) + b


def rand_weights(n, scale=0.1):
    random.seed(42)
    return [random.gauss(0, scale) for _ in range(n)]


def main():
    w = GGUFWriter()

    # Metadata
    w.add_string("general.architecture", "llama")
    w.add_string("general.name", "test-tiny-32")
    w.add_uint32("llama.embedding_length", DIM)
    w.add_uint32("llama.feed_forward_length", HIDDEN_DIM)
    w.add_uint32("llama.block_count", N_LAYERS)
    w.add_uint32("llama.attention.head_count", N_HEADS)
    w.add_uint32("llama.attention.head_count_kv", KV_HEADS)
    w.add_uint32("llama.context_length", MAX_SEQ)

    # Tokenizer: byte-level (256 tokens)
    tokens = []
    scores = []
    for i in range(VOCAB):
        if 32 <= i < 127:
            tokens.append(chr(i))
        else:
            tokens.append(f"<{i:02x}>")
        scores.append(0.0)
    w.add_string_array("tokenizer.ggml.tokens", tokens)
    w.add_float_array("tokenizer.ggml.scores", scores)

    # Tensors
    random.seed(42)
    w.add_tensor("token_embd.weight", [DIM, VOCAB], rand_weights(VOCAB * DIM))

    for l in range(N_LAYERS):
        w.add_tensor(f"blk.{l}.attn_norm.weight", [DIM], [1.0] * DIM)  # init to 1
        w.add_tensor(f"blk.{l}.attn_q.weight", [DIM, DIM], rand_weights(DIM * DIM))
        w.add_tensor(f"blk.{l}.attn_k.weight", [KV_DIM, DIM], rand_weights(DIM * KV_DIM))
        w.add_tensor(f"blk.{l}.attn_v.weight", [KV_DIM, DIM], rand_weights(DIM * KV_DIM))
        w.add_tensor(f"blk.{l}.attn_output.weight", [DIM, DIM], rand_weights(DIM * DIM))
        w.add_tensor(f"blk.{l}.ffn_norm.weight", [DIM], [1.0] * DIM)
        w.add_tensor(f"blk.{l}.ffn_gate.weight", [HIDDEN_DIM, DIM], rand_weights(DIM * HIDDEN_DIM))
        w.add_tensor(f"blk.{l}.ffn_up.weight", [HIDDEN_DIM, DIM], rand_weights(DIM * HIDDEN_DIM))
        w.add_tensor(f"blk.{l}.ffn_down.weight", [DIM, HIDDEN_DIM], rand_weights(HIDDEN_DIM * DIM))

    w.add_tensor("output_norm.weight", [DIM], [1.0] * DIM)
    w.add_tensor("output.weight", [VOCAB, DIM], rand_weights(DIM * VOCAB))

    # Write GGUF
    gguf_file = "test-model.gguf"
    total = w.write(gguf_file)

    # Write to disk.img
    disk_img = "disk.img"
    disk_size = ((total // (1024 * 1024)) + 2) * 1024 * 1024
    with open(disk_img, 'wb') as f:
        # Write model at LBA 0
        with open(gguf_file, 'rb') as m:
            f.write(m.read())
        # Pad to disk size
        remaining = disk_size - total
        f.write(b'\x00' * remaining)

    print(f"Created {disk_img}: {disk_size/1024:.0f} KiB")
    print(f"\nTo test: make run-disk")
    print(f"  merlion> ai-load disk")


if __name__ == '__main__':
    main()
