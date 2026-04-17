"""
DFlash Draft Model — MLX implementation for Apple Silicon inference.

Mirrors the PyTorch DFlashDraftModel architecture:
  - 6 decoder layers (self-attn + cross-attn + SwiGLU FFN)
  - Shared frozen word_embeddings + lm_head from Sarvam-30B
  - KV fusion from 5-layer target hidden states (20480 → K,V)
  - Block-diagonal causal self-attention (block_size=16)
  - RoPE θ=8M, QK-norm, GQA 16Q:4KV
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class DFlashConfig:
    hidden_size: int = 4096
    num_draft_layers: int = 6
    num_self_attn_heads: int = 16
    num_self_attn_kv_heads: int = 4
    num_cross_attn_heads: int = 16
    num_cross_attn_kv_heads: int = 4
    head_dim: int = 64
    ffn_intermediate: int = 2048
    block_size: int = 16
    injection_dim: int = 20480
    rope_theta: float = 8_000_000.0
    rms_norm_eps: float = 1e-6
    vocab_size: int = 262144


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x):
        dtype = x.dtype
        x = x.astype(mx.float32)
        variance = mx.mean(x * x, axis=-1, keepdims=True)
        x = x * mx.rsqrt(variance + self.eps)
        return (self.weight * x).astype(dtype)


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------

class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, theta: float = 8_000_000.0):
        super().__init__()
        self.head_dim = head_dim
        self.theta = theta
        inv_freq = 1.0 / (theta ** (mx.arange(0, head_dim, 2).astype(mx.float32) / head_dim))
        self._inv_freq = inv_freq

    def __call__(self, seq_len: int):
        t = mx.arange(seq_len).astype(mx.float32)
        freqs = mx.outer(t, self._inv_freq)  # [seq_len, head_dim/2]
        emb = mx.concatenate([freqs, freqs], axis=-1)  # [seq_len, head_dim]
        return mx.cos(emb), mx.sin(emb)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return mx.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    # q, k: [B, heads, L, head_dim]
    # cos, sin: [L, head_dim]
    cos = cos[None, None, :, :]  # [1, 1, L, head_dim]
    sin = sin[None, None, :, :]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# ---------------------------------------------------------------------------
# KV Fusion
# ---------------------------------------------------------------------------

class DFlashKVFusion(nn.Module):
    def __init__(self, config: DFlashConfig):
        super().__init__()
        self.num_kv_heads = config.num_cross_attn_kv_heads
        self.head_dim = config.head_dim
        kv_dim = 2 * self.num_kv_heads * self.head_dim  # 512
        self.kv_proj = nn.Linear(config.injection_dim, kv_dim, bias=False)

    def __call__(self, injection_features: mx.array) -> Tuple[mx.array, mx.array]:
        B, S, _ = injection_features.shape
        kv = self.kv_proj(injection_features)  # [B, S, 512]
        kv = kv.reshape(B, S, 2, self.num_kv_heads, self.head_dim)
        kv = kv.transpose(2, 0, 3, 1, 4)  # [2, B, heads, S, dim]
        return kv[0], kv[1]


# ---------------------------------------------------------------------------
# Self-Attention (GQA + RoPE + QK-Norm)
# ---------------------------------------------------------------------------

class DFlashSelfAttention(nn.Module):
    def __init__(self, config: DFlashConfig):
        super().__init__()
        self.num_heads = config.num_self_attn_heads
        self.num_kv_heads = config.num_self_attn_kv_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.scale = self.head_dim ** -0.5

        total = (self.num_heads + 2 * self.num_kv_heads) * self.head_dim
        self.query_key_value = nn.Linear(config.hidden_size, total, bias=False)
        self.dense = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)
        self.query_layernorm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.key_layernorm = RMSNorm(self.head_dim, config.rms_norm_eps)

    def __call__(self, x, cos, sin, mask=None):
        B, L, _ = x.shape

        qkv = self.query_key_value(x)
        qkv = qkv.reshape(B, L, self.num_heads + 2 * self.num_kv_heads, self.head_dim)

        q = qkv[:, :, :self.num_heads]
        k = qkv[:, :, self.num_heads:self.num_heads + self.num_kv_heads]
        v = qkv[:, :, self.num_heads + self.num_kv_heads:]

        # [B, L, heads, dim] -> [B, heads, L, dim]
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        q = self.query_layernorm(q)
        k = self.key_layernorm(k)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Expand KV for GQA
        if self.num_kv_groups > 1:
            k = mx.repeat(k, self.num_kv_groups, axis=1)
            v = mx.repeat(v, self.num_kv_groups, axis=1)

        # Scaled dot-product attention
        attn = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        if mask is not None:
            attn = attn + mask
        attn = mx.softmax(attn, axis=-1)
        out = attn @ v  # [B, heads, L, dim]

        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.dense(out)


# ---------------------------------------------------------------------------
# Cross-Attention
# ---------------------------------------------------------------------------

class DFlashCrossAttention(nn.Module):
    def __init__(self, config: DFlashConfig):
        super().__init__()
        self.num_heads = config.num_cross_attn_heads
        self.num_kv_heads = config.num_cross_attn_kv_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.dense = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

    def __call__(self, x, cross_k, cross_v):
        B, L, _ = x.shape

        q = self.q_proj(x).reshape(B, L, self.num_heads, self.head_dim)
        q = q.transpose(0, 2, 1, 3)  # [B, heads, L, dim]

        k = cross_k
        v = cross_v
        if self.num_kv_groups > 1:
            k = mx.repeat(k, self.num_kv_groups, axis=1)
            v = mx.repeat(v, self.num_kv_groups, axis=1)

        attn = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        attn = mx.softmax(attn, axis=-1)
        out = attn @ v

        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.dense(out)


# ---------------------------------------------------------------------------
# FFN (SwiGLU)
# ---------------------------------------------------------------------------

class DFlashFFN(nn.Module):
    def __init__(self, config: DFlashConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.ffn_intermediate, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.ffn_intermediate, bias=False)
        self.down_proj = nn.Linear(config.ffn_intermediate, config.hidden_size, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# Decoder Layer
# ---------------------------------------------------------------------------

class DFlashDecoderLayer(nn.Module):
    def __init__(self, config: DFlashConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = DFlashSelfAttention(config)
        self.post_self_attn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.cross_attn = DFlashCrossAttention(config)
        self.post_cross_attn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.ffn = DFlashFFN(config)

    def __call__(self, x, cos, sin, cross_k, cross_v, mask=None):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, mask)
        x = x + self.cross_attn(self.post_self_attn_norm(x), cross_k, cross_v)
        x = x + self.ffn(self.post_cross_attn_norm(x))
        return x


# ---------------------------------------------------------------------------
# Block Causal Mask
# ---------------------------------------------------------------------------

def make_block_causal_mask(seq_len: int, block_size: int = 16) -> mx.array:
    """Block-diagonal causal mask. 0 = attend, -inf = mask."""
    num_blocks = seq_len // block_size
    block = mx.tril(mx.ones((block_size, block_size)))
    # Build block-diagonal
    mask = mx.zeros((seq_len, seq_len))
    for i in range(num_blocks):
        s = i * block_size
        e = s + block_size
        mask = mask.at[s:e, s:e].add(block)
    # Convert: 1 -> 0, 0 -> -inf
    mask = mx.where(mask > 0, mx.array(0.0), mx.array(float("-inf")))
    return mask


# ---------------------------------------------------------------------------
# Draft Model
# ---------------------------------------------------------------------------

class DFlashDraftModel(nn.Module):
    def __init__(self, config: DFlashConfig):
        super().__init__()
        self.config = config

        # Shared frozen weights (loaded separately)
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Trainable
        self.kv_fusion = DFlashKVFusion(config)
        self.layers = [DFlashDecoderLayer(config) for _ in range(config.num_draft_layers)]
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rope = RotaryEmbedding(config.head_dim, config.rope_theta)

    def __call__(
        self,
        input_ids: mx.array,            # [B, seq_len]
        injection_features: mx.array,    # [B, ctx_len, 20480]
        mask: Optional[mx.array] = None, # [seq_len, seq_len]
    ) -> mx.array:
        B, S = input_ids.shape

        h = self.word_embeddings(input_ids)  # [B, S, 4096]
        cos, sin = self.rope(S)               # [S, 64], [S, 64]
        cross_k, cross_v = self.kv_fusion(injection_features)

        for layer in self.layers:
            h = layer(h, cos, sin, cross_k, cross_v, mask)

        h = self.norm(h)
        return self.lm_head(h)  # [B, S, vocab_size]
