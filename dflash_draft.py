"""
DFlash Block Diffusion Draft Model for Sarvam-30B.

A lightweight 6-layer dense transformer (~275M trainable parameters) that generates
16-token blocks in parallel, conditioned on KV-injected hidden states from the
frozen Sarvam-30B target model.

Architecture:
- Shares frozen word_embeddings (1.07B) and lm_head (1.07B) from target
- 6 decoder layers, each with: self-attention + cross-attention + SwiGLU FFN
- Self-attention: 16 Q heads, 4 KV heads (4:1 GQA), head_dim=64, RoPE + QK-norm
- Cross-attention: 16 Q heads, 4 KV heads, attends to fused target hidden states
- KV fusion: projects concatenated 5-layer target features (20480) → K,V (512)
- Block-diagonal causal mask for self-attention (block_size=16)
"""

import math
import sys
import os
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import reusable components from Sarvam-30B via our wrapper (handles relative imports)
from modeling_sarvam_moe_dflash import SarvamMoEConfig

# Access the already-imported modeling module
import sarvam_30b.modeling_sarvam_moe as _sarvam_modeling

SarvamMoERMSNorm = _sarvam_modeling.SarvamMoERMSNorm
SarvamMoERotaryEmbedding = _sarvam_modeling.SarvamMoERotaryEmbedding
apply_rotary_pos_emb = _sarvam_modeling.apply_rotary_pos_emb
repeat_kv = _sarvam_modeling.repeat_kv


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DFlashConfig:
    """Configuration for the DFlash draft model."""
    hidden_size: int = 4096          # Must match target embedding dim
    num_draft_layers: int = 6
    num_self_attn_heads: int = 16
    num_self_attn_kv_heads: int = 4  # 4:1 GQA
    num_cross_attn_heads: int = 16
    num_cross_attn_kv_heads: int = 4
    head_dim: int = 64               # Match target for RoPE
    ffn_intermediate: int = 2048
    block_size: int = 16
    num_injection_layers: int = 5
    injection_dim: int = 20480       # 5 * 4096
    rope_theta: float = 8_000_000.0  # Same as Sarvam-30B
    rms_norm_eps: float = 1e-6
    vocab_size: int = 262144
    max_position_embeddings: int = 131072
    initializer_range: float = 0.006

    def to_rope_config(self) -> SarvamMoEConfig:
        """Create a minimal SarvamMoEConfig for RoPE initialization."""
        return SarvamMoEConfig(
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_self_attn_heads,
            head_dim=self.head_dim,
            rope_theta=self.rope_theta,
            max_position_embeddings=self.max_position_embeddings,
        )


# ---------------------------------------------------------------------------
# KV Fusion Module
# ---------------------------------------------------------------------------

class DFlashKVFusion(nn.Module):
    """
    Projects concatenated target hidden states into cross-attention K and V.

    Input:  [B, ctx_len, 20480] (5 layers * 4096)
    Output: cross_k [B, num_kv_heads, ctx_len, head_dim]
            cross_v [B, num_kv_heads, ctx_len, head_dim]
    """

    def __init__(self, config: DFlashConfig):
        super().__init__()
        self.num_kv_heads = config.num_cross_attn_kv_heads
        self.head_dim = config.head_dim
        kv_dim = 2 * self.num_kv_heads * self.head_dim  # 2 * 4 * 64 = 512
        self.kv_proj = nn.Linear(config.injection_dim, kv_dim, bias=False)

    def forward(self, injection_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            injection_features: [B, ctx_len, injection_dim]
        Returns:
            cross_k: [B, num_kv_heads, ctx_len, head_dim]
            cross_v: [B, num_kv_heads, ctx_len, head_dim]
        """
        B, S, _ = injection_features.shape
        kv = self.kv_proj(injection_features)  # [B, S, 512]
        kv = kv.view(B, S, 2, self.num_kv_heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)  # [2, B, num_kv_heads, S, head_dim]
        cross_k, cross_v = kv[0], kv[1]
        return cross_k, cross_v


# ---------------------------------------------------------------------------
# Self-Attention (GQA + RoPE + QK-Norm)
# ---------------------------------------------------------------------------

class DFlashSelfAttention(nn.Module):
    """
    Grouped-query self-attention with RoPE and QK-normalization.
    Uses block-diagonal causal mask for intra-block attention.
    """

    def __init__(self, config: DFlashConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = config.num_self_attn_heads          # 16
        self.num_kv_heads = config.num_self_attn_kv_heads    # 4
        self.head_dim = config.head_dim                       # 64
        self.num_kv_groups = self.num_heads // self.num_kv_heads  # 4
        self.scaling = self.head_dim ** -0.5

        total_head_dim = (self.num_heads + 2 * self.num_kv_heads) * self.head_dim  # 1536
        self.query_key_value = nn.Linear(config.hidden_size, total_head_dim, bias=False)
        self.dense = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

        # QK normalization (same as Sarvam-30B)
        self.query_layernorm = SarvamMoERMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.key_layernorm = SarvamMoERMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,      # [B, L, hidden_size]
        attention_mask: Optional[torch.Tensor] = None,  # [B, 1, L, L] or [L, L]
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        B, L, _ = hidden_states.shape

        # Fused QKV projection
        qkv = self.query_key_value(hidden_states)  # [B, L, 1536]
        qkv = qkv.view(B, L, self.num_heads + 2 * self.num_kv_heads, self.head_dim)

        query, key, value = qkv.split(
            [self.num_heads, self.num_kv_heads, self.num_kv_heads], dim=-2
        )
        # query: [B, L, 16, 64], key: [B, L, 4, 64], value: [B, L, 4, 64]

        query = query.transpose(1, 2)  # [B, 16, L, 64]
        key = key.transpose(1, 2)      # [B, 4, L, 64]
        value = value.transpose(1, 2)  # [B, 4, L, 64]

        # QK normalization
        query = self.query_layernorm(query)
        key = self.key_layernorm(key)

        # RoPE
        cos, sin = position_embeddings
        query, key = apply_rotary_pos_emb(query, key, cos, sin)

        # Expand KV for GQA
        key = repeat_kv(key, self.num_kv_groups)      # [B, 16, L, 64]
        value = repeat_kv(value, self.num_kv_groups)  # [B, 16, L, 64]

        # Attention with block-diagonal mask
        # Ensure mask dtype matches query (MPS may upcast to float32)
        if attention_mask is not None and attention_mask.dtype != query.dtype:
            attention_mask = attention_mask.to(query.dtype)
        attn_output = F.scaled_dot_product_attention(
            query, key, value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,  # We provide explicit mask
        )  # [B, 16, L, 64]

        attn_output = attn_output.transpose(1, 2).contiguous()  # [B, L, 16, 64]
        attn_output = attn_output.reshape(B, L, -1)             # [B, L, 1024]
        return self.dense(attn_output)                           # [B, L, 4096]


# ---------------------------------------------------------------------------
# Cross-Attention (to fused target KV)
# ---------------------------------------------------------------------------

class DFlashCrossAttention(nn.Module):
    """
    Cross-attention to KV-injected target features.
    No RoPE (target features already encode position).
    No QK-norm (cross-attention is simpler).
    Full attention (every draft token attends to full target context).
    """

    def __init__(self, config: DFlashConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = config.num_cross_attn_heads        # 16
        self.num_kv_heads = config.num_cross_attn_kv_heads  # 4
        self.head_dim = config.head_dim                      # 64
        self.num_kv_groups = self.num_heads // self.num_kv_heads  # 4
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.dense = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,  # [B, L, hidden_size]
        cross_k: torch.Tensor,        # [B, num_kv_heads, ctx_len, head_dim]
        cross_v: torch.Tensor,        # [B, num_kv_heads, ctx_len, head_dim]
    ) -> torch.Tensor:
        B, L, _ = hidden_states.shape

        # Query from draft hidden states
        query = self.q_proj(hidden_states)             # [B, L, 1024]
        query = query.view(B, L, self.num_heads, self.head_dim)
        query = query.transpose(1, 2)                  # [B, 16, L, 64]

        # Expand KV for GQA
        key = repeat_kv(cross_k, self.num_kv_groups)   # [B, 16, ctx_len, 64]
        value = repeat_kv(cross_v, self.num_kv_groups) # [B, 16, ctx_len, 64]

        # Ensure matching dtypes (MPS may upcast query via prior RMSNorm)
        if key.dtype != query.dtype:
            key = key.to(query.dtype)
            value = value.to(query.dtype)

        # Full attention (no mask — draft sees entire target context)
        attn_output = F.scaled_dot_product_attention(
            query, key, value,
            dropout_p=0.0,
            is_causal=False,
        )  # [B, 16, L, 64]

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(B, L, -1)  # [B, L, 1024]
        return self.dense(attn_output)                 # [B, L, 4096]


# ---------------------------------------------------------------------------
# Feed-Forward Network (SwiGLU)
# ---------------------------------------------------------------------------

class DFlashFFN(nn.Module):
    """SwiGLU FFN matching Sarvam-30B's MLP pattern."""

    def __init__(self, config: DFlashConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.ffn_intermediate, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.ffn_intermediate, bias=False)
        self.down_proj = nn.Linear(config.ffn_intermediate, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# Decoder Layer
# ---------------------------------------------------------------------------

class DFlashDecoderLayer(nn.Module):
    """
    Pre-norm decoder layer with self-attention, cross-attention, and FFN.

    x = x + self_attn(input_layernorm(x))
    x = x + cross_attn(post_self_attn_norm(x), cross_kv)
    x = x + ffn(post_cross_attn_norm(x))
    """

    def __init__(self, config: DFlashConfig, layer_idx: int):
        super().__init__()
        self.input_layernorm = SarvamMoERMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = DFlashSelfAttention(config, layer_idx)
        self.post_self_attn_norm = SarvamMoERMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.cross_attn = DFlashCrossAttention(config, layer_idx)
        self.post_cross_attn_norm = SarvamMoERMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn = DFlashFFN(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        block_mask: Optional[torch.Tensor],
        cross_k: torch.Tensor,
        cross_v: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        # Self-attention with block-diagonal causal mask
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask=block_mask, position_embeddings=position_embeddings)
        hidden_states = residual + hidden_states

        # Cross-attention to target KV
        residual = hidden_states
        hidden_states = self.post_self_attn_norm(hidden_states)
        hidden_states = self.cross_attn(hidden_states, cross_k, cross_v)
        hidden_states = residual + hidden_states

        # FFN
        residual = hidden_states
        hidden_states = self.post_cross_attn_norm(hidden_states)
        hidden_states = self.ffn(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


# ---------------------------------------------------------------------------
# Block-Diagonal Causal Mask
# ---------------------------------------------------------------------------

def make_block_causal_mask(
    seq_len: int,
    block_size: int = 16,
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Create a block-diagonal causal attention mask for SDPA.

    Within each block of `block_size` tokens, causal masking applies
    (token k can attend to positions 0..k). Across blocks, no attention.

    Args:
        seq_len: total sequence length (must be divisible by block_size)
        block_size: size of each block (default 16)
        dtype: output dtype (use float for SDPA mask)
        device: target device

    Returns:
        mask: [seq_len, seq_len] where 0 = attend, -inf = mask
    """
    assert seq_len % block_size == 0, f"seq_len {seq_len} must be divisible by block_size {block_size}"
    num_blocks = seq_len // block_size

    # Causal mask for one block [block_size, block_size]
    block = torch.tril(torch.ones(block_size, block_size, dtype=torch.bool, device=device))

    # Build block-diagonal mask
    full_mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
    for i in range(num_blocks):
        start = i * block_size
        end = start + block_size
        full_mask[start:end, start:end] = block

    # Convert to SDPA format: 0 for attend, -inf for mask
    attn_mask = torch.where(full_mask, torch.tensor(0.0, dtype=dtype, device=device),
                            torch.tensor(float("-inf"), dtype=dtype, device=device))
    return attn_mask  # [seq_len, seq_len]


# ---------------------------------------------------------------------------
# Top-Level Draft Model
# ---------------------------------------------------------------------------

class DFlashDraftModel(nn.Module):
    """
    DFlash block diffusion draft model for Sarvam-30B.

    Shares frozen word_embeddings and lm_head from the target model.
    Trains ~275M parameters: KV fusion + 6 decoder layers + final norm.
    """

    def __init__(self, config: DFlashConfig):
        super().__init__()
        self.config = config

        # These will be set by share_target_embeddings() — placeholders for now
        self.word_embeddings: Optional[nn.Embedding] = None
        self.lm_head: Optional[nn.Linear] = None

        # Trainable components
        self.kv_fusion = DFlashKVFusion(config)
        self.layers = nn.ModuleList([
            DFlashDecoderLayer(config, layer_idx=i)
            for i in range(config.num_draft_layers)
        ])
        self.norm = SarvamMoERMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # RoPE (same config as Sarvam-30B: theta=8M, head_dim=64)
        rope_config = config.to_rope_config()
        self.rotary_emb = SarvamMoERotaryEmbedding(config=rope_config)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def share_target_embeddings(self, target_model):
        """
        Share frozen embedding and LM head from the target Sarvam-30B model.
        Must be called before forward().
        """
        self.word_embeddings = target_model.model.word_embeddings
        self.lm_head = target_model.lm_head

        # Freeze shared parameters
        for param in self.word_embeddings.parameters():
            param.requires_grad = False
        for param in self.lm_head.parameters():
            param.requires_grad = False

    def get_trainable_parameters(self):
        """Return only the trainable parameters (excludes frozen embed/head)."""
        trainable = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable.append(param)
        return trainable

    def count_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_total_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        input_ids: torch.LongTensor,                          # [B, seq_len]
        injection_features: torch.Tensor,                      # [B, ctx_len, 20480]
        block_mask: Optional[torch.Tensor] = None,             # [seq_len, seq_len]
        position_ids: Optional[torch.LongTensor] = None,       # [B, seq_len]
    ) -> torch.Tensor:
        """
        Forward pass of the draft model.

        Args:
            input_ids: token IDs [B, seq_len]
            injection_features: target hidden states [B, ctx_len, 20480]
            block_mask: block-diagonal causal mask [seq_len, seq_len]
            position_ids: position IDs for RoPE [B, seq_len]

        Returns:
            logits: [B, seq_len, vocab_size]
        """
        assert self.word_embeddings is not None, "Call share_target_embeddings() first"
        B, S = input_ids.shape

        # Token embedding (frozen) — may be on different device with device_map="auto"
        hidden_states = self.word_embeddings(input_ids)  # [B, S, 4096]

        # Move to draft model's device (layers may be on MPS/CUDA while embed is on CPU)
        draft_device = self.kv_fusion.kv_proj.weight.device
        hidden_states = hidden_states.to(draft_device)
        injection_features = injection_features.to(draft_device)

        # Position IDs
        if position_ids is None:
            position_ids = torch.arange(S, device=draft_device).unsqueeze(0).expand(B, -1)
        else:
            position_ids = position_ids.to(draft_device)

        # RoPE embeddings
        position_embeddings = self.rotary_emb(hidden_states, position_ids)  # (cos, sin)

        # KV fusion (computed once, shared across all layers)
        cross_k, cross_v = self.kv_fusion(injection_features)
        # cross_k, cross_v: [B, 4, ctx_len, 64]

        # Prepare block mask for SDPA: [1, 1, S, S] for broadcasting
        if block_mask is not None:
            if block_mask.dim() == 2:
                block_mask = block_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, S, S]

        # Decoder layers
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                block_mask=block_mask,
                cross_k=cross_k,
                cross_v=cross_v,
                position_embeddings=position_embeddings,
            )

        # Final norm + LM head (frozen — may be on different device)
        hidden_states = self.norm(hidden_states)
        lm_head_device = self.lm_head.weight.device
        logits = self.lm_head(hidden_states.to(lm_head_device))  # [B, S, 262144]

        return logits


# ---------------------------------------------------------------------------
# Utility: Position-Weighted Loss
# ---------------------------------------------------------------------------

def make_position_weights(block_size: int = 16, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """
    Exponentially decaying position weights for DFlash loss.

    weight[k] = exp(-k / block_size) for k = 0..block_size-1
    weight[0] = 0 (no loss on anchor token)
    Normalized so sum = 1.

    Returns:
        weights: [block_size] tensor
    """
    k = torch.arange(block_size, dtype=torch.float32, device=device)
    weights = torch.exp(-k / block_size)
    weights[0] = 0.0  # No loss on anchor
    weights = weights / weights.sum()
    return weights


def make_sequence_position_weights(
    seq_len: int, block_size: int = 16, device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    """
    Tile block-level position weights across the full sequence.

    Returns:
        weights: [seq_len] tensor with repeated block weights
    """
    block_weights = make_position_weights(block_size, device)
    num_blocks = seq_len // block_size
    return block_weights.repeat(num_blocks)
