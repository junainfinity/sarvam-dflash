# Converting DFlash Draft Model to MLX for Mac Inference

## Overview

After training the DFlash draft model on GPU, convert it to MLX format for
efficient inference on Apple Silicon. The draft model is tiny (~275M trainable
params) and runs well on M-series chips.

## Step 1: Export Draft Weights to Safetensors

```python
import torch
from safetensors.torch import save_file

ckpt = torch.load("checkpoints/dflash_draft_best.pt", weights_only=False)
state_dict = {k: v.to(torch.float16) for k, v in ckpt["state_dict"].items()}
save_file(state_dict, "dflash_draft_weights.safetensors")
```

This exports only the ~275M trainable parameters (KV fusion + 6 layers + final norm).

## Step 2: Convert Sarvam-30B Shared Weights

Extract just the embedding and LM head from the target model:

```python
from safetensors.torch import load_file, save_file

# These are in the Sarvam-30B safetensors shards
# Check model.safetensors.index.json for which shard contains each tensor
shared_weights = {}

# Load word_embeddings from the appropriate shard
shard = load_file("sarvam-30b/model-00001-of-00026.safetensors")
shared_weights["word_embeddings"] = shard["model.word_embeddings.weight"].to(torch.float16)

# Load lm_head from its shard (check index.json)
shard = load_file("sarvam-30b/model-00026-of-00026.safetensors")
shared_weights["lm_head"] = shard["lm_head.weight"].to(torch.float16)

save_file(shared_weights, "dflash_shared_weights.safetensors")
```

## Step 3: Rewrite Draft Model in MLX

Create `dflash_draft_mlx.py`:

```python
import mlx.core as mx
import mlx.nn as nn
import math


class RMSNorm(nn.Module):
    def __init__(self, dims, eps=1e-6):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x):
        variance = mx.mean(x * x, axis=-1, keepdims=True)
        x = x * mx.rsqrt(variance + self.eps)
        return self.weight * x


class DFlashKVFusion(nn.Module):
    def __init__(self, injection_dim=20480, num_kv_heads=4, head_dim=64):
        super().__init__()
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.kv_proj = nn.Linear(injection_dim, 2 * num_kv_heads * head_dim, bias=False)

    def __call__(self, x):
        B, S, _ = x.shape
        kv = self.kv_proj(x)  # [B, S, 512]
        kv = kv.reshape(B, S, 2, self.num_kv_heads, self.head_dim)
        kv = mx.transpose(kv, (2, 0, 3, 1, 4))
        return kv[0], kv[1]  # cross_k, cross_v


class DFlashSelfAttention(nn.Module):
    def __init__(self, hidden=4096, num_heads=16, num_kv_heads=4, head_dim=64, eps=1e-6):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.query_key_value = nn.Linear(hidden, (num_heads + 2 * num_kv_heads) * head_dim, bias=False)
        self.dense = nn.Linear(num_heads * head_dim, hidden, bias=False)
        self.query_layernorm = RMSNorm(head_dim, eps)
        self.key_layernorm = RMSNorm(head_dim, eps)
        self.rope = nn.RoPE(head_dim, base=8_000_000)

    def __call__(self, x, mask=None):
        B, L, _ = x.shape
        qkv = self.query_key_value(x).reshape(B, L, -1, self.head_dim)
        q = qkv[:, :, :self.num_heads]
        k = qkv[:, :, self.num_heads:self.num_heads + self.num_kv_heads]
        v = qkv[:, :, self.num_heads + self.num_kv_heads:]
        q = self.query_layernorm(q)
        k = self.key_layernorm(k)
        q = self.rope(q)
        k = self.rope(k)
        out = mx.fast.scaled_dot_product_attention(
            q.transpose(0, 2, 1, 3), k.transpose(0, 2, 1, 3),
            v.transpose(0, 2, 1, 3), scale=self.scale, mask=mask
        )
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.dense(out)


# ... (similar pattern for CrossAttention, FFN, DecoderLayer, DraftModel)
```

## Step 4: Load Weights

```python
from safetensors import safe_open

def load_dflash_mlx(draft_path, shared_path):
    model = DFlashDraftModelMLX(...)

    # Load draft weights
    with safe_open(draft_path, framework="numpy") as f:
        for key in f.keys():
            # Map PyTorch key names to MLX model attributes
            ...

    # Load shared weights
    with safe_open(shared_path, framework="numpy") as f:
        model.word_embeddings = mx.array(f.get_tensor("word_embeddings"))
        model.lm_head_weight = mx.array(f.get_tensor("lm_head"))

    return model
```

## Step 5: Quantize (Optional)

```python
# Quantize draft model to 4-bit for faster inference
nn.quantize(model, group_size=64, bits=4)

# The shared embedding/head are large (1.07B each) - quantize those too
# or keep them in float16 if memory allows
```

## Step 6: Inference

```python
def speculative_decode(target_model, draft_model, prompt_tokens, num_blocks=4):
    """Generate tokens using DFlash speculative decoding."""
    # 1. Run target on prompt
    target_hidden = target_model(prompt_tokens, output_hidden_states=True)
    injection_feats = extract_injection_features(target_hidden)

    # 2. Draft generates block_size=16 tokens in parallel
    draft_logits = draft_model(prompt_tokens[-1:], injection_feats)
    draft_tokens = mx.argmax(draft_logits, axis=-1)  # [16] tokens

    # 3. Verify with target model (single forward pass on all 16 tokens)
    target_logits = target_model(draft_tokens)
    # Accept tokens where draft matches target (standard spec decode verification)
    ...
```

## Memory Requirements on Mac

| Component | FP16 | INT4 |
|-----------|------|------|
| Draft trainable (275M) | 0.55 GB | 0.14 GB |
| Shared embedding (1.07B) | 2.14 GB | 0.54 GB |
| Shared LM head (1.07B) | 2.14 GB | 0.54 GB |
| **Draft total** | **4.83 GB** | **1.22 GB** |

The draft model alone fits comfortably on any M-series Mac. For full speculative
decoding, you also need the target Sarvam-30B model (INT4: ~16 GB), which fits
on M2 Pro/Max/Ultra with 32+ GB unified memory.
