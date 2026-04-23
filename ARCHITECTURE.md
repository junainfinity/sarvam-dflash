# ARCHITECTURE.md — Sarvam-30B Internals, DFlash, and the Corrected Application Architecture

**Purpose:** a dense, honest reference for anyone building on this project. Captures everything we learned while working with Sarvam-30B, the block-diffusion speculative-decoding literature, and the application-level design that glues the two together. This is not a tutorial — it assumes familiarity with transformers, MoE, RoPE, and speculative decoding.

---

## 1. Sarvam-30B — target model internals

### 1.1 Top-level shape

- **Family:** Sarvam-M (HuggingFace id `sarvamai/sarvam-m`), 30B total parameters, **Mixture-of-Experts**.
- **Vocabulary:** 262,144 tokens. Multilingual coverage skewed toward Indic scripts (Devanagari, Tamil, Kannada, Bengali, …) plus English. The huge vocab is the reason sparse KL distillation (top-64) is essential — dense KL over the full vocab would dominate training memory.
- **Hidden dim:** 4,096.
- **Layers:** **19 decoder layers** only. Surprisingly shallow for a 30B model — the parameter budget sits in the MoE experts (many experts per layer, each with its own FFN), not in depth. Depth-19 is why injection layers `[0, 4, 9, 13, 18]` span the stack well with just 5 probes.
- **Head config:** 16 query heads, 4 KV heads (GQA 4:1), head_dim = 64.
- **RoPE:** θ = 8,000,000 (unusually large; consistent with long-context training). Uses the standard rotary position embedding applied to Q and K before attention.
- **Norm:** RMSNorm (`SarvamMoERMSNorm`, eps = 1e-6), pre-norm architecture.
- **Pre-training objective:** causal LM. Base model, **not instruction-tuned** — this is why naïve prompts like "Explain speculative decoding:" produce repetition loops or drift into other scripts.

### 1.2 Per-layer anatomy

Each decoder layer follows the pre-norm SwiGLU MoE pattern. Approximately:

```
x → RMSNorm → GQA self-attn (RoPE, QK-norm) → + residual
  → RMSNorm → MoE-FFN (router → top-k experts → weighted sum of expert outputs) → + residual
```

Key implications for draft-model design:
- **QK-norm** (normalization applied to queries and keys before attention) is standard in this family. Our draft mirrors it for consistency.
- **MoE router is on the target side only** — the draft never routes; it's dense.
- **Residual stream is hidden_dim = 4096** at every layer. This is what we extract for KV injection.

### 1.3 Hidden states indexing (important gotcha)

When calling `target.model(input_ids, output_hidden_states=True)`, the returned `hidden_states` is a tuple:

```
hidden_states[0]   = the embedding output (before any decoder layer)
hidden_states[1]   = output of decoder layer 0
hidden_states[2]   = output of decoder layer 1
...
hidden_states[19]  = output of decoder layer 18 (the last)
```

So our injection-layer indices `[0, 4, 9, 13, 18]` correspond to `hidden_states` indices `[1, 5, 10, 14, 19]`. The off-by-one bites if you forget. Constant lives in `modeling_sarvam_moe_dflash.py` as `KV_INJECTION_HIDDEN_STATE_INDICES`.

### 1.4 Weight storage

- **7,122 safetensors shards** when downloaded from HuggingFace. Each shard is small (a few MB). Loading takes ~1–2 min on NVMe (bottleneck is shard-count overhead, not total bytes).
- **Shared weights** we use from Sarvam in the draft:
  - `target.model.word_embeddings` — shape `[262144, 4096]`, ~1.07 B params
  - `target.lm_head` — shape `[4096, 262144]`, ~1.07 B params
  These are **frozen** (not trained) and **device-pinned** (not moved by draft's `.to(device)`).
- On MPS with `device_map="auto"`, Transformers may split the model across MPS and CPU. Our code handles this by querying `next(target.parameters()).device` and moving draft tensors to match.

### 1.5 Inference performance (M3 Max, MPS, bfloat16, batch = 1)

Measured in our quick_test.py runs:
- **~3–5 tok/s** for standard autoregressive generation at context lengths 9–100 tokens
- **~0.2–0.3 s per forward pass** at those context lengths
- Prefill (running on a prompt of ~10 tokens): ~1–2 s (first-token cost)
- Memory: comfortably fits in 128 GB unified memory in bf16; no quantization needed for correctness

### 1.6 Custom loader (`modeling_sarvam_moe_dflash.py`)

The Sarvam-M package on HuggingFace uses **relative imports** in its `modeling_sarvam_moe.py` (e.g., `from .configuration_sarvam_moe import …`). These break when imported from outside the package. We patch them at load time with `importlib.util.spec_from_file_location` + a custom finder that rewrites relative imports to absolute. See `modeling_sarvam_moe_dflash.py` top for the mechanism.

On top of that, we subclass to `SarvamMoEForCausalLMWithKVInjection`, which adds:
- `get_kv_injection_features(input_ids)` → `[B, S, 20480]` (5 layers concatenated)
- `get_teacher_logits(input_ids, top_k=64)` → top-k logits for sparse KL
- `get_injection_and_logits(input_ids)` — **single forward pass** that produces both (used in data generation to halve target-inference cost)

---

## 2. Block-diffusion language models — the foundation DFlash builds on

### 2.1 Why block diffusion exists

Two decoding regimes dominate language modeling:

| | Autoregressive (AR) | Diffusion (pure) |
|---|---|---|
| Decoding | One token at a time, left-to-right | Iteratively denoise all positions in parallel |
| Parallelism | None during generation | Full position parallelism at each step |
| Quality at equal compute | Gold standard | Slightly behind AR for text |
| KV cache | Yes, cheap | No, each step re-runs the whole model |
| Training | Standard causal CE | Denoising CE over randomly masked tokens |

**Block diffusion** is a compromise: the sequence is chopped into fixed-size blocks (e.g., 16 tokens). Blocks are generated **autoregressively** (block `k+1` depends on all of blocks `≤ k`). Within a block, tokens are generated **in parallel via denoising** — attention is **bidirectional within a block**, and the block is produced by filling in mask tokens given the block's anchor and prior context.

The parent paper: Arriola et al., "Block Diffusion" / BD3-LMs, arXiv 2503.09573 (ICLR 2025 Oral). BD3-LMs use a stochastic masking schedule with multiple denoising steps per block.

### 2.2 "Flash" = a single denoising step

The expensive part of diffusion is the iteration. BD3-LMs do `T` denoising steps per block (`T ≈ 8` in practice). DFlash's key observation: for **speculative decoding**, you don't need high-quality draft output — you just need *plausible enough* candidates for the target to verify. So DFlash **collapses the schedule to a single step**: one draft forward per block, argmax, done. If the target rejects, the algorithm falls back — you can afford a lossier draft. This is where the 4–6× speedups come from.

---

## 3. DFlash — corrected architecture

Primary source: `arXiv 2602.06036` (Chen / Liang / Liu, Z-Lab, Feb 2026). Secondary: `vllm-project/speculators` RFC #248 (most precise algorithmic spec), `github.com/z-lab/dflash` (reference impl), the Z-Lab blog. The points below are **what the paper actually says**, not what our original implementation did.

### 3.1 The learned mask embedding

A single learned parameter `m ∈ ℝ^{hidden_dim}` is added to the draft model. It represents the "unknown token" placed at every non-anchor position in the block, at both training and inference time. Not a vocabulary token — a pure embedding. This is what our original implementation lacked entirely.

### 3.2 Within-block attention: bidirectional

Our mask: `torch.tril(torch.ones(16,16))` — causal. **Wrong.** DFlash uses `torch.ones(16,16)` — every position in a block sees every other position. This is the whole point of mask-filling — position `i` needs context from positions on both sides (specifically, from the anchor at position 0 and from the other mask positions) to make a good guess.

Across blocks, the mask is still 0 (no cross-block attention in self-attn). Block `k` can only see its own anchor + masks.

### 3.3 Prediction alignment: same-position, not next-token

Our training: `shift_logits[:, :-1]` predicts `labels[:, 1:]` — the classic causal-LM shift. **Wrong for DFlash.** DFlash loss: `logit[i]` predicts `token[i]` — no shift. Reason: at inference position `i` holds the mask embedding, and the model must output the probability distribution of the actual token that belongs at position `i`. This is mask-filling / BERT-style, not next-token causal LM.

### 3.4 KV injection: concatenation into self-attention, not a separate cross-attn block

This is the subtle one. The RFC describes it precisely:

```
queries   = mask_token_embeddings         # [B, L, D]
keys      = [target_features | masks]     # [B, 2L, D]
values    = [target_features | masks]     # [B, 2L, D]
mask      = [L, 2L] bidirectional         # (not causal)
```

So the draft's "self-attention" actually operates over a **doubled K/V sequence**: half of K/V is the target's hidden features for the block positions, half is the draft's own mask-token embeddings. Queries attend across all 2L positions bidirectionally. This is one attention op per layer, not self-attn plus cross-attn.

Our implementation has **separate self-attn and cross-attn layers**. Functionally similar (the cross-attn keys are the projected target features, same information), but parameter-heavier. Keeping our cross-attn structure should still work if the other three mismatches are fixed; it's just not the most compute-efficient form.

### 3.5 Feature extraction: features of what?

In DFlash, features are the target's hidden states **at the block's positions**, captured during the target's own prefill on the accepted context. For a block at absolute positions `[P, P+1, …, P+15]`:

- During **prefill**, target sees tokens `[0, …, P−1]` and produces hidden states for those positions.
- But the target's hidden state at position `P−1` encodes information useful for predicting position `P` (that's how LMs work). So the target's last-layer hidden state at `P−1` (plus intermediate layer snapshots) *is* the conditioning signal the draft uses for the whole block.
- Specifically: in the single-step mask-fill formulation, features are sourced from the target's existing hidden states **up to and including the anchor position** (which is `P` once we've accepted `P−1` and target has run on `[0..P]`). Nothing from the future is ever available — and nothing from the future is needed for the draft to make its parallel guess.

### 3.6 Single-forward block generation at inference

```
anchor = last_accepted_token           # from target
draft_input_embeds = cat([embed(anchor),
                          m.expand(block_size-1, hidden_dim)], dim=0)  # [L, D]
features = target_hidden_features_up_to_anchor                          # [ctx, 20480]
logits = draft(input_embeds=draft_input_embeds,
               injection_features=features,
               block_mask=block_bidirectional_mask)                     # [L, V]
candidates = logits.argmax(-1)                                           # [L]
# target verifies in one forward; accept longest prefix
```

One draft forward per block. 4–6× published speedup comes from this + a high acceptance length (typical τ ≈ 5.8–6.5 on a 16-wide block).

### 3.7 Training regime (for our corrected retrain)

For each training sample of 2048 tokens:

1. Sample a random anchor position `P ∈ [0, 2048 − block_size]`.
2. **Inputs:** word-embed(token at `P`) at position 0 of the draft block; `m` at positions 1..15. In-sequence RoPE positions `[P, P+1, ..., P+15]`.
3. **Cross-attn conditioning:** slice `injection_features[:, :P+1, :]` — target features strictly up to (and including) the anchor. **Never** slice features beyond `P` — that's the leakage our original training had.
4. **Self-attn mask:** bidirectional `[L, L]` within the block.
5. **Loss:** CE(`logit[i]`, `token[P+i]`) for `i = 1..L−1`. Anchor (`i = 0`) has weight 0. Exponentially decaying position weights `w_k = exp(−(k−1)/γ)` for `k = 1..L−1`, with γ around 4–8 (tune). Same-position, no shift.
6. **Distillation:** sparse KL against teacher's top-64 logits (already stored per-position in our shards). KL weight 0.5 (unchanged from V1/V2).

**Important data reuse:** our existing 3.9 TB of shards remain valid. The teacher logits at each position were computed causally by Sarvam-30B on the full sequence, but what matters is that `injection_features[:, :P+1, :]` reflects only tokens `0..P` (which is true because the target is causal). We just **slice correctly at train time** and the leakage disappears — no regeneration needed.

---

## 4. Application architecture for Sarvam-30B specifically

This section describes the full runtime system tailored to Sarvam-30B on M3 Max.

### 4.1 Draft model sizing

- **6 layers, hidden_dim = 4,096** (matches Sarvam so we can share embed/head)
- **GQA 16:4** (matches target) — keeps KV-cache / KV-projection symmetric
- **SwiGLU FFN intermediate = 2,048** (half of Sarvam's per-expert size, roughly)
- **Per-layer KV fusion** (or concat-into-self-attn if we refactor): single linear `20,480 → 512` (2 × 4 heads × 64 dim) shared across layers
- **Learned mask embedding:** `nn.Parameter(torch.zeros(hidden_dim))` with `nn.init.normal_(std=0.02)`

**Total params:**
| Component | Count | Trainable? |
|---|---|---|
| Word embedding (shared) | 262,144 × 4,096 = 1.07 B | ❌ frozen |
| LM head (shared) | 4,096 × 262,144 = 1.07 B | ❌ frozen |
| Mask embedding | 4,096 | ✅ |
| KV fusion | 20,480 × 512 ≈ 10.5 M | ✅ |
| 6 × (self-attn + cross-attn + FFN + 3 norms) | ≈ 265 M | ✅ |
| **Trainable total** | **≈ 275 M** | |
| **All-up parameters** | **≈ 2.42 B** | (shared weights included) |

### 4.2 Memory layout on M3 Max (128 GB unified)

| Resident | Dtype | Size |
|---|---|---|
| Sarvam-30B | bf16 | ~60 GB |
| Draft trainable | bf16 | ~0.55 GB |
| Draft shared (embed + head) | bf16 | ~4.3 GB (but **same memory as target's — no duplication**) |
| Working tensors (ctx ~2048, grad_accum 8) | bf16 | a few GB |
| **Total working set during inference** | | **~65 GB, comfortable** |

Inference during speculative decoding peaks at ~65 GB. Training peaks higher because of optimizer state (Adam: ~1.6 GB extra) plus activations, but still fits well inside 128 GB.

### 4.3 Device-map gotchas

- `device_map="auto"` from Transformers may place Sarvam across MPS and CPU. Draft must query `next(target.parameters()).device` and move its own parameters to match, not assume MPS.
- Draft's cross-attention keys/values come from target — if target is on MPS and draft is on MPS, free; if target is on CPU, the features must be moved to MPS before the draft uses them (our `DFlashKVFusion` handles this implicitly via `.to(device)`).
- MPS does not support pinned memory. The `pin_memory=True` warning from DataLoader is benign; leave it as-is (ignoring is cheaper than branching).

### 4.4 Inference runtime loop

```
ids = tokenize(prompt)
loop until max_new_tokens:
    # --- one speculative round ---
    out = target.model(ids, output_hidden_states=True)
    features = concat(hidden_states[i] for i in KV_INJECTION_HIDDEN_STATE_INDICES)  # [1, ctx, 20480]
    anchor   = argmax(target.lm_head(out.last_hidden_state[:, -1, :]))              # scalar

    draft_in = cat([embed(anchor), mask_embedding × 15])                            # [1, 16, 4096]
    logits   = draft(input_embeds=draft_in,
                     injection_features=features[:, :ctx, :],
                     block_mask=block_bidirectional)                                # [1, 16, V]
    cand     = logits.argmax(-1)                                                    # [1, 16]

    verify   = target(cat([ids, cand]))
    tgt_pred = verify.logits[:, ctx-1:ctx+15, :].argmax(-1)                         # [1, 16]

    # accept longest matching prefix (anchor always matches trivially)
    n = longest_match(cand, tgt_pred)
    ids = cat([ids, cand[:n]])
```

Expected numbers with a correctly trained DFlash on Sarvam-30B (extrapolating from the paper's Qwen3-8B results, discounted for Sarvam's lower baseline tok/s):
- Baseline: ~4 tok/s
- DFlash: ~15–25 tok/s (3.5–6× speedup)
- Acceptance length: 5–7 tokens per 16-wide block

### 4.5 Training runtime loop (corrected)

```
for batch in dataloader:
    ids, feats, teach_logits, teach_idx = batch
    B, S = ids.shape  # S = 2048

    # Sample a random anchor per sample
    P = torch.randint(0, S - block_size + 1, (B,))

    # Build draft input: anchor at pos 0, mask embedding at pos 1..L-1
    block_ids = stack([ids[b, P[b]:P[b]+block_size] for b in range(B)])  # [B, L]
    input_embeds = embed(block_ids)                                      # [B, L, D]
    input_embeds[:, 1:, :] = mask_embedding                              # broadcast

    # Slice features to context-up-to-anchor
    # (features tensor is [B, S, 20480]; we slice each sample's P+1 prefix)
    # Use a packed + cu_seqlens-style batch, OR pad to max P+1 and mask.
    feats_ctx = pad_and_mask(feats, P + 1)                               # [B, max_P+1, 20480]

    # Forward
    logits = draft(input_embeds=input_embeds,
                   injection_features=feats_ctx,
                   block_mask=block_bidirectional)                       # [B, L, V]

    # Same-position loss, anchor weight 0, exp-decay weights
    labels = block_ids                                                   # [B, L]
    w = exp(-(arange(L) - 1) / gamma);  w[0] = 0                         # [L]
    ce = cross_entropy(logits, labels, reduction='none')                 # [B, L]
    ce_loss = (ce * w).sum() / (w.sum() * B)

    # Sparse KL against teacher top-64
    kl = sparse_kl(logits, teach_logits, teach_idx)                      # [B, L]
    kl_loss = (kl * w).sum() / (w.sum() * B)

    loss = 0.5 * ce_loss + 0.5 * kl_loss
    loss.backward()
```

Per-step compute is ~128× lighter than our original full-sequence training (16 tokens/sample instead of 2048). To match the effective gradient signal, either: (a) run ~128× more steps, or (b) sample multiple anchors per batch element. (b) is cleaner — sample 8 random anchors per sample, treat as micro-batches of 8B, one forward each. 4 epochs of this should be comparable in quality to the original 4-epoch run.

Wall-clock estimate on M3 Max: ~3–5 days with `batch_size = 2, grad_accum = 8, 8 anchors/sample`. Stick with `AdamW(β1=0.9, β2=0.95, wd=0.1)`, cosine LR decay with 5% warmup, lr = 6e-4. Those hyperparameters were fine; it was only the objective that was wrong.

---

## 5. Learnings we carry forward

- **Read the paper before implementing.** We implemented an architecture that sounded right but never matched any published system. One careful afternoon with the DFlash paper PDF would have prevented four days of wasted training.
- **Acceptance rate is the only metric that validates a speculative-decoding draft.** Training loss going down is not a proof of correctness — it just proves you're training *something*. The moment training saves a first checkpoint, run a 30-second inference smoke test and check acceptance. If it's 0%, something is structurally wrong, and more training won't help.
- **Name your masks after their behavior, not their shape.** Calling a function `make_block_causal_mask` baked a wrong assumption into every caller. Rename the corrected version `make_block_bidirectional_mask` so the next person reads the intent off the call-site.
- **Verify data doesn't leak.** Any time target-model hidden states are fed into a draft as conditioning, verify that slicing or masking prevents the draft from attending to positions beyond the one it's predicting. The causal structure of transformer hidden states gives you free leakage-prevention *only if you actually slice*.
- **Sarvam-30B is a remarkably forgiving target.** Its very large vocab, multilingual distribution, and base-model (non-instruct) nature mean even a good draft will be verified against a target that itself loops on out-of-distribution English prompts. The benchmark must use prompts Sarvam is known to handle (Indic-language factual queries, shorter continuations) to get meaningful acceptance-length numbers.
- **Unified memory is a real advantage.** Being able to hold the full 30B target + draft + activations in one bf16 address space removes a whole class of engineering — no tensor-parallel slicing, no PCIe-boundary considerations. The speedup math becomes honest: saved target forwards directly reduce wall-clock.

---

## 6. Related reading

Essential:
- DFlash: Block Diffusion for Flash Speculative Decoding — Chen, Liang, Liu, 2026. [`arXiv 2602.06036`](https://arxiv.org/abs/2602.06036).
- Block Diffusion (BD3-LMs) — Arriola et al., ICLR 2025 Oral. [`arXiv 2503.09573`](https://arxiv.org/abs/2503.09573).
- Speculators RFC #248 — the most precise public algorithmic description. [github.com/vllm-project/speculators/issues/248](https://github.com/vllm-project/speculators/issues/248).
- z-lab/dflash — reference implementation. [github.com/z-lab/dflash](https://github.com/z-lab/dflash).
- DDTree — multi-candidate tree decoding built on DFlash's per-position marginals. [`arXiv 2604.12989`](https://arxiv.org/abs/2604.12989).

Comparators worth reading but not directly applicable:
- EAGLE-3 — autoregressive draft with target-feature fusion into the draft's token embeddings (not KV). Different shape but same family of ideas.
- Medusa, Hydra, Kangaroo — shallow multi-head drafts; no mask-fill.
- SEDD, DiffuSeq, DiffuLLaMA — pure diffusion LMs; DFlash is a deliberate one-step collapse of this class.

Not applicable (but cited by mistake in our original README):
- Hong et al., "Training Domain Draft Models for Speculative Decoding" — [`arXiv 2503.07807`](https://arxiv.org/abs/2503.07807). Domain distillation for generic autoregressive drafters, unrelated to block diffusion.
