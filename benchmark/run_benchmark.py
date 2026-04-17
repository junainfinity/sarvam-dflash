#!/usr/bin/env python3
"""
DFlash Draft Model Benchmark for Sarvam-30B.

Runs two modes:
  1. Baseline: standard autoregressive generation with Sarvam-30B
  2. DFlash: speculative decoding with draft model + target verification

Measures per-question:
  - Time to First Token (TTFT)
  - Total generation time
  - Tokens per second (tok/s)
  - Total tokens generated
  - Output text
  - DFlash-specific: acceptance rate, draft tokens/step, speedup vs baseline

Outputs: JSON log + human-readable summary.

Usage:
    python3 run_benchmark.py --mode baseline   # Sarvam-30B only
    python3 run_benchmark.py --mode dflash     # DFlash speculative decoding
    python3 run_benchmark.py --mode both       # Run both sequentially
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from mmlu_questions import get_all_prompts, MMLU_QUESTIONS


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class InferenceMetrics:
    mode: str = ""                   # "baseline" or "dflash"
    subject: str = ""
    question_idx: int = 0
    prompt_tokens: int = 0
    generated_tokens: int = 0
    total_tokens: int = 0
    time_to_first_token_ms: float = 0.0
    total_time_s: float = 0.0
    tokens_per_second: float = 0.0
    prompt_tokens_per_second: float = 0.0
    output_text: str = ""
    correct_answer: str = ""

    # Generation config
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 1
    max_new_tokens: int = 128

    # DFlash-specific
    draft_forward_passes: int = 0
    target_forward_passes: int = 0
    total_draft_tokens_proposed: int = 0
    total_draft_tokens_accepted: int = 0
    acceptance_rate: float = 0.0
    avg_accepted_per_step: float = 0.0
    speedup_vs_baseline: float = 0.0

    # System info
    device: str = ""
    peak_memory_mb: float = 0.0


def print_metrics_table(metrics_list: List[InferenceMetrics]):
    """Print a formatted summary table."""
    print("\n" + "=" * 100)
    print(f"{'Mode':<10} {'Subject':<22} {'Tokens':<8} {'TTFT(ms)':<10} {'Time(s)':<9} {'Tok/s':<8} {'Accept%':<9} {'Speedup':<8}")
    print("-" * 100)
    for m in metrics_list:
        accept = f"{m.acceptance_rate:.1%}" if m.mode == "dflash" else "N/A"
        speedup = f"{m.speedup_vs_baseline:.2f}x" if m.speedup_vs_baseline > 0 else "N/A"
        print(
            f"{m.mode:<10} {m.subject:<22} {m.generated_tokens:<8} "
            f"{m.time_to_first_token_ms:<10.1f} {m.total_time_s:<9.2f} "
            f"{m.tokens_per_second:<8.1f} {accept:<9} {speedup:<8}"
        )
    print("=" * 100)

    # Averages
    for mode in ["baseline", "dflash"]:
        group = [m for m in metrics_list if m.mode == mode]
        if not group:
            continue
        avg_ttft = sum(m.time_to_first_token_ms for m in group) / len(group)
        avg_tps = sum(m.tokens_per_second for m in group) / len(group)
        avg_time = sum(m.total_time_s for m in group) / len(group)
        total_tokens = sum(m.generated_tokens for m in group)
        print(f"\n  {mode.upper()} AVERAGES:")
        print(f"    Avg TTFT:     {avg_ttft:.1f} ms")
        print(f"    Avg tok/s:    {avg_tps:.1f}")
        print(f"    Avg time:     {avg_time:.2f}s")
        print(f"    Total tokens: {total_tokens}")
        if mode == "dflash":
            avg_accept = sum(m.acceptance_rate for m in group) / len(group)
            avg_speedup = sum(m.speedup_vs_baseline for m in group) / len(group) if group[0].speedup_vs_baseline > 0 else 0
            print(f"    Avg accept:   {avg_accept:.1%}")
            if avg_speedup > 0:
                print(f"    Avg speedup:  {avg_speedup:.2f}x")


# ---------------------------------------------------------------------------
# Baseline: Standard Autoregressive Sarvam-30B
# ---------------------------------------------------------------------------

def run_baseline(
    model, tokenizer, prompts, device,
    max_new_tokens=128, temperature=0.0, top_p=1.0, top_k=1,
) -> List[InferenceMetrics]:
    """Run standard autoregressive generation with Sarvam-30B."""
    results = []

    for idx, (subject, prompt, answer) in enumerate(prompts):
        print(f"\n[Baseline {idx+1}/{len(prompts)}] {subject}...", flush=True)

        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
        prompt_len = input_ids.shape[1]

        # Warmup MPS
        if idx == 0 and device.type == "mps":
            with torch.no_grad():
                _ = model(input_ids[:, :2])
                if device.type == "mps":
                    torch.mps.synchronize()

        generated_ids = input_ids.clone()
        first_token_time = None
        t_start = time.perf_counter()

        with torch.no_grad():
            for step in range(max_new_tokens):
                outputs = model(generated_ids)
                if device.type == "mps":
                    torch.mps.synchronize()

                if first_token_time is None:
                    first_token_time = time.perf_counter()

                next_logits = outputs.logits[:, -1, :]  # [1, V]

                if temperature == 0.0:
                    next_token = next_logits.argmax(dim=-1, keepdim=True)
                else:
                    next_logits = next_logits / temperature
                    if top_k > 0:
                        topk_vals, topk_idx = next_logits.topk(top_k, dim=-1)
                        next_logits = torch.full_like(next_logits, float("-inf"))
                        next_logits.scatter_(1, topk_idx, topk_vals)
                    probs = torch.softmax(next_logits, dim=-1)
                    next_token = torch.multinomial(probs, 1)

                generated_ids = torch.cat([generated_ids, next_token], dim=1)

                # Check EOS
                if next_token.item() == tokenizer.eos_token_id:
                    break

        t_end = time.perf_counter()
        if device.type == "mps":
            torch.mps.synchronize()

        gen_tokens = generated_ids.shape[1] - prompt_len
        total_time = t_end - t_start
        ttft = (first_token_time - t_start) * 1000 if first_token_time else 0
        tps = gen_tokens / total_time if total_time > 0 else 0

        output_text = tokenizer.decode(generated_ids[0, prompt_len:], skip_special_tokens=True)

        m = InferenceMetrics(
            mode="baseline",
            subject=subject,
            question_idx=idx,
            prompt_tokens=prompt_len,
            generated_tokens=gen_tokens,
            total_tokens=prompt_len + gen_tokens,
            time_to_first_token_ms=ttft,
            total_time_s=total_time,
            tokens_per_second=tps,
            prompt_tokens_per_second=prompt_len / (ttft / 1000) if ttft > 0 else 0,
            output_text=output_text[:200],
            correct_answer=answer,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            device=str(device),
        )
        results.append(m)
        print(f"  {gen_tokens} tokens in {total_time:.2f}s ({tps:.1f} tok/s), TTFT={ttft:.0f}ms", flush=True)

    return results


# ---------------------------------------------------------------------------
# DFlash: Speculative Decoding with Draft Model
# ---------------------------------------------------------------------------

def run_dflash(
    target_model, draft_model, tokenizer, prompts, device,
    block_size=16, max_new_tokens=128, temperature=0.0,
    baseline_results=None,
) -> List[InferenceMetrics]:
    """
    Run DFlash block speculative decoding.

    Algorithm:
    1. Target forward on current context → extract KV injection features
    2. Draft generates block_size tokens in parallel (using injection features)
    3. Target verifies all block_size candidates in one forward pass
    4. Accept longest matching prefix
    5. Append accepted tokens, repeat
    """
    from dflash_draft import make_block_causal_mask
    from modeling_sarvam_moe_dflash import KV_INJECTION_HIDDEN_STATE_INDICES

    results = []

    for idx, (subject, prompt, answer) in enumerate(prompts):
        print(f"\n[DFlash {idx+1}/{len(prompts)}] {subject}...", flush=True)

        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
        prompt_len = input_ids.shape[1]
        generated_ids = input_ids.clone()

        first_token_time = None
        draft_passes = 0
        target_passes = 0
        total_proposed = 0
        total_accepted = 0

        # Pre-compute block mask for draft
        block_mask = make_block_causal_mask(block_size, block_size, dtype=torch.bfloat16, device=device)

        t_start = time.perf_counter()

        with torch.no_grad():
            tokens_generated = 0

            while tokens_generated < max_new_tokens:
                # --- Step 1: Target forward on current context ---
                target_outputs = target_model.model(
                    input_ids=generated_ids,
                    output_hidden_states=True,
                    use_cache=False,
                    return_dict=True,
                )
                target_passes += 1

                # Extract injection features from 5 layers
                selected = [target_outputs.hidden_states[i] for i in KV_INJECTION_HIDDEN_STATE_INDICES]
                injection_feats = torch.cat(selected, dim=-1)  # [1, ctx_len, 20480]

                # Target's prediction for the last token (our first guaranteed token)
                target_logits_last = target_model.lm_head(target_outputs.last_hidden_state[:, -1:, :])
                if temperature == 0.0:
                    first_token = target_logits_last.argmax(dim=-1)  # [1, 1]
                else:
                    first_token = torch.multinomial(
                        torch.softmax(target_logits_last.squeeze(1) / temperature, dim=-1), 1
                    )

                if first_token_time is None:
                    if device.type == "mps":
                        torch.mps.synchronize()
                    first_token_time = time.perf_counter()

                if first_token.item() == tokenizer.eos_token_id:
                    generated_ids = torch.cat([generated_ids, first_token], dim=1)
                    tokens_generated += 1
                    break

                # --- Step 2: Draft generates block_size-1 more candidates ---
                # Draft input: the first_token + (block_size-1) positions to predict
                # We feed the draft the first token and let it predict the rest autoregressively
                # But in DFlash, the draft predicts ALL block_size tokens in parallel
                # Given the first token as anchor, predict tokens 2..block_size

                # Build draft input: start with first_token, pad remaining with zeros
                draft_input = torch.zeros(1, block_size, dtype=torch.long, device=device)
                draft_input[0, 0] = first_token.item()

                # For teacher forcing during speculation, we need to do iterative draft
                # Actually in DFlash parallel mode: feed anchor, get all predictions at once
                draft_logits = draft_model(
                    input_ids=draft_input,
                    injection_features=injection_feats[:, :block_size, :] if injection_feats.shape[1] >= block_size
                        else torch.nn.functional.pad(injection_feats, (0, 0, 0, block_size - injection_feats.shape[1])),
                    block_mask=block_mask,
                )  # [1, block_size, V]
                draft_passes += 1

                if temperature == 0.0:
                    draft_tokens = draft_logits.argmax(dim=-1)  # [1, block_size]
                else:
                    B, S, V = draft_logits.shape
                    probs = torch.softmax(draft_logits.float().reshape(-1, V) / temperature, dim=-1)
                    draft_tokens = torch.multinomial(probs, 1).reshape(B, S)

                # Build candidate sequence: first_token + draft predictions for positions 1..block_size-1
                candidates = torch.cat([first_token, draft_tokens[:, 1:]], dim=1)  # [1, block_size]
                total_proposed += block_size - 1  # first token is guaranteed

                # --- Step 3: Target verifies candidates ---
                verify_input = torch.cat([generated_ids, candidates], dim=1)
                verify_outputs = target_model(
                    input_ids=verify_input,
                    use_cache=False,
                    return_dict=True,
                )
                target_passes += 1

                # Target's predictions at positions where we placed candidates
                # For each candidate position i, target predicts from position (ctx_len + i - 1)
                ctx_len = generated_ids.shape[1]
                verify_logits = verify_outputs.logits[:, ctx_len - 1 : ctx_len + block_size - 1, :]

                if temperature == 0.0:
                    target_predictions = verify_logits.argmax(dim=-1)  # [1, block_size]
                else:
                    B, S, V = verify_logits.shape
                    probs = torch.softmax(verify_logits.float().reshape(-1, V) / temperature, dim=-1)
                    target_predictions = torch.multinomial(probs, 1).reshape(B, S)

                # --- Step 4: Accept longest matching prefix ---
                accepted = 0
                for i in range(block_size):
                    if candidates[0, i].item() == target_predictions[0, i].item():
                        accepted += 1
                    else:
                        # Accept up to here, use target's token at rejection point
                        break

                if accepted > 0:
                    accepted_tokens = candidates[:, :accepted]
                else:
                    accepted_tokens = target_predictions[:, :1]
                    accepted = 1

                total_accepted += accepted - 1  # subtract the guaranteed first token

                generated_ids = torch.cat([generated_ids, accepted_tokens], dim=1)
                tokens_generated += accepted

                if device.type == "mps":
                    torch.mps.synchronize()

                # Check for EOS in accepted tokens
                if tokenizer.eos_token_id in accepted_tokens[0].tolist():
                    break

        t_end = time.perf_counter()
        if device.type == "mps":
            torch.mps.synchronize()

        gen_tokens = generated_ids.shape[1] - prompt_len
        total_time = t_end - t_start
        ttft = (first_token_time - t_start) * 1000 if first_token_time else 0
        tps = gen_tokens / total_time if total_time > 0 else 0
        accept_rate = total_accepted / max(total_proposed, 1)

        output_text = tokenizer.decode(generated_ids[0, prompt_len:], skip_special_tokens=True)

        # Compute speedup vs baseline
        speedup = 0.0
        if baseline_results and idx < len(baseline_results):
            baseline_tps = baseline_results[idx].tokens_per_second
            if baseline_tps > 0:
                speedup = tps / baseline_tps

        m = InferenceMetrics(
            mode="dflash",
            subject=subject,
            question_idx=idx,
            prompt_tokens=prompt_len,
            generated_tokens=gen_tokens,
            total_tokens=prompt_len + gen_tokens,
            time_to_first_token_ms=ttft,
            total_time_s=total_time,
            tokens_per_second=tps,
            prompt_tokens_per_second=prompt_len / (ttft / 1000) if ttft > 0 else 0,
            output_text=output_text[:200],
            correct_answer=answer,
            temperature=temperature,
            top_p=1.0,
            top_k=1 if temperature == 0 else 0,
            max_new_tokens=max_new_tokens,
            draft_forward_passes=draft_passes,
            target_forward_passes=target_passes,
            total_draft_tokens_proposed=total_proposed,
            total_draft_tokens_accepted=total_accepted,
            acceptance_rate=accept_rate,
            avg_accepted_per_step=total_accepted / max(draft_passes, 1),
            speedup_vs_baseline=speedup,
            device=str(device),
        )
        results.append(m)
        print(
            f"  {gen_tokens} tokens in {total_time:.2f}s ({tps:.1f} tok/s), "
            f"TTFT={ttft:.0f}ms, accept={accept_rate:.0%}, "
            f"speedup={speedup:.2f}x" if speedup > 0 else "",
            flush=True,
        )

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="DFlash Benchmark")
    parser.add_argument("--mode", choices=["baseline", "dflash", "both"], default="both")
    parser.add_argument("--target_model_path", default="./sarvam-30b")
    parser.add_argument("--draft_checkpoint", default="./checkpoints/dflash_draft_best.pt")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num_questions", type=int, default=10)
    parser.add_argument("--output_dir", default="./benchmark/results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")

    prompts = get_all_prompts()[:args.num_questions]
    all_results = []

    # -----------------------------------------------------------------------
    # Load target model (needed for both modes)
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("Loading Sarvam-30B target model...")
    print("=" * 60, flush=True)

    from transformers import AutoTokenizer
    from modeling_sarvam_moe_dflash import SarvamMoEForCausalLMWithKVInjection, SarvamMoEConfig

    tokenizer = AutoTokenizer.from_pretrained(args.target_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    target_config = SarvamMoEConfig.from_pretrained(args.target_model_path)
    target_model = SarvamMoEForCausalLMWithKVInjection.from_pretrained(
        args.target_model_path,
        config=target_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    target_model.eval()
    print("Target model loaded.", flush=True)

    first_device = next(target_model.parameters()).device

    # -----------------------------------------------------------------------
    # Baseline
    # -----------------------------------------------------------------------
    baseline_results = None
    if args.mode in ["baseline", "both"]:
        print("\n" + "=" * 60)
        print("BASELINE: Standard Autoregressive Generation")
        print("=" * 60, flush=True)

        baseline_results = run_baseline(
            target_model, tokenizer, prompts, first_device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        all_results.extend(baseline_results)

    # -----------------------------------------------------------------------
    # DFlash
    # -----------------------------------------------------------------------
    if args.mode in ["dflash", "both"]:
        print("\n" + "=" * 60)
        print("DFLASH: Speculative Decoding with Draft Model")
        print("=" * 60, flush=True)

        # Load draft model
        from dflash_draft import DFlashDraftModel
        from train_dflash_sarvam import load_draft_from_checkpoint

        draft_model = load_draft_from_checkpoint(args.draft_checkpoint, target_model)
        draft_device = first_device
        draft_model.kv_fusion = draft_model.kv_fusion.to(draft_device, dtype=torch.bfloat16)
        draft_model.layers = draft_model.layers.to(draft_device, dtype=torch.bfloat16)
        draft_model.norm = draft_model.norm.to(draft_device, dtype=torch.bfloat16)
        draft_model.rotary_emb = draft_model.rotary_emb.to(draft_device)
        draft_model.eval()
        print(f"Draft model loaded from {args.draft_checkpoint}", flush=True)

        dflash_results = run_dflash(
            target_model, draft_model, tokenizer, prompts, first_device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            baseline_results=baseline_results,
        )
        all_results.extend(dflash_results)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print_metrics_table(all_results)

    # Save results
    results_path = os.path.join(args.output_dir, "benchmark_results.json")
    with open(results_path, "w") as f:
        json.dump([asdict(m) for m in all_results], f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
