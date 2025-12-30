#!/usr/bin/env python3
"""
CLI script for testing steering vectors interactively.

This script applies steering vectors at runtime and compares baseline vs
steered model outputs.
"""

import argparse
import sys
import os

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from activation_steering import (
    test_steering,
    load_best_layers_from_correlations,
    load_actual_refusal_prompts
)


def main():
    parser = argparse.ArgumentParser(
        description="Test steering with correlation-informed layer selection"
    )
    parser.add_argument("--model",
                       default="../ai-backends/models/huggingface/Qwen/Qwen3-8B-FP8")
    parser.add_argument("--steering-vectors",
                       default="steering_vectors_md.pt")
    parser.add_argument("--correlations",
                       default="layer_correlations.json")
    parser.add_argument("--results-dir",
                       default="LLM-Refusal-Evaluation/results/qwen3_8b_baseline")

    # Layer selection modes
    layer_group = parser.add_mutually_exclusive_group()
    layer_group.add_argument("--top-k", type=int, default=None,
                            help="Test top K layers individually")
    layer_group.add_argument("--layers", type=int, nargs="+", default=None,
                            help="Manually specify layers to test together")

    parser.add_argument("--alpha", type=float, default=-2.0,
                       help="Steering coefficient (negative = reduce refusal)")
    parser.add_argument("--num-prompts", type=int, default=3)
    parser.add_argument("--min-refusal-score", type=float, default=0.5)
    parser.add_argument("--max-tokens", type=int, default=500,
                       help="Max tokens to generate")

    args = parser.parse_args()

    # Load prompts
    print("[LOAD] Loading prompts that refused in baseline evaluation...")
    refusal_prompts = load_actual_refusal_prompts(
        args.results_dir,
        min_score=args.min_refusal_score,
        max_prompts=args.num_prompts
    )

    if not refusal_prompts:
        print(f"❌ No prompts found with refusal score > {args.min_refusal_score}")
        sys.exit(1)

    print(f"[OK] Found {len(refusal_prompts)} high-refusal prompts\n")

    # Determine which layers to test
    layer_configs = []

    if args.layers is not None:
        layer_configs = [args.layers]
        print(f"[TARGET] Manual mode: Testing layers {args.layers} together\n")

    elif args.top_k is not None:
        best_layers = load_best_layers_from_correlations(args.correlations, args.top_k)
        if best_layers is None:
            sys.exit(1)
        layer_configs = [[layer] for layer in best_layers]
        print(f"\n[TARGET] Top-{args.top_k} mode: Testing {len(layer_configs)} layers individually\n")

    else:
        best_layers = load_best_layers_from_correlations(args.correlations, top_k=1)
        if best_layers is None:
            sys.exit(1)
        layer_configs = [[best_layers[0]]]
        print(f"\n[TARGET] Default mode: Testing only best layer ({best_layers[0]})\n")

    # Run tests
    test_steering(
        model_name=args.model,
        steering_file=args.steering_vectors,
        layer_configs=layer_configs,
        alpha=args.alpha,
        test_prompts=refusal_prompts,
        max_new_tokens=args.max_tokens
    )

    print("\n" + "="*80)
    print("[OK] All tests complete!")
    print("="*80)


if __name__ == "__main__":
    main()
