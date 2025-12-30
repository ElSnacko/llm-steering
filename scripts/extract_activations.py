#!/usr/bin/env python3
"""
CLI script for extracting activations from language models.

This script provides a command-line interface to the ActivationExtractor class.
"""

import argparse
import sys
import os

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from activation_steering import (
    ActivationExtractor,
    load_prompts_from_judge_scores,
    analyze_dataset_quality,
    extract_model_name,
    generate_run_id,
    setup_model_run_dirs,
    ensure_dir
)


def main():
    parser = argparse.ArgumentParser(description="Extract activations using judge scores")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--results-dir", required=True,
                        help="Path to evaluation results (e.g., LLM-Refusal-Evaluation/results/qwen3_8b_baseline)")
    parser.add_argument("--output", default="activations_qwen_7b_judged.pt")
    parser.add_argument("--output-dir", default=None,
                        help="Custom output directory (default: outputs/{model_name}/{run_id}/extract_activations/)")
    parser.add_argument("--run-id", default=None,
                        help="Custom run ID (default: auto-generated timestamp)")
    parser.add_argument("--refusal-threshold", type=float, default=0.1,
                        help="Judge score above this = refusal")
    parser.add_argument("--compliance-threshold", type=float, default=-0.1,
                        help="Judge score below this = compliant")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    # Set up output directory
    if args.output_dir is None:
        model_name = extract_model_name(args.model)
        run_id = args.run_id or generate_run_id()
        output_dirs = setup_model_run_dirs(model_name=model_name, run_id=run_id)
        output_dir = output_dirs['extract_activations']
        print(f"📁 Using output directory: {output_dir}")
        print(f"   Model: {model_name}")
        print(f"   Run ID: {run_id}")
    else:
        output_dir = ensure_dir(args.output_dir)

    # Load prompts based on actual judge scores
    prompts, labels, metadata = load_prompts_from_judge_scores(
        args.results_dir,
        args.refusal_threshold,
        args.compliance_threshold
    )

    # Analyze dataset quality
    analyze_dataset_quality(metadata)

    # Optional: limit samples
    if args.max_samples:
        print(f"\n[WARN]  Limiting to {args.max_samples} samples for testing")
        prompts = prompts[:args.max_samples]
        labels = labels[:args.max_samples]
        metadata = metadata[:args.max_samples]

    # Extract activations
    extractor = ActivationExtractor(args.model)
    extractor.extract_dataset(prompts, labels, args.output, metadata, output_dir)

    # Determine the actual output path
    if not os.path.isabs(args.output):
        actual_output_path = os.path.join(output_dir, args.output)
    else:
        actual_output_path = args.output

    print(f"\n[OK] Done! Activations saved to {actual_output_path}")


if __name__ == "__main__":
    main()
