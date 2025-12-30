#!/usr/bin/env python3
"""
CLI script for merging steering vectors into model weights.

This script permanently modifies a model by merging steering vectors
into the MLP bias terms, creating a model with built-in steering behavior.

WARNING: This permanently modifies the model weights!
"""

import argparse
import sys
import os
import json

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from activation_steering import (
    merge_steering_into_model,
    verify_merged_model,
    load_best_layers_from_correlations
)


def main():
    parser = argparse.ArgumentParser(
        description="Merge steering vectors permanently into model weights",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Merge steering into specific layers
  python scripts/merge_steering.py \\
      --model Qwen/Qwen2.5-7B-Instruct \\
      --steering-vectors steering_vectors_wrmd.pt \\
      --layers 10 11 12 \\
      --alpha -2.0 \\
      --output-dir Qwen2.5-7B-Steered

  # Use best layers from correlation analysis
  python scripts/merge_steering.py \\
      --model Qwen/Qwen2.5-7B-Instruct \\
      --steering-vectors steering_vectors_wrmd.pt \\
      --correlations layer_correlations.json \\
      --top-k 3 \\
      --alpha -2.0 \\
      --output-dir Qwen2.5-7B-Steered

  # Verify a merged model
  python scripts/merge_steering.py \\
      --verify \\
      --merged-model Qwen2.5-7B-Steered \\
      --original-model Qwen/Qwen2.5-7B-Instruct \\
      --layers 10 11 12
        """
    )

    # Mode selection
    parser.add_argument("--verify", action="store_true",
                       help="Verify a previously merged model instead of merging")

    # Model paths
    parser.add_argument("--model", "--original-model",
                       help="Path to base/original HuggingFace model")
    parser.add_argument("--merged-model",
                       help="Path to merged model (for verification)")

    # Steering configuration
    parser.add_argument("--steering-vectors", required=False,
                       help="Path to steering vectors .pt file")
    parser.add_argument("--alpha", type=float, default=-2.0,
                       help="Steering coefficient (negative = reduce refusal, default: -2.0)")

    # Layer selection (mutually exclusive)
    layer_group = parser.add_mutually_exclusive_group()
    layer_group.add_argument("--layers", type=int, nargs="+",
                            help="Specific layer indices to modify")
    layer_group.add_argument("--correlations",
                            help="Path to layer_correlations.json (to use with --top-k)")

    parser.add_argument("--top-k", type=int, default=1,
                       help="Number of top layers to use from correlations (default: 1)")

    # Output
    parser.add_argument("--output-dir", "--output",
                       help="Directory to save merged model")

    args = parser.parse_args()

    # Verification mode
    if args.verify:
        if not args.merged_model or not args.model or not args.layers:
            print("[ERROR] Verification requires --merged-model, --model, and --layers", file=sys.stderr)
            return 1

        print("="*80)
        print("VERIFYING MERGED MODEL")
        print("="*80)

        results = verify_merged_model(
            model_dir=args.merged_model,
            original_model_path=args.model,
            steering_layers=args.layers
        )

        if results["verified"]:
            print("\n[OK] Verification successful!")
            return 0
        else:
            print("\n[WARN] Verification failed or incomplete!")
            print(f"[INFO] See results: {results}")
            return 1

    # Merge mode
    else:
        # Validate required arguments
        if not args.model:
            print("[ERROR] --model is required for merging", file=sys.stderr)
            return 1
        if not args.steering_vectors:
            print("[ERROR] --steering-vectors is required for merging", file=sys.stderr)
            return 1
        if not args.output_dir:
            print("[ERROR] --output-dir is required for merging", file=sys.stderr)
            return 1

        # Determine layers to modify
        if args.layers:
            target_layers = args.layers
            print(f"[INFO] Using manually specified layers: {target_layers}")
        elif args.correlations:
            best_layers = load_best_layers_from_correlations(args.correlations, args.top_k)
            if best_layers is None:
                print("[ERROR] Failed to load layer correlations", file=sys.stderr)
                return 1
            target_layers = best_layers
            print(f"[INFO] Using top-{args.top_k} layers from correlations: {target_layers}")
        else:
            print("[ERROR] Must specify either --layers or --correlations", file=sys.stderr)
            return 1

        print("\n" + "="*80)
        print("MERGING STEERING INTO MODEL WEIGHTS")
        print("="*80)
        print(f"WARNING: This will permanently modify the model!")
        print(f"Base model: {args.model}")
        print(f"Steering vectors: {args.steering_vectors}")
        print(f"Target layers: {target_layers}")
        print(f"Alpha: {args.alpha}")
        print(f"Output: {args.output_dir}")
        print("="*80)

        # Confirm with user
        try:
            response = input("\nContinue? [y/N]: ")
            if response.lower() != 'y':
                print("[INFO] Merge cancelled by user")
                return 0
        except KeyboardInterrupt:
            print("\n[INFO] Merge cancelled by user")
            return 0

        # Perform merge
        try:
            metadata = merge_steering_into_model(
                base_model_path=args.model,
                steering_vectors_file=args.steering_vectors,
                target_layers=target_layers,
                alpha=args.alpha,
                output_dir=args.output_dir
            )

            print("\n" + "="*80)
            print("MERGE COMPLETE")
            print("="*80)
            print(f"Modified model saved to: {args.output_dir}")
            print(f"Metadata saved to: {os.path.join(args.output_dir, 'steering_metadata.json')}")
            print("\nMetadata:")
            print(json.dumps(metadata, indent=2))
            print("="*80)

            return 0

        except Exception as e:
            print(f"\n[ERROR] Merge failed: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            return 1


if __name__ == "__main__":
    sys.exit(main())
