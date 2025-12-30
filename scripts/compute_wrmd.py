#!/usr/bin/env python3
"""
CLI script for computing steering vectors using WRMD and related methods.

This script provides a command-line interface to the WRMDCalculator class.
"""

import argparse
import sys
import os

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from activation_steering import (
    WRMDCalculator,
    compare_methods,
    extract_model_name,
    generate_run_id,
    setup_model_run_dirs,
    ensure_dir
)


def main():
    parser = argparse.ArgumentParser(description="Compute WRMD steering vectors")
    parser.add_argument("--activations", required=True)
    parser.add_argument("--output", default="steering_vectors_wrmd.pt")
    parser.add_argument("--output-dir", default=None,
                        help="Custom output directory (default: outputs/{model_name}/{run_id}/compute_wrmd/)")
    parser.add_argument("--run-id", default=None,
                        help="Custom run ID (default: auto-generated timestamp)")
    parser.add_argument("--method", choices=['md', 'rmd', 'wrmd'], default='wrmd')
    parser.add_argument("--lambda-ridge", type=float, default=0.1)
    parser.add_argument("--no-score-weighting", action="store_true",
                        help="Don't weight by judge scores (uniform weights)")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--compare-methods", action="store_true")
    args = parser.parse_args()

    # Set up output directory
    if args.output_dir is None:
        # Try to extract model name from activation file path
        model_name = "unknown_model"
        if '/' in args.activations:
            path_parts = args.activations.split('/')
            for part in path_parts:
                if part and not part.startswith('outputs') and not part.startswith('extract_activations'):
                    model_name = part
                    break

        run_id = args.run_id or generate_run_id()
        output_dirs = setup_model_run_dirs(model_name=model_name, run_id=run_id)
        output_dir = output_dirs['compute_wrmd']
        print(f"📁 Using output directory: {output_dir}")
        print(f"   Model: {model_name}")
        print(f"   Run ID: {run_id}")
    else:
        output_dir = ensure_dir(args.output_dir)

    calculator = WRMDCalculator(args.activations)

    if args.compare_methods:
        all_vectors = compare_methods(calculator, args.lambda_ridge, output_dir)

        # Save the best one (WRMD with score weighting)
        calculator.save_vectors(
            all_vectors['WRMD (weighted)'],
            args.output,
            method='wrmd',
            lambda_ridge=args.lambda_ridge,
            use_score_weighting=True,
            output_dir=output_dir
        )
    else:
        vectors = calculator.compute_steering_vectors(
            args.method,
            args.lambda_ridge,
            use_score_weighting=not args.no_score_weighting,
            normalize=args.normalize
        )

        calculator.analyze_vectors(vectors, output_dir)

        calculator.save_vectors(
            vectors,
            args.output,
            method=args.method,
            lambda_ridge=args.lambda_ridge,
            use_score_weighting=not args.no_score_weighting,
            output_dir=output_dir
        )

    # Determine the actual output path
    if not os.path.isabs(args.output):
        actual_output_path = os.path.join(output_dir, args.output)
    else:
        actual_output_path = args.output

    print(f"\n[OK] Done! Steering vectors saved to {actual_output_path}")


if __name__ == "__main__":
    main()
