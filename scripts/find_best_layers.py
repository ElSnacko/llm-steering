#!/usr/bin/env python3
"""
CLI script for finding the best layers for steering.

This script analyzes layer correlations to identify which layers are most
effective for steering interventions.
"""

import argparse
import sys
import os
import json

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from activation_steering import (
    compute_layer_correlations,
    plot_correlations,
    find_best_layers,
    visualize_layer_projections,
    extract_model_name,
    generate_run_id,
    setup_model_run_dirs,
    ensure_dir
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--activations", required=True)
    parser.add_argument("--steering-vectors", required=True)
    parser.add_argument("--judge-scores", default=None,
                       help="Optional separate judge scores file")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output-dir", default=None,
                       help="Custom output directory (default: outputs/{model_name}/{run_id}/find_best_layers/)")
    parser.add_argument("--run-id", default=None,
                       help="Custom run ID (default: auto-generated timestamp)")
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
        output_dir = output_dirs['find_best_layers']
        print(f"📁 Using output directory: {output_dir}")
        print(f"   Model: {model_name}")
        print(f"   Run ID: {run_id}")
    else:
        output_dir = ensure_dir(args.output_dir)

    # Compute correlations
    correlations, projections_all, judge_scores = compute_layer_correlations(
        args.activations,
        args.steering_vectors,
        args.judge_scores
    )

    # Plot overall correlations
    plot_correlations(correlations, output_dir=output_dir, output_file='layer_correlations.png')

    # Find best layers
    best_layers = find_best_layers(correlations, args.top_k)

    # Visualize top 3 layers
    print("\n[INFO] Generating scatter plots for top 3 layers...")
    sorted_corrs = sorted(correlations, key=lambda x: x['abs_correlation'], reverse=True)
    for i, c in enumerate(sorted_corrs[:3]):
        layer = c['layer']
        visualize_layer_projections(
            layer,
            projections_all[layer],
            judge_scores,
            c['correlation'],
            output_dir=output_dir,
            output_file=f'layer_{layer}_projection_scatter.png'
        )

    # Save results
    output = {
        'best_layers': best_layers,
        'all_correlations': correlations
    }

    json_output_path = os.path.join(output_dir, 'layer_correlations.json')
    with open(json_output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n[OK] Results saved to {json_output_path}")
    print(f"[OK] Best layers for steering: {best_layers}")


if __name__ == "__main__":
    main()
