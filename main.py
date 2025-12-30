#!/usr/bin/env python3
"""
Main pipeline script for activation steering.

This script runs the complete pipeline:
1. Extract activations from a model based on judge scores
2. Compute steering vectors using WRMD
3. Find best layers for steering
4. Test steering on sample prompts

Usage:
    python main.py --config config.yaml

Or with command-line arguments:
    python main.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --results-dir LLM-Refusal-Evaluation/results/baseline \
        --output-dir outputs/my-run
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from activation_steering import (
    ActivationExtractor,
    WRMDCalculator,
    load_prompts_from_judge_scores,
    analyze_dataset_quality,
    compute_layer_correlations,
    plot_correlations,
    find_best_layers,
    visualize_layer_projections,
    test_steering,
    load_actual_refusal_prompts,
    extract_model_name,
    generate_run_id,
    setup_model_run_dirs,
)
import json
import torch


def run_pipeline(
    model_name,
    results_dir,
    output_dir=None,
    run_id=None,
    refusal_threshold=0.1,
    compliance_threshold=-0.1,
    max_samples=None,
    method='wrmd',
    lambda_ridge=0.1,
    normalize=False,
    top_k_layers=5,
    test_alpha=-2.0,
    test_num_prompts=3,
    test_min_score=0.5,
    test_max_tokens=500,
    skip_extraction=False,
    skip_computation=False,
    skip_analysis=False,
    skip_testing=False,
):
    """
    Run the complete activation steering pipeline.

    Args:
        model_name: HuggingFace model identifier
        results_dir: Path to LLM-Refusal-Evaluation results
        output_dir: Custom output directory (optional)
        run_id: Custom run ID (optional)
        refusal_threshold: Judge score threshold for refusals
        compliance_threshold: Judge score threshold for compliance
        max_samples: Limit number of samples (optional)
        method: Steering vector method ('md', 'rmd', 'wrmd')
        lambda_ridge: Ridge regularization parameter
        normalize: Whether to normalize steering vectors
        top_k_layers: Number of top layers to identify
        test_alpha: Steering coefficient for testing
        test_num_prompts: Number of prompts to test
        test_min_score: Minimum refusal score for test prompts
        test_max_tokens: Max tokens to generate during testing
        skip_extraction: Skip activation extraction step
        skip_computation: Skip steering vector computation step
        skip_analysis: Skip layer analysis step
        skip_testing: Skip steering testing step

    Returns:
        Dictionary with paths to generated files
    """

    # Setup output directories
    if output_dir is None:
        model_base_name = extract_model_name(model_name)
        run_id = run_id or generate_run_id()
        output_dirs = setup_model_run_dirs(model_name=model_base_name, run_id=run_id)
    else:
        output_dirs = {
            'extract_activations': os.path.join(output_dir, 'extract_activations'),
            'compute_wrmd': os.path.join(output_dir, 'compute_wrmd'),
            'find_best_layers': os.path.join(output_dir, 'find_best_layers'),
        }
        for dir_path in output_dirs.values():
            os.makedirs(dir_path, exist_ok=True)

    print("="*80)
    print("ACTIVATION STEERING PIPELINE")
    print("="*80)
    print(f"Model: {model_name}")
    print(f"Results directory: {results_dir}")
    print(f"Output directory: {output_dirs['extract_activations'].rsplit('/', 2)[0]}")
    print("="*80)

    results = {}

    # Step 1: Extract Activations
    if not skip_extraction:
        print("\n" + "="*80)
        print("STEP 1: EXTRACTING ACTIVATIONS")
        print("="*80)

        # Load prompts from judge scores
        prompts, labels, metadata = load_prompts_from_judge_scores(
            results_dir,
            refusal_threshold,
            compliance_threshold
        )

        analyze_dataset_quality(metadata)

        # Limit samples if requested
        if max_samples:
            print(f"\n[WARN] Limiting to {max_samples} samples for testing")
            prompts = prompts[:max_samples]
            labels = labels[:max_samples]
            metadata = metadata[:max_samples]

        # Extract activations
        extractor = ActivationExtractor(model_name)
        activation_file = "activations.pt"
        extractor.extract_dataset(
            prompts, labels, activation_file, metadata,
            output_dir=output_dirs['extract_activations']
        )

        activation_path = os.path.join(output_dirs['extract_activations'], activation_file)
        results['activations'] = activation_path
        print(f"\n[OK] Activations saved to: {activation_path}")

    else:
        # Look for existing activation file
        activation_path = os.path.join(output_dirs['extract_activations'], 'activations.pt')
        if not os.path.exists(activation_path):
            raise FileNotFoundError(f"Activation file not found: {activation_path}")
        results['activations'] = activation_path
        print(f"\n[INFO] Using existing activations: {activation_path}")

    # Step 2: Compute Steering Vectors
    if not skip_computation:
        print("\n" + "="*80)
        print("STEP 2: COMPUTING STEERING VECTORS")
        print("="*80)

        calculator = WRMDCalculator(results['activations'])

        vectors = calculator.compute_steering_vectors(
            method=method,
            lambda_ridge=lambda_ridge,
            use_score_weighting=True,
            normalize=normalize
        )

        calculator.analyze_vectors(vectors, output_dir=output_dirs['compute_wrmd'])

        vector_file = f"steering_vectors_{method}.pt"
        calculator.save_vectors(
            vectors,
            vector_file,
            method=method,
            lambda_ridge=lambda_ridge,
            use_score_weighting=True,
            output_dir=output_dirs['compute_wrmd']
        )

        vector_path = os.path.join(output_dirs['compute_wrmd'], vector_file)
        results['steering_vectors'] = vector_path
        print(f"\n[OK] Steering vectors saved to: {vector_path}")

    else:
        # Look for existing steering vector file
        vector_path = os.path.join(output_dirs['compute_wrmd'], f'steering_vectors_{method}.pt')
        if not os.path.exists(vector_path):
            raise FileNotFoundError(f"Steering vectors not found: {vector_path}")
        results['steering_vectors'] = vector_path
        print(f"\n[INFO] Using existing steering vectors: {vector_path}")

    # Step 3: Find Best Layers
    if not skip_analysis:
        print("\n" + "="*80)
        print("STEP 3: ANALYZING BEST LAYERS")
        print("="*80)

        correlations, projections_all, judge_scores = compute_layer_correlations(
            results['activations'],
            results['steering_vectors']
        )

        plot_correlations(
            correlations,
            output_dir=output_dirs['find_best_layers'],
            output_file='layer_correlations.png'
        )

        best_layers = find_best_layers(correlations, top_k=top_k_layers)

        # Generate scatter plots for top 3 layers
        print("\n[INFO] Generating scatter plots for top 3 layers...")
        sorted_corrs = sorted(correlations, key=lambda x: x['abs_correlation'], reverse=True)
        for c in sorted_corrs[:3]:
            layer = c['layer']
            visualize_layer_projections(
                layer,
                projections_all[layer],
                judge_scores,
                c['correlation'],
                output_dir=output_dirs['find_best_layers'],
                output_file=f'layer_{layer}_projection_scatter.png'
            )

        # Save results
        analysis_output = {
            'best_layers': best_layers,
            'all_correlations': correlations
        }

        json_path = os.path.join(output_dirs['find_best_layers'], 'layer_correlations.json')
        with open(json_path, 'w') as f:
            json.dump(analysis_output, f, indent=2)

        results['layer_analysis'] = json_path
        results['best_layers'] = best_layers
        print(f"\n[OK] Layer analysis saved to: {json_path}")
        print(f"[OK] Best layers for steering: {best_layers}")

    else:
        # Load existing analysis
        json_path = os.path.join(output_dirs['find_best_layers'], 'layer_correlations.json')
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Layer analysis not found: {json_path}")
        with open(json_path) as f:
            analysis_output = json.load(f)
        results['layer_analysis'] = json_path
        results['best_layers'] = analysis_output['best_layers'][:top_k_layers]
        print(f"\n[INFO] Using existing layer analysis: {json_path}")
        print(f"[INFO] Best layers: {results['best_layers']}")

    # Step 4: Test Steering
    if not skip_testing:
        print("\n" + "="*80)
        print("STEP 4: TESTING STEERING")
        print("="*80)

        # Load test prompts
        print("[LOAD] Loading prompts that refused in baseline evaluation...")
        refusal_prompts = load_actual_refusal_prompts(
            results_dir,
            min_score=test_min_score,
            max_prompts=test_num_prompts
        )

        if not refusal_prompts:
            print(f"[WARN] No prompts found with refusal score > {test_min_score}")
            print("[WARN] Skipping testing step")
        else:
            print(f"[OK] Found {len(refusal_prompts)} high-refusal prompts\n")

            # Test top layer only
            layer_configs = [[results['best_layers'][0]]]

            test_steering(
                model_name=model_name,
                steering_file=results['steering_vectors'],
                layer_configs=layer_configs,
                alpha=test_alpha,
                test_prompts=refusal_prompts,
                max_new_tokens=test_max_tokens
            )

            print("\n[OK] Steering tests complete!")

    # Final Summary
    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    print("\nGenerated Files:")
    for key, path in results.items():
        if key != 'best_layers':
            print(f"  {key}: {path}")
    print("\nBest Layers:", results.get('best_layers', 'N/A'))
    print("="*80)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run complete activation steering pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Required arguments
    parser.add_argument("--model", required=True,
                       help="HuggingFace model identifier or path")
    parser.add_argument("--results-dir", required=True,
                       help="Path to LLM-Refusal-Evaluation results directory")

    # Output configuration
    parser.add_argument("--output-dir", default=None,
                       help="Custom output directory (default: outputs/{model_name}/{run_id})")
    parser.add_argument("--run-id", default=None,
                       help="Custom run ID (default: auto-generated timestamp)")

    # Extraction parameters
    parser.add_argument("--refusal-threshold", type=float, default=0.1,
                       help="Judge score above this = refusal (default: 0.1)")
    parser.add_argument("--compliance-threshold", type=float, default=-0.1,
                       help="Judge score below this = compliant (default: -0.1)")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Limit number of samples for testing")

    # Computation parameters
    parser.add_argument("--method", choices=['md', 'rmd', 'wrmd'], default='wrmd',
                       help="Steering vector computation method (default: wrmd)")
    parser.add_argument("--lambda-ridge", type=float, default=0.1,
                       help="Ridge regularization parameter (default: 0.1)")
    parser.add_argument("--normalize", action="store_true",
                       help="Normalize steering vectors to unit length")

    # Analysis parameters
    parser.add_argument("--top-k", type=int, default=5,
                       help="Number of top layers to identify (default: 5)")

    # Testing parameters
    parser.add_argument("--test-alpha", type=float, default=-2.0,
                       help="Steering coefficient for testing (default: -2.0)")
    parser.add_argument("--test-num-prompts", type=int, default=3,
                       help="Number of prompts to test (default: 3)")
    parser.add_argument("--test-min-score", type=float, default=0.5,
                       help="Minimum refusal score for test prompts (default: 0.5)")
    parser.add_argument("--test-max-tokens", type=int, default=500,
                       help="Max tokens to generate during testing (default: 500)")

    # Pipeline control
    parser.add_argument("--skip-extraction", action="store_true",
                       help="Skip activation extraction (use existing)")
    parser.add_argument("--skip-computation", action="store_true",
                       help="Skip steering vector computation (use existing)")
    parser.add_argument("--skip-analysis", action="store_true",
                       help="Skip layer analysis (use existing)")
    parser.add_argument("--skip-testing", action="store_true",
                       help="Skip steering testing")

    args = parser.parse_args()

    try:
        results = run_pipeline(
            model_name=args.model,
            results_dir=args.results_dir,
            output_dir=args.output_dir,
            run_id=args.run_id,
            refusal_threshold=args.refusal_threshold,
            compliance_threshold=args.compliance_threshold,
            max_samples=args.max_samples,
            method=args.method,
            lambda_ridge=args.lambda_ridge,
            normalize=args.normalize,
            top_k_layers=args.top_k,
            test_alpha=args.test_alpha,
            test_num_prompts=args.test_num_prompts,
            test_min_score=args.test_min_score,
            test_max_tokens=args.test_max_tokens,
            skip_extraction=args.skip_extraction,
            skip_computation=args.skip_computation,
            skip_analysis=args.skip_analysis,
            skip_testing=args.skip_testing,
        )

        return 0

    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
