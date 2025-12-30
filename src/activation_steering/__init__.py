"""
Activation Steering for Language Models.

A toolkit for extracting activations, computing steering vectors, and analyzing
effective layers for modifying LLM refusal behavior.
"""

__version__ = "0.1.0"

from .extraction import (
    ActivationExtractor,
    load_prompts_from_judge_scores,
    analyze_dataset_quality
)

from .computation import (
    WRMDCalculator,
    compute_md,
    compute_rmd,
    compute_wrmd,
    compare_methods
)

from .analysis import (
    compute_layer_correlations,
    plot_correlations,
    find_best_layers,
    visualize_layer_projections
)

from .steering import (
    SteeringHook,
    load_best_layers_from_correlations,
    load_actual_refusal_prompts,
    test_steering
)

from .utils import (
    ensure_dir,
    extract_model_name,
    generate_run_id,
    get_run_output_dir,
    get_output_path,
    setup_model_run_dirs
)

from .merge_steering_into_weights import (
    merge_steering_into_model,
    verify_merged_model
)

__all__ = [
    # Core classes
    "ActivationExtractor",
    "WRMDCalculator",
    "SteeringHook",

    # Extraction functions
    "load_prompts_from_judge_scores",
    "analyze_dataset_quality",

    # Computation functions
    "compute_md",
    "compute_rmd",
    "compute_wrmd",
    "compare_methods",

    # Analysis functions
    "compute_layer_correlations",
    "plot_correlations",
    "find_best_layers",
    "visualize_layer_projections",

    # Steering functions
    "load_best_layers_from_correlations",
    "load_actual_refusal_prompts",
    "test_steering",

    # Merge functions
    "merge_steering_into_model",
    "verify_merged_model",

    # Utility functions
    "ensure_dir",
    "extract_model_name",
    "generate_run_id",
    "get_run_output_dir",
    "get_output_path",
    "setup_model_run_dirs",
]
