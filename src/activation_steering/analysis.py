"""
Layer correlation analysis to identify effective steering layers.

Computes correlations between activation projections and judge scores to determine
which layers are most effective for steering interventions.
"""

import torch
import json
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import os


def compute_layer_correlations(activations_file, steering_vectors_file, judge_scores_file=None):
    """
    Compute correlation between activation projections and judge scores.

    This measures: "If we project activations onto the steering vector,
    does that projection predict the judge's refusal score?"

    High correlation = this layer is good for steering.

    Args:
        activations_file: Path to activation .pt file
        steering_vectors_file: Path to steering vector .pt file
        judge_scores_file: Optional separate judge scores JSON file

    Returns:
        Tuple of (correlations, projections_all_layers, judge_scores)
    """
    print("[LOAD] Loading data...")

    # Load activations
    act_data = torch.load(activations_file)
    activations = act_data['activations']  # [N, num_layers, hidden_size]
    labels = act_data['labels']

    # Load steering vectors
    vec_data = torch.load(steering_vectors_file)
    steering_vectors = vec_data['steering_vectors']  # [num_layers, hidden_size]

    print("🔧 Normalizing steering vectors to unit length...")
    steering_vectors = steering_vectors / steering_vectors.norm(dim=1, keepdim=True)

    # Get judge scores from metadata or separate file
    if 'metadata' in act_data:
        judge_scores = torch.tensor([m['score'] for m in act_data['metadata']])
        print("[OK] Using judge scores from activations metadata")
    elif judge_scores_file:
        with open(judge_scores_file) as f:
            data = json.load(f)
        judge_scores = torch.tensor([item['answer_censor_score'] for item in data])
        print("[OK] Using judge scores from separate file")
    else:
        raise ValueError("No judge scores found!")

    print(f"   Samples: {len(activations)}")
    print(f"   Layers: {activations.shape[1]}")
    print(f"   Judge score range: [{judge_scores.min():.2f}, {judge_scores.max():.2f}]")

    # Compute correlations for each layer
    print("\n[COMPUTE] Computing layer-wise correlations...")

    correlations = []
    projections_all_layers = []

    for layer in range(activations.shape[1]):
        # Project activations onto steering vector
        layer_acts = activations[:, layer, :]  # [N, hidden_size]
        steering_vec = steering_vectors[layer]  # [hidden_size]

        # Projection: how much each activation aligns with steering direction
        projections = (layer_acts @ steering_vec).float().numpy()  # [N]
        projections_all_layers.append(projections)

        # Correlation with judge scores
        corr, p_value = pearsonr(projections, judge_scores.numpy())

        correlations.append({
            'layer': int(layer),
            'correlation': float(corr),
            'abs_correlation': float(abs(corr)),
            'p_value': float(p_value),
            'projection_mean': float(projections.mean()),
            'projection_std': float(projections.std())
        })

        print(f"  Layer {layer:2d}: r={corr:+.4f} (p={p_value:.2e})")

    return correlations, projections_all_layers, judge_scores.numpy()


def plot_correlations(correlations, output_dir=None, output_file='layer_correlations.png'):
    """
    Plot correlation by layer.

    Args:
        correlations: List of correlation dicts from compute_layer_correlations
        output_dir: Directory to save plot
        output_file: Filename for plot
    """
    from .utils import get_output_path

    if output_dir is None:
        output_dir = get_output_path(script_name="find_best_layers")

    layers = [c['layer'] for c in correlations]
    corrs = [c['correlation'] for c in correlations]
    abs_corrs = [c['abs_correlation'] for c in correlations]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

    # Plot 1: Signed correlation
    ax1.plot(layers, corrs, marker='o', linewidth=2, markersize=6)
    ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax1.set_xlabel('Layer Index', fontsize=12)
    ax1.set_ylabel('Correlation (r)', fontsize=12)
    ax1.set_title('Layer-wise Correlation: Activation Projection vs Judge Score', fontsize=14)
    ax1.grid(True, alpha=0.3)

    # Highlight top layers
    top_5_idx = sorted(range(len(abs_corrs)), key=lambda i: abs_corrs[i], reverse=True)[:5]
    for idx in top_5_idx:
        ax1.axvline(x=layers[idx], color='r', linestyle='--', alpha=0.3)
        ax1.text(layers[idx], corrs[idx], f' L{layers[idx]}', fontsize=9, color='red')

    # Plot 2: Absolute correlation
    ax2.bar(layers, abs_corrs, alpha=0.7)
    ax2.set_xlabel('Layer Index', fontsize=12)
    ax2.set_ylabel('|Correlation|', fontsize=12)
    ax2.set_title('Absolute Correlation by Layer', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')

    # Highlight top 5
    for idx in top_5_idx:
        ax2.bar(layers[idx], abs_corrs[idx], color='red', alpha=0.7)

    plt.tight_layout()

    # Save plot
    if not os.path.isabs(output_file):
        output_path = os.path.join(output_dir, output_file)
    else:
        output_path = output_file

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n[PLOT] Saved plot: {output_path}")


def find_best_layers(correlations, top_k=5):
    """
    Find top K layers by absolute correlation.

    Args:
        correlations: List of correlation dicts
        top_k: Number of top layers to return

    Returns:
        List of best layer indices
    """
    # Sort by absolute correlation
    sorted_corrs = sorted(correlations, key=lambda x: x['abs_correlation'], reverse=True)

    print(f"\n[TOP] Top {top_k} Layers by Correlation:")
    print(f"{'Layer':<8} {'r':<10} {'|r|':<10} {'p-value':<12}")
    print("-" * 45)

    best_layers = []
    for i, c in enumerate(sorted_corrs[:top_k]):
        print(f"{c['layer']:<8} {c['correlation']:+.4f}    {c['abs_correlation']:.4f}    {c['p_value']:.2e}")
        best_layers.append(c['layer'])

    return best_layers


def visualize_layer_projections(layer, projections, judge_scores,
                                correlation, output_dir=None, output_file=None):
    """
    Scatter plot: projection vs judge score for a specific layer.

    Args:
        layer: Layer index
        projections: Projection values for this layer
        judge_scores: Judge scores
        correlation: Correlation coefficient
        output_dir: Directory to save plot
        output_file: Filename for plot
    """
    from .utils import get_output_path

    if output_dir is None:
        output_dir = get_output_path(script_name="find_best_layers")

    plt.figure(figsize=(10, 6))
    plt.scatter(projections, judge_scores, alpha=0.5, s=30)
    plt.xlabel('Activation Projection onto Steering Vector', fontsize=12)
    plt.ylabel('Judge Refusal Score', fontsize=12)
    plt.title(f'Layer {layer}: Projection vs Judge Score (r={correlation:+.4f})', fontsize=14)
    plt.grid(True, alpha=0.3)

    # Add regression line
    z = np.polyfit(projections, judge_scores, 1)
    p = np.poly1d(z)
    x_line = np.linspace(projections.min(), projections.max(), 100)
    plt.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)

    if output_file:
        if not os.path.isabs(output_file):
            output_path = os.path.join(output_dir, output_file)
        else:
            output_path = output_file

        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"[PLOT] Saved scatter plot: {output_path}")
    else:
        plt.show()
