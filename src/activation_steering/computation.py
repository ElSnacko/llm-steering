"""
Steering vector computation using various statistical methods.

Implements Mean Difference (MD), Ridge Mean Difference (RMD), and
Weighted Ridge Mean Difference (WRMD) for computing steering vectors.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os


def compute_md(refusal_acts, compliant_acts):
    """
    Simple Mean Difference.

    Args:
        refusal_acts: [N_refusal, hidden_size]
        compliant_acts: [N_compliant, hidden_size]

    Returns:
        Steering vector: [hidden_size]
    """
    return refusal_acts.mean(dim=0) - compliant_acts.mean(dim=0)


def compute_weighted_mean(activations, weights):
    """
    Compute weighted mean of activations.

    Args:
        activations: [N, hidden_size]
        weights: [N] - will be normalized to sum to 1

    Returns:
        weighted_mean: [hidden_size]
    """
    weights = weights.to(activations.dtype)
    weights = weights / weights.sum()
    return (activations.T @ weights).squeeze()


def compute_rmd(refusal_acts, compliant_acts, lambda_ridge=0.1):
    """
    Ridge Mean Difference.

    Args:
        refusal_acts: [N_refusal, hidden_size]
        compliant_acts: [N_compliant, hidden_size]
        lambda_ridge: Ridge regularization parameter

    Returns:
        Steering vector: [hidden_size]
    """
    compliant_mean = compliant_acts.mean(dim=0)
    centered = compliant_acts - compliant_mean
    cov = (centered.T @ centered) / len(compliant_acts)

    hidden_size = cov.shape[0]

    # Upcast to float32 for matrix inversion
    cov_f32 = cov.float()
    ridge_inv = torch.linalg.inv(cov_f32 + lambda_ridge * torch.eye(
        hidden_size,
        dtype=torch.float32,
        device=cov.device
    ))
    ridge_inv = ridge_inv.to(cov.dtype)

    mean_diff = refusal_acts.mean(dim=0) - compliant_mean

    return ridge_inv @ mean_diff


def compute_wrmd(refusal_acts, compliant_acts,
                 refusal_weights=None, compliant_weights=None,
                 neutral_acts=None, neutral_weights=None,
                 lambda_ridge=0.1):
    """
    Weighted Ridge Mean Difference (WRMD).

    Weights samples by judge confidence scores and accounts for covariance structure.

    Args:
        refusal_acts: [N_refusal, hidden_size]
        compliant_acts: [N_compliant, hidden_size]
        refusal_weights: [N_refusal] - judge confidence scores
        compliant_weights: [N_compliant] - judge confidence scores
        neutral_acts: Optional neutral baseline activations
        neutral_weights: Optional weights for neutral samples
        lambda_ridge: Ridge regularization parameter

    Returns:
        Steering vector: [hidden_size]
    """
    # If no neutral provided, use compliant as neutral
    if neutral_acts is None:
        neutral_acts = compliant_acts
        neutral_weights = compliant_weights

    # Default to uniform weights if none provided
    if refusal_weights is None:
        refusal_weights = torch.ones(len(refusal_acts))
    if compliant_weights is None:
        compliant_weights = torch.ones(len(compliant_acts))
    if neutral_weights is None:
        neutral_weights = torch.ones(len(neutral_acts))

    # Convert weights to match activation dtype
    refusal_weights = refusal_weights.to(refusal_acts.dtype)
    compliant_weights = compliant_weights.to(compliant_acts.dtype)
    neutral_weights = neutral_weights.to(neutral_acts.dtype)

    # Compute weighted neutral mean
    neutral_mean = compute_weighted_mean(neutral_acts, neutral_weights)

    # Center activations by neutral
    refusal_centered = refusal_acts - neutral_mean
    compliant_centered = compliant_acts - neutral_mean

    # Compute weighted means of centered activations
    refusal_mean_centered = compute_weighted_mean(refusal_centered, refusal_weights)
    compliant_mean_centered = compute_weighted_mean(compliant_centered, compliant_weights)

    # Compute weighted covariance of compliant distribution
    centered_from_mean = compliant_centered - compliant_mean_centered
    weighted_centered = centered_from_mean * compliant_weights.unsqueeze(1)
    cov = (weighted_centered.T @ centered_from_mean) / compliant_weights.sum()

    # Ridge inverse (upcast to float32)
    hidden_size = cov.shape[0]
    cov_f32 = cov.float()
    ridge_inv = torch.linalg.inv(cov_f32 + lambda_ridge * torch.eye(
        hidden_size,
        dtype=torch.float32,
        device=cov.device
    ))
    ridge_inv = ridge_inv.to(cov.dtype)

    # Weighted mean difference
    mean_diff = refusal_mean_centered - compliant_mean_centered

    return ridge_inv @ mean_diff


class WRMDCalculator:
    """Calculate steering vectors using various methods."""

    def __init__(self, activation_file):
        """
        Initialize calculator from activation file.

        Args:
            activation_file: Path to .pt file with activations
        """
        print(f"[LOAD] Loading activations from: {activation_file}")
        data = torch.load(activation_file)

        self.activations = data['activations']  # [N, num_layers, hidden_size]
        self.labels = data['labels']  # [N]
        self.prompts = data['prompts']
        self.num_layers = data['num_layers']
        self.hidden_size = data['hidden_size']

        # Extract judge scores from metadata
        if 'metadata' in data:
            self.metadata = data['metadata']
            self.judge_scores = torch.tensor([m['score'] for m in self.metadata])
            print(f"[OK] Found judge scores in metadata")
        else:
            self.metadata = None
            self.judge_scores = None
            print(f"[WARN]  No metadata found - will use uniform weights")

        print(f"[OK] Loaded {len(self.labels)} samples")
        print(f"   Layers: {self.num_layers}, Hidden size: {self.hidden_size}")
        print(f"   Refusal: {(self.labels == 1).sum()}, Compliant: {(self.labels == 0).sum()}")

        if self.judge_scores is not None:
            print(f"   Judge score range: [{self.judge_scores.min():.2f}, {self.judge_scores.max():.2f}]")

    def get_weights_from_scores(self, scores, label):
        """
        Convert judge scores to weights for WRMD.

        For refusals (label=1): higher score = higher weight
        For compliances (label=0): more negative score = higher weight

        Args:
            scores: Judge confidence scores
            label: 0 for compliant, 1 for refusal

        Returns:
            Positive weights proportional to confidence
        """
        if label == 1:
            # Refusals: score > 0, higher is better
            weights = scores.clone()
        else:
            # Compliances: score < 0, more negative is better
            weights = -scores.clone()

        # Ensure all weights are positive
        weights = torch.clamp(weights, min=0.01)

        return weights

    def compute_steering_vectors(self, method='wrmd', lambda_ridge=0.1,
                                 use_score_weighting=True, normalize=False):
        """
        Compute steering vectors for each layer.

        Args:
            method: 'md', 'rmd', or 'wrmd'
            lambda_ridge: Ridge regularization parameter
            use_score_weighting: Whether to weight by judge scores (WRMD only)
            normalize: Whether to normalize vectors to unit length

        Returns:
            steering_vectors: [num_layers, hidden_size]
        """
        print(f"\n[COMPUTE] Computing steering vectors using {method.upper()}")
        print(f"   Ridge λ: {lambda_ridge}")
        print(f"   Score weighting: {use_score_weighting and method == 'wrmd'}")
        print(f"   Normalize: {normalize}")

        steering_vectors = []

        # Separate by label
        refusal_mask = self.labels == 1
        compliant_mask = self.labels == 0

        # Prepare weights if using score weighting
        if use_score_weighting and self.judge_scores is not None and method == 'wrmd':
            refusal_scores = self.judge_scores[refusal_mask]
            compliant_scores = self.judge_scores[compliant_mask]

            refusal_weights = self.get_weights_from_scores(refusal_scores, label=1)
            compliant_weights = self.get_weights_from_scores(compliant_scores, label=0)

            print(f"   Refusal weights: mean={refusal_weights.mean():.2f}, std={refusal_weights.std():.2f}")
            print(f"   Compliant weights: mean={compliant_weights.mean():.2f}, std={compliant_weights.std():.2f}")
        else:
            refusal_weights = None
            compliant_weights = None

        for layer in range(self.num_layers):
            layer_acts = self.activations[:, layer, :]

            refusal_acts = layer_acts[refusal_mask]
            compliant_acts = layer_acts[compliant_mask]

            if method == 'md':
                vec = compute_md(refusal_acts, compliant_acts)
            elif method == 'rmd':
                vec = compute_rmd(refusal_acts, compliant_acts, lambda_ridge)
            elif method == 'wrmd':
                vec = compute_wrmd(
                    refusal_acts, compliant_acts,
                    refusal_weights, compliant_weights,
                    neutral_acts=compliant_acts,
                    neutral_weights=compliant_weights,
                    lambda_ridge=lambda_ridge
                )
            else:
                raise ValueError(f"Unknown method: {method}")

            # Optional: normalize to unit length
            if normalize:
                vec = vec / vec.norm()

            steering_vectors.append(vec)

        steering_vectors = torch.stack(steering_vectors)

        print(f"[OK] Computed steering vectors")
        print(f"   Shape: {steering_vectors.shape}")
        print(f"   Norm range: [{steering_vectors.norm(dim=1).min():.2f}, {steering_vectors.norm(dim=1).max():.2f}]")

        return steering_vectors

    def analyze_vectors(self, steering_vectors, output_dir=None):
        """
        Analyze properties of steering vectors.

        Args:
            steering_vectors: [num_layers, hidden_size]
            output_dir: Directory to save plots

        Returns:
            Array of norms per layer
        """
        from .utils import get_output_path

        print("\n[INFO] Vector Analysis:")

        # Compute norms per layer
        norms = steering_vectors.norm(dim=1).float().numpy()

        print(f"   Mean norm: {norms.mean():.2f}")
        print(f"   Std norm: {norms.std():.2f}")
        print(f"   Max norm layer: {norms.argmax()} (norm={norms.max():.2f})")
        print(f"   Min norm layer: {norms.argmin()} (norm={norms.min():.2f})")

        # Plot norms by layer
        plt.figure(figsize=(12, 4))
        plt.plot(norms, marker='o', linewidth=2, markersize=6)
        plt.xlabel('Layer Index', fontsize=12)
        plt.ylabel('Vector Norm', fontsize=12)
        plt.title('Steering Vector Magnitude by Layer', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.axhline(y=norms.mean(), color='r', linestyle='--', alpha=0.5, label=f'Mean: {norms.mean():.2f}')
        plt.legend()
        plt.tight_layout()

        # Save plot
        if output_dir is None:
            output_path = get_output_path(script_name="compute_wrmd", filename="steering_vector_norms.png")
        else:
            output_path = os.path.join(output_dir, "steering_vector_norms.png")

        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"   [PLOT] Saved plot: {output_path}")

        return norms

    def save_vectors(self, vectors, output_file, method='wrmd', lambda_ridge=0.1,
                    use_score_weighting=False, output_dir=None):
        """
        Save steering vectors to file.

        Args:
            vectors: [num_layers, hidden_size]
            output_file: Filename or full path
            method: Method used ('md', 'rmd', 'wrmd')
            lambda_ridge: Ridge parameter used
            use_score_weighting: Whether score weighting was used
            output_dir: Optional output directory
        """
        from .utils import get_output_path

        if output_dir is None:
            output_dir = get_output_path(script_name="compute_wrmd")

        if not os.path.isabs(output_file):
            output_path = os.path.join(output_dir, output_file)
        else:
            output_path = output_file

        print(f"\n[SAVE] Saving to {output_path}...")

        torch.save({
            'steering_vectors': vectors,
            'num_layers': self.num_layers,
            'hidden_size': self.hidden_size,
            'method': method,
            'lambda_ridge': lambda_ridge,
            'use_score_weighting': use_score_weighting,
            'num_refusal_samples': (self.labels == 1).sum().item(),
            'num_compliant_samples': (self.labels == 0).sum().item(),
        }, output_path)

        print(f"[OK] Saved steering vectors")


def compare_methods(calculator, lambda_ridge=0.1, output_dir=None):
    """
    Compare different steering vector computation methods.

    Args:
        calculator: WRMDCalculator instance
        lambda_ridge: Ridge parameter
        output_dir: Directory to save comparison plots

    Returns:
        Dictionary of steering vectors by method name
    """
    from .utils import get_output_path

    print("\n[COMPUTE] Comparing Methods...")

    comparisons = [
        ('MD', 'md', False),
        ('RMD', 'rmd', False),
        ('WRMD (uniform)', 'wrmd', False),
        ('WRMD (weighted)', 'wrmd', True),
    ]

    all_vectors = {}

    for name, method, use_weighting in comparisons:
        print(f"\n--- {name} ---")
        vectors = calculator.compute_steering_vectors(
            method, lambda_ridge, use_score_weighting=use_weighting
        )
        all_vectors[name] = vectors

    # Compare vector norms
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    for i, (name, _, _) in enumerate(comparisons):
        norms = all_vectors[name].norm(dim=1).float().numpy()
        axes[i].plot(norms, marker='o', linewidth=2, markersize=6)
        axes[i].set_xlabel('Layer Index')
        axes[i].set_ylabel('Vector Norm')
        axes[i].set_title(f'{name} Vector Norms')
        axes[i].grid(True, alpha=0.3)
        axes[i].axhline(y=norms.mean(), color='r', linestyle='--', alpha=0.5)

    plt.tight_layout()

    # Save plot
    if output_dir is None:
        output_path = get_output_path(script_name="compute_wrmd", filename="method_comparison.png")
    else:
        output_path = os.path.join(output_dir, "method_comparison.png")

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n[PLOT] Saved comparison plot: {output_path}")

    return all_vectors
