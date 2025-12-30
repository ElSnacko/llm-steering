"""
Merge steering vectors permanently into model weights.

This module provides functionality to permanently modify a model's weights
by merging steering vectors into the MLP bias terms. This creates a model
that has the steering behavior built-in without needing runtime hooks.

WARNING: This permanently modifies the model weights!
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import json


def merge_steering_into_model(
    base_model_path,
    steering_vectors_file,
    target_layers,
    alpha,
    output_dir
):
    """
    Permanently merge steering vectors into model weights.

    This function loads a base model and steering vectors, then modifies
    the model's MLP bias terms to incorporate the steering behavior.
    The modified model is saved to disk.

    WARNING: This permanently modifies the model! The output model will
    have different behavior than the base model.

    Args:
        base_model_path: Path to base HuggingFace model
        steering_vectors_file: Path to .pt file with steering vectors
        target_layers: List of layer indices to modify
        alpha: Steering coefficient (negative = reduce refusal)
        output_dir: Directory to save modified model

    Returns:
        Dictionary with metadata about the modification
    """

    print("[MERGE] Merging steering into model weights...")
    print(f"   Base model: {base_model_path}")
    print(f"   Layers: {target_layers}")
    print(f"   Alpha: {alpha}")

    # Load model
    print("[LOAD] Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    # Load steering vectors
    print("[LOAD] Loading steering vectors...")
    vec_data = torch.load(steering_vectors_file)
    steering_vectors = vec_data['steering_vectors'].to(model.device)

    print(f"\n[COMPUTE] Modifying {len(target_layers)} layer(s)...")

    # Merge steering into layer MLP biases
    for layer_idx in target_layers:
        layer = model.model.layers[layer_idx]
        steering_vec = steering_vectors[layer_idx]

        # Add steering as a bias term to the MLP down projection
        # This ensures the steering is applied to all tokens uniformly
        if hasattr(layer.mlp, 'down_proj'):
            with torch.no_grad():
                # Create bias if it doesn't exist
                if layer.mlp.down_proj.bias is None:
                    layer.mlp.down_proj.bias = torch.nn.Parameter(
                        torch.zeros(
                            layer.mlp.down_proj.out_features,
                            dtype=steering_vec.dtype,
                            device=steering_vec.device
                        )
                    )

                # Add steering vector scaled by alpha
                layer.mlp.down_proj.bias.data += alpha * steering_vec

            print(f"   * Layer {layer_idx}: Added steering to MLP output bias")
        else:
            print(f"   [WARN] Layer {layer_idx}: No down_proj found, skipping")

    # Save modified model
    print(f"\n[SAVE] Saving modified model to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)

    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)

    # Save metadata about the modification
    metadata = {
        "base_model": base_model_path,
        "steering_vectors_file": steering_vectors_file,
        "steering_method": vec_data.get('method', 'unknown'),
        "steering_layers": target_layers,
        "steering_alpha": alpha,
        "num_layers": vec_data.get('num_layers'),
        "hidden_size": vec_data.get('hidden_size'),
        "lambda_ridge": vec_data.get('lambda_ridge'),
        "use_score_weighting": vec_data.get('use_score_weighting'),
        "modified": True,
        "modification_type": "mlp_bias_injection",
        "note": "Steering vectors permanently merged into MLP down_proj biases"
    }

    metadata_path = os.path.join(output_dir, "steering_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[OK] Model saved successfully!")
    print(f"[INFO] Metadata saved to: {metadata_path}")
    print(f"\nLoad the modified model with:")
    print(f'   from transformers import AutoModelForCausalLM')
    print(f'   model = AutoModelForCausalLM.from_pretrained("{output_dir}")')

    return metadata


def verify_merged_model(model_dir, original_model_path, steering_layers):
    """
    Verify that a merged model has the expected modifications.

    Args:
        model_dir: Directory containing merged model
        original_model_path: Path to original base model
        steering_layers: List of layer indices that should be modified

    Returns:
        Dictionary with verification results
    """

    print("[INFO] Verifying merged model...")

    # Check metadata exists
    metadata_path = os.path.join(model_dir, "steering_metadata.json")
    if not os.path.exists(metadata_path):
        return {"verified": False, "error": "No steering_metadata.json found"}

    with open(metadata_path) as f:
        metadata = json.load(f)

    # Load both models to compare
    print("[LOAD] Loading original and merged models...")
    original_model = AutoModelForCausalLM.from_pretrained(
        original_model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu"
    )
    merged_model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        device_map="cpu"
    )

    results = {
        "verified": True,
        "metadata": metadata,
        "modified_layers": [],
        "unmodified_layers": [],
    }

    # Check each specified layer
    for layer_idx in steering_layers:
        orig_layer = original_model.model.layers[layer_idx]
        merged_layer = merged_model.model.layers[layer_idx]

        # Check MLP down_proj bias
        if hasattr(orig_layer.mlp, 'down_proj') and hasattr(merged_layer.mlp, 'down_proj'):
            orig_bias = orig_layer.mlp.down_proj.bias
            merged_bias = merged_layer.mlp.down_proj.bias

            # Check if they differ
            if orig_bias is None and merged_bias is not None:
                results["modified_layers"].append(layer_idx)
            elif orig_bias is not None and merged_bias is not None:
                if not torch.allclose(orig_bias, merged_bias, rtol=1e-5):
                    results["modified_layers"].append(layer_idx)
                else:
                    results["unmodified_layers"].append(layer_idx)
            else:
                results["unmodified_layers"].append(layer_idx)

    print(f"[INFO] Modified layers: {results['modified_layers']}")
    print(f"[INFO] Unmodified layers: {results['unmodified_layers']}")

    if len(results['modified_layers']) == len(steering_layers):
        print("[OK] Verification passed: All specified layers modified")
    else:
        print("[WARN] Verification incomplete: Not all layers modified")
        results["verified"] = False

    return results
