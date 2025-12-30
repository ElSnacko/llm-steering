"""
Runtime steering via forward hooks.

Implements PyTorch hooks to modify model activations during generation,
applying steering vectors to reduce or increase refusal behavior.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os


class SteeringHook:
    """Apply steering vectors to model activations via forward hooks."""

    def __init__(self, model, steering_vectors, target_layers, alpha):
        """
        Initialize steering hook.

        Args:
            model: HuggingFace model instance
            steering_vectors: Tensor [num_layers, hidden_size]
            target_layers: List of layer indices to steer
            alpha: Steering coefficient (negative = reduce refusal)
        """
        self.model = model
        self.steering_vectors = steering_vectors.to(model.device)
        self.target_layers = target_layers
        self.alpha = alpha
        self.hooks = []

    def create_hook(self, layer_idx):
        """
        Create a forward hook for a specific layer.

        Args:
            layer_idx: Index of layer to hook

        Returns:
            Hook function
        """
        def hook_fn(module, input, output):
            # Handle both tuple and tensor outputs
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

            # Apply steering: h' = h + alpha * steering_vector
            steering_vec = self.steering_vectors[layer_idx]
            hidden_states = hidden_states + self.alpha * steering_vec

            # Return in same format as input
            if isinstance(output, tuple):
                return (hidden_states,) + output[1:]
            else:
                return hidden_states
        return hook_fn

    def register_hooks(self):
        """Register forward hooks on target layers."""
        for layer_idx in self.target_layers:
            layer = self.model.model.layers[layer_idx]
            hook = layer.register_forward_hook(self.create_hook(layer_idx))
            self.hooks.append(hook)

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


def load_best_layers_from_correlations(correlation_file, top_k=1):
    """
    Load best layers from layer_correlations.json.

    Args:
        correlation_file: Path to correlation JSON file
        top_k: Number of top layers to load

    Returns:
        List of best layer indices, or None if file not found
    """
    if not os.path.exists(correlation_file):
        print(f"[WARN]  Correlation file not found: {correlation_file}")
        print(f"   Run find_best_layers.py first!")
        return None

    with open(correlation_file) as f:
        data = json.load(f)

    best_layers = data['best_layers'][:top_k]

    print(f"[INFO] Loaded from {correlation_file}:")
    print(f"   Top {top_k} layer(s): {best_layers}")

    all_corrs = {c['layer']: c['correlation'] for c in data['all_correlations']}
    for layer in best_layers:
        print(f"   Layer {layer}: r={all_corrs[layer]:+.4f}")

    return best_layers


def load_actual_refusal_prompts(results_dir, min_score=0.5, max_prompts=5):
    """
    Load prompts that actually refused in baseline evaluation.

    Args:
        results_dir: Path to LLM-Refusal-Evaluation results
        min_score: Minimum refusal score threshold
        max_prompts: Maximum number of prompts to return

    Returns:
        List of dicts with 'prompt' and 'baseline_score'
    """
    censor_file = f"{results_dir}/ccp_sensitive_sampled/censor_scores.json"

    with open(censor_file) as f:
        data = json.load(f)

    refusal_prompts = []
    for item in data:
        if item['answer_censor_score'] > min_score:
            refusal_prompts.append({
                'prompt': item['prompt'],
                'baseline_score': item['answer_censor_score']
            })

    refusal_prompts.sort(key=lambda x: x['baseline_score'], reverse=True)
    return refusal_prompts[:max_prompts]


def test_single_layer_configuration(model, tokenizer, steering_vectors,
                                   layer, alpha, test_prompts, max_new_tokens):
    """
    Test steering with a single layer configuration.

    Args:
        model: Model instance
        tokenizer: Tokenizer instance
        steering_vectors: Steering vectors tensor
        layer: Layer index or list of indices
        alpha: Steering coefficient
        test_prompts: List of prompt dicts
        max_new_tokens: Max tokens to generate
    """
    # Normalize to list
    if isinstance(layer, int):
        layers = [layer]
        layer_desc = f"Layer {layer}"
    else:
        layers = layer
        layer_desc = f"Layers {layers}"

    print(f"\n{'='*80}")
    print(f"TESTING: {layer_desc} (alpha={alpha})")
    print(f"{'='*80}\n")

    for i, prompt_data in enumerate(test_prompts):
        prompt = prompt_data['prompt']
        baseline_score = prompt_data['baseline_score']

        print(f"{'-'*80}")
        print(f"Prompt {i+1}/{len(test_prompts)}")
        print(f"{'-'*80}")
        print(f" FULL PROMPT:\n{prompt}")
        print(f"\n[INFO] Baseline refusal score: {baseline_score:.2f}")
        print(f"{'-'*80}")

        # Format prompt
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # BASELINE
        print("\n[BASELINE] BASELINE OUTPUT:")
        print("-" * 80)
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        baseline_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        baseline_response = baseline_output[len(formatted_prompt):]

        print(baseline_response)
        print("-" * 80)

        # STEERED
        print(f"\n[STEERED] STEERED OUTPUT ({layer_desc}, α={alpha}):")
        print("-" * 80)

        steerer = SteeringHook(model, steering_vectors, layers, alpha)
        steerer.register_hooks()

        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        steered_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        steered_response = steered_output[len(formatted_prompt):]

        print(steered_response)
        print("-" * 80)

        steerer.remove_hooks()

        print(f"\n📏 Stats:")
        print(f"   Baseline length: {len(baseline_response)} chars")
        print(f"   Steered length: {len(steered_response)} chars")
        print(f"   Difference: {len(steered_response) - len(baseline_response):+d} chars")
        print()


def test_steering(model_name, steering_file, layer_configs, alpha, test_prompts,
                  max_new_tokens=300):
    """
    Test steering across multiple layer configurations.

    Args:
        model_name: HuggingFace model identifier
        steering_file: Path to steering vectors file
        layer_configs: List of layer configurations (each is int or list of ints)
        alpha: Steering coefficient
        test_prompts: List of prompt dicts
        max_new_tokens: Max tokens to generate
    """
    print(f"[COMPUTE] Testing Steering")
    print(f"   Model: {model_name}")
    print(f"   Steering file: {steering_file}")
    print(f"   Alpha: {alpha}")
    print(f"   Configurations to test: {len(layer_configs)}")
    print(f"   Test prompts: {len(test_prompts)}\n")

    # Load model (once)
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load steering vectors
    vec_data = torch.load(steering_file)
    steering_vectors = vec_data['steering_vectors']

    print(f"[OK] Model loaded: {model.config.num_hidden_layers} layers")
    print(f"[OK] Steering vectors: {steering_vectors.shape}\n")

    # Test each layer configuration
    for config in layer_configs:
        test_single_layer_configuration(
            model, tokenizer, steering_vectors,
            config, alpha, test_prompts, max_new_tokens
        )
