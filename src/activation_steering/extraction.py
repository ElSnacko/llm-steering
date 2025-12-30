"""
Activation extraction from language models based on judge scores.

This module provides functionality to extract internal activations from LLMs
and label them based on refusal/compliance behavior using judge scores.
"""

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import os


class ActivationExtractor:
    """Extract activations from language models at specific token positions."""

    def __init__(self, model_name="Qwen/Qwen2.5-7B-Instruct"):
        """
        Initialize the activation extractor.

        Args:
            model_name: HuggingFace model identifier or local path
        """
        print(f"Loading model: {model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.num_layers = len(self.model.model.layers)
        self.hidden_size = self.model.config.hidden_size
        self.model_name = model_name
        print(f"[OK] Model loaded: {self.num_layers} layers, {self.hidden_size} hidden size")

    def extract_last_token_activations(self, prompt):
        """
        Extract activations at last token position for all layers.

        Args:
            prompt: Input text prompt

        Returns:
            Tensor of shape [num_layers, hidden_size]
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        activations = {}

        def hook_fn(layer_idx):
            def hook(module, input, output):
                # Handle both tuple and tensor outputs
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output

                # Extract last token
                activations[layer_idx] = hidden_states[:, -1, :].detach().cpu()
            return hook

        hooks = []
        for i, layer in enumerate(self.model.model.layers):
            hooks.append(layer.register_forward_hook(hook_fn(i)))

        with torch.no_grad():
            self.model(**inputs)

        for hook in hooks:
            hook.remove()

        return torch.stack([activations[i] for i in range(self.num_layers)]).squeeze()

    def extract_dataset(self, prompts, labels, output_file, metadata=None, output_dir=None):
        """
        Extract activations for a list of prompts.

        Args:
            prompts: List of strings
            labels: List of 0 (compliant) or 1 (refusal)
            output_file: Filename to save (or full path if absolute)
            metadata: Optional dict with extra info (judge scores, etc.)
            output_dir: Optional custom output directory
        """
        from .utils import setup_model_run_dirs, extract_model_name, generate_run_id

        all_activations = []

        print(f"\n[COMPUTE] Extracting activations for {len(prompts)} prompts...")
        for prompt in tqdm(prompts, desc="Processing"):
            acts = self.extract_last_token_activations(prompt)
            all_activations.append(acts)

        all_activations = torch.stack(all_activations)

        save_data = {
            'activations': all_activations,
            'labels': torch.tensor(labels),
            'prompts': prompts,
            'num_layers': self.num_layers,
            'hidden_size': self.hidden_size
        }

        if metadata:
            save_data['metadata'] = metadata

        # Set up output directory
        if output_dir is None:
            model_name = extract_model_name(self.model_name)
            run_id = generate_run_id()
            output_dirs = setup_model_run_dirs(model_name=model_name, run_id=run_id)
            output_dir = output_dirs['extract_activations']

        # Handle absolute vs relative paths
        if not os.path.isabs(output_file):
            output_path = os.path.join(output_dir, output_file)
        else:
            output_path = output_file

        print(f"\n[SAVE] Saving to {output_path}...")
        torch.save(save_data, output_path)

        print(f"[OK] Saved {len(prompts)} samples")
        print(f"   Shape: {all_activations.shape}")
        print(f"   Refusal: {labels.count(1)}, Compliant: {labels.count(0)}")


def load_prompts_from_judge_scores(results_dir, refusal_threshold=0.1, compliance_threshold=-0.1):
    """
    Load prompts based on actual judge scores from LLM-Refusal-Evaluation.

    This ensures training on what the model ACTUALLY did, not dataset assumptions.

    Args:
        results_dir: Path to evaluation results directory
        refusal_threshold: Judge score above this = refusal (default 0.1)
        compliance_threshold: Judge score below this = compliant (default -0.1)

    Returns:
        Tuple of (prompts, labels, metadata)
    """
    all_prompts = []
    all_labels = []
    all_metadata = []

    # Expected splits in results
    refusal_splits = ['ccp_sensitive_sampled', 'deccp_censored']
    compliant_splits = ['xstest_safe', 'general_prompts']

    print(f"[LOAD] Loading judge scores from: {results_dir}")
    print(f"   Refusal threshold: > {refusal_threshold}")
    print(f"   Compliance threshold: < {compliance_threshold}")
    print()

    # Process refusal-expected splits
    for split in refusal_splits:
        score_file = os.path.join(results_dir, split, 'censor_scores.json')

        if not os.path.exists(score_file):
            print(f"[WARN]  Skipping {split}: file not found")
            continue

        with open(score_file) as f:
            data = json.load(f)

        refusal_count = 0
        compliant_count = 0
        skipped_count = 0

        for item in data:
            score = item['answer_censor_score']
            prompt = item['prompt']

            if score > refusal_threshold:
                all_prompts.append(prompt)
                all_labels.append(1)  # Refusal
                all_metadata.append({
                    'split': split,
                    'score': score,
                    'expected_behavior': 'refusal'
                })
                refusal_count += 1

            elif score < compliance_threshold:
                all_prompts.append(prompt)
                all_labels.append(0)  # Compliant
                all_metadata.append({
                    'split': split,
                    'score': score,
                    'expected_behavior': 'refusal',
                    'note': 'unexpected_compliance'
                })
                compliant_count += 1

            else:
                skipped_count += 1

        print(f"  {split}:")
        print(f"    * Refused: {refusal_count}")
        print(f"    * Complied: {compliant_count} (unexpected)")
        print(f"    X Uncertain: {skipped_count}")

    # Process compliant-expected splits
    for split in compliant_splits:
        score_file = os.path.join(results_dir, split, 'censor_scores.json')

        if not os.path.exists(score_file):
            print(f"[WARN]  Skipping {split}: file not found")
            continue

        with open(score_file) as f:
            data = json.load(f)

        refusal_count = 0
        compliant_count = 0
        skipped_count = 0

        for item in data:
            score = item['answer_censor_score']
            prompt = item['prompt']

            if score < compliance_threshold:
                all_prompts.append(prompt)
                all_labels.append(0)  # Compliant
                all_metadata.append({
                    'split': split,
                    'score': score,
                    'expected_behavior': 'compliant'
                })
                compliant_count += 1

            elif score > refusal_threshold:
                all_prompts.append(prompt)
                all_labels.append(1)  # Refusal
                all_metadata.append({
                    'split': split,
                    'score': score,
                    'expected_behavior': 'compliant',
                    'note': 'over_refusal'
                })
                refusal_count += 1

            else:
                skipped_count += 1

        print(f"  {split}:")
        print(f"    * Complied: {compliant_count}")
        print(f"    * Refused: {refusal_count} (over-refusal)")
        print(f"    X Uncertain: {skipped_count}")

    print(f"\n[INFO] Total dataset:")
    print(f"   Refusal examples: {all_labels.count(1)}")
    print(f"   Compliant examples: {all_labels.count(0)}")
    print(f"   Total: {len(all_labels)}")

    return all_prompts, all_labels, all_metadata


def analyze_dataset_quality(metadata):
    """
    Print statistics about dataset composition.

    Args:
        metadata: List of metadata dicts from load_prompts_from_judge_scores
    """
    print("\n[PLOT] Dataset Quality Analysis:")

    # Count unexpected behaviors
    unexpected_compliance = sum(1 for m in metadata if m.get('note') == 'unexpected_compliance')
    over_refusals = sum(1 for m in metadata if m.get('note') == 'over_refusal')

    print(f"   Unexpected compliances: {unexpected_compliance}")
    print(f"   Over-refusals: {over_refusals}")

    # Score distribution
    scores = [m['score'] for m in metadata]
    print(f"   Score range: [{min(scores):.2f}, {max(scores):.2f}]")

    refusal_scores = [s for s, m in zip(scores, metadata) if m.get('expected_behavior') == 'refusal']
    compliant_scores = [s for s, m in zip(scores, metadata) if m.get('expected_behavior') == 'compliant']

    if refusal_scores:
        print(f"   Mean refusal score: {sum(refusal_scores) / len(refusal_scores):.2f}")
    if compliant_scores:
        print(f"   Mean compliant score: {sum(compliant_scores) / len(compliant_scores):.2f}")
