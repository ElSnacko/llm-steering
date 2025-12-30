# Activation Steering

A Python toolkit for extracting activations, computing steering vectors, and analyzing effective layers for modifying LLM refusal behavior.

This project implements the methodology from the paper [**"Refusal Steering: Fine-grained Control over LLM Refusal Behaviour for Sensitive Topics"**](https://arxiv.org/abs/2512.16602).

## Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage Guide](#usage-guide)
- [Inner Workings](#inner-workings)
- [Using as a Library](#using-as-a-library)
- [Citation](#citation)

---

## Quick Start

Get started in 5 steps:

### 1. Clone and Setup

```bash
# Clone repository
git clone --recurse-submodules https://github.com/ElSnacko/llm-steering
cd llm-steering

# If already cloned without submodules, initialize them
git submodule update --init --recursive

# Install dependencies
pip install -e .
pip install uv
```

### 2. Generate Baseline Evaluation

```bash
cd LLM-Refusal-Evaluation
uv run python -m src.compute_refusal_score --config configs/your_model.yaml
cd ..
```

### 3. Extract Activations

```bash
python scripts/extract_activations.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --results-dir LLM-Refusal-Evaluation/results/baseline \
    --refusal-threshold 0.1 \
    --compliance-threshold -0.1
```

### 4. Compute Steering Vectors & Find Best Layers

```bash
# Compute WRMD steering vectors
python scripts/compute_wrmd.py \
    --activations outputs/.../activations_*.pt \
    --method wrmd \
    --lambda-ridge 0.1

# Identify effective layers
python scripts/find_best_layers.py \
    --activations outputs/.../activations_*.pt \
    --steering-vectors outputs/.../steering_vectors_wrmd.pt \
    --top-k 5
```

### 5. Optimize Alpha & Test Steering

```bash
# Find optimal alpha automatically (recommended)
python scripts/optimize_alpha.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --steering-vectors outputs/.../steering_vectors_wrmd.pt \
    --correlations outputs/.../layer_correlations.json \
    --baseline-results LLM-Refusal-Evaluation/results/baseline \
    --top-k 3 \
    --num-prompts 100

# Test interactively with optimal alpha
python scripts/test_steering.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --steering-vectors outputs/.../steering_vectors_wrmd.pt \
    --correlations outputs/.../layer_correlations.json \
    --top-k 3 \
    --alpha -2.5  # Use optimal alpha from previous step
```

**That's it!** You now have a steered model. For permanent steering, see [Merging Into Weights]([#6-merge-steering-into-model-weights-optional](https://github.com/ElSnacko/activation-steering-llm/tree/readme-consolidation?tab=readme-ov-file#multi-stage-pipeline-overview ))

---

## Installation

### Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.0.0 with CUDA support
- Transformers ≥ 4.30.0
- 16GB+ GPU memory (for 7B models)

### Step-by-Step Installation

#### 1. Clone Repository

```bash
# Clone with submodules
git clone --recurse-submodules https://github.com/ELSnacko/activation-steering.git
cd activation-steering
```

#### 2. Initialize Git Submodule

This project depends on [LLM-Refusal-Evaluation](https://github.com/CompactifAI/LLM-Refusal-Evaluation) for generating judge scores.

If you already cloned without `--recurse-submodules`:

```bash
git submodule update --init --recursive
```

Alternatively, clone separately:

```bash
git clone https://github.com/CompactifAI/LLM-Refusal-Evaluation.git
```

#### 3. Install Dependencies

Using pip:

```bash
pip install -r requirements.txt
pip install uv
```

Or install as editable package:

```bash
pip install -e .
pip install uv
```

---

## Project Structure

### Codebase Organization

The codebase is organized as a Python package:

```
activation_steering/
├── src/activation_steering/      # Main package
│   ├── extraction.py             # ActivationExtractor class and helpers
│   ├── computation.py            # WRMDCalculator and computation methods
│   ├── analysis.py               # Layer correlation analysis
│   ├── steering.py               # SteeringHook for runtime steering
│   └── utils.py                  # Output utilities
├── scripts/                      # CLI entry points
│   ├── extract_activations.py   # Step 1: Extract activations
│   ├── compute_wrmd.py           # Step 2: Compute steering vectors
│   ├── find_best_layers.py      # Step 3: Find effective layers
│   ├── optimize_alpha.py        # Step 4: Find optimal alpha (recommended)
│   ├── test_steering.py         # Step 5: Test steering interactively
│   └── merge_steering.py        # Optional: Merge into weights
├── docs/                         # Additional documentation
├── LLM-Refusal-Evaluation/      # External submodule (judge scores)
└── outputs/                      # Results directory (auto-created)
```

### Output Directory Structure

All scripts use a consistent output hierarchy:

```
outputs/
└── {model_name}/           # Auto-extracted (e.g., qwen2.5-7b-instruct)
    └── {run_id}/           # Auto-generated timestamp (YYYYMMDD-HHMMSS)
        ├── extract_activations/
        │   └── activations_*.pt
        ├── compute_wrmd/
        │   ├── steering_vectors_*.pt
        │   ├── steering_vector_norms.png
        │   └── method_comparison.png
        ├── find_best_layers/
        │   ├── layer_correlations.json
        │   ├── layer_correlations.png
        │   └── layer_*_projection_scatter.png
        └── optimize_alpha/
            ├── optimization_summary.json
            ├── alpha_optimization.png
            └── alpha_*/responses.json
```

**Tip:** Use `--output-dir` and `--run-id` to override defaults.

---

## Usage Guide

### Multi-Stage Pipeline Overview

The codebase follows a sequential pipeline:

```
1. extract_activations.py → activations_*.pt
2. compute_wrmd.py        → steering_vectors_*.pt
3. find_best_layers.py    → layer_correlations.json
4. optimize_alpha.py      → optimal alpha value (recommended)
5. test_steering.py       → interactive testing
6. merge_steering.py      → permanent model (optional)
```

Each stage consumes outputs from previous stages and produces inputs for the next.

---

### 1. Extract Activations from Model

Extract internal activations labeled by judge scores:

```bash
python scripts/extract_activations.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --results-dir LLM-Refusal-Evaluation/results/qwen3_8b_baseline \
    --output activations_qwen_8b_judged.pt \
    --refusal-threshold 0.1 \
    --compliance-threshold -0.1
```

**What it does:**
- Loads judge scores from LLM-Refusal-Evaluation results
- Extracts model activations at the last token position for all layers
- Filters samples: only includes clear refusal (score > 0.1) or compliance (score < -0.1)
- Saves metadata with judge scores for downstream use

**Arguments:**
- `--model`: HuggingFace model name or path
- `--results-dir`: Path to LLM-Refusal-Evaluation baseline results
- `--refusal-threshold`: Minimum score to label as refusal (default: 0.1)
- `--compliance-threshold`: Maximum score to label as compliance (default: -0.1)
- `--output-dir`: Custom output directory (optional)
- `--run-id`: Custom run ID (optional)
- `--max-samples`: Limit samples for testing (optional)

---

### 2. Compute Steering Vectors

Compute steering vectors using weighted methods:

```bash
# Single method (WRMD recommended)
python scripts/compute_wrmd.py \
    --activations outputs/qwen2.5-7b-instruct/20231227-035148/extract_activations/activations_qwen_8b_judged.pt \
    --output steering_vectors_wrmd.pt \
    --method wrmd \
    --lambda-ridge 0.1

# Compare all methods (MD, RMD, WRMD)
python scripts/compute_wrmd.py \
    --activations outputs/.../activations_qwen_8b_judged.pt \
    --output steering_vectors_wrmd.pt \
    --compare-methods
```

**Methods:**
- **MD (Mean Difference)**: Simple difference of means
- **RMD (Ridge Mean Difference)**: Adds ridge regularization
- **WRMD (Weighted Ridge Mean Difference)**: Weights samples by judge confidence scores (recommended)

**Arguments:**
- `--activations`: Path to activations .pt file
- `--method`: Choose 'md', 'rmd', or 'wrmd' (default: wrmd)
- `--lambda-ridge`: Ridge regularization parameter (default: 0.1)
- `--normalize`: Normalize vectors to unit length
- `--no-score-weighting`: Disable judge score weighting (uniform weights)
- `--compare-methods`: Generate comparison plots for all methods
- `--output-dir`: Custom output directory (optional)
- `--run-id`: Custom run ID (optional)

---

### 3. Find Best Layers for Steering

Identify which layers are most effective:

```bash
python scripts/find_best_layers.py \
    --activations outputs/qwen2.5-7b-instruct/20231227-035148/extract_activations/activations_qwen_8b_judged.pt \
    --steering-vectors outputs/qwen2.5-7b-instruct/20231227-035148/compute_wrmd/steering_vectors_wrmd.pt \
    --top-k 5
```

**What it does:**
- Computes Pearson correlation between activation projections and judge scores for each layer
- Higher absolute correlation = better steering effectiveness
- Generates visualizations showing layer effectiveness

**Arguments:**
- `--activations`: Path to activations .pt file
- `--steering-vectors`: Path to steering vectors .pt file
- `--top-k`: Number of best layers to identify (default: 5)
- `--output-dir`: Custom output directory (optional)
- `--run-id`: Custom run ID (optional)

---

### 4. Test Steering Interactively

Apply steering vectors at runtime and compare outputs:

```bash
# Test top-3 layers individually
python scripts/test_steering.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --steering-vectors steering_vectors_wrmd.pt \
    --correlations layer_correlations.json \
    --top-k 3 \
    --alpha -2.0 \
    --num-prompts 3

# Test specific layers together
python scripts/test_steering.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --steering-vectors steering_vectors_wrmd.pt \
    --layers 9 10 11 \
    --alpha -2.0
```

**Alpha parameter controls steering strength:**
- **Negative alpha** (e.g., -2.0): Reduce refusal behavior
- **Positive alpha** (e.g., +2.0): Increase refusal behavior

**Arguments:**
- `--model`: HuggingFace model name or path
- `--steering-vectors`: Path to steering vectors .pt file
- `--correlations`: Path to layer correlations JSON (for --top-k)
- `--layers`: Specific layers to test (alternative to --top-k)
- `--top-k`: Use top K layers from correlations
- `--alpha`: Steering strength (negative=reduce refusal)
- `--num-prompts`: Number of prompts to test (default: 5)

---

### 4a. Optimize Alpha Parameter (Recommended)

Automatically find the optimal alpha value using judge scores with early stopping:

```bash
# Recommended: automatic optimization with early stopping
python scripts/optimize_alpha.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --steering-vectors outputs/.../steering_vectors_wrmd.pt \
    --correlations outputs/.../layer_correlations.json \
    --baseline-results LLM-Refusal-Evaluation/results/qwen3_8b_baseline \
    --top-k 3 \
    --alpha-start -2.0 \
    --alpha-min -5.0 \
    --alpha-max 0.0 \
    --alpha-step 0.5 \
    --num-prompts 100 \
    --stopping-metric mean_score \
    --stopping-tolerance 0.1 \
    --judge-model openai/gpt-oss-20b

# Fine-grained search around specific alpha
python scripts/optimize_alpha.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --steering-vectors steering_vectors_wrmd.pt \
    --baseline-results LLM-Refusal-Evaluation/results/qwen3_8b_baseline \
    --layers 10 11 12 \
    --alpha-start -2.0 \
    --alpha-min -2.5 \
    --alpha-max -1.5 \
    --alpha-step 0.1 \
    --num-prompts 200
```

**Key Features:**
- Loads baseline (alpha=0) metrics from existing evaluation results (no re-run needed)
- Tests alphas starting from `--alpha-start` (default: -2.0), sweeping outward
- Early stopping when performance drops below best result + tolerance
- Tracks best performing alpha during sweep and compares against it
- Can continue testing alphas closer to zero even if far alphas fail
- Generates visualizations and summary with optimal alpha recommendation

**Early Stopping Behavior:**
- Baseline loaded from `--baseline-results` (no inference needed)
- Tests alphas in order: -2.0 → -2.5 → -3.0 → ... (away from zero)
- Tracks the best performing alpha throughout the sweep
- Stops if performance degrades beyond tolerance from EITHER baseline OR best prior result
- Example: If alpha=-2.5 achieves mean_score=-0.5 (best so far), stops at alpha=-3.0 if it scores worse than -0.5 + tolerance
- If far alphas fail, continues testing -1.5 → -1.0 → -0.5 (toward zero)
- Skips alpha=0.0 entirely (already have baseline metrics)

**Outputs:**
- `optimization_summary.json`: Full results with optimal alpha and best alpha during sweep
- `alpha_optimization.png`: 4-panel visualization (scores, rates, distribution, uncertainty)
- `alpha_*/responses.json`: Generated responses for each alpha tested

The summary includes both the optimal alpha (based on objective function) and the best alpha found during the sweep (for early stopping reference).

**Arguments:**
- `--model`: HuggingFace model name or path
- `--steering-vectors`: Path to steering vectors .pt file
- `--baseline-results`: Path to baseline evaluation results
- `--correlations`: Path to layer correlations JSON (for --top-k)
- `--layers`: Specific layers to test (alternative to --top-k)
- `--top-k`: Use top K layers from correlations
- `--alpha-start`: Starting alpha value (default: -2.0)
- `--alpha-min`: Minimum alpha to test (default: -5.0)
- `--alpha-max`: Maximum alpha to test (default: 0.0)
- `--alpha-step`: Alpha step size (default: 0.5)
- `--num-prompts`: Number of test prompts (default: 50)
- `--stopping-metric`: Metric for early stopping (default: mean_score)
- `--stopping-tolerance`: Tolerance before stopping (default: 0.1)
- `--objective`: Optimization objective (default: minimize_refusal)
- `--judge-model`: Judge model for scoring (default: openai/gpt-oss-20b)
- `--output-dir`: Custom output directory (optional)
- `--run-id`: Custom run ID (optional)

---

### 5. Merge Steering Into Model Weights (Optional)

Permanently merge steering vectors into model weights to create a standalone steered model:

```bash
# Merge using best layers from correlation analysis
python scripts/merge_steering.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --steering-vectors steering_vectors_wrmd.pt \
    --correlations layer_correlations.json \
    --top-k 3 \
    --alpha -2.0 \
    --output-dir Qwen2.5-7B-Steered

# Verify the merged model
python scripts/merge_steering.py \
    --verify \
    --merged-model Qwen2.5-7B-Steered \
    --original-model Qwen/Qwen2.5-7B-Instruct \
    --layers 10 11 12
```

**What it does:**
- Modifies the model's MLP bias terms to include steering vectors
- Creates a standalone model with built-in steering behavior
- No runtime hooks required for the merged model

**WARNING:** This permanently modifies the model weights!

**Arguments:**
- `--model`: HuggingFace model name or path
- `--steering-vectors`: Path to steering vectors .pt file
- `--correlations`: Path to layer correlations JSON (for --top-k)
- `--layers`: Specific layers to merge (alternative to --top-k)
- `--top-k`: Merge top K layers from correlations
- `--alpha`: Steering strength to merge
- `--output-dir`: Directory to save merged model
- `--verify`: Verify merged model against original
- `--merged-model`: Path to merged model (for verification)
- `--original-model`: Path to original model (for verification)

---

### 6. Run LLM-Refusal-Evaluation (Generate Judge Scores)

Generate baseline judge scores needed for activation extraction:

```bash
cd LLM-Refusal-Evaluation
uv run python -m src.compute_refusal_score --config configs/Qwen3-4B-Instruct-2507.yaml
```

See `LLM-Refusal-Evaluation/README.md` for configuration details.

---

## Inner Workings

### Architecture & Data Flow

The project uses judge scores from LLM-Refusal-Evaluation to train steering vectors:

1. **LLM-Refusal-Evaluation/** generates judge scores for model responses
2. **extract_activations.py** uses judge scores to label activations as refusal/compliant
3. **compute_wrmd.py** computes steering vectors weighted by judge confidence scores
4. **find_best_layers.py** correlates activation projections with judge scores to identify effective steering layers
5. **optimize_alpha.py** automatically finds optimal alpha using judge scores with early stopping
6. **test_steering.py** applies steering vectors at runtime via forward hooks with optimal alpha

### Steering Vector Methods

#### Mean Difference (MD)
Simple difference of refusal and compliant activation means:
```
v = mean(activations_refusal) - mean(activations_compliant)
```

#### Ridge Mean Difference (RMD)
Adds ridge regularization to stabilize computation:
```
v = (C + λI)^(-1) * (mean_refusal - mean_compliant)
```

#### Weighted Ridge Mean Difference (WRMD)
Weights samples by judge confidence scores and accounts for covariance structure:
```
v = (C_weighted + λI)^(-1) * (weighted_mean_refusal - weighted_mean_compliant)
```

**WRMD is the recommended method** as it leverages judge confidence scores to weight examples by certainty.

### Judge Score Interpretation

Judge scores range from -1 to 1:
- **score > 0.1**: Model refused (higher = stronger refusal)
- **score < -0.1**: Model complied (more negative = stronger compliance)
- **-0.1 ≤ score ≤ 0.1**: Uncertain/mixed behavior

Scores are used to:
1. **Filter samples** during activation extraction (only clear behavior)
2. **Weight samples** during WRMD computation (higher confidence = higher weight)
3. **Evaluate steering** during alpha optimization (measure effectiveness)

### Implementation Details

#### Activation Extraction (`src/activation_steering/extraction.py`)
- Extracts activations at the **last token position** for all layers
- Uses judge scores from LLM-Refusal-Evaluation metadata to filter samples
- Only includes samples with clear behavior (score > refusal_threshold OR score < compliance_threshold)
- Skips uncertain samples (scores near zero)
- Saves metadata alongside activations for downstream use

#### WRMD Computation (`src/activation_steering/computation.py`)
- Converts judge scores to weights: higher magnitude = higher confidence = higher weight
- For refusals (label=1): weight = score (already positive)
- For compliances (label=0): weight = -score (convert negative to positive)
- Computes weighted covariance matrix from compliant distribution
- Uses float32 for matrix inversion, then casts back to original dtype (bfloat16)
- Ridge regularization (lambda) stabilizes inversion of covariance matrix

#### Layer Correlation Analysis (`src/activation_steering/analysis.py`)
- Normalizes steering vectors to unit length before projection
- Computes projection = activations @ steering_vector for each layer
- Correlates projections with judge scores using Pearson correlation
- High positive/negative correlation = layer is effective for steering
- Top layers by absolute correlation are recommended for steering

#### Steering Application (`src/activation_steering/steering.py`)
- Uses PyTorch forward hooks to modify activations at runtime
- Hook modifies hidden states: `h' = h + alpha * steering_vector`
- Can apply to single layer or multiple layers simultaneously
- Loads "actual refusal prompts" from baseline evaluation results
- Compares baseline vs steered outputs side-by-side

### File Format Details

#### Activation Files (.pt)
```python
{
    'activations': Tensor[N, num_layers, hidden_size],
    'labels': Tensor[N],  # 0=compliant, 1=refusal
    'prompts': List[str],
    'num_layers': int,
    'hidden_size': int,
    'metadata': List[dict]  # Contains judge scores and split info
}
```

#### Steering Vector Files (.pt)
```python
{
    'steering_vectors': Tensor[num_layers, hidden_size],
    'num_layers': int,
    'hidden_size': int,
    'method': str,  # 'md', 'rmd', or 'wrmd'
    'lambda_ridge': float,
    'use_score_weighting': bool,
    'num_refusal_samples': int,
    'num_compliant_samples': int
}
```

#### Layer Correlation Files (.json)
```python
{
    'best_layers': [int, ...],  # Top K layers by absolute correlation
    'all_correlations': [
        {
            'layer': int,
            'correlation': float,
            'abs_correlation': float,
            'p_value': float,
            'projection_mean': float,
            'projection_std': float
        },
        ...
    ]
}
```

### Model Loading Pattern

All scripts use this pattern for efficient loading:

```python
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,  # Reduces memory usage
    device_map="auto"             # Automatic multi-GPU distribution
)
```

### Utility Functions (`src/activation_steering/utils.py`)

- `extract_model_name(model_string)`: Converts "Qwen/Qwen2.5-7B-Instruct" → "qwen2.5-7b-instruct"
- `generate_run_id()`: Creates timestamp-based ID (YYYYMMDD-HHMMSS)
- `setup_model_run_dirs(model_name, run_id)`: Creates full output directory structure
- `ensure_dir(path)`: Creates directory if it doesn't exist

All scripts use these utilities for consistent output organization.

### Typical Workflow

1. **Evaluate baseline model** using LLM-Refusal-Evaluation to generate judge scores
2. **Extract activations** from the baseline model, filtering by judge scores
3. **Compute steering vectors** using WRMD with judge score weighting
4. **Find best layers** by correlating projections with judge scores
5. **Optimize alpha parameter** using automated search with early stopping (recommended)
6. **Test steering** interactively with optimal alpha on actual refusal prompts

This workflow ensures that steering is trained on actual model behavior (not dataset assumptions), alpha is optimized using judge scores, and the final configuration is tested on prompts that the model actually refused in baseline evaluation.

---

## Using as a Library

The package can be imported directly in Python code:

### Basic Usage

```python
from activation_steering import (
    ActivationExtractor,
    WRMDCalculator,
    SteeringHook,
    load_prompts_from_judge_scores
)

# Extract activations
extractor = ActivationExtractor("Qwen/Qwen2.5-7B-Instruct")
prompts, labels, metadata = load_prompts_from_judge_scores("results/baseline")
extractor.extract_dataset(prompts, labels, "activations.pt", metadata)

# Compute steering vectors
calculator = WRMDCalculator("activations.pt")
vectors = calculator.compute_steering_vectors(method='wrmd', use_score_weighting=True)
calculator.save_vectors(vectors, "steering_vectors.pt")

# Apply steering at runtime
import torch
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", ...)
steerer = SteeringHook(model, vectors, target_layers=[10, 11], alpha=-2.0)
steerer.register_hooks()
# ... run generation ...
steerer.remove_hooks()
```

### Permanent Steering via Weight Merging

```python
from activation_steering import merge_steering_into_model

# Merge steering permanently into model weights
metadata = merge_steering_into_model(
    base_model_path="Qwen/Qwen2.5-7B-Instruct",
    steering_vectors_file="steering_vectors.pt",
    target_layers=[10, 11],
    alpha=-2.0,
    output_dir="Qwen2.5-7B-Steered"
)

# Load and use the merged model
from transformers import AutoModelForCausalLM
steered_model = AutoModelForCausalLM.from_pretrained("Qwen2.5-7B-Steered")
# No hooks needed - steering is built-in!
```

---

## Dependencies

### Core Dependencies

- PyTorch ≥ 2.0.0 (with bfloat16 support)
- Transformers ≥ 4.30.0 (HuggingFace models)
- NumPy ≥ 1.24.0 (numerical operations)
- SciPy ≥ 1.10.0 (statistical computations)
- Matplotlib ≥ 3.7.0 (visualization)
- tqdm ≥ 4.65.0 (progress bars)

### Additional Dependencies

- vLLM (for LLM-Refusal-Evaluation batch inference)
- See `requirements.txt` for full list

---

## Citation

This toolkit implements the methodology from:
```bibtex
@misc{garciaferrero2025Refusal,
      title={Refusal Steering: Fine-grained Control over LLM Refusal Behaviour for Sensitive Topics},
      author={Iker García-Ferrero and David Montero and Roman Orus},
      year={2025},
      eprint={2512.16602},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2512.16602},
}
```

If you use this implementation in your research, please cite both the original paper above and this repository.

## License

MIT License

---

## Related Projects

- [LLM-Refusal-Evaluation](https://github.com/CompactifAI/LLM-Refusal-Evaluation): Inference-time evaluation framework for measuring refusal behavior
- [Refusal Steering Paper](https://arxiv.org/abs/2512.16602): Original research paper

---

## Troubleshooting

### Common Issues

**GPU Memory Errors:**
- Use smaller batch sizes or models
- Enable `device_map="auto"` for multi-GPU distribution
- Use `torch_dtype=torch.bfloat16` for memory efficiency

**Judge Score Files Not Found:**
- Ensure LLM-Refusal-Evaluation ran successfully
- Check that `--results-dir` points to the correct baseline results
- Verify `judge_scores.json` or `aggregated_results.json` exists

**Import Errors:**
- Install package in editable mode: `pip install -e .`
- Ensure all dependencies are installed: `pip install -r requirements.txt`

**Correlation Analysis Shows Weak Correlations:**
- Try different steering vector methods (MD vs RMD vs WRMD)
- Adjust ridge regularization parameter `--lambda-ridge`
- Ensure sufficient samples with clear refusal/compliance behavior

---

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## Contact

For questions or issues:
- Open an issue on GitHub
- See the [paper](https://arxiv.org/abs/2512.16602) for research details
