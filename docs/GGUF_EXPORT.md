# GGUF Export Feature

## Overview

Added GGUF export functionality to convert HuggingFace models (including steered models) to GGUF format for use with llama.cpp and compatible inference engines.

## Changes Made

### 1. Core Implementation (`src/activation_steering/merge_steering_into_weights.py`)

- Added `export_to_gguf()` function with support for multiple quantization types
- Added `_try_convert_with_hf_to_gguf()` for llama.cpp conversion script integration
- Added `_try_convert_with_llama_cpp_python()` stub for future llama-cpp-python support
- Extended `merge_steering_into_model()` with `export_gguf` and `gguf_quantization` parameters
- Automatic GGUF export after model merging when enabled

### 2. Package Exports (`src/activation_steering/__init__.py`)

- Exported `export_to_gguf` function for library usage

### 3. CLI Script Updates (`scripts/merge_steering.py`)

- Added `--export-gguf` flag to enable GGUF export during merge
- Added `--gguf-quantization` parameter (choices: f32, f16, q4_0, q4_1, q5_0, q5_1, q8_0)
- Updated help text with GGUF export example

### 4. New Standalone Script (`scripts/export_to_gguf.py`)

- Standalone tool for converting any HuggingFace model to GGUF
- Supports all quantization types
- Detailed help and examples
- Verbose mode for debugging conversions

### 5. Documentation Updates

#### CLAUDE.md
- Added GGUF export commands to common commands section
- Added GGUF export section with usage examples and quantization guide
- Added library import for `export_to_gguf`

#### README.md
- Added `export_to_gguf.py` to project structure
- Updated Step 5 (merge) with GGUF export example
- Added new Step 6 dedicated to GGUF export
- Documented all quantization options and requirements
- Renumbered subsequent steps

## Usage Examples

### During Model Merge

```bash
python scripts/merge_steering.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --steering-vectors steering_vectors_wrmd.pt \
    --layers 10 11 12 \
    --alpha -2.0 \
    --output-dir Qwen2.5-7B-Steered \
    --export-gguf \
    --gguf-quantization q4_0
```

### Standalone Export

```bash
python scripts/export_to_gguf.py \
    --model-dir Qwen2.5-7B-Steered \
    --quantization q4_0 \
    --output model-q4_0.gguf
```

### Library Usage

```python
from activation_steering import export_to_gguf

result = export_to_gguf(
    model_dir="Qwen2.5-7B-Steered",
    output_path="model-q4_0.gguf",
    quantization="q4_0",
    verbose=True
)
print(f"GGUF saved to: {result['output_path']}")
```

## Quantization Types

- **f32**: Full 32-bit precision (largest, highest quality)
- **f16**: Half precision (recommended default, good balance)
- **q8_0**: 8-bit quantization (good quality, 4x smaller than f32)
- **q5_0/q5_1**: 5-bit quantization (decent quality, smaller)
- **q4_0/q4_1**: 4-bit quantization (lowest quality, smallest size)

## Requirements

One of the following:

1. **llama.cpp** with `convert-hf-to-gguf.py` script available
   - Clone: `git clone https://github.com/ggerganov/llama.cpp`
   - Script locations checked: `~/llama.cpp`, `/opt/llama.cpp`, `./llama.cpp`, `../llama.cpp`

2. **llama-cpp-python** (future support)
   - Install: `pip install llama-cpp-python`

## Error Handling

- Tries multiple conversion methods automatically
- Falls back gracefully if one method fails
- Detailed error messages guide users to installation
- GGUF export failures don't affect HuggingFace model saving

## Files Modified

1. `src/activation_steering/merge_steering_into_weights.py` - Core implementation
2. `src/activation_steering/__init__.py` - Exports
3. `scripts/merge_steering.py` - CLI integration
4. `scripts/export_to_gguf.py` - New standalone script
5. `CLAUDE.md` - Usage documentation
6. `README.md` - Feature documentation
7. `docs/GGUF_EXPORT.md` - This file

## Testing Recommendations

To test the implementation:

```bash
# Install dependencies
pip install -e .
pip install uv

# Ensure llama.cpp is available
git clone https://github.com/ggerganov/llama.cpp ~/llama.cpp

# Test with a small model or existing merged model
python scripts/export_to_gguf.py \
    --model-dir <path-to-model> \
    --quantization f16 \
    --verbose
```

## Future Enhancements

- Direct llama-cpp-python conversion support
- Automatic llama.cpp installation/setup
- Additional quantization formats (k-quants)
- GGUF metadata preservation
- Batch conversion support
