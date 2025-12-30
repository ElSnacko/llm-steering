# Output Folder Structure Plan

## Current Structure
Currently, all output files are saved directly in the root directory:
- activations_qwen_4b_judged.pt
- activations_qwen_7b_judged.pt
- activations_qwen_8b_judged.pt
- steering_vectors_md.pt
- steering_vectors_wrmd_normalized.pt
- steering_vectors_wrmd_qwen3_4b.pt
- steering_vectors_wrmd_weighted.pt
- layer_correlations.json
- layer_correlations.png
- method_comparison.png
- steering_vector_norms.png
- layer_9_projection_scatter.png
- layer_10_projection_scatter.png
- ... (more layer projection scatter plots)

## Proposed Structure
```
outputs/
├── compute_wrmd/
│   ├── steering_vectors_md.pt
│   ├── steering_vectors_wrmd_normalized.pt
│   ├── steering_vectors_wrmd_qwen3_4b.pt
│   ├── steering_vectors_wrmd_weighted.pt
│   ├── steering_vector_norms.png
│   └── method_comparison.png
├── extract_activations/
│   ├── activations_qwen_4b_judged.pt
│   ├── activations_qwen_7b_judged.pt
│   └── activations_qwen_8b_judged.pt
└── find_best_layers/
    ├── layer_correlations.json
    ├── layer_correlations.png
    ├── layer_9_projection_scatter.png
    ├── layer_10_projection_scatter.png
    ├── layer_11_projection_scatter.png
    └── ... (other layer projection scatter plots)
```

## Implementation Details

### 1. Utility Module (output_utils.py)
Create a shared utility module with functions to:
- Create output directories if they don't exist
- Handle path joining for different operating systems
- Provide consistent naming conventions

### 2. File Modifications

#### compute_wrmd.py
- Import utility functions
- Create outputs/compute_wrmd/ directory
- Update save_vectors method to save to outputs/compute_wrmd/
- Update plot saving paths for:
  - steering_vector_norms.png
  - method_comparison.png
- Update default argument for --output to "outputs/compute_wrmd/steering_vectors_wrmd.pt"

#### extract_activations.py
- Import utility functions
- Create outputs/extract_activations/ directory
- Update extract_dataset method to save to outputs/extract_activations/
- Update default argument for --output to "outputs/extract_activations/activations_qwen_7b_judged.pt"

#### find_best_layers.py
- Import utility functions
- Create outputs/find_best_layers/ directory
- Update plot saving paths for:
  - layer_correlations.png
  - layer_*_projection_scatter.png
- Update JSON output path for layer_correlations.json
- Update default argument for --output to "outputs/find_best_layers/"

### 3. Command Line Arguments
Add optional --output-dir argument to each script to allow custom output directories:
```bash
python compute_wrmd.py --output-dir custom/path/outputs ...
python extract_activations.py --output-dir custom/path/outputs ...
python find_best_layers.py --output-dir custom/path/outputs ...
```

### 4. Backward Compatibility
Ensure the scripts still work with absolute paths provided via existing arguments.

## Benefits
1. **Organization**: All outputs are organized by script/function
2. **Clean Root Directory**: Reduces clutter in the project root
3. **Easy Cleanup**: Can easily clean outputs by deleting the outputs folder
4. **Scalability**: Easy to add new scripts with their own output folders
5. **Consistency**: Standardized approach to handling outputs across all scripts