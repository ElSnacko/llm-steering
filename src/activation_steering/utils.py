# output_utils.py - Utility functions for managing output directories and paths

import os
import re
from datetime import datetime
from pathlib import Path

def ensure_dir(dir_path):
    """
    Create directory if it doesn't exist
    
    Args:
        dir_path: Path to directory (string or Path object)
    
    Returns:
        Path object of the created directory
    """
    path = Path(dir_path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def extract_model_name(model_string):
    """
    Extract a clean model name from a model string
    
    Args:
        model_string: Full model name (e.g., "Qwen/Qwen2.5-7B-Instruct")
    
    Returns:
        Clean model name (e.g., "qwen2.5-7b-instruct")
    """
    # Extract the last part after the slash
    if '/' in model_string:
        model_name = model_string.split('/')[-1]
    else:
        model_name = model_string
    
    # Convert to lowercase and replace special characters
    model_name = model_name.lower()
    model_name = re.sub(r'[^a-z0-9\-]', '-', model_name)
    model_name = re.sub(r'-+', '-', model_name)  # Replace multiple hyphens with single
    
    return model_name

def generate_run_id():
    """
    Generate a unique run ID based on timestamp
    
    Returns:
        String run ID (e.g., "20231227-035148")
    """
    return datetime.now().strftime("%Y%m%d-%H%M%S")

def get_run_output_dir(base_dir="outputs", model_name=None, run_id=None):
    """
    Get output directory path for a specific model and run
    
    Args:
        base_dir: Base output directory (default: "outputs")
        model_name: Name of the model (e.g., "qwen2.5-7b-instruct")
        run_id: Unique run identifier (e.g., "20231227-035148")
    
    Returns:
        Path object for the output directory
    """
    if model_name and run_id:
        output_path = Path(base_dir) / model_name / run_id
    elif model_name:
        output_path = Path(base_dir) / model_name
    else:
        output_path = Path(base_dir)
    
    return ensure_dir(output_path)

def get_output_path(base_dir="outputs", model_name=None, run_id=None, filename=None):
    """
    Get full path for an output file
    
    Args:
        base_dir: Base output directory (default: "outputs")
        model_name: Name of the model (e.g., "qwen2.5-7b-instruct")
        run_id: Unique run identifier (e.g., "20231227-035148")
        filename: Name of the output file
    
    Returns:
        Path object for the output file
    """
    output_dir = get_run_output_dir(base_dir, model_name, run_id)
    
    if filename:
        return output_dir / filename
    else:
        return output_dir

def setup_model_run_dirs(base_dir="outputs", model_name=None, run_id=None):
    """
    Setup output directories for a specific model run
    
    Args:
        base_dir: Base output directory (default: "outputs")
        model_name: Name of the model (e.g., "qwen2.5-7b-instruct")
        run_id: Unique run identifier (e.g., "20231227-035148")
    
    Returns:
        Dictionary mapping script types to their output directories
    """
    run_dir = get_run_output_dir(base_dir, model_name, run_id)
    
    output_dirs = {
        'extract_activations': ensure_dir(run_dir / 'extract_activations'),
        'compute_wrmd': ensure_dir(run_dir / 'compute_wrmd'),
        'find_best_layers': ensure_dir(run_dir / 'find_best_layers')
    }
    
    return output_dirs

# Legacy functions for backward compatibility
def get_output_dir(base_dir="outputs", script_name=None):
    """
    Legacy function - use get_run_output_dir instead
    """
    if script_name:
        output_path = Path(base_dir) / script_name
    else:
        output_path = Path(base_dir)
    
    return ensure_dir(output_path)