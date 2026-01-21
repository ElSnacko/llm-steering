#!/usr/bin/env python3
"""
CLI script for exporting HuggingFace models to GGUF format.

This script converts any HuggingFace model (including merged models) to GGUF format
for use with llama.cpp and compatible inference engines.
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from activation_steering import export_to_gguf


def main():
    parser = argparse.ArgumentParser(
        description="Export HuggingFace models to GGUF format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export with default FP16 quantization
  python scripts/export_to_gguf.py \\
      --model-dir Qwen2.5-7B-Steered \\
      --output model-f16.gguf

  # Export with Q4_0 quantization for smaller size
  python scripts/export_to_gguf.py \\
      --model-dir Qwen2.5-7B-Steered \\
      --output model-q4_0.gguf \\
      --quantization q4_0

  # Export with auto-naming
  python scripts/export_to_gguf.py \\
      --model-dir Qwen2.5-7B-Steered \\
      --quantization q8_0

Quantization types:
  - f32: Full 32-bit precision (largest, highest quality)
  - f16: Half precision (good quality, moderate size)
  - q8_0: 8-bit quantization (good quality, smaller)
  - q5_1: 5-bit quantization (decent quality, small)
  - q5_0: 5-bit quantization (alternative)
  - q4_1: 4-bit quantization (lower quality, very small)
  - q4_0: 4-bit quantization (lowest quality, smallest)

Requirements:
  - llama.cpp installed and convert-hf-to-gguf.py script available
  - Or llama-cpp-python: pip install llama-cpp-python
        """
    )

    parser.add_argument("--model-dir", required=True,
                       help="Directory containing HuggingFace model")
    parser.add_argument("--output", "--output-path",
                       help="Output path for GGUF file (default: model-dir/model-{quantization}.gguf)")
    parser.add_argument("--quantization", default="f16",
                       choices=["f32", "f16", "q4_0", "q4_1", "q5_0", "q5_1", "q8_0"],
                       help="Quantization type (default: f16)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Print detailed conversion output")

    args = parser.parse_args()

    print("="*80)
    print("GGUF EXPORT")
    print("="*80)
    print(f"Model directory: {args.model_dir}")
    print(f"Quantization: {args.quantization}")
    if args.output:
        print(f"Output path: {args.output}")
    print("="*80)

    try:
        result = export_to_gguf(
            model_dir=args.model_dir,
            output_path=args.output,
            quantization=args.quantization,
            verbose=args.verbose
        )

        print("\n" + "="*80)
        print("EXPORT COMPLETE")
        print("="*80)
        print(f"Output file: {result['output_path']}")
        print(f"Quantization: {result['quantization']}")
        print(f"Method: {result['method']}")
        print("="*80)
        print("\nUse with llama.cpp:")
        print(f'   ./main -m {result["output_path"]} -p "Your prompt here"')
        print("="*80)

        return 0

    except Exception as e:
        print(f"\n[ERROR] GGUF export failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
