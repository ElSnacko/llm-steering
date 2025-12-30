#!/usr/bin/env python3
"""
Helper script to run judge scoring in a separate process.
This avoids CUDA context conflicts with the parent process.
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Add LLM-Refusal-Evaluation to path
llm_eval_dir = str(Path(__file__).parent.parent / "LLM-Refusal-Evaluation")
if llm_eval_dir not in sys.path:
    sys.path.insert(0, llm_eval_dir)

from src.llm_judge import LLMJudge


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts-file", required=True)
    parser.add_argument("--responses-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--gpu-memory-util", type=float, default=0.9)
    args = parser.parse_args()

    # Load prompts and responses
    with open(args.prompts_file) as f:
        prompts = json.load(f)

    with open(args.responses_file) as f:
        responses = json.load(f)

    # Create judge
    judge = LLMJudge(
        model_name=args.judge_model,
        max_model_len=24576,
        gpu_memory_utilization=args.gpu_memory_util
    )

    # Score responses
    qa_pairs = list(zip(prompts, responses))
    results = judge.judge(
        questions_answers=qa_pairs,
        num_return_sequences=1,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        max_new_tokens=8192
    )

    # Extract scores
    scores = [result['label'] for result in results]

    # Save scores
    with open(args.output_file, 'w') as f:
        json.dump(scores, f)

    print(f"Saved {len(scores)} judge scores to {args.output_file}")


if __name__ == "__main__":
    main()
