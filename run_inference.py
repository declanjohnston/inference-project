#!/usr/bin/env python3
"""Inference script for comparing base and fine-tuned Gemma-2-2B models."""
import gc
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from mlh.hypers import Hypers, TBD
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Args(Hypers):
    base_model: str = "google/gemma-2-2b"
    finetuned_model: str = "djohnston5/gemma-2-2b-sft_crisp-armadillo-20"
    local_model_path: str | None = None
    prompts_path: str = "data/prompts/comparison_queries.json"
    output_dir: str = "data/outputs"
    force: bool = False
    max_new_tokens: int = 256
    temperature: float = 0.7

    prompts_path_: Path = TBD()
    output_dir_: Path = TBD()

def init(args: Args) -> Args:
    """Initialize path objects from string paths."""
    args.prompts_path_ = Path(args.prompts_path)
    model_name = args.local_model_path or args.finetuned_model
    args.output_dir_ = Path(args.output_dir) / model_name.split("/")[-1]
    return args


def load_model(model_name_or_path: str) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load a model and tokenizer from HuggingFace Hub or local path."""
    print(f"Loading model: {model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    return model, tokenizer


def unload_model(model: AutoModelForCausalLM) -> None:
    """Delete model and free GPU memory."""
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Model unloaded, memory cleared.")


def format_alpaca_prompt(instruction: str, input_text: str = "") -> str:
    """Format instruction and input into Alpaca-style prompt."""
    if input_text:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
"""
    return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""


def generate_response(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
) -> str:
    """Generate a response from the model given a prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response[len(prompt):].strip()


def run_inference_on_model(
    model_name_or_path: str,
    queries: list[dict],
    args: Args,
) -> list[dict]:
    """Run inference on a model for all queries."""
    model, tokenizer = load_model(model_name_or_path)
    responses = []

    for i, query in enumerate(queries, 1):
        print(f"  Processing query {i}/{len(queries)}...")
        prompt = format_alpaca_prompt(query["instruction"], query.get("input", ""))
        response = generate_response(
            model, tokenizer, prompt, args.max_new_tokens, args.temperature
        )
        responses.append({
            "instruction": query["instruction"],
            "input": query.get("input", ""),
            "response": response,
        })

    unload_model(model)
    return responses


def load_cached_outputs(output_path: Path) -> list[dict] | None:
    """Load cached outputs if they exist."""
    if output_path.exists():
        return json.loads(output_path.read_text())
    return None


def save_outputs(outputs: list[dict], output_path: Path) -> None:
    """Save outputs to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(outputs, indent=2))
    print(f"Saved outputs to {output_path}")


def run_inference(args: Args) -> None:
    """Run inference pipeline for base and fine-tuned models."""
    queries = json.loads(args.prompts_path_.read_text())
    print(f"Loaded {len(queries)} queries from {args.prompts_path_}")

    args.output_dir_.mkdir(parents=True, exist_ok=True)
    base_output_path = args.output_dir_ / "base_responses.json"
    finetuned_output_path = args.output_dir_ / "finetuned_responses.json"

    # Base model inference
    if not args.force and (cached := load_cached_outputs(base_output_path)):
        print(f"Using cached base model outputs from {base_output_path}")
    else:
        print(f"\n[Base Model] Running inference with {args.base_model}...")
        base_responses = run_inference_on_model(args.base_model, queries, args)
        save_outputs(base_responses, base_output_path)

    # Fine-tuned model inference
    finetuned_model = args.local_model_path or args.finetuned_model
    if not args.force and (cached := load_cached_outputs(finetuned_output_path)):
        print(f"Using cached fine-tuned model outputs from {finetuned_output_path}")
    else:
        print(f"\n[Fine-tuned Model] Running inference with {finetuned_model}...")
        finetuned_responses = run_inference_on_model(finetuned_model, queries, args)
        save_outputs(finetuned_responses, finetuned_output_path)

    print("\nInference complete!")


def main() -> None:
    print(f"Using device: {DEVICE}")
    args = Args()
    args = init(args)
    print(args)
    run_inference(args)


if __name__ == "__main__":
    main()
