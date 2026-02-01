---
title: Gemma-2 Fine-tuning Comparison
emoji: 🔬
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.30.0
app_file: dashboard.py
pinned: false
---

# Base vs Fine-tuned Model Comparison

This dashboard compares responses from the base **Gemma-2-2B** model with a version fine-tuned on the [Alpaca dataset](https://huggingface.co/datasets/tatsu-lab/alpaca) using SFTTrainer.

## Features

- Side-by-side comparison of base vs fine-tuned model outputs
- 10 curated test queries covering various instruction types
- Interactive query selection

## Project Structure

- `run_inference.py` - Run inference on base and fine-tuned models
- `run_experiments.py` - Training script using SFTTrainer
- `dashboard.py` - Streamlit comparison dashboard
- `data/outputs/` - Cached model responses
