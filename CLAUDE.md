# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research project for time-series anomaly detection using Large Language Models (LLMs). The codebase extends the [AnomLLM](https://github.com/rose-stl-lab/anomllm) framework and supports two benchmarks:
- **AnomLLM**: Synthetic anomaly detection benchmark
- **TSB-AD-U**: Real-world time-series benchmark

The project implements various prompting strategies (zero-shot, one-shot, text-only, vision, Chain-of-Thought) to evaluate LLM performance on anomaly detection tasks.

## Environment Setup

**Dependency Management**: This project uses Poetry (see [pyproject.toml](pyproject.toml)).

**Installation**:
```bash
poetry install
```

**Required Setup**:
1. Must set up the AnomLLM environment first (see AnomLLM repository)
2. Create `credentials.yml` in root directory with API keys for online API usage
3. Download datasets:
   - AnomLLM: Download `anomllm.zip` and extract to `data/synthetic/`
   - TSB-AD-U: Download from TSB-AD repository and place in `TSB-AD/Datasets/`

**Generate Synthetic Data**:
```bash
./synthesize.sh
```

## Common Commands

### Running Experiments

**AnomLLM Benchmark**:
```bash
# Run single experiment
python src/LLM-TSAD-AnomLLM_api.py --model gemini-1.5-flash --data trend --variant 0shot-text-vision

# Aggregate results
python src/result_agg_by_model.py --model gemini-1.5-flash --benchmark anomllm
```

**TSB-AD-U Benchmark**:
```bash
# Run single experiment
python src/LLM-TSAD-TSB_api.py --model gemini-1.5-flash --datadir ./TSB-AD/Datasets

# Aggregate results
python src/result_agg_by_model.py --model gemini-1.5-flash --benchmark tsb-ad-u
```

**Batch Processing** (multiple datasets/models/variants):
```bash
./run_script.sh  # Edit the script to configure which experiments to run
```

**Baselines**:
```bash
python src/baselines/isoforest.py --data trend --model isolation-forest
```

## Code Architecture

### Core Components

- **[neurips_our/AnoAgent.py](src/neurips_our/AnoAgent.py)**: Main agent class that orchestrates the anomaly detection pipeline. Handles model selection, preprocessing, prompt generation, and inference.

- **[config.py](src/config.py)**: Defines experiment variants (configurations for different prompting strategies). Key function: `create_batch_api_configs()` returns a dictionary mapping variant names to request generators.

- **[neurips_our/prompts.py](src/neurips_our/prompts.py)**: Contains prompt templates and generation logic for different experiment variants (text-only, vision, CoT, etc.).

- **[neurips_our/preprocessing_seq.py](src/neurips_our/preprocessing_seq.py)**: Time-series preprocessing utilities including period detection, deseasonalization, and sequence-to-image conversion.

### API Modules

- **[openai_api.py](src/openai_api.py)**: OpenAI API wrapper (also used for compatible APIs)
- **[gemini_api.py](src/gemini_api.py)**: Google Gemini API wrapper
- **[batch_api.py](src/batch_api.py)**: Batch processing with retry logic
- **[online_api.py](src/online_api.py)**: Online/streaming API interface

### Experiment Runners

- **[LLM-TSAD-AnomLLM_api.py](src/LLM-TSAD-AnomLLM_api.py)**: Main script for running experiments on AnomLLM benchmark
- **[LLM-TSAD-TSB_api.py](src/LLM-TSAD-TSB_api.py)**: Main script for running experiments on TSB-AD-U benchmark

Both scripts:
1. Load datasets from appropriate directories
2. Initialize AnoAgent with specified model
3. Run inference with retry logic (exponential backoff)
4. Save results to `results/` directory as JSONL files
5. Handle API errors and rate limiting

### Utilities

- **[utils.py](src/utils.py)**: Core utility functions including:
  - `compute_metrics()`: Calculate precision, recall, F1 for anomaly detection
  - `interval_to_vector()` / `vector_to_interval()`: Convert between interval and binary vector representations
  - `parse_output()`: Extract anomaly predictions from LLM responses
  - Visualization functions for plotting predictions

- **[result_agg.py](src/result_agg.py)**: Aggregate metrics across multiple experiments
- **[result_agg_by_model.py](src/result_agg_by_model.py)**: Aggregate and compare results by model

## Experiment Variants

The project supports multiple experiment configurations (defined in [config.py](src/config.py#L4-L153)):

- **Shot types**: `0shot` (zero-shot), `1shot` (one-shot with example)
- **Modality**: `text` (text-only), `vision` (includes time-series visualization)
- **Prompting**: `cot` (Chain-of-Thought reasoning)
- **Preprocessing**: `calc` (with calculations), `dyscalc` (without calculations), `s0.3` (scaled by 0.3)
- **Format**: `csv` (CSV format), `tpd` (token-per-digit), `pap` (point-adjusted precision)

Example variants: `0shot-text-vision`, `1shot-vision-cot`, `0shot-text-s0.3-csv`

## Data Organization

```
LLM-TSAD/
├── data/synthetic/          # AnomLLM synthetic datasets
│   ├── trend/
│   ├── point/
│   ├── freq/
│   ├── range/
│   └── ...
├── TSB-AD/Datasets/        # TSB-AD-U real-world datasets
├── results/                # Experiment results (JSONL format)
│   ├── synthetic/
│   └── tsb-ad-u/
└── credentials.yml         # API credentials (not in repo)
```

## Supported Models

- **OpenAI**: `gpt-4o`, `gpt-4o-mini`
- **Google Gemini**: `gemini-1.5-flash`
- **Azure OpenAI**: Use format `azure-{deployment-name}` (e.g., `azure-gpt-4o-mini`)
- **Open-source VLMs** (via LMDeploy): `OpenGVLab/InternVL2-Llama3-76B`, `Qwen/Qwen-VL-Chat`, `Qwen/Qwen2.5-VL-3B-Instruct`, `Qwen/Qwen2.5-VL-72B-Instruct`

See [AnoAgent.py](src/neurips_our/AnoAgent.py#L35-L48) for model initialization logic.

### Azure OpenAI Setup

Azure OpenAI models use a different configuration in `credentials.yml`:
- Model name format: `azure-{deployment-name}`
- Requires: `endpoint`, `deployment`, `api_key`, `api_version`
- Implementation: [azure_api.py](src/azure_api.py)

Example usage:
```bash
python src/LLM-TSAD-AnomLLM_api.py --model azure-gpt-4o-mini --data trend --variant 0shot-text-vision
```

## Result Format

Results are saved as JSONL files in `results/` with structure:
```json
{
  "custom_id": "dataset_model_variant_00001",
  "request": {...},
  "response": {...},
  "pred_intervals": [[start, end], ...],
  "metrics": {"precision": 0.85, "recall": 0.90, ...}
}
```
