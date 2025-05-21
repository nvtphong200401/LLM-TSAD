# README for Supplementary Material

This document provides an overview of the supplementary materials submitted in support of our paper:

**Paper Title**: *Delving into Large Language Models for Effective Time-Series Anomaly Detection* 


# Environment Setup
* This repository is built upon [AnomLLM](https://github.com/rose-stl-lab/anomllm), and therefore **must be set up using the AnomLLM environment**.  
Please follow the installation instructions provided in the AnomLLM repository before proceeding.

* Dataset Download
  * AnomLLM: Downloaad "anomllm.zip" in the provied link in README of https://github.com/rose-stl-lab/anomllm
  * TSB-AD-U: Download "Datasets" directory in https://github.com/TheDatumOrg/TSB-AD/tree/main/Datasets
 
* Requirements
```

```

* API setting
```

``` 

# Run Our Method

## Experiemntal Results on AnomLLM Benchmark

1. Run online api
```
python src/LLM-TSAD-AnomLLM_api.py --model gemini-1.5-flash --data trend --variant 0shot-text-vision
```

2. Aggregate evaluation results
```
python ./src/result_agg_by_model.py --model gemini-1.5-flash --benchmark anomllm
```

## Experiemntal Results on TSB-AD-U Benchmark

1. Run online api
```
python src/LLM-TSAD-TSB_api.py --model gemini-1.5-flash --datadir ./TSB-AD/Datasets
```

2. Aggregate evaluation results
```
python ./src/result_agg_by_model.py --model gemini-1.5-flash  --benchmark tsb-ad-u
```
