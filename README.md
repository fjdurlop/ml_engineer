# Applied Speech ML Engineer Assessment

This repository contains solutions for the Applied Speech ML Engineer technical assessment. It includes three parts:
- Part 1: TTS Optimization
- Part 2: ASR Evaluation Framework
- Part 3: Streaming ASR System

## Table of Contents
- [Applied Speech ML Engineer Assessment](#applied-speech-ml-engineer-assessment)
  - [Table of Contents](#table-of-contents)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Setup](#setup)
  - [Part 1: TTS Optimization](#part-1-tts-optimization)
  - [Part 2: ASR Evaluation](#part-2-asr-evaluation)
  - [Part 3: Streaming ASR System](#part-3-streaming-asr-system)
  - [Future improvement suggestions and roadmap](#future-improvement-suggestions-and-roadmap)
  - [Production considerations and monitoring strategy](#production-considerations-and-monitoring-strategy)
  - [Troubleshooting](#troubleshooting)

## Prerequisites
- Python 3.12+
- [uv CLI](https://github.com/xthexder/uv) (for package management and script execution)
- NVIDIA GPU with CUDA (optional, recommended for benchmarking)

## Installation
Install the `uv` CLI and project dependencies:
```bash
pip install uv
uv install
```

## Setup
Make the setup script executable and run it:
```bash
chmod +x setup.sh
./setup.sh | tee setup.out
```

Use pyproject for uv sync in order to reproduce

## Part 1: TTS Optimization
Report: [Report_p1](p1.md)

Solution: [`baseline_inference.py`](baseline_inference.py:1)

Usage examples:

Generate comparison original vs improved
```bash
uv run baseline_inference.py --num_runs 5 \
  2>&1 | tee logs/p1/baseline_inference.out
```

Run with implemented batching.
todo: Not all improvements are enabled when batching
```bash
uv run baseline_inference.py --num_runs 5 --batching True\
  2>&1 | tee logs/p1/baseline_inference_batching.out
```

Optimizations implemented:
- Batching
- Automated quantization according to device
  - Supported: fp32, fp16
- Improvements in memory management
- KV caching in decoding step, reducing quadratic inference time to linear

## Part 2: ASR Evaluation
Report: [Report_p2](p2.md)

Solution: [`asr_evaluation_framework.py`](asr_evaluation_framework.py:1)

Usage:
```bash
uv run asr_evaluation_framework.py \
  2>&1 | tee logs/p2/asr_evaluation_framework.out
```
Get statistical testing report:
```bash
uv run asr_eval_models.py
uv run asr_eval_dialects.py
```

Simulate significant differences accross models
```bash
uv run asr_evaluation_framework.py  --runs 2 --sig_model True 2>&1 | tee logs/p2/asr_evaluation_framework_significant_model.out

```
Get statistical testing report:

```bash
uv run asr_eval_models.py
uv run asr_eval_dialects.py
```

Simulate significant differences accross dialects

```bash
uv run asr_evaluation_framework.py  --runs 2 --sig_dialect True 2>&1 | tee logs/p2/asr_evaluation_framework_significant_dialect.out
```
Get statistical testing report:

```bash
uv run asr_eval_models.py
uv run asr_eval_dialects.py
```

Features:
- WER/CER per dialect and overall
- Statistical significance testing accross models and accross dialects
- Trade-off analysis: accuracy vs speed vs memory # todo

## Part 3: Streaming ASR System
Report: [Report_p3](p3.md)

Solution: [`streaming_asr_system.py`](streaming_asr_system.py:1)

Usage:
```bash
uv run streaming_asr_system.py 2>&1 | tee logs/p3/streaming_asr_system.out
```

Added to simulate input for streaming
```bash
# ran from streaming_asr_system.py
# streaming_ingress.py
```


## Future improvement suggestions and roadmap
p1
- Explore additional quantization techniques (e.g., INT4, mixed-precision pruning).
- Improve heterogeneous hardware support and their policies:
  - only CPU, different GPUs
- continuous batching
- adaptive batching strategies and dynamic load balancing.

p2
- Statistical testing reports for different metrics
- Improve trade-off analysis

## Production considerations and monitoring strategy
- Containerize services using Docker and orchestrate with Kubernetes for scalability.
- Establish CI/CD pipelines for automated testing, model validation, and deployment.
- Implement centralized logging and real-time monitoring with Prometheus & Grafana.
- Define service-level objectives (SLOs) and set up alerting for latency, throughput, and error thresholds.
- Employ model versioning and experiment tracking (e.g., MLflow, DVC) for reproducibility.
- Schedule periodic retraining and data drift detection to maintain model accuracy.

## Troubleshooting
- Check logs under `logs/` for output examples

