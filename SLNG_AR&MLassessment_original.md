# Applied Speech ML Engineer - Technical Assessment
## Complete Assessment Materials

---

# Table of Contents

1. [Assessment Overview](#assessment-overview)
2. [Setup Instructions](#setup-instructions)
3. [Part 1: TTS Optimization Challenge](#part-1-tts-optimization-challenge)
4. [Part 2: ASR Evaluation Framework](#part-2-asr-evaluation-framework)
5. [Part 3: Streaming ASR System](#part-3-streaming-asr-system)
6. [Supporting Materials](#supporting-materials)
7. [Evaluation Criteria](#evaluation-criteria)
8. [Submission Guidelines](#submission-guidelines)

---

# Assessment Overview

## Role Context
You'll make speech models fast. From TTS voices to multilingual ASR, you'll benchmark, optimize, and productionize speech inference at scale.

## Technical Challenge
This assessment evaluates your ability to:
- Quantize neural TTS models for minimal latency on heterogeneous GPU hardware
- Benchmark ASR models across dialectal variations, measuring WER and latency trade-offs
- Implement continuous batching and KV-cache reuse for streaming inference
- Profile GPU utilization with CUDA kernels to identify bottlenecks

## Assessment Structure
- **Duration:** 4-5 hours (take-home with follow-up discussion)
- **Parts:** 3 technical challenges of increasing complexity
- **Format:** Hands-on coding with performance optimization

---

# Setup Instructions

## Requirements

### System Dependencies
```bash
pip install torch torchaudio numpy matplotlib seaborn
pip install python-Levenshtein scipy GPUtil psutil
pip install jupyter notebook  # Optional, for analysis
```

### GPU Setup (Recommended)
- NVIDIA GPU with CUDA support
- At least 4GB GPU memory
- CUDA toolkit installed

### Directory Structure
```
speech_ml_assessment/
├── baseline_inference.py
├── profiling_utils.py  
├── asr_evaluation_framework.py
├── streaming_asr_system.py
├── sample_text_inputs.json
├── models/                    # (created automatically)
├── results/                   # (for outputs)
└── profiler_output/          # (for profiling results)
```

---

# Part 1: TTS Optimization Challenge

## Overview
You're given a PyTorch TTS model that's too slow for production. Your task is to optimize it for real-time inference.

## Current Performance Baseline
- **Latency:** 800ms average for 50-character input
- **Throughput:** 15 samples/second on V100
- **Memory:** 4GB GPU memory usage
- **Target:** <200ms latency, >50 samples/second

## Code: baseline_inference.py

```python
"""
Baseline TTS Inference Implementation
This is the current slow implementation that needs optimization.
"""

import torch
import torch.nn as nn
import time
import numpy as np
from typing import List, Dict, Tuple
import json

class SimpleTTSModel(nn.Module):
    """
    Simplified TTS model for testing purposes.
    This represents a typical autoregressive TTS architecture.
    """
    def __init__(self, vocab_size=256, hidden_dim=512, num_layers=6, mel_dim=80):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.mel_dim = mel_dim
        
        # Text encoder
        self.text_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.text_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_dim, nhead=8, batch_first=True),
            num_layers=num_layers
        )
        
        # Decoder (autoregressive mel spectrogram generation)
        self.mel_embedding = nn.Linear(mel_dim, hidden_dim)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(hidden_dim, nhead=8, batch_first=True),
            num_layers=num_layers
        )
        
        # Output projection
        self.mel_projection = nn.Linear(hidden_dim, mel_dim)
        self.stop_token = nn.Linear(hidden_dim, 1)
        
        # Vocoder (simplified)
        self.vocoder = nn.Sequential(
            nn.ConvTranspose1d(mel_dim, 256, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose1d(256, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 1, 4, 2, 1),
            nn.Tanh()
        )
    
    def encode_text(self, text_tokens):
        """Encode input text tokens"""
        text_emb = self.text_embedding(text_tokens)
        text_encoded = self.text_encoder(text_emb)
        return text_encoded
    
    def decode_mel(self, text_encoded, mel_input=None, max_length=500):
        """Autoregressive mel spectrogram generation"""
        batch_size = text_encoded.size(0)
        device = text_encoded.device
        
        if mel_input is None:
            # Start with zero frame
            mel_input = torch.zeros(batch_size, 1, self.mel_dim, device=device)
        
        outputs = []
        stop_probs = []
        
        for step in range(max_length):
            # Embed previous mel frames
            mel_emb = self.mel_embedding(mel_input)
            
            # Decode next frame
            decoder_out = self.decoder(mel_emb, text_encoded)
            
            # Predict next mel frame and stop probability
            next_mel = self.mel_projection(decoder_out[:, -1:, :])
            stop_prob = torch.sigmoid(self.stop_token(decoder_out[:, -1:, :]))
            
            outputs.append(next_mel)
            stop_probs.append(stop_prob)
            
            # Update input for next step
            mel_input = torch.cat([mel_input, next_mel], dim=1)
            
            # Stop if all sequences predict stop
            if torch.all(stop_prob > 0.5):
                break
        
        mel_output = torch.cat(outputs, dim=1)
        stop_probs = torch.cat(stop_probs, dim=1)
        
        return mel_output, stop_probs
    
    def vocoder_inference(self, mel_spec):
        """Convert mel spectrogram to audio waveform"""
        # Transpose for conv1d: (batch, mel_dim, time)
        mel_spec = mel_spec.transpose(1, 2)
        audio = self.vocoder(mel_spec)
        return audio.squeeze(1)  # Remove channel dimension
    
    def forward(self, text_tokens, mel_target=None):
        """Full forward pass"""
        # Encode text
        text_encoded = self.encode_text(text_tokens)
        
        if mel_target is not None:
            # Training mode: teacher forcing
            mel_emb = self.mel_embedding(mel_target[:, :-1, :])
            decoder_out = self.decoder(mel_emb, text_encoded)
            mel_pred = self.mel_projection(decoder_out)
            stop_pred = self.stop_token(decoder_out)
            return mel_pred, stop_pred
        else:
            # Inference mode: autoregressive generation
            mel_output, stop_probs = self.decode_mel(text_encoded)
            audio = self.vocoder_inference(mel_output)
            return audio, mel_output, stop_probs


class BaselineTTSInference:
    """
    Current inference implementation - SLOW and needs optimization
    """
    def __init__(self, model_path: str = "pretrained_tts_model.pt"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model(model_path)
        self.model.eval()
        
    def load_model(self, model_path: str):
        """Load pretrained model"""
        # For testing, create a model instead of loading
        model = SimpleTTSModel()
        model.to(self.device)
        return model
    
    def text_to_tokens(self, text: str) -> torch.Tensor:
        """Convert text to token indices (simplified tokenization)"""
        # Simple character-level tokenization
        tokens = [ord(c) % 256 for c in text.lower()]
        return torch.tensor(tokens, device=self.device).unsqueeze(0)
    
    def synthesize(self, text: str) -> Tuple[np.ndarray, Dict]:
        """
        Synthesize speech from text - CURRENT SLOW IMPLEMENTATION
        """
        start_time = time.time()
        
        # Tokenize text
        tokens = self.text_to_tokens(text)
        
        # INEFFICIENCY 1: No batching, processing one sample at a time
        with torch.no_grad():
            # INEFFICIENCY 2: Full precision inference (FP32)
            audio, mel_spec, stop_probs = self.model(tokens)
        
        # INEFFICIENCY 3: Synchronous GPU operations
        torch.cuda.synchronize()  # Force wait for GPU
        
        # INEFFICIENCY 4: Inefficient memory usage - keeping all intermediate results
        audio_np = audio.cpu().numpy()
        mel_np = mel_spec.cpu().numpy()
        
        end_time = time.time()
        
        # Return audio and metrics
        metrics = {
            "latency": end_time - start_time,
            "text_length": len(text),
            "audio_length": audio_np.shape[-1],
            "mel_frames": mel_np.shape[1],
            "rtf": (end_time - start_time) / (audio_np.shape[-1] / 22050)  # Real-time factor
        }
        
        return audio_np[0], metrics
    
    def batch_synthesize(self, texts: List[str]) -> Tuple[List[np.ndarray], List[Dict]]:
        """
        Batch synthesis - CURRENT INEFFICIENT IMPLEMENTATION
        """
        # INEFFICIENCY: Processing each text individually instead of true batching
        results = []
        metrics = []
        
        for text in texts:
            audio, metric = self.synthesize(text)
            results.append(audio)
            metrics.append(metric)
        
        return results, metrics


def benchmark_baseline(inference_engine: BaselineTTSInference, test_texts: List[str], num_runs: int = 5):
    """
    Benchmark the baseline implementation
    """
    print("Benchmarking baseline TTS inference...")
    
    latencies = []
    rtfs = []
    
    for run in range(num_runs):
        run_latencies = []
        run_rtfs = []
        
        for text in test_texts:
            audio, metrics = inference_engine.synthesize(text)
            run_latencies.append(metrics["latency"])
            run_rtfs.append(metrics["rtf"])
        
        latencies.extend(run_latencies)
        rtfs.extend(run_rtfs)
    
    # Calculate statistics
    avg_latency = np.mean(latencies)
    p95_latency = np.percentile(latencies, 95)
    avg_rtf = np.mean(rtfs)
    throughput = 1.0 / avg_latency  # samples per second
    
    results = {
        "avg_latency": avg_latency,
        "p95_latency": p95_latency,
        "avg_rtf": avg_rtf,
        "throughput": throughput,
        "total_samples": len(latencies)
    }
    
    print(f"Baseline Results:")
    print(f"  Average Latency: {avg_latency:.3f}s")
    print(f"  95th Percentile Latency: {p95_latency:.3f}s")
    print(f"  Average RTF: {avg_rtf:.3f}")
    print(f"  Throughput: {throughput:.1f} samples/second")
    
    return results


if __name__ == "__main__":
    # Load test texts
    with open("sample_text_inputs.json", "r") as f:
        test_data = json.load(f)
    
    test_texts = test_data["texts"]
    
    # Initialize baseline inference
    inference = BaselineTTSInference()
    
    # Benchmark
    baseline_results = benchmark_baseline(inference, test_texts)
    
    print("\nIdentified bottlenecks to optimize:")
    print("1. No batching - processing samples individually")
    print("2. Full precision (FP32) - no quantization")
    print("3. Synchronous operations - not overlapping computation")
    print("4. Inefficient memory usage - storing unnecessary intermediate results")
    print("5. No caching - recomputing static components")
```

## Tasks for Part 1

### 1. Performance Analysis (45 mins)
Using the provided profiling tools:
1. Identify the top 3 bottlenecks in the inference pipeline
2. Measure GPU utilization and memory bandwidth
3. Analyze batch size vs latency trade-offs
4. Document findings with concrete metrics

### 2. Optimization Implementation (90 mins)
Implement 3 optimization techniques:
1. **Model quantization** (INT8 or mixed precision)
2. **Continuous batching** for variable-length inputs
3. **KV-cache optimization** for autoregressive generation

**Requirements:**
- Maintain audio quality (MOS score within 5% of baseline)
- Measure latency/throughput improvements
- Handle edge cases (very short/long inputs)

### 3. Production Deployment (15 mins)
Design a deployment strategy addressing:
- How to handle multiple concurrent requests
- Memory management across different GPU types
- Fallback strategies for optimization failures

---

# Part 2: ASR Evaluation Framework

## Overview
You need to benchmark ASR models across different dialects and measure WER vs latency trade-offs.

## Code: asr_evaluation_framework.py

```python
"""
ASR Evaluation Framework for Multi-Dialect Benchmarking
This framework provides tools to evaluate ASR models across different dialects
and measure WER vs latency trade-offs.
"""

import torch
import torch.nn as nn
import torchaudio
import numpy as np
import time
import json
import Levenshtein
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import scipy.stats as stats


class SimpleASRModel(nn.Module):
    """
    Simplified ASR model for testing purposes
    Represents different model sizes (small/medium/large)
    """
    def __init__(self, input_dim=80, hidden_dim=256, num_layers=4, vocab_size=1000, model_size="medium"):
        super().__init__()
        
        # Scale model based on size
        size_multipliers = {"small": 0.5, "medium": 1.0, "large": 2.0}
        multiplier = size_multipliers[model_size]
        
        self.hidden_dim = int(hidden_dim * multiplier)
        self.num_layers = int(num_layers * multiplier)
        self.model_size = model_size
        
        # Feature extraction (CNN frontend)
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((None, self.hidden_dim // 4))
        )
        
        # Recurrent layers
        self.encoder = nn.LSTM(
            self.hidden_dim // 4,
            self.hidden_dim,
            self.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1 if self.num_layers > 1 else 0
        )
        
        # Output projection
        self.classifier = nn.Linear(self.hidden_dim * 2, vocab_size)
        
        # CTC loss for sequence alignment
        self.ctc_loss = nn.CTCLoss(blank=0, reduction='mean')
    
    def forward(self, mel_specs, spec_lengths=None):
        """Forward pass through ASR model"""
        batch_size = mel_specs.size(0)
        
        # Add channel dimension for CNN
        if mel_specs.dim() == 3:
            mel_specs = mel_specs.unsqueeze(1)
        
        # Feature extraction
        features = self.feature_extractor(mel_specs)
        features = features.squeeze(3).transpose(1, 2)  # (batch, time, features)
        
        # Sequence modeling
        if spec_lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                features, spec_lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            encoded, _ = self.encoder(packed)
            encoded, output_lengths = nn.utils.rnn.pad_packed_sequence(encoded, batch_first=True)
        else:
            encoded, _ = self.encoder(features)
            output_lengths = torch.full((batch_size,), encoded.size(1), dtype=torch.long)
        
        # Classification
        logits = self.classifier(encoded)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        
        return log_probs, output_lengths


class ASREvaluator:
    """Comprehensive ASR evaluation toolkit"""
    
    def __init__(self, models_dir: str = "./models"):
        self.models_dir = Path(models_dir)
        self.models = self._load_models()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Character to index mapping (simplified)
        self.char_to_idx = {chr(i): i for i in range(32, 127)}  # ASCII printable
        self.char_to_idx['<blank>'] = 0
        self.idx_to_char = {v: k for k, v in self.char_to_idx.items()}
    
    def _load_models(self) -> Dict[str, SimpleASRModel]:
        """Load pretrained ASR models of different sizes"""
        models = {}
        
        for size in ["small", "medium", "large"]:
            model = SimpleASRModel(model_size=size)
            model.eval()
            models[size] = model
        
        return models
    
    def text_to_indices(self, text: str) -> List[int]:
        """Convert text to character indices"""
        return [self.char_to_idx.get(c, 1) for c in text.lower()]
    
    def indices_to_text(self, indices: List[int]) -> str:
        """Convert indices back to text"""
        chars = [self.idx_to_char.get(idx, '?') for idx in indices if idx != 0]  # Remove blanks
        return ''.join(chars)
    
    def ctc_decode(self, log_probs: torch.Tensor, lengths: torch.Tensor) -> List[str]:
        """Simple CTC decoding (greedy)"""
        batch_size = log_probs.size(0)
        predictions = []
        
        for i in range(batch_size):
            length = lengths[i].item()
            sequence = log_probs[i, :length].argmax(dim=-1)
            
            # Remove consecutive duplicates and blanks
            decoded = []
            prev_token = -1
            for token in sequence:
                token = token.item()
                if token != prev_token and token != 0:  # 0 is blank
                    decoded.append(token)
                prev_token = token
            
            text = self.indices_to_text(decoded)
            predictions.append(text)
        
        return predictions
    
    def compute_wer(self, reference: str, hypothesis: str) -> float:
        """Compute Word Error Rate"""
        ref_words = reference.strip().split()
        hyp_words = hypothesis.strip().split()
        
        if len(ref_words) == 0:
            return 1.0 if len(hyp_words) > 0 else 0.0
        
        # Use Levenshtein distance for word-level alignment
        distance = Levenshtein.distance(ref_words, hyp_words)
        return distance / len(ref_words)
    
    def compute_cer(self, reference: str, hypothesis: str) -> float:
        """Compute Character Error Rate"""
        if len(reference) == 0:
            return 1.0 if len(hypothesis) > 0 else 0.0
        
        distance = Levenshtein.distance(reference, hypothesis)
        return distance / len(reference)
    
    def synthesize_audio_features(self, text: str, dialect: str = "standard") -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Synthesize mel spectrogram features from text
        (In real scenario, this would be actual audio files)
        """
        # Simulate different dialects with different feature patterns
        dialect_noise = {
            "standard": 0.1,
            "southern": 0.15,
            "british": 0.12,
            "australian": 0.14,
            "indian": 0.18
        }
        
        # Generate synthetic mel spectrogram
        text_length = len(text)
        time_steps = max(50, text_length * 3)  # Rough approximation
        mel_bins = 80
        
        # Base spectrogram with some structure
        mel_spec = torch.randn(mel_bins, time_steps) * 0.5
        
        # Add dialect-specific noise
        noise_level = dialect_noise.get(dialect, 0.1)
        mel_spec += torch.randn_like(mel_spec) * noise_level
        
        # Add some temporal structure
        for i in range(0, time_steps, 10):
            end_idx = min(i + 5, time_steps)
            mel_spec[:, i:end_idx] += torch.randn(mel_bins, end_idx - i) * 0.3
        
        length = torch.tensor(time_steps)
        return mel_spec.transpose(0, 1).unsqueeze(0), length.unsqueeze(0)  # (1, time, mel)
    
    def evaluate_model_on_dataset(self, model_name: str, test_data: List[Dict], num_runs: int = 3) -> Dict:
        """Evaluate a single model on test dataset"""
        model = self.models[model_name].to(self.device)
        
        results = {
            "model_name": model_name,
            "total_samples": len(test_data),
            "per_dialect": defaultdict(list),
            "overall": {
                "wer": [],
                "cer": [],
                "latency": [],
                "rtf": []
            }
        }
        
        for run in range(num_runs):
            print(f"Run {run + 1}/{num_runs} for model {model_name}")
            
            for sample in test_data:
                text = sample["text"]
                dialect = sample["dialect"]
                
                # Generate audio features
                mel_spec, length = self.synthesize_audio_features(text, dialect)
                mel_spec = mel_spec.to(self.device)
                length = length.to(self.device)
                
                # Measure inference time
                start_time = time.time()
                
                with torch.no_grad():
                    log_probs, output_lengths = model(mel_spec, length)
                    predictions = self.ctc_decode(log_probs, output_lengths)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                end_time = time.time()
                inference_time = end_time - start_time
                
                # Compute metrics
                predicted_text = predictions[0] if predictions else ""
                wer = self.compute_wer(text, predicted_text)
                cer = self.compute_cer(text, predicted_text)
                
                # Real-time factor (assuming 16kHz audio)
                audio_duration = length.item() * 0.01  # 10ms per frame
                rtf = inference_time / audio_duration if audio_duration > 0 else float('inf')
                
                # Store results
                results["per_dialect"][dialect].append({
                    "wer": wer,
                    "cer": cer,
                    "latency": inference_time,
                    "rtf": rtf,
                    "text": text,
                    "prediction": predicted_text
                })
                
                results["overall"]["wer"].append(wer)
                results["overall"]["cer"].append(cer)
                results["overall"]["latency"].append(inference_time)
                results["overall"]["rtf"].append(rtf)
        
        # Compute aggregate statistics
        for dialect in results["per_dialect"]:
            dialect_data = results["per_dialect"][dialect]
            results["per_dialect"][dialect] = {
                "samples": len(dialect_data),
                "wer": {
                    "mean": np.mean([d["wer"] for d in dialect_data]),
                    "std": np.std([d["wer"] for d in dialect_data]),
                    "p95": np.percentile([d["wer"] for d in dialect_data], 95)
                },
                "cer": {
                    "mean": np.mean([d["cer"] for d in dialect_data]),
                    "std": np.std([d["cer"] for d in dialect_data])
                },
                "latency": {
                    "mean": np.mean([d["latency"] for d in dialect_data]),
                    "p95": np.percentile([d["latency"] for d in dialect_data], 95)
                },
                "rtf": {
                    "mean": np.mean([d["rtf"] for d in dialect_data])
                }
            }
        
        # Overall statistics
        for metric in ["wer", "cer", "latency", "rtf"]:
            data = results["overall"][metric]
            results["overall"][metric] = {
                "mean": np.mean(data),
                "std": np.std(data),
                "p95": np.percentile(data, 95),
                "min": np.min(data),
                "max": np.max(data)
            }
        
        return results
    
    def benchmark_all_models(self, test_data: List[Dict]) -> Dict:
        """Benchmark all models on the test dataset"""
        all_results = {}
        
        for model_name in self.models.keys():
            print(f"Evaluating model: {model_name}")
            results = self.evaluate_model_on_dataset(model_name, test_data)
            all_results[model_name] = results
        
        return all_results
    
    def analyze_statistical_significance(self, results: Dict) -> Dict:
        """Perform statistical significance testing between models"""
        model_names = list(results.keys())
        significance_tests = {}
        
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names[i+1:], i+1):
                wer1 = [sample["wer"] for dialect_data in results[model1]["per_dialect"].values() 
                       for sample in dialect_data if isinstance(dialect_data, list)]
                wer2 = [sample["wer"] for dialect_data in results[model2]["per_dialect"].values() 
                       for sample in dialect_data if isinstance(dialect_data, list)]
                
                if len(wer1) > 0 and len(wer2) > 0:
                    # Perform t-test
                    t_stat, p_value = stats.ttest_ind(wer1, wer2)
                    
                    significance_tests[f"{model1}_vs_{model2}"] = {
                        "t_statistic": t_stat,
                        "p_value": p_value,
                        "significant": p_value < 0.05,
                        "effect_size": abs(np.mean(wer1) - np.mean(wer2)) / np.sqrt((np.var(wer1) + np.var(wer2)) / 2)
                    }
        
        return significance_tests


# Test data generator
def generate_test_dataset(num_samples_per_dialect: int = 20) -> List[Dict]:
    """Generate synthetic test dataset with multiple dialects"""
    
    dialects = ["standard", "southern", "british", "australian", "indian"]
    
    sample_texts = [
        "hello world how are you today",
        "the quick brown fox jumps over the lazy dog",
        "artificial intelligence is changing our world",
        "please call me back when you get this message",
        "weather forecast shows rain for the next three days",
        "machine learning models require large amounts of data",
        "speech recognition accuracy depends on audio quality",
        "natural language processing enables human computer interaction",
        "deep learning has revolutionized speech technology",
        "voice assistants are becoming increasingly popular"
    ]
    
    test_data = []
    
    for dialect in dialects:
        for i in range(num_samples_per_dialect):
            text = sample_texts[i % len(sample_texts)]
            test_data.append({
                "text": text,
                "dialect": dialect,
                "id": f"{dialect}_{i:03d}"
            })
    
    return test_data
```

## Tasks for Part 2

### 1. Comprehensive Evaluation Pipeline (60 mins)
Build an evaluation system that measures:

**Performance Metrics:**
- WER per dialect and overall
- Character Error Rate (CER)
- Real-time factor (RTF)
- 95th percentile latency

**Quality Analysis:**
- Error breakdown by phoneme/word type
- Performance correlation with audio duration
- Statistical significance testing across dialects

### 2. Optimization Strategy (30 mins)
Based on your benchmarks, propose:
1. Which model to use for each dialect
2. Dynamic model selection strategy
3. Specific optimizations for worst-performing cases
4. Trade-off analysis: accuracy vs speed vs memory

---

# Part 3: Streaming ASR System

## Overview
Design and implement a streaming ASR system with continuous batching support for real-time speech recognition.

## Challenge Requirements
- Process audio chunks in real-time (streaming)
- Support multiple concurrent streams
- Implement dynamic batching across streams
- Maintain context across chunks per stream
- Handle variable-length inputs efficiently

## Constraints
- <100ms end-to-end latency per chunk
- Support 50+ concurrent streams
- Memory usage <8GB on single GPU

## Code: streaming_asr_system.py

```python
"""
Streaming ASR System with Continuous Batching
Template for implementing real-time speech recognition with multiple concurrent streams.
"""

import torch
import torch.nn as nn
import numpy as np
import threading
import time
import queue
import uuid
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict, deque
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import asyncio
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AudioChunk:
    """Represents an audio chunk from a stream"""
    stream_id: str
    chunk_id: int
    audio_data: np.ndarray
    timestamp: float
    is_final: bool = False


@dataclass
class StreamState:
    """Maintains state for each audio stream"""
    stream_id: str
    created_at: float
    last_activity: float
    audio_buffer: deque
    context_buffer: deque
    partial_transcript: str
    final_transcript: str
    processed_chunks: int
    
    def __post_init__(self):
        if not hasattr(self, 'audio_buffer') or self.audio_buffer is None:
            self.audio_buffer = deque(maxlen=50)  # Keep last 50 chunks
        if not hasattr(self, 'context_buffer') or self.context_buffer is None:
            self.context_buffer = deque(maxlen=10)  # Keep context for continuity


class StreamBatch:
    """Manages batching of multiple streams for efficient processing"""
    
    def __init__(self, max_batch_size: int = 8, max_wait_time: float = 0.05):
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.pending_chunks: List[AudioChunk] = []
        self.last_batch_time = time.time()
    
    def add_chunk(self, chunk: AudioChunk) -> bool:
        """Add chunk to batch. Returns True if batch is ready."""
        self.pending_chunks.append(chunk)
        
        # Batch is ready if we hit size limit or time limit
        current_time = time.time()
        time_elapsed = current_time - self.last_batch_time
        
        return (len(self.pending_chunks) >= self.max_batch_size or 
                time_elapsed >= self.max_wait_time)
    
    def get_batch(self) -> List[AudioChunk]:
        """Get current batch and reset"""
        batch = self.pending_chunks.copy()
        self.pending_chunks.clear()
        self.last_batch_time = time.time()
        return batch


class StreamingASREngine:
    """
    Main streaming ASR engine with continuous batching support
    """
    
    def __init__(self, model_path: str, device: str = "auto"):
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else device)
        self.model = self._load_model(model_path)
        
        # Stream management
        self.active_streams: Dict[str, StreamState] = {}
        self.stream_lock = threading.RLock()
        
        # Batching configuration
        self.batch_manager = StreamBatch(max_batch_size=8, max_wait_time=0.05)
        self.processing_queue = queue.Queue(maxsize=100)
        
        # Worker threads
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.processing_thread = None
        self.is_running = False
        
        # Performance monitoring
        self.stats = {
            "total_chunks_processed": 0,
            "average_latency": 0.0,
            "active_streams_count": 0,
            "batch_utilization": 0.0
        }
        
        # Start processing
        self.start()
    
    def _load_model(self, model_path: str) -> nn.Module:
        """Load the ASR model"""
        # For this template, we'll use a simplified model
        # In practice, load your actual model here
        from asr_evaluation_framework import SimpleASRModel
        
        model = SimpleASRModel(model_size="medium")
        model.to(self.device)
        model.eval()
        return model
    
    def start(self):
        """Start the processing engine"""
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.start()
        logger.info("Streaming ASR engine started")
    
    def stop(self):
        """Stop the processing engine"""
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join()
        self.executor.shutdown(wait=True)
        logger.info("Streaming ASR engine stopped")
    
    def add_stream(self, stream_id: Optional[str] = None) -> str:
        """Add a new audio stream"""
        if stream_id is None:
            stream_id = str(uuid.uuid4())
        
        with self.stream_lock:
            if stream_id in self.active_streams:
                logger.warning(f"Stream {stream_id} already exists")
                return stream_id
            
            current_time = time.time()
            self.active_streams[stream_id] = StreamState(
                stream_id=stream_id,
                created_at=current_time,
                last_activity=current_time,
                audio_buffer=deque(maxlen=50),
                context_buffer=deque(maxlen=10),
                partial_transcript="",
                final_transcript="",
                processed_chunks=0
            )
            
            self.stats["active_streams_count"] = len(self.active_streams)
            logger.info(f"Added stream {stream_id}")
        
        return stream_id
    
    def process_chunk(self, stream_id: str, audio_chunk: np.ndarray) -> str:
        """
        Process an audio chunk and return partial transcript
        
        Args:
            stream_id: Unique identifier for the stream
            audio_chunk: Audio data (numpy array)
        
        Returns:
            Partial transcript for this chunk
        """
        with self.stream_lock:
            if stream_id not in self.active_streams:
                raise ValueError(f"Stream {stream_id} not found")
            
            stream_state = self.active_streams[stream_id]
            stream_state.last_activity = time.time()
        
        # Create chunk object
        chunk = AudioChunk(
            stream_id=stream_id,
            chunk_id=stream_state.processed_chunks,
            audio_data=audio_chunk,
            timestamp=time.time()
        )
        
        # Add to processing queue
        try:
            self.processing_queue.put(chunk, timeout=0.1)
        except queue.Full:
            logger.warning(f"Processing queue full, dropping chunk for stream {stream_id}")
            return stream_state.partial_transcript
        
        # Return current partial transcript
        return stream_state.partial_transcript
    
    def finalize_stream(self, stream_id: str) -> str:
        """
        Finalize a stream and return the complete transcript
        
        Args:
            stream_id: Unique identifier for the stream
        
        Returns:
            Final complete transcript
        """
        with self.stream_lock:
            if stream_id not in self.active_streams:
                raise ValueError(f"Stream {stream_id} not found")
            
            stream_state = self.active_streams[stream_id]
            final_transcript = stream_state.final_transcript + " " + stream_state.partial_transcript
            
            # Clean up stream
            del self.active_streams[stream_id]
            self.stats["active_streams_count"] = len(self.active_streams)
            
            logger.info(f"Finalized stream {stream_id}")
            return final_transcript.strip()
```

## Supporting Materials

### Sample Text Inputs (sample_text_inputs.json)

```json
{
  "texts": [
    "Hello world!",
    "Good morning, how are you today?",
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is transforming the way we interact with technology.",
    "In a hole in the ground there lived a hobbit. Not a nasty, dirty, wet hole, filled with the ends of worms and an oozy smell.",
    "To be or not to be, that is the question. Whether 'tis nobler in the mind to suffer the slings and arrows of outrageous fortune.",
    "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness, it was the epoch of belief, it was the epoch of incredulity, it was the season of Light, it was the season of Darkness.",
    "Call me Ishmael. Some years ago—never mind how long precisely—having little or no money in my purse, and nothing particular to interest me on shore, I thought I would sail about a little and see the watery part of the world.",
    "In the beginning was the Word, and the Word was with God, and the Word was God. The same was in the beginning with God. All things were made by him; and without him was not any thing made that was made.",
    "Four score and seven years ago our fathers brought forth on this continent, a new nation, conceived in Liberty, and dedicated to the proposition that all men are created equal. Now we are engaged in a great civil war, testing whether that nation or any nation so conceived and so dedicated, can long endure."
  ],
  "test_categories": {
    "short": ["Hello world!", "Good morning, how are you today?", "The quick brown fox jumps over the lazy dog."],
    "medium": [
      "Artificial intelligence is transforming the way we interact with technology.",
      "In a hole in the ground there lived a hobbit. Not a nasty, dirty, wet hole, filled with the ends of worms and an oozy smell.",
      "To be or not to be, that is the question. Whether 'tis nobler in the mind to suffer the slings and arrows of outrageous fortune."
    ],
    "long": [
      "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness, it was the epoch of belief, it was the epoch of incredulity, it was the season of Light, it was the season of Darkness.",
      "Call me Ishmael. Some years ago—never mind how long precisely—having little or no money in my purse, and nothing particular to interest me on shore, I thought I would sail about a little and see the watery part of the world.",
      "In the beginning was the Word, and the Word was with God, and the Word was God. The same was in the beginning with God. All things were made by him; and without him was not any thing made that was made.",
      "Four score and seven years ago our fathers brought forth on this continent, a new nation, conceived in Liberty, and dedicated to the proposition that all men are created equal. Now we are engaged in a great civil war, testing whether that nation or any nation so conceived and so dedicated, can long endure."
    ]
  },
  "metadata": {
    "character_counts": {
      "min": 12,
      "max": 573,
      "average": 197
    },
    "expected_performance": {
      "baseline_latency_range": "200-800ms",
      "target_latency": "<200ms",
      "target_throughput": ">50 samples/sec"
    }
  }
}
```

### GPU Profiling Utilities (profiling_utils.py)

```python
"""
GPU Profiling Utilities for Performance Analysis
Use these tools to identify bottlenecks in your TTS inference pipeline.
"""

import torch
import time
import psutil
import GPUtil
import numpy as np
from typing import Dict, List, Any, Optional, Callable
from contextlib import contextmanager
from collections import defaultdict


class GPUProfiler:
    """Comprehensive GPU profiling utilities"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.reset_stats()
    
    def reset_stats(self):
        """Reset all profiling statistics"""
        self.timings = defaultdict(list)
        self.memory_usage = defaultdict(list)
        self.gpu_utilization = []
        self.batch_sizes = []
        self.throughput_data = []
        
    @contextmanager
    def profile_section(self, section_name: str):
        """Context manager to profile a code section"""
        # Clear GPU cache and synchronize
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Record initial state
        start_time = time.time()
        start_memory = self.get_gpu_memory_usage()
        
        try:
            yield
        finally:
            # Record final state
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end_time = time.time()
            end_memory = self.get_gpu_memory_usage()
            
            # Store measurements
            self.timings[section_name].append(end_time - start_time)
            self.memory_usage[section_name].append(end_memory - start_memory)
    
    def get_gpu_memory_usage(self) -> float:
        """Get current GPU memory usage in MB"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        return 0.0
    
    def get_gpu_utilization(self) -> float:
        """Get current GPU utilization percentage"""
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                return gpus[0].load * 100
        except:
            pass
        return 0.0
    
    def profile_inference_batch(self, inference_fn: Callable, inputs: List[Any], batch_sizes: List[int]):
        """Profile inference across different batch sizes"""
        results = {}
        
        for batch_size in batch_sizes:
            print(f"Profiling batch size: {batch_size}")
            
            # Prepare batched inputs
            batched_inputs = []
            for i in range(0, len(inputs), batch_size):
                batch = inputs[i:i + batch_size]
                batched_inputs.append(batch)
            
            batch_times = []
            batch_memory = []
            batch_utilization = []
            
            for batch in batched_inputs:
                # Warm up
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                start_mem = self.get_gpu_memory_usage()
                start_util = self.get_gpu_utilization()
                
                with self.profile_section(f"batch_size_{batch_size}"):
                    _ = inference_fn(batch)
                
                end_util = self.get_gpu_utilization()
                peak_mem = self.get_gpu_memory_usage()
                
                batch_times.append(self.timings[f"batch_size_{batch_size}"][-1])
                batch_memory.append(peak_mem - start_mem)
                batch_utilization.append(max(start_util, end_util))
            
            # Calculate metrics
            avg_latency = np.mean(batch_times)
            throughput = batch_size / avg_latency
            avg_memory = np.mean(batch_memory)
            avg_utilization = np.mean(batch_utilization)
            
            results[batch_size] = {
                "latency": avg_latency,
                "throughput": throughput,
                "memory_usage": avg_memory,
                "gpu_utilization": avg_utilization,
                "samples": len(batch_times)
            }
        
        return results
```

---

# Evaluation Criteria

## Technical Implementation (40%)

### Code Quality
- **Clean Architecture:** Well-structured, modular code with clear separation of concerns
- **Performance:** Measurable improvements over baseline implementation
- **Error Handling:** Robust handling of edge cases and failure modes
- **Documentation:** Clear comments and explanations of optimization decisions

### Speech-Specific Technical Depth
- **Model Optimization:** Understanding of quantization, pruning, and efficiency techniques
- **Audio Processing:** Proper handling of variable-length sequences and streaming audio
- **Real-time Constraints:** Awareness of latency requirements and memory limitations
- **GPU Utilization:** Effective use of CUDA and parallel processing

## Optimization Results (30%)

### Performance Improvements
**TTS Optimization Targets:**
- Latency reduction: >4x improvement (800ms → <200ms)
- Throughput increase: >3x improvement (15 → >50 samples/sec)
- Memory reduction: >50% improvement (4GB → <2GB)
- Maintain audio quality within 5% of baseline

**ASR Evaluation Quality:**
- Comprehensive WER/CER analysis across dialects
- Statistical significance testing between models
- Latency vs accuracy trade-off quantification
- Model selection strategy with clear rationale

**Streaming System Performance:**
- Sub-100ms chunk processing latency
- Support for 50+ concurrent streams
- Efficient memory utilization (<8GB total)
- Graceful degradation under load

## Problem-Solving Approach (20%)

### Systematic Optimization Process
- **Profiling First:** Use of profiling tools to identify actual bottlenecks
- **Iterative Improvement:** Step-by-step optimization with measurements
- **Trade-off Analysis:** Understanding of speed vs accuracy vs memory trade-offs
- **Validation:** Proper testing of optimizations to ensure correctness

### Research and Analysis Skills
- **Benchmarking Methodology:** Proper experimental design and statistical analysis
- **Performance Characterization:** Understanding of model behavior across conditions
- **Documentation:** Clear explanation of findings and recommendations

## System Design Understanding (10%)

### Production Readiness
- **Scalability:** How would the system handle increased load?
- **Monitoring:** What metrics would you track in production?
- **Deployment:** Understanding of edge cases and failure modes
- **Maintenance:** How would you debug and update the system?

### Architecture Decisions
- **Component Design:** Proper separation of concerns and modularity
- **Resource Management:** Efficient use of GPU memory and compute
- **Concurrency:** Safe handling of multiple concurrent streams
- **Recovery:** Graceful handling of failures and resource exhaustion

---

# Submission Guidelines

## Required Deliverables

### 1. Optimized Code Implementation

**File: `optimized_tts_inference.py`**
- Your improved TTS inference system
- Must demonstrate significant performance improvements
- Include benchmarking and comparison with baseline
- Document all optimization techniques used

**File: `comprehensive_asr_evaluation.py`**
- Enhanced ASR evaluation framework
- Statistical analysis and significance testing
- Model comparison and selection strategy
- Performance vs accuracy trade-off analysis

**File: `production_streaming_asr.py`**
- Production-ready streaming ASR system
- Support for concurrent streams and batching
- Performance monitoring and error handling
- Comprehensive test suite

### 2. Performance Analysis Report

**Requirements:**
- Before/after performance benchmarks with specific metrics
- Detailed explanation of optimization strategies employed
- Trade-off analysis (speed vs accuracy vs memory)
- Production deployment recommendations and considerations

**Format:** PDF or Markdown document, 3-5 pages

### 3. Technical Documentation

**Implementation Notes:**
- Key optimization decisions and rationale
- Performance bottlenecks identified through profiling
- Future improvement suggestions and roadmap
- Production considerations and monitoring strategy

## Expected Performance Results

### TTS Optimization Benchmarks
```
Metric           | Baseline | Target    | Your Result
-----------------|----------|-----------|------------
Avg Latency      | 800ms    | <200ms    | _____ ms
P95 Latency      | 1200ms   | <300ms    | _____ ms
Throughput       | 15/sec   | >50/sec   | _____ /sec
Memory Usage     | 4GB      | <2GB      | _____ GB
GPU Utilization  | ~30%     | >70%      | _____ %
```

### ASR Evaluation Results
```
Model Comparison Matrix:
Model    | WER (%) | CER (%) | Latency (ms) | RTF  | Memory (GB)
---------|---------|---------|--------------|------|------------
Small    | ____    | ____    | ____         | ____ | ____
Medium   | ____    | ____    | ____         | ____ | ____
Large    | ____    | ____    | ____         | ____ | ____

Dialect Performance Analysis:
- Standard: ___% WER
- Southern: ___% WER  
- British: ___% WER
- Australian: ___% WER
- Indian: ___% WER
```

### Streaming System Metrics
```
Performance Requirements:
- Chunk Latency: <100ms (achieved: _____ ms)
- Concurrent Streams: >50 (achieved: _____ streams)
- Memory Usage: <8GB (achieved: _____ GB)
- Batch Utilization: >80% (achieved: _____ %)
```

## Submission Format

### Code Organization
```
submission/
├── optimized_tts_inference.py
├── comprehensive_asr_evaluation.py  
├── production_streaming_asr.py
├── performance_report.pdf
├── technical_documentation.md
├── results/
│   ├── tts_benchmarks.json
│   ├── asr_evaluation_results.json
│   └── streaming_performance.json
└── README.md
```

### README Requirements
- Setup instructions and dependencies
- How to run each component
- Brief explanation of optimizations implemented
- Key results summary

## Tips for Success

### 1. Start with Profiling
- Use the provided profiling tools to understand where time is spent
- Measure before optimizing to avoid premature optimization
- Profile on representative workloads and data

### 2. Systematic Optimization
- Implement one optimization at a time
- Measure the impact of each change
- Keep a log of what works and what doesn't

### 3. Think Production
- Consider real-world deployment challenges
- Handle edge cases and error conditions gracefully
- Design for monitoring and debugging

### 4. Document Trade-offs
- Explain why you made specific optimization choices
- Quantify the trade-offs between speed, accuracy, and memory
- Consider the business impact of your decisions

### 5. Validate Thoroughly
- Test with different input sizes and types
- Verify that optimizations don't break functionality
- Measure performance under realistic conditions

## Common Pitfalls to Avoid

### Technical Pitfalls
1. **Optimizing without profiling** - Wasting time on non-bottlenecks
2. **Ignoring accuracy degradation** - Sacrificing quality for speed
3. **Memory leaks in streaming** - Poor resource management
4. **Inefficient batching** - Increasing latency instead of reducing it
5. **Poor error handling** - System failures in production scenarios

### Evaluation Pitfalls
1. **Cherry-picking metrics** - Not reporting negative results
2. **Insufficient statistical testing** - Drawing conclusions from noise
3. **Unrealistic benchmarks** - Testing on non-representative data
4. **Ignoring variance** - Not accounting for performance variability

### System Design Pitfalls
1. **Over-engineering** - Adding unnecessary complexity
2. **Under-engineering** - Ignoring production requirements
3. **Poor monitoring** - No visibility into system behavior
4. **Rigid architecture** - Difficult to modify or extend

## Follow-up Technical Discussion

After submission, be prepared to discuss:

### Deep Dive Questions
1. **"Walk me through your optimization decisions. What would you try next?"**
2. **"How would you handle a sudden 10x increase in traffic?"**
3. **"What monitoring metrics would you track in production?"**
4. **"How would you approach optimizing for mobile/edge deployment?"**
5. **"Describe a time when you had to balance model accuracy vs inference speed."**

### Technical Challenges
- Explain the theoretical foundations of your optimizations
- Discuss alternative approaches and why you chose your solution
- Demonstrate understanding of the broader speech ML ecosystem
- Show awareness of current research and industry trends

---

**Good luck with the assessment!**