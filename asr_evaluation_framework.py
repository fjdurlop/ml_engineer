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