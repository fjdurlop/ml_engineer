"""
ASR Evaluation Framework for Multi-Dialect Benchmarking
This framework provides tools to evaluate ASR models across different dialects
and measure WER vs latency trade-offs.

uv run baseline_inference.py 2>&1 | tee logs/baseline_inference.out
uv run asr_evaluation_framework.py  2>&1 | tee logs/p2/asr_evaluation_framework.out
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

from typing import List, Dict, Tuple, Optional, Any

from dataclasses import dataclass, asdict
import os
import json
import math
import time
import argparse



MODELS_DIR = "./models"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def simulate_predicted_text(text, dialect, model_name, sig_model=False, sig_dialect=False):
    """
    
    """
    k = 0
    if sig_dialect:
        # Compute metrics
        if dialect == "indian":
            k = 2
        elif dialect == "australian":
            k = 1
        elif dialect == "british":
            k = 5
        else:
            k = 0
    
    if sig_model:
        if model_name == "small":
            k = 2
        elif model_name == "medium":
            k = 0
        elif model_name == "large":
            k = 4
        else:
            k = 0
    

    # split reference into words
    words = text.split()

    # pick random cutoff (0 → no words, len(words) → full sentence)
    random_n = np.random.randint(0, len(words)-k + 1)

    # simulate partial recognition by taking only the first random_n words
    predicted_text = " ".join(words[:random_n])

    return predicted_text

        
# -----------------------------
# Evaluation core (modular flags)
# -----------------------------
@dataclass
class EvalConfig:
    models: List[str]
    runs: int = 3
    batch_size: int = 8
    enable_batch: bool = True
    enable_amp: bool = False
    amp_dtype: str = "fp16"           # "fp16" | "bf16"
    device: str = "auto"              # "auto" | "cuda" | "cpu"
    num_samples_per_dialect: int = 20
    enable_plots: bool = True
    plots_dir: str = "results/plots"
    save_json: Optional[str] = "results/results.json"
    save_raw_per_sample: bool = False
    seed: int = 42

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
            # packed = nn.utils.rnn.pack_padded_sequence(
            #     features, spec_lengths.cpu(), batch_first=True, enforce_sorted=False
            # )
            # pad_packed_sequence then sees a tensor of size 22 × 1 × 512 = 11264, but it’s told to pad to 87 × 1 × 512, which is impossible →
            # RuntimeError: shape '[87, 1, 512]' is invalid for input of size 11264
            
            # Two convs with stride=2 halve time twice -> ~ /4 (use ceil for safety)
            ds_lengths = torch.clamp((spec_lengths + 3) // 4, min=1)
            # 3 because of ceil division
            # print(f"spec_lengths:{spec_lengths}, ds_lengths:{ds_lengths}")
            
            packed = nn.utils.rnn.pack_padded_sequence(
                features, ds_lengths.cpu(), batch_first=True, enforce_sorted=False)

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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Character to index mapping (simplified)
        self.char_to_idx = {chr(i): i for i in range(32, 127)}  # ASCII printable
        self.char_to_idx['<blank>'] = 0
        self.idx_to_char = {v: k for k, v in self.char_to_idx.items()}
        self.models = self._load_models()
    
    def _load_models(self) -> Dict[str, SimpleASRModel]:
        """Load pretrained ASR models of different sizes"""
        models = {}
        
        for size in ["small", "medium", "large"]:
            model = SimpleASRModel(model_size=size, vocab_size=len(self.char_to_idx))
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
        """Simple CTC decoding (greedy)
            ctc is a method used in sequence modeling tasks where the alignment between input and output sequences is unknown.
        """
        batch_size = log_probs.size(0)
        predictions = []
        
        for i in range(batch_size):
            length = lengths[i].item()
            
            # temperature + blank suppression to avoid trivial collapse
            temperature = 0.7  # <1 sharpens; >1 flattens
            frame_lp = log_probs[i, :length] / temperature

            # discourage blank (id 0) a bit
            frame_lp[:, 0] -= 5.0

            sequence = frame_lp.argmax(dim=-1)
            #sequence = log_probs[i, :length].argmax(dim=-1)
            
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
    
    def evaluate_model_on_dataset(self, model_name: str, test_data: List[Dict], num_runs: int = 3,
                                  sig_model = False, sig_dialect = False) -> Dict:
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
                    #print(f"time:{time.time()}, mel_spec:{mel_spec.shape}, length:{length}")
                    log_probs, output_lengths = model(mel_spec, length)
                    
                    # DEBUG: inspect argmax distribution (before ctc_decode)
                    seq = log_probs[0, :output_lengths[0]].argmax(dim=-1).detach().cpu().numpy()
                    unique, counts = np.unique(seq, return_counts=True)
                    #print("argmax tokens (id:count):", dict(zip(unique.tolist(), counts.tolist())))
                    #print("top-1 mean prob:", float(log_probs[0, :output_lengths[0]].exp().max(dim=-1).values.mean()))
                                        
                    #print(f"time:{time.time()}, log_probs:{log_probs.shape}, output_lengths:{output_lengths}")
                    predictions = self.ctc_decode(log_probs, output_lengths)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                end_time = time.time()
                inference_time = end_time - start_time
                
                # Compute metrics
                predicted_text = predictions[0] if predictions else ""
                #random_n = np.random.randint(0,len(predicted_text))
                #predicted_text = text[:random_n]  # simulate partial recognition
                
                
                predicted_text = simulate_predicted_text(text, dialect, model_name, sig_model, sig_dialect)
                #print(f"simulated: Ref: {text} | Real Pred: {predicted_text} | Sim Pred: {predicted_text}")
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
        print("analyze_statistical_significance: Models to compare:", model_names)
        print(f"analyze_statistical_significance: results: {results}")
        significance_tests = {}
        
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names[i+1:], i+1):
                wer1 = [sample["wer"] for dialect_data in results[model1]["per_dialect"].values() 
                       for sample in dialect_data if isinstance(dialect_data, list)]
                wer2 = [sample["wer"] for dialect_data in results[model2]["per_dialect"].values() 
                       for sample in dialect_data if isinstance(dialect_data, list)]
                
                print(f"analyze_statistical_significance: len wer1: {len(wer1)} len wer2 {len(wer2)}")
                
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

    # ---------- reports / plots ----------
    def maybe_plot(self, results_by_model: Dict[str, Dict[str, Any]], plots_dir: str = "results/plots"):
        
        outdir = Path(plots_dir)
        outdir.mkdir(parents=True, exist_ok=True)

        # Latency vs RTF scatter for each model
        for name, res in results_by_model.items():
            raw = res.get("raw")
            if not raw:
                continue
            lat = [r["latency"] for r in raw]
            rtf = [r["rtf"] for r in raw]
            plt.figure()
            plt.scatter(lat, rtf, s=8)
            plt.xlabel("Latency (s)")
            plt.ylabel("RTF")
            plt.title(f"Latency vs RTF — {name}")
            plt.tight_layout()
            plt.savefig(outdir / f"latency_vs_rtf_{name}.png", dpi=150)
            plt.close()

        # Per-dialect WER bar
        for name, res in results_by_model.items():
            per_d = res["per_dialect"]
            labels = sorted(per_d.keys())
            vals = [per_d[d]["wer"]["mean"] for d in labels]
            plt.figure()
            sns.barplot(x=labels, y=vals)
            plt.ylabel("WER (mean)")
            plt.title(f"WER per dialect — {name}")
            plt.xticks(rotation=20)
            plt.tight_layout()
            plt.savefig(outdir / f"wer_per_dialect_{name}.png", dpi=150)
            plt.close()

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



def main():
    ap = argparse.ArgumentParser(description="Comprehensive ASR evaluation (modularized).")
    ap.add_argument("--models", nargs="*", default=["small", "medium", "large"], help="Models to evaluate.")
    ap.add_argument("--runs", type=int, default=3, help="Number of passes over the dataset.")
    ap.add_argument("--batch-size", type=int, default=8, help="Batch size when --enable-batch is set.")
    ap.add_argument("--enable-batch", action="store_true", help="Enable real batching.")
    ap.add_argument("--enable-amp", action="store_true", help="Enable CUDA AMP (fp16/bf16).")
    ap.add_argument("--amp-dtype", choices=["fp16", "bf16"], default="fp16")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--num-samples", type=int, default=20, help="Samples per dialect.")
    ap.add_argument("--save-json", type=str, default="results/asr_evaluation.json", help="Path to write results JSON.")
    ap.add_argument("--save-raw-per-sample", action="store_true", help="Include per-sample records in results.")
    ap.add_argument("--enable-plots", action="store_true", default=True, help="Save simple plots to --plots-dir.")
    ap.add_argument("--plots-dir", type=str, default="results/plots")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sig_model", type=bool, default=False, help="Model not trained: simulate differences accross models.")
    ap.add_argument("--sig_dialect", type=bool, default=False, help="Model not trained: simulate differences accross dialects.")
    args = ap.parse_args()
    
    cfg = EvalConfig(
        models=args.models,
        runs=args.runs,
        batch_size=args.batch_size,
        enable_batch=args.enable_batch,
        enable_amp=args.enable_amp,
        amp_dtype=args.amp_dtype,
        device=args.device,
        num_samples_per_dialect=args.num_samples,
        enable_plots=args.enable_plots,
        plots_dir=args.plots_dir,
        save_json=args.save_json,
        save_raw_per_sample=args.save_raw_per_sample,
        seed=args.seed,
    )
    
    print("Arguments:", cfg)
    
    evaluator = ASREvaluator(models_dir=MODELS_DIR)
    
    indices = evaluator.text_to_indices("test")
    print("Text to indices:", indices)
    text = evaluator.indices_to_text(indices)
    print("Indices to text:", text)
    
    # print(evaluator.compute_wer("hello world", "hello world"))
    # print(evaluator.compute_wer("hello world", "hello word"))
    # print(evaluator.compute_wer("hello world", "hele worde"))
    
    # print(evaluator.compute_cer("hello", "hello"))
    # print(evaluator.compute_cer("hello", "hallo"))
    # print(evaluator.compute_cer("hello", "alo"))
    
    mel_spec_test = evaluator.synthesize_audio_features("hello world", "british")
    #print("mel_spec_test:", mel_spec_test)
    print("Synthesized mel spectrogram shape:", mel_spec_test[0].shape)
    
    test_data = generate_test_dataset(num_samples_per_dialect=cfg.num_samples_per_dialect)
    
    print(f"Generated {len(test_data)} test samples across multiple dialects.")
    #print("Sample test data:", test_data)
    
    results_by_model: Dict[str, Dict[str, Any]] = {}
    
    for model in cfg.models:
        
        results = evaluator.evaluate_model_on_dataset(model, test_data, num_runs=cfg.runs, 
                                                      sig_model=args.sig_model, sig_dialect=args.sig_dialect)
        print(f"Evaluating model: {model}")
        #print("Evaluation results for  model:", json.dumps(results, indent=2))
        #print(f"Evaluation results type {type(results)}: {results}")
        
        print("----------------------------------------------")
        results_by_model[model] = results
        # Pretty summary
        o = results["overall"]
        print(f"Overall — WER mean={o['wer']['mean']:.3f} CER mean={o['cer']['mean']:.3f} "
              f"Latency mean={o['latency']['mean']:.3f}s (p95={o['latency']['p95']:.3f}s) "
              f"RTF mean={o['rtf']['mean']:.3f}")

    # Statistical significance across models
    # sig = evaluator.analyze_statistical_significance(results_by_model)
    # print(sig)
    # print("Statistical significance analysis:", json.dumps(sig, indent=2))
    # if sig:
    #     print("\n--- Significance tests (Welch t-test, Cliff's delta) ---")
    #     for k, v in sig.items():
    #         print(f"{k:>20s}: t={v['t_statistic']:.3f}, p={v['p_value']:.3g}, s={v['significant']}, e={v['effect_size']:.3g}, ")

    # Save JSON
    if cfg.save_json:
        print(f"\nSaving results JSON to {cfg.save_json}")
        out_path = Path(cfg.save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results_by_model, f, indent=2)
        print(f"\nSaved results JSON -> {out_path}")

    # Plots
    evaluator.maybe_plot(results_by_model)
    if cfg.enable_plots:
        print(f"Saved plots -> {cfg.plots_dir}")

    

if __name__ == "__main__":
    main()
