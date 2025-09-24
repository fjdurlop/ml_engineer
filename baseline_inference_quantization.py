"""
Baseline TTS Inference Implementation
This aselineTTSInference(device=device)is the current slow implementation that needs optimization.
uv run baseline_inference.py 2>&1 | tee logs/baseline_inference.out

"""

import torch
import torch.nn as nn
import time
import numpy as np
from typing import List, Dict, Tuple
import json
import os

from profiling_utils import GPUProfiler

MODELS_DIR = "models"

profiler = GPUProfiler()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



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
    
    def decode_mel_original(self, text_encoded, mel_input=None, max_length=500):
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
    
    def decode_mel(self, text_encoded, mel_input=None, max_length=500):
        """Autoregressive mel spectrogram generation with per-item early stop."""
        B = text_encoded.size(0)
        device = text_encoded.device

        if mel_input is None:
            mel_input = torch.zeros(B, 1, self.mel_dim, device=device)

        outputs = []
        stop_probs_all = []
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for step in range(max_length):
            mel_emb = self.mel_embedding(mel_input)            # (B, T, H)
            decoder_out = self.decoder(mel_emb, text_encoded)  # (B, T, H)
            last = decoder_out[:, -1:, :]                      # (B, 1, H)

            next_mel = self.mel_projection(last)               # (B, 1, mel_dim)
            stop_prob = torch.sigmoid(self.stop_token(last))   # (B, 1, 1)

            # For finished items, keep feeding back the last frame (no-op growth)
            # so shapes stay consistent but their content doesn't change.
            if finished.any():
                # zero-out updates for finished items
                mask = (~finished).view(B, 1, 1).to(next_mel.dtype)
                next_mel = next_mel * mask + mel_input[:, -1:, :] * (1 - mask)
                # lock stop prob high so they remain finished
                stop_prob = stop_prob * mask + torch.ones_like(stop_prob) * (1 - mask)

            outputs.append(next_mel)
            stop_probs_all.append(stop_prob)

            mel_input = torch.cat([mel_input, next_mel], dim=1)

            # Update finished mask
            newly_finished = (stop_prob.squeeze(-1).squeeze(-1) > 0.5)
            finished = finished | newly_finished
            if finished.all():
                break

        mel_output = torch.cat(outputs, dim=1) if outputs else mel_input
        stop_probs = torch.cat(stop_probs_all, dim=1) if stop_probs_all else torch.empty(B, 0, 1, device=device)
        return mel_output, stop_probs

    
    def vocoder_inference(self, mel_spec):
        """Convert mel spectrogram to audio waveform"""
        # Transpose for conv1d: (batch, mel_dim, time)
        mel_spec = mel_spec.transpose(1, 2)
        audio = self.vocoder(mel_spec)
        return audio.squeeze(1)  # Remove channel dimension
    
    def forward(self, text_tokens, mel_target=None):
        """Full forward pass"""
        # [profile_section("encode_text")]
        # Encode text
        #profiler.reset_stats()
        with profiler.profile_section("encode_text"):
            text_encoded = self.encode_text(text_tokens)
        
        if mel_target is not None:
            # Training mode: teacher forcing
            mel_emb = self.mel_embedding(mel_target[:, :-1, :])
            decoder_out = self.decoder(mel_emb, text_encoded)
            mel_pred = self.mel_projection(decoder_out)
            stop_pred = self.stop_token(decoder_out)
            return mel_pred, stop_pred
        else:
            # [profile_section("decode_mel")]
            #print("SimplifiedTTSModel.forward: Inference mode")
            #profiler.reset_stats()
            with profiler.profile_section("decode_mel"):
                # Inference mode: autoregressive generation
                mel_output, stop_probs = self.decode_mel(text_encoded)
            # [profile_section("vocoder_inference")]
            #profiler.reset_stats()
            with profiler.profile_section("vocoder_inference"):
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
    
    def text_to_tokens_batch(self, texts: List[str]) -> torch.Tensor:
        """Pad texts to the same length and return a (B, T) LongTensor on device."""
        if len(texts) == 0:
            #return torch.zeros(0, 1, dtype=torch.long, device=self.device)
            
            return torch.zeros(0, 1, dtype=torch.long, device=self.device)
        seqs = [[ord(c) % 256 for c in t.lower()] for t in texts]
        T = max(len(s) for s in seqs)
        tokens = torch.zeros(len(seqs), T, dtype=torch.long, device=self.device)
        for i, s in enumerate(seqs):
            if s:
                tokens[i, :len(s)] = torch.tensor(s, dtype=torch.long, device=self.device)
        return tokens

    
    def synthesize(self, text: str, quantization: bool = False, eff_3: bool = False, eff_4: bool = False) -> Tuple[np.ndarray, Dict]:
        """
        Synthesize speech from text - CURRENT SLOW IMPLEMENTATION
        """
        #eff_3 = True
        #eff_4 = True
        
        print(f"Synthesizing text of length {len(text)} with quantization={quantization}, eff_3={eff_3}, eff_4={eff_4}")
        print(f"Device: {self.device}")
        
        start_time = time.time()
        
        use_cuda = (self.device.type == "cuda")
        
        # Tokenize text
        # [profile_section("text_to_tokens")]
        #profiler.reset_stats()
        with profiler.profile_section("text_to_tokens"):
            tokens = self.text_to_tokens(text)
        
        if eff_3 and use_cuda:
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt   = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()  # clean timing start
            start_evt.record()
        
        if quantization:
            with torch.no_grad():
                
                # Autocast for matmuls/conv — keeps numerics stable
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    audio, mel_spec, stop_probs = self.model(tokens)
                    print(f"Quantization enabled: {audio.dtype}, {mel_spec.dtype}, {stop_probs.dtype}")
                
        else:
            # INEFFICIENCY 1: No batching, processing one sample at a time
            with torch.no_grad():
                # INEFFICIENCY 2: Full precision inference (FP32)
                audio, mel_spec, stop_probs = self.model(tokens)
                print(f"Quantization disabled: {audio.dtype}, {mel_spec.dtype}, {stop_probs.dtype}")
        
        if eff_3 and use_cuda:
            end_evt.record()
        else: 
            # INEFFICIENCY 3: Synchronous GPU operations
            torch.cuda.synchronize()  # Force wait for GPU
        
        # INEFFICIENCY 4: Inefficient memory usage - keeping all intermediate results
        if not eff_4:
            audio_np = audio.cpu().numpy()
            mel_np = mel_spec.cpu().numpy()
            mel_frames = mel_np.shape[1]
        else:
            # 3) Compute metrics without copying large tensors to CPU
            #    -> mel frames directly from tensor shape on device
            mel_frames = int(mel_spec.size(1))
            #    -> audio length from tensor shape (still on device)
            audio_len_samples = int(audio.size(-1))
        
        
            # Move ONLY audio to CPU (non_blocking=True). We don't copy mel_spec at all.
            if use_cuda:
                audio_cpu = audio.detach().to("cpu", non_blocking=True)
                if eff_3: end_evt.synchronize() # lightweight wait for *this* work only (Fix #3)
            else:
                audio_cpu = audio.detach().cpu()
            
            audio_np = audio_cpu.numpy()
            
        end_time = time.time()
        
        # todo: eff_4 first and eff_3 second, not supported only eff_3 
        
        # Return audio and metrics
        metrics = {
            "latency": end_time - start_time,
            "text_length": len(text),
            "audio_length": audio_np.shape[-1],
            #"mel_frames": mel_np.shape[1],
            "mel_frames": mel_frames,
            "rtf": (end_time - start_time) / (audio_np.shape[-1] / 22050)  # Real-time factor
        }
        
        return audio_np[0], metrics
    
    def batch_synthesize_original(self, texts: List[str]) -> Tuple[List[np.ndarray], List[Dict]]:
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
    
    def batch_synthesize(self, texts: List[str]) -> Tuple[List[np.ndarray], List[Dict]]:
        """
        True batched synthesis: tokenize together, encode once, decode with per-item stop.
        """
        if len(texts) == 0:
            return [], []

        start_wall = time.time()

        with torch.no_grad():
            tokens = self.text_to_tokens_batch(texts)  # (B, T)
            assert type
            audio, mel_spec, stop_probs = self.model(tokens)  # uses updated decode_mel

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_wall = time.time()
        wall = end_wall - start_wall

        audio_np = audio.detach().cpu().numpy()
        mel_np = mel_spec.detach().cpu().numpy()

        results = []
        metrics = []
        for i, t in enumerate(texts):
            # per-item RTF based on wall time and that sample's audio length
            dur_s = max(audio_np[i].shape[-1] / 22050.0, 1e-6)
            metrics.append({
                "latency": wall,                         # batch wall time attributed per item
                "text_length": len(t),
                "audio_length": int(audio_np[i].shape[-1]),
                "mel_frames": int(mel_np.shape[1]),
                #"rtf": wall / dur_s,
                "rtf": (wall) / (audio_np[i].shape[-1] / 22050)  # Real-time factor
                
            })
            results.append(audio_np[i])

        return results, metrics


    def save_model(self, model_path: str):
        """Save the pretrained model to the specified path."""
        # Move model to CPU for compatibility
        model_to_save = self.model.cpu()
        torch.save(model_to_save.state_dict(), model_path,)
        # Move back to original device
        self.model.to(self.device)
        print(f"Model saved to {model_path}")


def benchmark_baseline(inference_engine: BaselineTTSInference, test_texts: List[str], num_runs: int = 5, quantization: bool = False,
                       eff_3: bool = False, eff_4: bool = False) -> Dict:
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
            audio, metrics = inference_engine.synthesize(text, quantization=quantization, eff_3 = eff_3, eff_4=eff_4)
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
    print(f"  Num of requests: {num_runs*len(test_texts):.3f}s")
    print(f"  Average Latency: {avg_latency:.3f}s")
    print(f"  95th Percentile Latency: {p95_latency:.3f}s")
    print(f"  Average RTF: {avg_rtf:.3f}")
    print(f"  Throughput: {throughput:.2f} samples/second")
    
    return results


if __name__ == "__main__":
    # Load test texts
    with open("sample_text_inputs.json", "r") as f:
        test_data = json.load(f)
        
    num_runs = 1
    
    test_texts = test_data["texts"]
    
    # Initialize baseline inference
    #inference = BaselineTTSInference()
    
    profiler.reset_stats()
    
    # Initialize baseline inference
    #inference = BaselineTTSInference()
    
    print("--------- Load model ---------")
    #profiler.reset_stats()
    with profiler.profile_section("load_model"):
        engine = BaselineTTSInference()
    
    # once after creating engine
    for _ in range(3):
        _ = engine.synthesize("warmup warmup warmup", quantization=False, eff_3=True, eff_4=True)
    torch.cuda.synchronize()
    
    print("Latency (s):", profiler.timings["load_model"])
    print("Memory delta (MB):", profiler.memory_usage["load_model"])

    profiler.reset_stats()
    print("--------- Run 1 ---------")
    with profiler.profile_section_improved("tts_batch"):
        #baseline_results = benchmark_baseline(engine, test_texts, num_runs=num_runs, quantization=False)
        baseline_results = benchmark_baseline(engine, test_texts, num_runs=num_runs, quantization=False, eff_3=False, eff_4=False)

    print("Latency (s):", profiler.timings["tts_batch"])
    print("Memory delta (MB):", profiler.memory_usage["tts_batch"])
    print("GPU utilization (%):", profiler.gpu_utilization_section["tts_batch"])

    # results = profiler.profile_inference_batch(
    #     inference_fn,
    #     test_texts,
    #     batch_sizes=[1] # [1, 2, 4, 8, 16] 
    # )

    # for bsz, stats in results.items():
    #     print(f"\nBatch size {bsz}:")
    #     print(f"  Avg latency: {stats['latency']:.4f} s")
    #     print(f"  Throughput: {stats['throughput']:.2f} samples/s")
    #     print(f"  GPU memory delta: {stats['memory_usage']:.2f} MB")
    #     print(f"  GPU utilization: {stats['gpu_utilization']:.1f}%")
    
    profiler.reset_stats()
    print("--------- Run 2 ---------")
    with profiler.profile_section_improved("tts_batch_quant"):
        #baseline_results = benchmark_baseline(engine, test_texts, num_runs=num_runs, quantization=True)
        baseline_results = benchmark_baseline(engine, test_texts, num_runs=num_runs, quantization=False, eff_4=True)
        

    print("Latency (s):", profiler.timings["tts_batch_quant"])
    print("Memory delta (MB):", profiler.memory_usage["tts_batch_quant"])
    print("GPU utilization (%):", profiler.gpu_utilization_section["tts_batch_quant"])
    
    
    
    profiler.reset_stats()
    print("--------- Run 2 ---------")
    with profiler.profile_section_improved("tts_eff34"):
        #baseline_results = benchmark_baseline(engine, test_texts, num_runs=num_runs, quantization=True)
        
        baseline_results = benchmark_baseline(engine, test_texts, num_runs=num_runs, quantization=False, eff_3=True, eff_4=True)
        

    print("Latency (s):", profiler.timings["tts_eff34"])
    print("Memory delta (MB):", profiler.memory_usage["tts_eff34"])
    print("GPU utilization (%):", profiler.gpu_utilization_section["tts_eff34"])
    
    # print("--------- END ---------")
 