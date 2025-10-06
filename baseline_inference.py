"""
Baseline TTS Inference Implementation
This is the current slow implementation that needs optimization.
uv run baseline_inference.py 2>&1 | tee logs/p1/baseline_inference.out
"""

import torch
import torch.nn as nn
import time
import numpy as np
from typing import List, Dict, Tuple
import json
import os, gc
import argparse

from profiling_utils import GPUProfiler
from utils import detect_auto_policy, prepare_model_for_precision, DevicePolicy

MODELS_DIR = "models"
DEBUG = True

profiler = GPUProfiler()
policy = DevicePolicy("cuda:0", "fp32", torch.float32)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    torch.cuda.synchronize()
    #torch.cuda.empty_cache()
    torch.cuda.set_device(0)
    torch.cuda.empty_cache()                    # free unused cached blocks
    torch.cuda.ipc_collect()                    # reclaim shared CUDA memory
    torch.cuda.reset_peak_memory_stats()        # optional: zero peak trackers
    torch.cuda.reset_accumulated_memory_stats() # optional: zero accum trackers
    
gc.collect()

individual_sections = ["text_to_tokens", "encode_text", "decode_mel",
                       "vocoder_inference", "synchronize", "to_cpu"]

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
        self.kv_cache = False
        
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
            mel_input = torch.zeros(batch_size, 1, self.mel_dim, device=device, dtype=text_encoded.dtype)
        
        print(f" mel_input.shape {batch_size}")
        
        print(f" mel_input.shape {mel_input.shape}")
        
        outputs = []
        stop_probs = []
        
        for step in range(max_length):
            # Embed previous mel frames
            mel_emb = self.mel_embedding(mel_input) # full prefix processing
            
            # Decode next frame
            decoder_out = self.decoder(mel_emb, text_encoded) # full prefix processing
            
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
        
        print(f" stop_probs.shape {stop_probs.shape}")
        return mel_output, stop_probs
    
    def decode_mel_kv_cache(self, text_encoded, mel_input=None, max_length=500):
        """
        Autoregressive mel generation with a simple per-layer KV cache.
        - Caches self-attn K/V for each decoder layer (grows with steps)
        - Precomputes encoder K/V once per layer for cross-attn
        Assumes self.decoder was created with batch_first=True.
        """
        import torch.nn.functional as F

        B = text_encoded.size(0)
        device = text_encoded.device
        dtype = text_encoded.dtype
        H = self.hidden_dim

        # --- helpers to access projection weights from MultiheadAttention ---
        def _split_qkv_weight_bias(mha):
            """
            Returns (W_q, W_k, W_v, b_q, b_k, b_v) from a torch.nn.MultiheadAttention.
            Works for the common 'in_proj_weight' style modules.
            """
            if hasattr(mha, "in_proj_weight") and mha.in_proj_weight is not None:
                Wq, Wk, Wv = mha.in_proj_weight.split(H, dim=0)
                bq, bk, bv = (mha.in_proj_bias.split(H, dim=0)
                            if mha.in_proj_bias is not None else (None, None, None))
                return Wq, Wk, Wv, bq, bk, bv
            # (PyTorch variant with separate q_proj_weight etc.)
            Wq = mha.q_proj_weight
            Wk = mha.k_proj_weight
            Wv = mha.v_proj_weight
            bq = mha.in_proj_bias[:H] if mha.in_proj_bias is not None else None
            bk = mha.in_proj_bias[H:2*H] if mha.in_proj_bias is not None else None
            bv = mha.in_proj_bias[2*H:] if mha.in_proj_bias is not None else None
            return Wq, Wk, Wv, bq, bk, bv

        def _proj(x, W, b):
            # x: [B, T, H], W: [H, H], b: [H] or None -> [B, T, H]
            if b is None:
                return F.linear(x, W)
            return F.linear(x, W, b)

        def _split_heads(x, n_heads):
            # x: [B, T, H] -> [B, heads, T, Dh]
            B, T, Htot = x.shape
            Dh = Htot // n_heads
            return x.view(B, T, n_heads, Dh).transpose(1, 2)

        def _merge_heads(x):
            # x: [B, heads, T, Dh] -> [B, T, H]
            B, h, T, Dh = x.shape
            return x.transpose(1, 2).contiguous().view(B, T, h * Dh)

        def _attend(q, k, v, is_causal=False):
            # q: [B,h,1,Dh], k/v: [B,h,T,Dh] -> [B,h,1,Dh]
            if hasattr(F, "scaled_dot_product_attention"):
                # sdpa handles causal masks internally if you pass attn_mask,
                # but for single-step we have no future positions anyway.
                out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
            else:
                Dh = q.size(-1)
                scores = torch.matmul(q, k.transpose(-2, -1)) / (Dh ** 0.5)
                w = scores.softmax(dim=-1)
                out = torch.matmul(w, v)
            return out

        # ---------- prepare inputs ----------
        if mel_input is None:
            mel_input = torch.zeros(B, 1, self.mel_dim, device=device, dtype=dtype)

        # text_encoded is the encoder memory for cross-attn (keys/values come from it)
        # We'll precompute encoder K/V once per decoder layer.
        # Also set up self-attn caches for each layer.
        enc_kv = []   # list of (K_mem, V_mem) each as [B, heads, T_enc, Dh]
        self_kv = []  # list of dicts per layer: {"k": [B,h,T,Dh], "v": [B,h,T,Dh]}

        # Gather layer modules for convenience
        layers = list(self.decoder.layers)
        n_layers = len(layers)
        n_heads = layers[0].self_attn.num_heads
        Dh = H // n_heads

        with torch.no_grad():
            # Precompute encoder K/V per layer (cross-attn)
            for l in range(n_layers):
                layer = layers[l]
                Wq_c, Wk_c, Wv_c, bq_c, bk_c, bv_c = _split_qkv_weight_bias(layer.multihead_attn)
                # Project encoder memory to K/V once
                K_mem = _proj(text_encoded, Wk_c, bk_c)  # [B, T_enc, H]
                V_mem = _proj(text_encoded, Wv_c, bv_c)
                K_mem = _split_heads(K_mem, n_heads)     # [B, h, T_enc, Dh]
                V_mem = _split_heads(V_mem, n_heads)
                enc_kv.append((K_mem.contiguous(), V_mem.contiguous()))
                self_kv.append({"k": None, "v": None})

            # Start loop
            outputs = []
            stop_probs = []

            # running token/frame embedding (decoder input at current step)
            # seed with the provided mel_input's last frame
            y_prev = self.mel_embedding(mel_input[:, -1:, :])  # [B,1,H]

            for t in range(max_length):
                x = y_prev  # [B,1,H] current step hidden going into first layer

                # Pass through each decoder layer with cache
                for l in range(n_layers):
                    layer = layers[l]
                    # Self-attn projections
                    Wq_s, Wk_s, Wv_s, bq_s, bk_s, bv_s = _split_qkv_weight_bias(layer.self_attn)

                    # 1) self-attn (cached K/V)
                    q = _proj(x, Wq_s, bq_s)                # [B,1,H]
                    k_new = _proj(x, Wk_s, bk_s)            # [B,1,H]
                    v_new = _proj(x, Wv_s, bv_s)            # [B,1,H]
                    q = _split_heads(q, n_heads)            # [B,h,1,Dh]
                    k_new = _split_heads(k_new, n_heads)    # [B,h,1,Dh]
                    v_new = _split_heads(v_new, n_heads)

                    k_cat = k_new if self_kv[l]["k"] is None else torch.cat([self_kv[l]["k"], k_new], dim=2)
                    v_cat = v_new if self_kv[l]["v"] is None else torch.cat([self_kv[l]["v"], v_new], dim=2)
                    
                    # q*k and then *v, for new Knew and Vnew
                    y = _attend(q, k_cat, v_cat, is_causal=True)             # [B,h,1,Dh]
                    y = _merge_heads(y)                                      # [B,1,H]
                    y = layer.self_attn.out_proj(y)                          # out_proj from MHA

                    # Add & Norm (use layer's Post-Norm/Pre-Norm behavior)
                    if hasattr(layer, "norm1"):
                        x = layer.norm1(x + y)
                    else:
                        x = x + y  # fallback

                    # 2) cross-attn (use precomputed encoder K/V)
                    K_mem, V_mem = enc_kv[l]
                    # Project query for cross-attn
                    q_c = _proj(x, Wq_c, bq_c)               # [B,1,H]
                    q_c = _split_heads(q_c, n_heads)         # [B,h,1,Dh]
                    z = _attend(q_c, K_mem, V_mem, is_causal=False)
                    z = _merge_heads(z)                       # [B,1,H]
                    z = layer.multihead_attn.out_proj(z)

                    if hasattr(layer, "norm2"):
                        x = layer.norm2(x + z)
                    else:
                        x = x + z

                    # 3) FFN
                    # Use the layer's feed-forward (linear1/activation/dropout/linear2) if present.
                    if hasattr(layer, "linear1") and hasattr(layer, "linear2"):
                        # Imitate PyTorch TransformerDecoderLayer FFN (pre/post-norm differences not critical here)
                        f = layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))
                    else:
                        f = x  # minimal fallback

                    if hasattr(layer, "norm3"):
                        x = layer.norm3(x + f)
                    else:
                        x = x + f

                    # update cache for this layer
                    self_kv[l]["k"] = k_cat.detach()
                    self_kv[l]["v"] = v_cat.detach()

                # x is the top-layer decoder hidden for the current step
                next_mel = self.mel_projection(x)                           # [B,1,mel_dim]
                stop_logit = self.stop_token(x.float())                     # do stop in fp32 for stability
                stop_prob = torch.sigmoid(stop_logit).to(dtype)             # [B,1,1] -> cast back

                outputs.append(next_mel)
                stop_probs.append(stop_prob)

                # prepare next step input
                y_prev = self.mel_embedding(next_mel)                       # teacher-free decoding

                # early stop if all say stop
                if torch.all(stop_prob > 0.5):
                    break

            mel_output = torch.cat(outputs, dim=1) if outputs else torch.zeros(B, 0, self.mel_dim, device=device, dtype=dtype)
            stop_probs = torch.cat(stop_probs, dim=1) if stop_probs else torch.zeros(B, 0, 1, device=device, dtype=dtype)
            return mel_output, stop_probs

    def vocoder_inference(self, mel_spec):
        """Convert mel spectrogram to audio waveform"""
        # Transpose for conv1d: (batch, mel_dim, time)
        mel_spec = mel_spec.transpose(1, 2)
        audio = self.vocoder(mel_spec)
        return audio.squeeze(1)  # Remove channel dimension
    
    def forward(self, text_tokens, mel_target=None):
        """Full forward pass"""
        
        section_name = "encode_text"
        with profiler.profile_section_improved(section_name):
            # task to profile
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
            section_name = "decode_mel"
            with profiler.profile_section_improved(section_name):
                # task to profile
                if self.kv_cache:
                    if DEBUG: print(f"Model kv cache: decode_mel_kv_cache")
                    if DEBUG: print(f"Model kv cache: {self.kv_cache}")
                    # Inference mode: autoregressive generation
                    mel_output, stop_probs = self.decode_mel_kv_cache(text_encoded)
                else:
                    if DEBUG: print(f"Model kv cache: {self.kv_cache}")
                    # Inference mode: autoregressive generation
                    mel_output, stop_probs = self.decode_mel(text_encoded)
                    
                
            section_name = "vocoder_inference"
            with profiler.profile_section_improved(section_name):
                # task to profile
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
    
    def synthesize(self, text: str, kv_cache: bool = False) -> Tuple[np.ndarray, Dict]:
        """
        Synthesize speech from text - CURRENT SLOW IMPLEMENTATION
        """
        start_time = time.time()
        
        section_name = "text_to_tokens"
        with profiler.profile_section_improved(section_name):
            # Tokenize text
            tokens = self.text_to_tokens(text)
            if DEBUG: print(f"Quantization precision: tokens {tokens.dtype}")
            
    
        
        
        # INEFFICIENCY 1: No batching, processing one sample at a time
        with torch.no_grad():
            # INEFFICIENCY 2: Full precision inference (FP32)
            if kv_cache:
                print(f"synthesize kv_cache: {kv_cache}")
                self.model.kv_cache = True
            audio, mel_spec, stop_probs = self.model(tokens)
            if DEBUG: print(f"Quantization precision: {audio.dtype}, {mel_spec.dtype}, {stop_probs.dtype}")
            
        
        section_name = "synchronize"
        with profiler.profile_section_improved(section_name):
            # task to profile
            # INEFFICIENCY 3: Synchronous GPU operations
            torch.cuda.synchronize()  # Force wait for GPU

        
        section_name = "to_cpu"
        with profiler.profile_section_improved(section_name):
            # task to profile
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
    
    def synthesize_test(self, text: str, quantization: bool = True, 
                        eff_3: bool = True, eff_4: bool = True, kv_cache: bool = False) -> Tuple[np.ndarray, Dict]:
        """
        Synthesize speech from text - CURRENT SLOW IMPLEMENTATION
        """
        #eff_3 = True
        #eff_4 = True
        
        if DEBUG: print(f"Synthesizing text of length {len(text)} with quantization={quantization}, eff_3={eff_3}, eff_4={eff_4}")
        if DEBUG: print(f"Device: {self.device}")
        
        start_time = time.time()
        
        use_cuda = (self.device.type == "cuda")
        
        # Tokenize text
        section_name = "text_to_tokens"
        with profiler.profile_section(section_name):
            tokens = self.text_to_tokens(text)
            
        
        
        if eff_3 and use_cuda:
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt   = torch.cuda.Event(enable_timing=True)
            #torch.cuda.synchronize()  # clean timing start
            start_evt.record()
        
        
        #if quantization:
        with torch.no_grad():
            
            if kv_cache:
                if DEBUG: print(f"synthesize kv_cache: {kv_cache}")
                self.model.kv_cache = True

            # Autocast for matmuls/conv — keeps numerics stable
            if DEBUG: print(f"Using precision: {policy.autocast_dtype}")
            
            with torch.autocast(device_type="cuda", dtype=policy.autocast_dtype):
                audio, mel_spec, stop_probs = self.model(tokens)
                if DEBUG: print(f"Quantization enabled: {audio.dtype}, {mel_spec.dtype}, {stop_probs.dtype}")
                
        
        section_name = "to_cpu"
        with profiler.profile_section_improved(section_name):
            # task to profile

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
                    #if eff_3: end_evt.synchronize() # lightweight wait for *this* work only (Fix #3)
                else:
                    audio_cpu = audio.detach().cpu()
                
                audio_np = audio_cpu.numpy()
                ## todo: use elapsed_time
                # "e2e_ms": e2e_s * 1000.0,
                #"gpu_forward_ms": gpu_ms,
                #"d2h_ms": d2h_ms,
            
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

    def batch_synthesize_new(self, texts: List[str]) -> Tuple[List[np.ndarray], List[Dict]]:
        """
        True batched synthesis: tokenize together, encode once, decode with per-item stop.
        """
        print(f"batch_synthesize_new")
        if len(texts) == 0:
            return [], []

        start_wall = time.time()

        with torch.no_grad():
            tokens = self.text_to_tokens_batch(texts)  # (B, T)
            assert type
            audio, mel_spec, stop_probs = self.model(tokens)  # uses updated decode_mel
            if DEBUG: print(f"batch_synthesize_new {audio.shape}")
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


def benchmark_baseline(inference_engine: BaselineTTSInference, test_texts: List[str], num_runs: int = 5,
                       optimize_mem: bool = False, kv_cache: bool = False):
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
            section_name = "inference"
            with profiler.profile_section_improved(section_name):
                # task to profile
                if optimize_mem:
                    if kv_cache:
                        audio, metrics = inference_engine.synthesize_test(text, kv_cache = True)
                    else:
                        audio, metrics = inference_engine.synthesize_test(text)
                else:
                    if kv_cache:
                        audio, metrics = inference_engine.synthesize(text, kv_cache = True)
                    else:
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
    print(f"  Num of requests: {num_runs*len(test_texts)}")
    print(f"  Average Latency: {avg_latency:.3f}s")
    print(f"  95th Percentile Latency: {p95_latency:.3f}s")
    print(f"  Average RTF: {avg_rtf:.3f}")
    print(f"  Throughput: {throughput:.1f} samples/second")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Baseline TTS Inference Benchmark")
    parser.add_argument("--model_path", type=str, default="pretrained_tts_model.pt", help="Path to the pretrained model file")
    parser.add_argument("--input_json", type=str, default="sample_text_inputs.json", help="Path to JSON file with test texts")
    parser.add_argument("--num_runs", type=int, default=5, help="Number of runs for benchmarking")
    # batch sizes for profiling
    parser.add_argument("--batch_sizes", type=int, nargs='+', default=[1, 2, 4], help="Batch sizes to profile")
    parser.add_argument("--optimize_mem", type=bool, default=False, help="Optimizations for GPU memory")
    # precision and device args
    parser.add_argument("--precision", type=str, choices=["auto", "fp32", "fp16", "bf16", "int8-cpu"], default="fp32", help="Precision")
    # kv cache
    parser.add_argument("--kv_cache", type=bool, default=False, help="KV cache")
    parser.add_argument("--batching", type=bool, default=False, help="Enable batching experiment")
    args = parser.parse_args()
    
    num_runs = args.num_runs
    batch_sizes = args.batch_sizes
    precision = args.precision
    kv_cache = args.kv_cache
    exp_batching = args.batching

    global policy
    exp = 6
    
    # Load test texts
    with open(args.input_json, "r") as f:
        test_data = json.load(f)
    test_texts = test_data["texts"]
    
    # change
    #test_texts = test_data["test_categories"]["long"]
    exp_name = "last_improved"
    

    # Initialize baseline inference
    inference = BaselineTTSInference(model_path=args.model_path)
    
    # warmup
    print("warmup")
    benchmark_baseline(inference, test_texts[:3], num_runs=1)
    
    if exp_batching == True:
        exp_name = "batching"

        print("--------- Exp 2 sequential batching---------")
        def inference_fn_batch(batch):
                return inference.batch_synthesize(batch)
        results = profiler.profile_inference_batch(
            inference_fn_batch,
            test_texts*num_runs,
            batch_sizes=batch_sizes # [1, 2, 4, 8, 16] 
        )

        for bsz, stats in results.items():
            print(f"\nBatch size {bsz}:")
            print(f"  Avg latency: {stats['latency']:.4f} s")
            print(f"  Throughput: {stats['throughput']:.2f} samples/s")
            print(f"  GPU memory delta: {stats['memory_usage']:.2f} MB")
            print(f"  GPU utilization: {stats['gpu_utilization']:.1f}%") 
        
        print("--------- Exp 2 real batching---------")
        def inference_fn_batch(batch):
                return inference.batch_synthesize_new(batch)
        results = profiler.profile_inference_batch(
            inference_fn_batch,
            test_texts*num_runs,
            batch_sizes=batch_sizes # [1, 2, 4, 8, 16] 
        )

        for bsz, stats in results.items():
            print(f"len: {len(stats)}")
            print(f"\nBatch size {bsz}:")
            print(f"  Avg latency: {stats['latency']:.4f} s")
            print(f"  Throughput: {stats['throughput']:.2f} samples/s")
            print(f"  GPU memory delta: {stats['memory_usage']:.2f} MB")
            print(f"  GPU utilization: {stats['gpu_utilization']:.1f}%")
    else:
    
    
        print("--------- Exp 6 precision and device---------")
        baseline_results = benchmark_baseline(inference, test_texts, num_runs=num_runs)
        
        # Profiler results
        print("\nProfiling Results:")
        for section in individual_sections:
            if section in profiler.timings:
                print(f"Section: {section}")
                print(f"  Latency (s): {np.mean(profiler.timings[section])}")
                print(f"  Memory delta (MB): {np.mean(profiler.memory_usage[section])}")
                print(f"  GPU utilization (%): {np.mean(profiler.gpu_utilization_section[section])}")
        
        print("\nProfiling Results:")
        section = "inference"
        print(f"Section: {section}")
        print(f"len: {len(profiler.timings[section])}")
        print(f"  Latency (s): {np.mean(profiler.timings[section])}")
        print(f"  Memory delta (MB): {np.mean(profiler.memory_usage[section])}")
        print(f"  GPU utilization (%): {np.mean(profiler.gpu_utilization_section[section])}")

        profiler.plot_per_section(sections=individual_sections, name = f"profiling_per_section_{exp_name}_6_original")
        
        profiler.reset_stats()
        print("--------- Exp 6 precision and device auto ---------")
        # Resolve device policy
        if precision == "auto": 
            policy = detect_auto_policy("auto")
            print(f"Auto-detected device policy: {policy}")
        # If precision explicitly set, override policy.precision
        elif policy.precision != "auto":
            policy.precision = precision
            policy.autocast_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(precision, None)

        # Build model
        inference.model, policy = prepare_model_for_precision(inference.model, policy)  

        baseline_results = benchmark_baseline(inference, test_texts, num_runs=num_runs, optimize_mem = True, kv_cache = True)
        profiler.plot_per_section(sections=individual_sections, name = f"profiling_per_section_{exp_name}_6_improved")
        
        # Profiler results
        print("\nProfiling Results:")
        for section in individual_sections:
            if section in profiler.timings:
                print(f"len: {len(profiler.timings[section])}")
                print(f"Section: {section}")
                print(f"  Latency (s): {np.mean(profiler.timings[section])}")
                print(f"  Memory delta (MB): {np.mean(profiler.memory_usage[section])}")
                print(f"  GPU utilization (%): {np.mean(profiler.gpu_utilization_section[section])}")

        print("\nProfiling Results:")
        section = "inference"
        print(f"Section: {section}")
        print(f"len: {len(profiler.timings[section])}")
        
        print(f"  Latency (s): {np.mean(profiler.timings[section])}")
        print(f"  Memory delta (MB): {np.mean(profiler.memory_usage[section])}")
        print(f"  GPU utilization (%): {np.mean(profiler.gpu_utilization_section[section])}")
        print("--------- Exp 6 ---------")


if __name__ == "__main__":
    main()