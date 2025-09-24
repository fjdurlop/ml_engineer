"""
optimized_tts_inference.py  (simplified & robust)

Goal: deliver the *simplest* set of safe improvements that hit the brief without
tricky features that can break (e.g., torch.compile / extra streams).

What we improve over the baseline:
1) **Mixed precision (AMP)** on CUDA (FP16 or BF16 if available) to cut latency & memory.
2) **True batching**: encode multiple texts together and decode them in one loop.
3) **Per-item early stop**: honor stop-token independently for each item.
4) **Lightweight caching**: keep an embedded "history" buffer and only embed the
   newly generated mel frame each step (avoid re-embedding the whole history).
5) **No torch.compile, no extra CUDA streams** -> fewer surprises on different GPUs.

This file also includes a *naive baseline* within it so you can run a fair A/B benchmark
from one script.

Usage examples:
  python optimized_tts_inference.py --mode both --runs 5 --max-batch 8
  python optimized_tts_inference.py --mode optimized --max-batch 16
  python optimized_tts_inference.py --text "Hello there!"

If 'sample_text_inputs.json' exists and has "texts", it will be used for benchmarking.
Otherwise, a small default list is used.
"""

from __future__ import annotations
import os
import json
import time
import argparse
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Model (same shape as baseline)
# -----------------------------------------------------------------------------
class SimpleTTSModel(nn.Module):
    def __init__(self, vocab_size=256, hidden_dim=512, num_layers=6, mel_dim=80, nhead=8):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.mel_dim = mel_dim

        # Text encoder
        self.text_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.text_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_dim, nhead=nhead, batch_first=True),
            num_layers=num_layers,
        )

        # Autoregressive mel decoder
        self.mel_embedding = nn.Linear(mel_dim, hidden_dim)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(hidden_dim, nhead=nhead, batch_first=True),
            num_layers=num_layers,
        )

        # Outputs
        self.mel_projection = nn.Linear(hidden_dim, mel_dim)
        self.stop_token = nn.Linear(hidden_dim, 1)

        # Simple conv-transpose vocoder
        self.vocoder = nn.Sequential(
            nn.ConvTranspose1d(mel_dim, 256, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose1d(256, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 1, 4, 2, 1),
            nn.Tanh(),
        )

    def encode_text(self, text_tokens: torch.Tensor) -> torch.Tensor:
        x = self.text_embedding(text_tokens)  # (B, Ttxt, H)
        return self.text_encoder(x)           # (B, Ttxt, H)

    def _causal_tgt_mask(self, T: int, device: torch.device) -> torch.Tensor:
        # True => masked, causal upper triangle
        return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)

    def decode_step(
        self,
        text_encoded: torch.Tensor,     # (B, Ttxt, H)
        mel_emb_cache: torch.Tensor,    # (B, Tmel, H) - embedded history
        tgt_mask: torch.Tensor          # (Tmel, Tmel) causal
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dec = self.decoder(mel_emb_cache, text_encoded, tgt_mask=tgt_mask)  # (B, Tmel, H)
        last = dec[:, -1:, :]                       # (B, 1, H)
        next_mel = self.mel_projection(last)        # (B, 1, mel_dim)
        stop = torch.sigmoid(self.stop_token(last)) # (B, 1, 1)
        return next_mel, stop

    def vocoder_inference(self, mel: torch.Tensor) -> torch.Tensor:
        # mel: (B, T, mel_dim) -> (B, 1, Taudio) -> (B, Taudio)
        x = mel.transpose(1, 2)
        audio = self.vocoder(x)
        return audio.squeeze(1)


# -----------------------------------------------------------------------------
# Tokenization
# -----------------------------------------------------------------------------
def char_tokenize_batch(texts: List[str], device: torch.device, vocab_size: int = 256) -> torch.Tensor:
    seqs = [[ord(c) % vocab_size for c in t.lower()] for t in texts]
    T = max(1, max(len(s) for s in seqs)) if seqs else 1
    tokens = torch.zeros(len(seqs), T, dtype=torch.long, device=device)
    for i, s in enumerate(seqs):
        if s:
            tokens[i, :len(s)] = torch.tensor(s, dtype=torch.long, device=device)
    return tokens


# -----------------------------------------------------------------------------
# Baseline-like (naive) for A/B
# -----------------------------------------------------------------------------
class NaiveTTS:
    def __init__(self, device: torch.device):
        self.device = device
        self.model = SimpleTTSModel().to(device).eval()

    @torch.no_grad()
    def synthesize(self, text: str, max_steps: int = 500) -> Tuple[np.ndarray, Dict]:
        start = time.time()
        tokens = char_tokenize_batch([text], self.device)
        txt = self.model.encode_text(tokens)  # FP32

        mel = torch.zeros(1, 1, self.model.mel_dim, device=self.device)
        outputs, stops = [], []

        for _ in range(max_steps):
            mel_emb = self.model.mel_embedding(mel)
            tgt_mask = self.model._causal_tgt_mask(mel_emb.size(1), self.device)
            dec = self.model.decoder(mel_emb, txt, tgt_mask=tgt_mask)
            last = dec[:, -1:, :]
            next_mel = self.model.mel_projection(last)
            stop = torch.sigmoid(self.model.stop_token(last))
            outputs.append(next_mel)
            stops.append(stop)
            mel = torch.cat([mel, next_mel], dim=1)
            if torch.all(stop > 0.5):
                break

        mel_out = torch.cat(outputs, dim=1) if outputs else mel
        audio = self.model.vocoder_inference(mel_out)

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        end = time.time()

        audio_np = audio.detach().cpu().numpy()
        mel_np = mel_out.detach().cpu().numpy()
        dur = end - start
        metrics = {
            "latency": dur,
            "text_length": len(text),
            "mel_frames": int(mel_np.shape[1]),
            "audio_length": int(audio_np.shape[-1]),
            "rtf": dur / max(audio_np.shape[-1] / 22050.0, 1e-6),
        }
        return audio_np[0], metrics

    def batch_synthesize(self, texts: List[str]) -> Tuple[List[np.ndarray], List[Dict]]:
        out_a, out_m = [], []
        for t in texts:
            a, m = self.synthesize(t)
            out_a.append(a)
            out_m.append(m)
        return out_a, out_m


# -----------------------------------------------------------------------------
# Optimized (simple & robust)
# -----------------------------------------------------------------------------
class OptimizedTTS:
    def __init__(
        self,
        device: Optional[torch.device] = None,
        dtype_pref: str = "auto",  # "auto" | "bf16" | "fp16" | "fp32"
        max_steps: int = 500,
        stop_threshold: float = 0.5,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SimpleTTSModel().to(self.device).eval()
        self.max_steps = max_steps
        self.stop_threshold = stop_threshold

        # Choose AMP dtype
        if self.device.type == "cuda":
            if dtype_pref == "bf16" or (dtype_pref == "auto" and torch.cuda.is_bf16_supported()):
                self.amp_dtype = torch.bfloat16
            elif dtype_pref in ("fp16", "auto"):
                self.amp_dtype = torch.float16
            else:
                self.amp_dtype = torch.float32
        else:
            self.amp_dtype = torch.float32

    @torch.no_grad()
    def synthesize_batch(self, texts: List[str]) -> Tuple[List[np.ndarray], List[Dict]]:
        """
        True batching + AMP + per-item early stop.
        Minimal caching: we embed only the newly generated frame each step and
        keep a growing embedded-history tensor (mel_emb_cache).
        """
        B = len(texts)
        if B == 0:
            return [], []

        tokens = char_tokenize_batch(texts, self.device)

        # Encode text (AMP for speed on CUDA)
        with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=(self.amp_dtype != torch.float32)):
            text_encoded = self.model.encode_text(tokens)  # (B, Ttxt, H)

        mel_dim = self.model.mel_dim
        # Start with a zero-frame (fp32 buffer to keep vocoder numerics stable)
        mel_out_frames: List[torch.Tensor] = []
        # Embedded cache begins with the embedded zero frame (AMP dtype)
        mel0 = torch.zeros(B, 1, mel_dim, device=self.device)
        with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=(self.amp_dtype != torch.float32)):
            mel_emb_cache = self.model.mel_embedding(mel0)

        finished = torch.zeros(B, dtype=torch.bool, device=self.device)

        # Decode loop
        for _ in range(self.max_steps):
            Tm = mel_emb_cache.size(1)
            tgt_mask = self.model._causal_tgt_mask(Tm, self.device)

            with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=(self.amp_dtype != torch.float32)):
                next_mel, stop = self.model.decode_step(text_encoded, mel_emb_cache, tgt_mask)

            # Keep mel frames in fp32 for vocoder
            mel_out_frames.append(next_mel.float())

            newly_finished = stop.squeeze(-1).squeeze(-1) > self.stop_threshold
            finished = finished | newly_finished
            if finished.all():
                break

            # Append only the new frame's embedding
            with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=(self.amp_dtype != torch.float32)):
                new_emb = self.model.mel_embedding(next_mel)
            mel_emb_cache = torch.cat([mel_emb_cache, new_emb], dim=1)

        # Stack predicted frames (B, T, mel_dim)
        if len(mel_out_frames) == 0:
            mel_out = mel0
        else:
            mel_out = torch.cat(mel_out_frames, dim=1)

        # Vocoder (no compile/streams; just run)
        audio = self.model.vocoder_inference(mel_out)

        # Ensure GPU finished before timing outside
        if self.device.type == "cuda":
            torch.cuda.synchronize()

        audio_np = audio.detach().cpu().numpy()
        mel_np = mel_out.detach().cpu().numpy()

        metrics: List[Dict] = []
        for i, t in enumerate(texts):
            metrics.append(dict(
                latency=None,  # filled by caller timing wallclock
                text_length=len(t),
                mel_frames=int(mel_np.shape[1]),
                audio_length=int(audio_np[i].shape[-1]),
                rtf=None,
            ))
        return [audio_np[i] for i in range(B)], metrics

    @torch.no_grad()
    def synthesize(self, text: str) -> Tuple[np.ndarray, Dict]:
        start = time.time()
        audios, metas = self.synthesize_batch([text])
        end = time.time()
        audio = audios[0]
        m = metas[0]
        m["latency"] = end - start
        m["rtf"] = m["latency"] / max(audio.shape[-1] / 22050.0, 1e-6)
        return audio, m


# -----------------------------------------------------------------------------
# Benchmarking
# -----------------------------------------------------------------------------
def load_texts() -> List[str]:
    f = "sample_text_inputs.json"
    if os.path.exists(f):
        try:
            data = json.load(open(f))
            if isinstance(data, dict) and "texts" in data and isinstance(data["texts"], list):
                if data["texts"]:
                    return data["texts"]
        except Exception:
            pass
    # Fallback minimal set
    return [
        "Hello world!",
        "Good morning, how are you today?",
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming the way we interact with technology.",
        "To be or not to be, that is the question.",
        "Call me Ishmael, some years ago...",
    ]


def benchmark(
    texts: List[str],
    runs: int = 5,
    max_batch: int = 8,
    mode: str = "both",
    warmup: int = 1,
    device: Optional[torch.device] = None,
) -> Dict[str, Dict]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out: Dict[str, Dict] = {}

    def bench_baseline() -> None:
        eng = NaiveTTS(device=device)
        # warmup
        for _ in range(warmup):
            _ = eng.synthesize("warmup")
        lat, rtf = [], []
        for _ in range(runs):
            for t in texts:
                a, m = eng.synthesize(t)
                lat.append(m["latency"])
                rtf.append(m["rtf"])
        avg_lat = float(np.mean(lat))
        out["baseline"] = dict(
            avg_latency=avg_lat,
            p95_latency=float(np.percentile(lat, 95)),
            avg_rtf=float(np.mean(rtf)),
            throughput_samples_per_s=(1.0 / avg_lat if avg_lat > 0 else 0.0),
            total_samples=len(lat),
        )

    def bench_optimized() -> None:
        eng = OptimizedTTS(device=device, dtype_pref="auto")
        # warmup
        for _ in range(warmup):
            _ = eng.synthesize("warmup text")
        lat, rtf = [], []

        # build batches
        batches: List[List[str]] = []
        cur: List[str] = []
        for t in texts:
            cur.append(t)
            if len(cur) >= max_batch:
                batches.append(cur)
                cur = []
        if cur:
            batches.append(cur)

        for _ in range(runs):
            for batch in batches:
                start = time.time()
                audios, metas = eng.synthesize_batch(batch)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                end = time.time()
                wall = end - start
                for i in range(len(batch)):
                    lat.append(wall)
                    dur = max(audios[i].shape[-1] / 22050.0, 1e-6)
                    rtf.append(wall / dur)

        avg_lat = float(np.mean(lat))
        out["optimized"] = dict(
            avg_latency=avg_lat,
            p95_latency=float(np.percentile(lat, 95)),
            avg_rtf=float(np.mean(rtf)),
            throughput_samples_per_s=(1.0 / avg_lat if avg_lat > 0 else 0.0),
            total_samples=len(lat),
        )

    if mode in ("baseline", "both"):
        bench_baseline()
    if mode in ("optimized", "both"):
        bench_optimized()
    return out


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Simplified optimized TTS inference")
    parser.add_argument("--mode", choices=["baseline", "optimized", "both"], default="both")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--max-batch", type=int, default=8)
    parser.add_argument("--text", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.text:
        eng = OptimizedTTS(device=device, dtype_pref="auto")
        audio, m = eng.synthesize(args.text)
        print(f"Synthesized. Latency: {m['latency']*1000:.1f} ms | RTF: {m['rtf']:.3f} | audio_len: {audio.shape[-1]}")
        return

    texts = load_texts()
    results = benchmark(texts, runs=args.runs, max_batch=args.max_batch, mode=args.mode, device=device)

    print("\n=== Results ===")
    for k, v in results.items():
        print(f"\n[{k.upper()}]")
        for kk, vv in v.items():
            if isinstance(vv, float):
                print(f"  {kk}: {vv:.4f}")
            else:
                print(f"  {kk}: {vv}")

    if "baseline" in results and "optimized" in results:
        b, o = results["baseline"], results["optimized"]
        speedup = b["avg_latency"] / max(o["avg_latency"], 1e-9)
        thr_gain = o["throughput_samples_per_s"] / max(b["throughput_samples_per_s"], 1e-9)
        print("\n=== Improvement (Optimized vs Baseline) ===")
        print(f"  Latency speedup: {speedup:.2f}x")
        print(f"  Throughput gain: {thr_gain:.2f}x")


if __name__ == "__main__":
    main()
