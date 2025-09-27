"""
Streaming ASR System with Continuous Batching (single-file)
-----------------------------------------------------------
Usage examples:

# 1) Real-time simulated audio (320 ms chunks)
python streaming_asr_system.py --simulate-stream --chunk-ms 320 --rt-speed 1.0

# 2) Faster-than-real-time demo (160 ms chunks, 2x speed)
python streaming_asr_system.py --simulate-stream --chunk-ms 160 --rt-speed 2.0

# 3) One-shot (non-streaming) single chunk
python streaming_asr_system.py --audio-length 32000
"""

import argparse
import asyncio
import logging
import math
import queue
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchaudio

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ----------------------------
# Frontend: audio -> log-mel
# ----------------------------
class FeatureFrontend:
    def __init__(self, sr: int = 16000, n_mels: int = 80, win_ms: int = 25, hop_ms: int = 10):
        self.sr = sr
        self.n_mels = n_mels
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=int(sr * win_ms / 1000),
            hop_length=int(sr * hop_ms / 1000),
            n_mels=n_mels,
            center=True,
            power=2.0,
            norm="slaney",
            mel_scale="htk",
        )
        self.db = torchaudio.transforms.AmplitudeToDB(stype="power")

    def to_mel(self, wav_np: np.ndarray) -> torch.Tensor:
        """
        wav_np: mono float32 [T]
        return: [F, T_frames] torch.float32 on CPU
        """
        wav = torch.from_numpy(wav_np).float().unsqueeze(0)              # [1, T]
        mel = self.melspec(wav)                                          # [1, F, T']
        mel_db = self.db(mel)                                            # [1, F, T']
        return mel_db.squeeze(0)                                         # [F, T']


# ----------------------------
# Simple ASR Model (CNN -> LSTM -> Linear -> CTC)
# Fixed to produce 3D features for LSTM.
# ----------------------------
class SimpleASRModel(nn.Module):
    """
    Simplified ASR model for streaming tests
    CNN downsamples time (x4 with 2 convs stride=2), averages frequency,
    then LSTM + Linear + log_softmax (CTC-style).
    """
    def __init__(self, input_dim=80, hidden_dim=256, num_layers=4, vocab_size=96, model_size="medium"):
        super().__init__()
        size_multipliers = {"small": 0.5, "medium": 1.0, "large": 2.0}
        m = size_multipliers.get(model_size, 1.0)

        self.hidden_dim = int(hidden_dim * m)
        self.num_layers = max(1, int(num_layers * m))
        self.model_size = model_size
        self.vocab_size = vocab_size  # 0=blank, 1..95 map to ASCII 32..126

        # CNN over [B, 1, T, F]
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(),
        )
        # Project channels -> LSTM input size (after averaging over frequency axis)
        self.proj = nn.Linear(128, self.hidden_dim // 4)

        # BiLSTM
        self.encoder = nn.LSTM(
            input_size=self.hidden_dim // 4,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1 if self.num_layers > 1 else 0.0,
        )

        self.classifier = nn.Linear(self.hidden_dim * 2, self.vocab_size)

    def forward(self, x_3d: torch.Tensor, spec_lengths: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x_3d: [B, T, F]  (time, mel)
        spec_lengths: [B] in frames (before CNN downsample)
        returns: (log_probs [B, T', V], output_lengths [B])
        """
        if x_3d.dim() != 3:
            raise ValueError(f"Expected [B,T,F], got {tuple(x_3d.shape)}")
        B, T, F = x_3d.shape

        # to [B,1,T,F] for CNN
        x = x_3d.unsqueeze(1)
        feat = self.conv(x)                     # [B, 128, T', F']
        # Average over frequency dim to make [B, 128, T']
        feat = feat.mean(dim=3)                 # [B, 128, T']
        feat = feat.transpose(1, 2)             # [B, T', 128]
        feat = self.proj(feat)                  # [B, T', D_in]

        # Compute downsampled lengths: 2 convs stride=2 -> /4 (ceil)
        if spec_lengths is None:
            out_T = feat.size(1)
            out_lengths = torch.full((B,), out_T, dtype=torch.long, device=feat.device)
            enc_out, _ = self.encoder(feat)
        else:
            ds_lengths = torch.clamp((spec_lengths + 3) // 4, min=1)  # ceil division by 4
            packed = nn.utils.rnn.pack_padded_sequence(feat, ds_lengths.cpu(), batch_first=True, enforce_sorted=False)
            enc_out, _ = self.encoder(packed)
            enc_out, out_lengths = nn.utils.rnn.pad_packed_sequence(enc_out, batch_first=True)

        logits = self.classifier(enc_out)       # [B, T', V]
        log_probs = torch.log_softmax(logits, dim=-1)
        return log_probs, out_lengths


# ----------------------------
# Real-time feeder (simulated audio)
# ----------------------------
class RealTimeFeeder:
    """
    Sends audio chunks to engine at real-time (or faster) pace.
    """
    def __init__(self, sr=16000, chunk_ms=320, overlap_ms=0, speed=1.0):
        self.sr = sr
        self.chunk = int(sr * chunk_ms / 1000)
        self.overlap = int(sr * overlap_ms / 1000)
        assert 0 <= self.overlap < self.chunk
        self.hop = self.chunk - self.overlap
        self.speed = max(1e-6, float(speed))

    def stream(self, audio: np.ndarray, send_fn):
        n = len(audio)
        t_per_chunk = (self.chunk / self.sr) / self.speed
        idx = 0
        next_deadline = time.time()
        cid = 0
        while idx < n:
            end = min(idx + self.chunk, n)
            chunk = audio[idx:end]
            if len(chunk) < self.chunk:
                pad = np.zeros(self.chunk - len(chunk), dtype=np.float32)
                chunk = np.concatenate([chunk, pad])
            send_fn(chunk.astype(np.float32))
            cid += 1
            next_deadline += t_per_chunk
            dt = next_deadline - time.time()
            if dt > 0:
                time.sleep(dt)
            idx += self.hop


# ----------------------------
# Stream data structures
# ----------------------------
@dataclass
class AudioChunk:
    stream_id: str
    chunk_id: int
    audio_data: np.ndarray
    timestamp: float
    is_final: bool = False


@dataclass
class StreamState:
    stream_id: str
    created_at: float
    last_activity: float
    partial_transcript: str
    final_transcript: str
    processed_chunks: int
    next_chunk_id: int = 0


class StreamBatch:
    def __init__(self, max_batch_size: int = 8, max_wait_time: float = 0.05):
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.pending: List[AudioChunk] = []
        self.last_batch_time = time.time()

    def add_chunk(self, c: AudioChunk) -> bool:
        self.pending.append(c)
        elapsed = time.time() - self.last_batch_time
        return (len(self.pending) >= self.max_batch_size) or (elapsed >= self.max_wait_time)

    def get_batch(self) -> List[AudioChunk]:
        b = self.pending[:]
        self.pending.clear()
        self.last_batch_time = time.time()
        return b


# ----------------------------
# Engine
# ----------------------------
class StreamingASREngine:
    def __init__(self, model_size="medium", device="auto", sr=16000, n_mels=80,
                 max_batch_size=8, max_wait_ms=50):
        self.device = torch.device("cuda" if (device == "auto" and torch.cuda.is_available()) else device)
        self.model = SimpleASRModel(model_size=model_size, vocab_size=96).to(self.device).eval()
        self.frontend = FeatureFrontend(sr=sr, n_mels=n_mels)

        self.active_streams: Dict[str, StreamState] = {}
        self.stream_lock = threading.RLock()

        self.batch_manager = StreamBatch(max_batch_size=max_batch_size, max_wait_time=max_wait_ms / 1000.0)
        self.processing_queue: "queue.Queue[AudioChunk]" = queue.Queue(maxsize=256)

        self.is_running = False
        self.processing_thread: Optional[threading.Thread] = None

        # Stats
        self.stats = {
            "total_chunks_processed": 0,
            "active_streams": 0,
            "avg_latency_ms": 0.0,
        }
        self._lat_ewma = 0.0

        self.start()

        # Build ASCII idx map: 0 blank, 1..95 -> ASCII 32..126
        self.idx_to_char = {i: chr(31 + i) for i in range(1, 96)}
        self.idx_to_char[0] = ""

    # --- lifecycle ---
    def start(self):
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        logger.info("Streaming ASR engine started")

    def stop(self):
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
        logger.info("Streaming ASR engine stopped")

    def drain(self, timeout_s: float = 2.0):
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            if self.processing_queue.empty() and not self.batch_manager.pending:
                break
            time.sleep(0.01)

    # --- stream API ---
    def add_stream(self, stream_id: Optional[str] = None) -> str:
        sid = stream_id or str(uuid.uuid4())
        with self.stream_lock:
            if sid in self.active_streams:
                return sid
            now = time.time()
            self.active_streams[sid] = StreamState(
                stream_id=sid,
                created_at=now,
                last_activity=now,
                partial_transcript="",
                final_transcript="",
                processed_chunks=0,
            )
            self.stats["active_streams"] = len(self.active_streams)
            logger.info(f"Added stream {sid}")
        return sid

    def process_chunk(self, stream_id: str, audio_chunk: np.ndarray) -> str:
        with self.stream_lock:
            if stream_id not in self.active_streams:
                raise ValueError(f"Unknown stream {stream_id}")
            st = self.active_streams[stream_id]
            st.last_activity = time.time()
            cid = st.next_chunk_id
            st.next_chunk_id += 1

        c = AudioChunk(stream_id=stream_id, chunk_id=cid, audio_data=audio_chunk, timestamp=time.time())
        try:
            self.processing_queue.put(c, timeout=0.05)
            logger.debug(f"Enqueued chunk {cid} for stream {stream_id}")
        except queue.Full:
            logger.warning("Global queue full, dropping chunk")
        return st.partial_transcript

    def finalize_stream(self, stream_id: str) -> str:
        with self.stream_lock:
            if stream_id not in self.active_streams:
                raise ValueError(f"Unknown stream {stream_id}")
            st = self.active_streams.pop(stream_id)
            self.stats["active_streams"] = len(self.active_streams)
            final = (st.final_transcript + " " + st.partial_transcript).strip()
            logger.info(f"Finalized stream {stream_id}")
            return final

    # --- internal ---
    def _ctc_greedy_decode(self, log_probs: torch.Tensor, lengths: torch.Tensor) -> List[str]:
        """
        Greedy CTC decode with blank=0, map 1..95 -> ASCII 32..126.
        log_probs: [B, T', V], lengths: [B]
        """
        B = log_probs.shape[0]
        out = []
        for i in range(B):
            T = int(lengths[i].item())
            seq = log_probs[i, :T].argmax(dim=-1).tolist()
            prev = -1
            chars = []
            for tok in seq:
                if tok != 0 and tok != prev:
                    chars.append(self.idx_to_char.get(tok, "?"))
                prev = tok
            out.append("".join(chars))
        return out

    def _inference(self, batch: List[AudioChunk]) -> List[Tuple[str, str, int]]:
        """
        Returns: list of tuples (stream_id, decoded_text, chunk_id)
        """
        print(f"Running inference on batch of size {batch}")
        # 1) audio -> mel
        mels = [self.frontend.to_mel(c.audio_data) for c in batch]  # each [F, T_i]
        lengths = torch.tensor([m.shape[1] for m in mels], device=self.device)
        B = len(mels)
        F = mels[0].shape[0]
        T_max = int(lengths.max().item())

        # 2) pad to [B, T, F]
        mel_3d = torch.zeros((B, T_max, F), dtype=torch.float32, device=self.device)
        for i, m in enumerate(mels):
            Ti = m.shape[1]
            mel_3d[i, :Ti, :] = m.T.to(self.device)

        # 3) model
        t0 = time.time()
        with torch.no_grad():
            log_probs, out_lengths = self.model(mel_3d, lengths)
        dt = (time.time() - t0) * 1000.0
        self._lat_ewma = 0.9 * self._lat_ewma + 0.1 * dt if self._lat_ewma else dt
        self.stats["avg_latency_ms"] = self._lat_ewma

        # 4) decode
        texts = self._ctc_greedy_decode(log_probs, out_lengths)
        logger.info(f"texts: {texts}")
        return [(c.stream_id, t, c.chunk_id) for c, t in zip(batch, texts)]

    def _processing_loop(self):
        while self.is_running:
            try:
                c = self.processing_queue.get(timeout=self.batch_manager.max_wait_time)
            except queue.Empty:
                continue

            ready = self.batch_manager.add_chunk(c)
            if not ready:
                continue

            batch = self.batch_manager.get_batch()
            logger.info(f"Processing batch of size {len(batch)} for streams {[x.stream_id for x in batch]}")
            results = self._inference(batch)

            with self.stream_lock:
                for sid, text, cid in results:
                    st = self.active_streams.get(sid)
                    if not st:
                        continue
                    # In a real system, you'd stitch partials; here we overwrite
                    st.partial_transcript = text
                    st.processed_chunks += 1
                    self.stats["total_chunks_processed"] += 1
                    logger.debug(f"Stream {sid}: processed chunk {cid} -> '{text}'")


# ----------------------------
# Helpers to synthesize audio
# ----------------------------
def synthesize_wave(sr: int, n_samples: int, dialect: str = "standard") -> np.ndarray:
    """
    Very simple synthetic audio:
    - base tone frequency depends on 'dialect'
    - slight AM + noise
    """
    t = np.arange(n_samples, dtype=np.float32) / sr
    base = {
        "standard": 220.0,
        "southern": 200.0,
        "british": 240.0,
        "australian": 260.0,
        "indian": 280.0,
    }.get(dialect, 220.0)
    
    
    # generate random number between -0.05 and 0.05
    num = np.random.rand(1)
    logger.info(f"num: {num}")
    
    tone = num * np.sin(2 * np.pi * base * t)
    am = 0.03 * np.sin(2 * np.pi * 3.0 * t)
    noise = 0.01 * np.random.randn(n_samples).astype(np.float32)
    x = tone * (1.0 + am) + noise
    return np.clip(x, -1.0, 1.0).astype(np.float32)


# ----------------------------
# Main (CLI)
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="Streaming ASR (single-file) with SimpleASRModel")
    ap.add_argument("--model-size", choices=["small", "medium", "large"], default="medium")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--n-mels", type=int, default=80)

    ap.add_argument("--max-batch-size", type=int, default=8)
    ap.add_argument("--max-wait-ms", type=int, default=50)

    ap.add_argument("--audio-length", type=int, default=32000, help="Samples for synthetic audio")
    ap.add_argument("--dialect", type=str, default="standard",
                    choices=["standard", "southern", "british", "australian", "indian"])

    ap.add_argument("--simulate-stream", action="store_true")
    ap.add_argument("--chunk-ms", type=int, default=320)
    ap.add_argument("--overlap-ms", type=int, default=0)
    ap.add_argument("--rt-speed", type=float, default=1.0)

    ap.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level))

    # Build engine
    engine = StreamingASREngine(
        model_size=args.model_size,
        device=args.device,
        sr=args.sr,
        n_mels=args.n_mels,
        max_batch_size=args.max_batch_size,
        max_wait_ms=args.max_wait_ms,
    )

    # One stream demo
    sid = engine.add_stream()

    if args.simulate_stream:
        audio = synthesize_wave(args.sr, args.audio_length, args.dialect)
        feeder = RealTimeFeeder(sr=args.sr, chunk_ms=args.chunk_ms, overlap_ms=args.overlap_ms, speed=args.rt_speed)

        def send_fn(chunk_np: np.ndarray):
            engine.process_chunk(sid, chunk_np)

        feeder.stream(audio, send_fn)
        engine.drain(timeout_s=2.0)
    else:
        # One-shot single chunk
        audio = synthesize_wave(args.sr, args.audio_length, args.dialect)
        engine.process_chunk(sid, audio)
        engine.drain(timeout_s=2.0)

    final = engine.finalize_stream(sid)
    logger.info(f"Final transcript: {final}")
    engine.stop()


if __name__ == "__main__":
    main()
