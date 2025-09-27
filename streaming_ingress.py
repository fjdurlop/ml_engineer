# streaming_ingress.py
import time
import numpy as np
from typing import Callable, Optional

class RealTimeFeeder:
    """
    Sends audio to engine.process_chunk(...) at real-time pace.
    Works with numpy float32 mono audio.
    """
    def __init__(self, sr: int = 16000, chunk_ms: int = 320, overlap_ms: int = 0, speed: float = 1.0):
        self.sr = sr
        self.chunk_ms = chunk_ms
        self.overlap_ms = overlap_ms
        self.speed = speed  # 1.0 = real time, 2.0 = 2x faster, etc.

        self.chunk_samples = int(sr * chunk_ms / 1000)
        self.overlap_samples = int(sr * overlap_ms / 1000)
        assert self.chunk_samples > 0 and self.overlap_samples >= 0
        assert self.overlap_samples < self.chunk_samples

        # Step size for sliding window with overlap
        self.hop = self.chunk_samples - self.overlap_samples

    def stream(
        self,
        audio: np.ndarray,
        send_fn: Callable[[np.ndarray], None],
        on_chunk_sent: Optional[Callable[[int], None]] = None
    ):
        """
        audio: 1D np.float32
        send_fn: function that accepts one chunk (np.ndarray) and forwards to engine
        """
        n = len(audio)
        t_per_chunk = (self.chunk_samples / self.sr) / self.speed  # seconds

        idx = 0
        chunk_idx = 0
        next_deadline = time.time()

        while idx < n:
            end = min(idx + self.chunk_samples, n)
            chunk = audio[idx:end]
            if len(chunk) < self.chunk_samples:
                # pad last chunk to fixed duration for simplicity
                pad = np.zeros(self.chunk_samples - len(chunk), dtype=np.float32)
                chunk = np.concatenate([chunk, pad], axis=0)

            send_fn(chunk.astype(np.float32))
            if on_chunk_sent:
                on_chunk_sent(chunk_idx)

            # real-time pacing
            next_deadline += t_per_chunk
            now = time.time()
            sleep_s = next_deadline - now
            if sleep_s > 0:
                time.sleep(sleep_s)

            idx += self.hop
            chunk_idx += 1
