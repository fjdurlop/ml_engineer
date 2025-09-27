"""
Streaming ASR System with Continuous Batching
Template for implementing real-time speech recognition with multiple concurrent streams.
uv run streaming_asr_system.py 2>&1 | tee logs/p3/streaming_asr_system.out
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
import argparse


MODELS_DIR = "models"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    
    def __init__(self, max_batch_size: int = 8, max_wait_time: float = 0.05, max_batch_frames: int = 20000):
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.pending_chunks: List[AudioChunk] = []
        self.last_batch_time = time.time()
        
        self.max_batch_frames = max_batch_frames
        self.total_frames = 0
    
    def add_chunk(self, chunk: AudioChunk) -> bool:
        """Add chunk to batch. Returns True if batch is ready."""
        #self.pending_chunks.append(chunk)
        
        self.pending_chunks.append((chunk))
        #self.total_frames += frames
        
        # Batch is ready if we hit size limit or time limit
        current_time = time.time()
        time_elapsed = current_time - self.last_batch_time
        
        
        
        return (len(self.pending_chunks) >= self.max_batch_size or 
               time_elapsed >= self.max_wait_time)
        
        # return (len(self.pending_chunks) >= self.max_batch_size or
        #         self.total_frames >= self.max_batch_frames or
        #         time_elapsed >= self.max_wait_time)
    
    def get_batch(self) -> List[AudioChunk]:
        """Get current batch and reset"""
        batch = self.pending_chunks.copy()
        self.pending_chunks.clear()
        self.total_frames = 0
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
        logger.info("Initializing Streaming ASR Engine")
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
        logger.info(f"Enqueuing chunk {chunk.chunk_id} for stream {stream_id} at {chunk.timestamp}")
        try:
            self.processing_queue.put(chunk, timeout=0.1)
            logger.info(f"Chunk {chunk.chunk_id} enqueued successfully")
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
    
    def _processing_loop(self):
        rr_idx = 0
        per_stream_latest = {}  # stream_id -> last dequeued chunk

        while self.is_running:
            # round-robin drain: try each active stream once
            with self.stream_lock:
                stream_ids = list(self.active_streams.keys())
            if not stream_ids:
                time.sleep(self.batch_manager.max_wait_time / 2)
                continue

            start_idx = rr_idx % len(stream_ids)
            for i in range(len(stream_ids)):
                sid = stream_ids[(start_idx + i) % len(stream_ids)]
                try:
                    chunk = self.processing_queue.get_nowait()  # or per-stream queue.get_nowait()
                except queue.Empty:
                    continue
                per_stream_latest[sid] = chunk

                # Convert to feats & add to batch manager
                feats = self.frontend.chunk_to_feats(chunk.audio_data)
                ready = self.batch_manager.add_chunk((chunk, feats), frames=feats.shape[0])
                if ready:
                    items = self.batch_manager.get_batch()
                    b_chunks, b_feats = zip(*items)
                    lengths = torch.tensor([f.shape[0] for f in b_feats], device=self.device)
                    padded = torch.nn.utils.rnn.pad_sequence(b_feats, batch_first=True)
                    t0 = time.time()
                    with torch.no_grad():
                        transcripts = self.model(padded.to(self.device, non_blocking=True), lengths)
                    self._tick_latency(t0)

                    with self.stream_lock:
                        for c, tr in zip(b_chunks, transcripts):
                            if c.stream_id in self.active_streams:
                                s = self.active_streams[c.stream_id]
                                s.partial_transcript = tr  # or merge with context
                                s.processed_chunks += 1
                                self.stats["total_chunks_processed"] += 1

            rr_idx += 1


    def _processing_loop(self):
        """Continuously process queued audio chunks."""
        while self.is_running:
            try:
                chunk = self.processing_queue.get(timeout=self.batch_manager.max_wait_time)
                logger.info(f"Dequeued chunk {chunk.chunk_id} from stream {chunk.stream_id}")
            except queue.Empty:
                continue
            # Add chunk to batch, check if batch ready
            ready = self.batch_manager.add_chunk(chunk)
            logger.info(f"Added chunk {chunk.chunk_id} to batch, batch size {len(self.batch_manager.pending_chunks)}, ready={ready}")
            if not ready:
                continue
            batch = self.batch_manager.get_batch()
            logger.info(f"Processing batch of size {len(batch)} for streams {[c.stream_id for c in batch]}")
            # Run model inference
            audio_tensors = [torch.from_numpy(c.audio_data).to(self.device) for c in batch]
            
            audio_tensors = [synthesize_audio_features(text="Hello world", dialect="standard", audio_gen=True) for c in batch]
            batch_input = torch.stack(audio_tensors)
            
            #batch_input = synthesize_audio_features(text="Hello world", dialect="standard", audio_gen=True)
            with torch.no_grad():
                logger.info(f"Inference")
                for c in batch_input:
                    transcripts = self.model(c)
                    logger.info(f"Model produced {transcripts} transcripts")
                    logger.info(f"Received {len(transcripts)} transcripts from ASR model")
            # Update stream states
            with self.stream_lock:
                for c, transcript in zip(batch, transcripts):
                    if c.stream_id in self.active_streams:
                        state = self.active_streams[c.stream_id]
                        state.partial_transcript = transcript
                        state.processed_chunks += 1
                        self.stats["total_chunks_processed"] += 1
                        logger.info(f"Stream {c.stream_id}: processed chunk {c.chunk_id}, total processed this stream={state.processed_chunks}")


# frontend.py
import numpy as np, torch, torchaudio

class FeatureFrontend:
    def __init__(self, sr=16000, n_mels=80, win_ms=25, hop_ms=10, dtype=torch.float16):
        self.sr = sr
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_mels=n_mels,
            n_fft=int(sr*win_ms/1000), hop_length=int(sr*hop_ms/1000)
        )
        self.db = torchaudio.transforms.AmplitudeToDB(stype="power")
        self.dtype = dtype

    def chunk_to_feats(self, x_np: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(x_np).float()
        with torch.no_grad():
            m = self.db(self.melspec(x.unsqueeze(0))).squeeze(0)  # [n_mels, T]
        return m.t().contiguous().to(dtype=self.dtype)           # [T, n_mels]


def synthesize_audio_features(text: str, dialect: str = "standard", audio_gen = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Synthesize mel spectrogram features from text
    (In real scenario, this would be actual audio files)

    """

    if audio_gen:
        t = np.arange(16000, dtype=np.float32) / 16000
        dummy_audio = 0.1 * np.sin(2 * np.pi * 220 * t).astype(np.float32)  # 220 Hz tone as placeholder
        # dummy_audio = np.zeros(args.audio_length, dtype=np.float32)
        
        # Generate synthetic mel spectrogram
        text_length = 20
        time_steps = max(50, text_length * 3)  # Rough approximation
        mel_bins = 80
        
        # Base spectrogram with some structure
        mel_spec = torch.randn(mel_bins, time_steps) * 0.5
        
        # Add dialect-specific noise
        noise_level = 0.1
        mel_spec += torch.randn_like(mel_spec) * noise_level
        
        # Add some temporal structure
        for i in range(0, time_steps, 10):
            end_idx = min(i + 5, time_steps)
            mel_spec[:, i:end_idx] += torch.randn(mel_bins, end_idx - i) * 0.3
        
        length = torch.tensor(time_steps)
        return mel_spec.transpose(0, 1).unsqueeze(0), length.unsqueeze(0)  # (1, time, mel)
        
        #pass
    else:
        return
    

# Basic usage test of [`StreamingASREngine`](streaming_asr_system.py:87)
def main():
    parser = argparse.ArgumentParser(description="Test StreamingASREngine basic functionality")
    parser.add_argument('--max-batch-size', type=int, default=8)
    parser.add_argument('--max-batch-frames', type=int, default=20000)
    parser.add_argument('--max-wait-ms', type=int, default=50)
    parser.add_argument('--precision', choices=['fp32','fp16','int8'], default='fp16')
    parser.add_argument('--no-amp', action='store_true')
    parser.add_argument('--chunk-overlap-ms', type=int, default=300)
    parser.add_argument('--max-stream-queue', type=int, default=3)
    parser.add_argument('--router-policy', choices=['rr','fifo'], default='rr')
    parser.add_argument('--tmax-frames', type=int, default=1600)
    parser.add_argument('--silence-ms', type=int, default=600)
    parser.add_argument('--min-utt-ms', type=int, default=300)
    parser.add_argument('--use-vad', action='store_true')
    parser.add_argument('--metrics-port', type=int, default=0)
    parser.add_argument('--log-level', default='INFO')

    parser.add_argument("--model-path", type=str, default=MODELS_DIR, help="Path to ASR model directory")
    parser.add_argument("--audio-length", type=int, default=16000, help="Length of dummy audio array") # lenght in samples, one sample = 1/16000 sec
    args = parser.parse_args()
    
    engine = StreamingASREngine(model_path=args.model_path)
    
    stream_id = engine.add_stream()
    dummy_audio = np.zeros(args.audio_length, dtype=np.float32)
    partial = engine.process_chunk(stream_id, dummy_audio)
    logger.info(f"Partial transcript: {partial}")
    final = engine.finalize_stream(stream_id)
    logger.info(f"Final transcript: {final}")
    engine.stop()

if __name__ == "__main__":
    main()
