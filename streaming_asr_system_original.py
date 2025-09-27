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
        


# Basic usage test of [`StreamingASREngine`](streaming_asr_system.py:87)
def main():
    parser = argparse.ArgumentParser(description="Test StreamingASREngine basic functionality")
    parser.add_argument("--model-path", type=str, default=MODELS_DIR, help="Path to ASR model directory")
    parser.add_argument("--audio-length", type=int, default=16000, help="Length of dummy audio array")
    args = parser.parse_args()
    engine = StreamingASREngine(model_path=args.model_path)
    stream_id = engine.add_stream()
    dummy_audio = np.zeros(args.audio_length, dtype=np.float32)
    partial = engine.process_chunk(stream_id, dummy_audio)
    print(f"Partial transcript: {partial}")
    final = engine.finalize_stream(stream_id)
    print(f"Final transcript: {final}")
    engine.stop()

if __name__ == "__main__":
    main()
