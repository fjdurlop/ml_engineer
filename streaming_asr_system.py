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

import torchaudio  # add this import

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
        # wav_np: [T] float32, mono
        wav = torch.from_numpy(wav_np).float().unsqueeze(0)            # [1, T]
        mel = self.melspec(wav)                                        # [1, n_mels, frames]
        mel_db = self.db(mel)                                          # [1, n_mels, frames]
        return mel_db.squeeze(0)                                       # [n_mels, frames]
    
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
    next_chunk_id: int = 0  # add this
    
    def __post_init__(self):
        if not hasattr(self, 'audio_buffer') or self.audio_buffer is None:
            self.audio_buffer = deque(maxlen=50)  # Keep last 50 chunks
        if not hasattr(self, 'context_buffer') or self.context_buffer is None:
            self.context_buffer = deque(maxlen=10)  # Keep context for continuity


class StreamBatch:
    """Manages batching of multiple streams for efficient processing"""
    
    def __init__(self, max_batch_size: int = 2, max_wait_time: float = 0.05):
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
        
        logger.info(f"Batch size pending: {len(self.pending_chunks)}, Time elapsed: {time_elapsed:.3f}s")
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
        self.batch_manager = StreamBatch(max_batch_size=2, max_wait_time=0.05)
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
        
        self.frontend = FeatureFrontend(sr=16000, n_mels=80) 
        
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
            
            chunk_id = stream_state.next_chunk_id
            stream_state.next_chunk_id += 1  # increment on enqueue
        
        # Create chunk object
        chunk = AudioChunk(
            stream_id=stream_id,
            chunk_id=stream_state.processed_chunks,
            audio_data=audio_chunk,
            timestamp=time.time()
        )
        
        # chunk = AudioChunk(
        #     stream_id=stream_id,
        #     chunk_id=chunk_id,
        #     audio_data=audio_chunk,
        #     timestamp=time.time()
        # )
        
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
        
    def _inference(self, batch: List[AudioChunk]) -> str:
        
        logger.info(f"Running inference on batch of size {len(batch)}")
        # Build [B, 1, T_max, F] instead of [B, 1, F, T_max]
        mels = []
        lengths = []
        for c in batch:
            mel_spec, length = self.frontend.to_mel(c.audio_data)            # [F, T_i]
            mels.append(mel_spec)
            lengths.append(length)
            
        

        with torch.no_grad():
            log_probs, output_lengths = self.model(mels, lengths)  # if supported: self.model(batch_input, lengths)

            #logger.info(f"Received {len(transcripts)} transcripts from ASR model")
            #logger.info(f"Transcripts: {transcripts}")
            
            # DEBUG: inspect argmax distribution (before ctc_decode)
            seq = log_probs[0, :output_lengths[0]].argmax(dim=-1).detach().cpu().numpy()
            unique, counts = np.unique(seq, return_counts=True)
            print("argmax tokens (id:count):", dict(zip(unique.tolist(), counts.tolist())))
            print("top-1 mean prob:", float(log_probs[0, :output_lengths[0]].exp().max(dim=-1).values.mean()))
                                
            #print(f"time:{time.time()}, log_probs:{log_probs.shape}, output_lengths:{output_lengths}")
            predictions = ctc_decode(log_probs, output_lengths)
            
        return predictions
    
    def _inference_1(self, batch_input) -> str:
        
        
        with torch.no_grad():
            logger.info("Running ASR model inference")
            dummy_audio = synthesize_audio_features("Hello world")
            
            mel_spec, length = synthesize_audio_features("test text", dialect="standard")
            mel_spec = mel_spec.to(self.device)
            length = length.to(self.device)
            
            # Measure inference time
            with torch.no_grad():
                #print(f"time:{time.time()}, mel_spec:{mel_spec.shape}, length:{length}")
                log_probs, output_lengths = self.model(mel_spec, length)
                
                # DEBUG: inspect argmax distribution (before ctc_decode)
                seq = log_probs[0, :output_lengths[0]].argmax(dim=-1).detach().cpu().numpy()
                unique, counts = np.unique(seq, return_counts=True)
                print("argmax tokens (id:count):", dict(zip(unique.tolist(), counts.tolist())))
                print("top-1 mean prob:", float(log_probs[0, :output_lengths[0]].exp().max(dim=-1).values.mean()))
                                    
                #print(f"time:{time.time()}, log_probs:{log_probs.shape}, output_lengths:{output_lengths}")
                predictions = ctc_decode(log_probs, output_lengths)
                logger.info(f"Model produced {len(predictions)} transcripts")
                transcripts = predictions[0]
                    
            #transcripts = self.model(batch_input)
            #logger.info(f"Received {len(transcripts)} transcripts from ASR model")
            #logger.info(f"Transcripts: {transcripts}")
        
            
        return transcripts
    
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
                logger.info("Batch not ready yet, continuing to accumulate")
                continue
            batch = self.batch_manager.get_batch()
            
            batch_input = self.preprocess_batch(batch)
            
            # logger.info(f"Processing batch of size {len(batch)} for streams {[c.stream_id for c in batch]}")
            # # Run model inference
            # audio_tensors = [torch.from_numpy(c.audio_data).to(self.device) for c in batch]
            # batch_input = torch.stack(audio_tensors)
            # with torch.no_grad():
            #     logger.info("Running ASR model inference")
            #     transcripts = self.model(batch_input, lengths=None)  # if supported: self.model(batch_input, lengths)
            #     logger.info(f"Received {len(transcripts)} transcripts from ASR model")
            #     logger.info(f"Transcripts: {transcripts}")
            
            #logger.info(f"batch: {batch}")
            transcripts = self._inference_1(batch_input)
            logger.info(f"Transcripts: {transcripts}")            

                
            # Update stream states
            with self.stream_lock:
                for c, transcript in zip(batch, transcripts):
                    logger.info(f"Updating stream {c.stream_id} with transcript: {transcript}")
                    logger.info(f"Updating stream {c.stream_id} active streams: {self.active_streams.keys()}")
                    
                    if c.stream_id in self.active_streams:
                        state = self.active_streams[c.stream_id]
                        state.partial_transcript = transcript
                        state.processed_chunks += 1
                        self.stats["total_chunks_processed"] += 1
                        logger.info(f"Stream {c.stream_id}: processed chunk {c.chunk_id}, total processed this stream={state.processed_chunks}")

    def preprocess_batch(self, batch: List[AudioChunk]) -> torch.Tensor:
        """Convert list of AudioChunks to padded tensor batch"""
        audio_tensors = [torch.from_numpy(c.audio_data).to(self.device) for c in batch]
        batch_input = torch.stack(audio_tensors)
        logger.info(f"Preprocessed batch input shape: {batch_input.shape}")
        return batch_input
    
from streaming_ingress import RealTimeFeeder

def synthesize_audio_features(text: str, dialect: str = "standard") -> Tuple[torch.Tensor, torch.Tensor]:
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


def ctc_decode(log_probs: torch.Tensor, lengths: torch.Tensor) -> List[str]:
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
            
            text = indices_to_text(decoded)
            predictions.append(text)
        
        return predictions

def indices_to_text(indices: List[int]) -> str:
    """Convert indices back to text"""
    # Character to index mapping (simplified)
    char_to_idx = {chr(i): i for i in range(32, 127)}  # ASCII printable
    char_to_idx['<blank>'] = 0
    idx_to_char = {v: k for k, v in char_to_idx.items()}
    chars = [idx_to_char.get(idx, '?') for idx in indices if idx != 0]  # Remove blanks
    return ''.join(chars)

# Basic usage test of [`StreamingASREngine`](streaming_asr_system.py:87)
def main():
    parser = argparse.ArgumentParser(description="Test StreamingASREngine basic functionality")

    parser.add_argument('--log-level', default='INFO')
    parser.add_argument("--model-path", type=str, default=MODELS_DIR, help="Path to ASR model directory")
    parser.add_argument("--audio-length", type=int, default=16000, help="Length of dummy audio array") # lenght in samples, one sample = 1/16000 sec
    
    parser.add_argument("--sr", type=int, default=16000, help="Sample rate")
    parser.add_argument("--chunk-ms", type=int, default=320, help="Chunk duration (ms)")
    parser.add_argument("--overlap-ms", type=int, default=0, help="Chunk overlap (ms)")
    parser.add_argument("--rt-speed", type=float, default=1.0, help="Real-time speed factor (1.0 = real time)")
    parser.add_argument("--simulate-stream", action="store_true",
                        help="Simulate real-time streaming of the dummy audio")

    args = parser.parse_args()
    

    engine = StreamingASREngine(model_path=args.model_path)
    stream_id = engine.add_stream()
    
    if args.simulate_stream:
        # Build a dummy “speech-like” waveform instead of pure zeros (optional).
        t = np.arange(args.audio_length, dtype=np.float32) / args.sr
        dummy_audio = 0.1 * np.sin(2 * np.pi * 220 * t).astype(np.float32)  # 220 Hz tone as placeholder
        feeder = RealTimeFeeder(sr=args.sr, chunk_ms=args.chunk_ms, overlap_ms=args.overlap_ms, speed=args.rt_speed)

        def send_fn(chunk_np: np.ndarray):
            engine.process_chunk(stream_id, chunk_np)

        feeder.stream(dummy_audio, send_fn)
    else:
        t = np.arange(args.audio_length, dtype=np.float32) / args.sr
        dummy_audio = 0.1 * np.sin(2 * np.pi * 220 * t).astype(np.float32)  # 220 Hz tone as placeholder
        #dummy_audio = np.zeros(args.audio_length, dtype=np.float32)
        
        #dummy_audio = synthesize_audio_features("Hello world")
        
        partial = engine.process_chunk(stream_id, dummy_audio)
        logger.info(f"Partial transcript: {partial}")
        
        dummy_audio = synthesize_audio_features("Hello world")
        dummy_audio = 0.1 * np.sin(2 * np.pi * 220 * t).astype(np.float32)  # 220 Hz tone as placeholder
        
        partial = engine.process_chunk(stream_id, dummy_audio)
        logger.info(f"Partial transcript: {partial}")
        
        dummy_audio = synthesize_audio_features("Hello world")
        dummy_audio = 0.1 * np.sin(2 * np.pi * 220 * t).astype(np.float32)  # 220 Hz tone as placeholder
        
        partial = engine.process_chunk(stream_id, dummy_audio)
        logger.info(f"Partial transcript: {partial}")
        
        dummy_audio = synthesize_audio_features("Hello world")
        dummy_audio = 0.1 * np.sin(2 * np.pi * 220 * t).astype(np.float32)  # 220 Hz tone as placeholder
        
        partial = engine.process_chunk(stream_id, dummy_audio)
        logger.info(f"Partial transcript: {partial}")
    
    final = engine.finalize_stream(stream_id)
    logger.info(f"Final transcript: {final}")
    engine.stop()

if __name__ == "__main__":
    main()
