"""
WebRTC VAD implementation for streaming audio processing
Provides high-performance voice activity detection with proper audio chunking
"""

import webrtcvad
import collections
import numpy as np
import logging
from typing import List, Tuple, Optional, Generator
import struct
import threading
import queue
import time

logger = logging.getLogger(__name__)

class WebRTCVADProcessor:
    """
    WebRTC-based Voice Activity Detection processor for streaming audio.
    
    Features:
    - Real-time VAD processing with WebRTC library
    - Proper audio chunking and buffering
    - Speech segment detection and extraction
    - Thread-safe operation for streaming applications
    """
    
    def __init__(self, aggressiveness=2, sample_rate=16000, frame_duration=30):
        """
        Initialize WebRTC VAD processor.
        
        Args:
            aggressiveness: VAD aggressiveness mode (0-3, higher = more aggressive)
            sample_rate: Audio sample rate (8000, 16000, 32000, or 48000 Hz)
            frame_duration: Frame duration in milliseconds (10, 20, or 30 ms)
        """
        self.vad = webrtcvad.Vad(aggressiveness)
        self.sample_rate = sample_rate
        self.frame_duration = frame_duration
        self.frame_size = int(sample_rate * frame_duration / 1000)
        
        # Circular buffer for frame management
        self.ring_buffer_size = max(10, int(500 / frame_duration))  # ~500ms buffer
        self.ring_buffer = collections.deque(maxlen=self.ring_buffer_size)
        
        # State tracking
        self.triggered = False
        self.speech_buffer = collections.deque()
        self.is_recording = False
        self.current_utterance_start = None
        
        # Configuration parameters
        self.silence_threshold = 0.8  # Ratio of silence frames to trigger end
        self.speech_threshold = 0.5   # Ratio of speech frames to trigger start
        self.min_speech_duration = 0.5  # Minimum speech duration in seconds
        self.max_speech_duration = 10.0  # Maximum speech duration in seconds
        self.max_silence_duration = 2.0  # Maximum silence before reset
        
        # Performance tracking
        self.total_frames_processed = 0
        self.speech_frames_detected = 0
        self.segments_extracted = 0
        
        # Thread-safe queue for streaming chunks
        self.audio_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.processing = False
        
        logger.info(f"WebRTC VAD initialized: aggressiveness={aggressiveness}, "
                   f"sample_rate={sample_rate}Hz, frame_duration={frame_duration}ms")
    
    def reset_state(self):
        """Reset VAD state for new processing session."""
        self.triggered = False
        self.is_recording = False
        self.ring_buffer.clear()
        self.speech_buffer.clear()
        self.current_utterance_start = None
        logger.debug("VAD state reset")
    
    def convert_audio_to_frames(self, audio_data: bytes) -> Generator[bytes, None, None]:
        """
        Convert audio data to properly sized frames for WebRTC VAD.
        
        Args:
            audio_data: Raw audio bytes (16-bit PCM)
            
        Yields:
            bytes: Frame data suitable for VAD processing
        """
        frame_size_bytes = self.frame_size * 2  # 16-bit = 2 bytes per sample
        
        for i in range(0, len(audio_data) - frame_size_bytes + 1, frame_size_bytes):
            frame = audio_data[i:i + frame_size_bytes]
            if len(frame) == frame_size_bytes:
                yield frame
    
    def is_speech_frame(self, frame: bytes) -> bool:
        """
        Determine if a frame contains speech using WebRTC VAD.
        
        Args:
            frame: Audio frame bytes
            
        Returns:
            bool: True if frame contains speech
        """
        try:
            if len(frame) != self.frame_size * 2:
                return False
            return self.vad.is_speech(frame, self.sample_rate)
        except Exception as e:
            logger.warning(f"VAD frame analysis failed: {e}")
            return False
    
    def process_audio_chunk(self, audio_data: bytes) -> List[bytes]:
        """
        Process audio chunk and return complete speech segments.
        
        Args:
            audio_data: Raw audio bytes (16-bit PCM)
            
        Returns:
            List[bytes]: List of detected speech segments
        """
        speech_segments = []
        
        for frame in self.convert_audio_to_frames(audio_data):
            self.total_frames_processed += 1
            is_speech = self.is_speech_frame(frame)
            
            if is_speech:
                self.speech_frames_detected += 1
            
            # Process frame through VAD collector
            collected_audio = self._vad_collector_step(frame, is_speech)
            
            if collected_audio is not None:
                # Complete speech segment detected
                speech_segments.append(collected_audio)
                self.segments_extracted += 1
                logger.debug(f"Speech segment extracted: {len(collected_audio)} bytes")
        
        return speech_segments
    
    def _vad_collector_step(self, frame: bytes, is_speech: bool) -> Optional[bytes]:
        """
        Single step of VAD collection algorithm.
        
        Args:
            frame: Audio frame
            is_speech: Whether frame contains speech
            
        Returns:
            bytes: Complete speech segment if detected, None otherwise
        """
        if not self.triggered:
            # Not currently in speech mode
            self.ring_buffer.append((frame, is_speech))
            num_voiced = sum(1 for f, speech in self.ring_buffer if speech)
            
            # Check if we should trigger speech detection
            if len(self.ring_buffer) == self.ring_buffer.maxlen:
                if num_voiced >= self.speech_threshold * self.ring_buffer.maxlen:
                    self.triggered = True
                    self.is_recording = True
                    self.current_utterance_start = time.time()
                    
                    # Output buffered frames to start speech segment
                    self.speech_buffer.clear()
                    for f, s in self.ring_buffer:
                        self.speech_buffer.append(f)
                    
                    self.ring_buffer.clear()
                    logger.debug("Speech triggered - starting collection")
                    
        else:
            # Currently in speech mode
            self.speech_buffer.append(frame)
            self.ring_buffer.append((frame, is_speech))
            
            # Check duration limits
            if self.current_utterance_start:
                utterance_duration = time.time() - self.current_utterance_start
                
                if utterance_duration > self.max_speech_duration:
                    # Force end due to maximum duration
                    logger.debug("Speech segment ended due to max duration")
                    return self._finalize_speech_segment()
            
            # Check for end of speech
            if len(self.ring_buffer) == self.ring_buffer.maxlen:
                num_unvoiced = sum(1 for f, speech in self.ring_buffer if not speech)
                
                if num_unvoiced >= self.silence_threshold * self.ring_buffer.maxlen:
                    # End of speech detected
                    logger.debug("Speech segment ended due to silence")
                    return self._finalize_speech_segment()
        
        return None
    
    def _finalize_speech_segment(self) -> Optional[bytes]:
        """
        Finalize and return current speech segment.
        
        Returns:
            bytes: Complete speech segment or None if too short
        """
        if not self.speech_buffer:
            self.triggered = False
            self.is_recording = False
            return None
        
        # Calculate duration
        total_frames = len(self.speech_buffer)
        duration = total_frames * self.frame_duration / 1000.0
        
        # Apply stricter minimum duration filter (0.1s minimum)
        min_duration = max(self.min_speech_duration, 0.1)  # At least 100ms
        
        # Check minimum duration
        if duration < min_duration:
            logger.debug(f"Speech segment too short: {duration:.2f}s < {min_duration}s")
            self.triggered = False
            self.is_recording = False
            self.speech_buffer.clear()
            self.ring_buffer.clear()
            return None
        
        # Create complete audio segment
        speech_data = b''.join(self.speech_buffer)
        
        # Reset state
        self.triggered = False
        self.is_recording = False
        self.speech_buffer.clear()
        self.ring_buffer.clear()
        self.current_utterance_start = None
        
        logger.info(f"Speech segment finalized: {duration:.2f}s, {len(speech_data)} bytes")
        return speech_data
    
    def process_numpy_audio(self, audio_array: np.ndarray) -> List[bytes]:
        """
        Process numpy audio array.
        
        Args:
            audio_array: Audio data as numpy array (float32, -1 to 1 range)
            
        Returns:
            List[bytes]: List of detected speech segments
        """
        # Convert to 16-bit PCM bytes
        if audio_array.dtype != np.int16:
            # Normalize and convert to int16
            audio_normalized = np.clip(audio_array, -1.0, 1.0)
            audio_int16 = (audio_normalized * 32767).astype(np.int16)
        else:
            audio_int16 = audio_array
        
        # Convert to bytes
        audio_bytes = audio_int16.tobytes()
        
        return self.process_audio_chunk(audio_bytes)
    
    def get_current_segment(self) -> Optional[bytes]:
        """
        Get current ongoing speech segment if any.
        
        Returns:
            bytes: Current speech segment or None
        """
        if self.is_recording and self.speech_buffer:
            current_duration = len(self.speech_buffer) * self.frame_duration / 1000.0
            if current_duration >= self.min_speech_duration:
                return b''.join(self.speech_buffer)
        return None
    
    def start_streaming_processing(self):
        """Start background thread for streaming audio processing."""
        if self.processing:
            return
        
        self.processing = True
        self.processing_thread = threading.Thread(target=self._streaming_worker, daemon=True)
        self.processing_thread.start()
        logger.info("Started streaming VAD processing")
    
    def stop_streaming_processing(self):
        """Stop background streaming processing."""
        self.processing = False
        if hasattr(self, 'processing_thread'):
            self.processing_thread.join(timeout=1.0)
        logger.info("Stopped streaming VAD processing")
    
    def add_audio_chunk(self, audio_data: bytes):
        """
        Add audio chunk to processing queue (thread-safe).
        
        Args:
            audio_data: Raw audio bytes
        """
        if self.processing:
            try:
                self.audio_queue.put_nowait(audio_data)
            except queue.Full:
                logger.warning("Audio queue full, dropping chunk")
    
    def get_speech_segments(self) -> List[bytes]:
        """
        Get all available speech segments from processing queue.
        
        Returns:
            List[bytes]: Available speech segments
        """
        segments = []
        try:
            while True:
                segment = self.output_queue.get_nowait()
                segments.append(segment)
        except queue.Empty:
            pass
        return segments
    
    def _streaming_worker(self):
        """Background worker for streaming audio processing."""
        while self.processing:
            try:
                # Get audio chunk with timeout
                audio_chunk = self.audio_queue.get(timeout=0.1)
                
                # Process chunk
                segments = self.process_audio_chunk(audio_chunk)
                
                # Add segments to output queue
                for segment in segments:
                    try:
                        self.output_queue.put_nowait(segment)
                    except queue.Full:
                        logger.warning("Output queue full, dropping segment")
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Streaming processing error: {e}")
    
    def get_stats(self) -> dict:
        """
        Get VAD processing statistics.
        
        Returns:
            dict: Processing statistics
        """
        return {
            'total_frames_processed': self.total_frames_processed,
            'speech_frames_detected': self.speech_frames_detected,
            'segments_extracted': self.segments_extracted,
            'speech_ratio': (
                self.speech_frames_detected / max(1, self.total_frames_processed)
            ),
            'is_recording': self.is_recording,
            'triggered': self.triggered,
            'buffer_size': len(self.speech_buffer),
            'ring_buffer_size': len(self.ring_buffer),
            'configuration': {
                'sample_rate': self.sample_rate,
                'frame_duration': self.frame_duration,
                'min_speech_duration': self.min_speech_duration,
                'max_speech_duration': self.max_speech_duration
            }
        }

class StreamingAudioBuffer:
    """
    Optimized audio buffer for streaming VAD processing.
    Thread-safe with memory pool for high performance.
    """
    
    def __init__(self, sample_rate=16000, max_duration=30):
        self.sample_rate = sample_rate
        self.max_samples = sample_rate * max_duration
        
        # Thread-safe circular buffer
        self.buffer = collections.deque(maxlen=self.max_samples)
        self.buffer_lock = threading.RLock()
        
        # Performance tracking
        self.total_samples_added = 0
        self.buffer_overruns = 0
    
    def add_audio(self, audio_data: np.ndarray):
        """
        Add audio data to buffer (thread-safe).
        
        Args:
            audio_data: Audio samples as numpy array
        """
        with self.buffer_lock:
            if len(self.buffer) + len(audio_data) > self.max_samples:
                self.buffer_overruns += 1
                # Remove old samples to make room
                samples_to_remove = len(audio_data)
                for _ in range(min(samples_to_remove, len(self.buffer))):
                    self.buffer.popleft()
            
            self.buffer.extend(audio_data)
            self.total_samples_added += len(audio_data)
    
    def get_recent_audio(self, duration_ms: int = 1000) -> np.ndarray:
        """
        Get recent audio with specified duration.
        
        Args:
            duration_ms: Duration in milliseconds
            
        Returns:
            np.ndarray: Recent audio samples
        """
        samples_needed = int(self.sample_rate * duration_ms / 1000)
        
        with self.buffer_lock:
            if len(self.buffer) >= samples_needed:
                return np.array(list(self.buffer)[-samples_needed:], dtype=np.float32)
            else:
                return np.array(list(self.buffer), dtype=np.float32)
    
    def clear(self):
        """Clear buffer contents."""
        with self.buffer_lock:
            self.buffer.clear()
    
    def get_stats(self) -> dict:
        """Get buffer statistics."""
        with self.buffer_lock:
            return {
                'buffer_size': len(self.buffer),
                'max_size': self.max_samples,
                'utilization': len(self.buffer) / self.max_samples,
                'total_added': self.total_samples_added,
                'overruns': self.buffer_overruns
            }