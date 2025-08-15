"""
Voice Activity Detection (VAD) for streaming audio processing
Detects speech segments and trims silence
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)

class VoiceActivityDetector:
    """Simple voice activity detector based on energy and zero-crossing rate."""
    
    def __init__(self):
        self.sample_rate = 16000
        self.frame_size = 512  # ~32ms frames at 16kHz
        self.hop_size = 256    # 50% overlap
        
        # VAD thresholds
        self.energy_threshold = 0.01  # Minimum energy for speech
        self.zcr_threshold = 0.3      # Zero crossing rate threshold
        self.min_speech_frames = 5    # Minimum frames for speech detection
        self.min_silence_frames = 8   # Minimum silence frames to end speech
        
        # State tracking
        self.is_speech_active = False
        self.speech_frames = 0
        self.silence_frames = 0
        self.speech_buffer = []
        
        logger.info("Voice Activity Detector initialized")
    
    def reset(self):
        """Reset VAD state."""
        self.is_speech_active = False
        self.speech_frames = 0
        self.silence_frames = 0
        self.speech_buffer = []
    
    def compute_energy(self, frame):
        """Compute energy of audio frame."""
        return np.mean(frame ** 2)
    
    def compute_zcr(self, frame):
        """Compute zero crossing rate of audio frame."""
        zcr = np.sum(np.abs(np.diff(np.sign(frame)))) / (2 * len(frame))
        return zcr
    
    def is_speech_frame(self, frame):
        """Determine if frame contains speech."""
        energy = self.compute_energy(frame)
        zcr = self.compute_zcr(frame)
        
        # Simple rule: speech has moderate energy and ZCR
        has_energy = energy > self.energy_threshold
        has_reasonable_zcr = zcr < self.zcr_threshold
        
        return has_energy and has_reasonable_zcr
    
    def process_chunk(self, audio_data):
        """
        Process audio chunk and return speech segments.
        
        Args:
            audio_data: numpy array of audio samples
            
        Returns:
            List of (start_sample, end_sample) tuples for speech segments
        """
        if len(audio_data) == 0:
            return []
        
        speech_segments = []
        num_frames = (len(audio_data) - self.frame_size) // self.hop_size + 1
        
        for i in range(num_frames):
            start_idx = i * self.hop_size
            end_idx = start_idx + self.frame_size
            
            if end_idx > len(audio_data):
                break
            
            frame = audio_data[start_idx:end_idx]
            is_speech = self.is_speech_frame(frame)
            
            if is_speech:
                self.speech_frames += 1
                self.silence_frames = 0
                
                if not self.is_speech_active and self.speech_frames >= self.min_speech_frames:
                    # Speech started
                    self.is_speech_active = True
                    self.speech_start_idx = max(0, start_idx - self.min_speech_frames * self.hop_size)
                    logger.debug(f"Speech started at sample {self.speech_start_idx}")
                    
            else:
                self.silence_frames += 1
                
                if self.is_speech_active and self.silence_frames >= self.min_silence_frames:
                    # Speech ended
                    speech_end_idx = start_idx
                    speech_segments.append((self.speech_start_idx, speech_end_idx))
                    logger.debug(f"Speech ended at sample {speech_end_idx}")
                    
                    # Reset for next speech segment
                    self.is_speech_active = False
                    self.speech_frames = 0
                    self.silence_frames = 0
        
        return speech_segments
    
    def extract_speech_segments(self, audio_data, segments):
        """Extract speech segments from audio data."""
        speech_chunks = []
        
        for start_idx, end_idx in segments:
            if end_idx > start_idx:
                segment = audio_data[start_idx:end_idx]
                # Trim silence from edges
                segment = self.trim_silence(segment)
                if len(segment) > self.sample_rate * 0.3:  # At least 300ms
                    speech_chunks.append(segment)
        
        return speech_chunks
    
    def trim_silence(self, audio_data, silence_threshold=0.01):
        """Trim silence from beginning and end of audio."""
        if len(audio_data) == 0:
            return audio_data
        
        # Find first and last non-silent samples
        energy = audio_data ** 2
        non_silent = energy > silence_threshold
        
        if not np.any(non_silent):
            return audio_data  # All silence, return as is
        
        first_sound = np.argmax(non_silent)
        last_sound = len(non_silent) - np.argmax(non_silent[::-1]) - 1
        
        return audio_data[first_sound:last_sound + 1]
    
    def get_current_speech_segment(self, audio_data):
        """Get current ongoing speech segment if any."""
        if self.is_speech_active and len(audio_data) > 0:
            current_segment = audio_data[self.speech_start_idx:]
            if len(current_segment) > self.sample_rate * 0.5:  # At least 500ms
                return self.trim_silence(current_segment)
        return None