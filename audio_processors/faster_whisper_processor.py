"""
Faster-Whisper processor with built-in VAD (2025 approach)
More reliable than manual WebRTC VAD + Whisper coordination
"""

import numpy as np
import io
import time
import logging
from typing import Dict, Any, Optional

try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False
    WhisperModel = None

from .base_processor import AudioProcessor

logger = logging.getLogger(__name__)

class FasterWhisperDigitProcessor(AudioProcessor):
    """
    Modern 2025 approach using faster-whisper with built-in VAD.
    Much more reliable than manual WebRTC VAD coordination.
    """
    
    def __init__(self):
        """Initialize faster-whisper processor with built-in VAD."""
        super().__init__("Faster-Whisper with VAD")
        
        if not FASTER_WHISPER_AVAILABLE:
            logger.error("faster-whisper not available. Install with: pip install faster-whisper")
            self.model = None
            return
        
        self.model = None
        self.device = "cuda" if self._cuda_available() else "cpu"
        
        # Digit mapping
        self.digit_map = {
            "zero": "0", "one": "1", "two": "2", "three": "3", 
            "four": "4", "five": "5", "six": "6", "seven": "7", 
            "eight": "8", "nine": "9",
            "oh": "0", "o": "0", "for": "4", "fore": "4", 
            "to": "2", "too": "2", "tu": "2", "tree": "3", 
            "free": "3", "ate": "8", "ait": "8"
        }
        
        # Statistics
        self.total_predictions = 0
        self.successful_predictions = 0
        self.failed_predictions = 0
        
        self._initialize_model()
    
    def _cuda_available(self) -> bool:
        """Check if CUDA is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def _initialize_model(self):
        """Initialize faster-whisper model with VAD."""
        if not FASTER_WHISPER_AVAILABLE:
            return
        
        try:
            logger.info("Initializing faster-whisper model with built-in VAD...")
            
            # Initialize faster-whisper model
            self.model = WhisperModel(
                "tiny",  # Use tiny model for speed
                device=self.device,
                compute_type="float16" if self.device == "cuda" else "int8"
            )
            
            logger.info(f"Faster-Whisper model initialized on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to initialize faster-whisper: {e}")
            self.model = None
    
    def is_configured(self) -> bool:
        """Check if processor is configured."""
        return self.model is not None and FASTER_WHISPER_AVAILABLE
    
    def process_audio(self, audio_data: bytes) -> str:
        """
        Process audio with built-in VAD and return predicted digit.
        
        Args:
            audio_data: Raw audio bytes
            
        Returns:
            str: Predicted digit (0-9) or error message
        """
        if not self.is_configured():
            return "error: Model not configured"
        
        try:
            # Convert audio to numpy array
            audio_array = self._convert_audio_bytes(audio_data)
            if audio_array is None:
                return "error: Audio conversion failed"
            
            # Use faster-whisper with built-in VAD
            segments, info = self.model.transcribe(
                audio_array,
                language="en",
                # Built-in VAD parameters - much better than manual VAD
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=100,  # 100ms minimum silence
                    speech_pad_ms=30  # 30ms padding around speech
                )
            )
            
            # Process transcription results
            transcriptions = []
            for segment in segments:
                text = segment.text.strip().lower()
                if text:
                    transcriptions.append(text)
            
            if not transcriptions:
                return "error: No speech detected"
            
            # Combine all segments and extract digit
            full_text = " ".join(transcriptions)
            digit = self._text_to_digit(full_text)
            
            logger.debug(f"Faster-Whisper: '{full_text}' -> '{digit}'")
            
            if digit in "0123456789":
                self.successful_predictions += 1
                return digit
            else:
                self.failed_predictions += 1
                return f"unclear: {full_text}"
                
        except Exception as e:
            logger.error(f"Faster-Whisper processing failed: {e}")
            self.failed_predictions += 1
            return f"error: {str(e)}"
        finally:
            self.total_predictions += 1
    
    def _convert_audio_bytes(self, audio_data: bytes) -> Optional[np.ndarray]:
        """Convert audio bytes to numpy array for faster-whisper."""
        try:
            # Check if it's a WAV file
            if audio_data.startswith(b'RIFF'):
                import soundfile as sf
                audio_buffer = io.BytesIO(audio_data)
                audio_array, sample_rate = sf.read(audio_buffer, dtype='float32')
                
                # Convert stereo to mono if needed
                if len(audio_array.shape) > 1:
                    audio_array = np.mean(audio_array, axis=1)
                
                return audio_array
            else:
                # Raw PCM data
                audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
                return audio_array / 32768.0
                
        except Exception as e:
            logger.error(f"Audio conversion failed: {e}")
            return None
    
    def _text_to_digit(self, text: str) -> str:
        """Convert transcribed text to digit."""
        text = text.strip().lower()
        
        # Remove common words
        text = text.replace("the", "").replace("number", "").replace("digit", "")
        text = text.strip()
        
        # Direct mapping
        if text in self.digit_map:
            return self.digit_map[text]
        
        # Word-by-word check
        for word in text.split():
            if word in self.digit_map:
                return self.digit_map[word]
        
        # Check for digits in text
        digits = [char for char in text if char.isdigit()]
        if digits:
            return digits[0]
        
        return text
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            'model_name': 'faster-whisper-tiny',
            'model_type': 'Speech-to-Text with VAD',
            'has_builtin_vad': True,
            'device': self.device,
            'available': FASTER_WHISPER_AVAILABLE
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        success_rate = self.successful_predictions / max(1, self.total_predictions)
        
        return {
            'total_predictions': self.total_predictions,
            'successful_predictions': self.successful_predictions,
            'failed_predictions': self.failed_predictions,
            'success_rate': round(success_rate, 3),
            'model_available': self.is_configured()
        }