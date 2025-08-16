"""
Whisper-based digit recognition processor
Specialized implementation for spoken digit recognition (0-9)
"""

import numpy as np
import io
import time
import logging
from typing import Dict, Any, Optional
import torch
from transformers import pipeline
import soundfile as sf

from .base_processor import AudioProcessor

logger = logging.getLogger(__name__)

class WhisperDigitProcessor(AudioProcessor):
    """
    Whisper-based digit recognition processor using Hugging Face transformers.
    Optimized for single digit recognition with mapping from text to numbers.
    """
    
    def __init__(self):
        """Initialize Whisper digit processor with optimized settings."""
        super().__init__("Whisper Digit Recognition")
        self.model = None
        self.device = 0 if torch.cuda.is_available() else -1
        
        # Digit mapping for text-to-number conversion
        self.digit_map = {
            "zero": "0", "one": "1", "two": "2", "three": "3", 
            "four": "4", "five": "5", "six": "6", "seven": "7", 
            "eight": "8", "nine": "9",
            # Common variations and alternatives
            "oh": "0", "o": "0",
            "for": "4", "fore": "4", "to": "2", "too": "2", "tu": "2",
            "tree": "3", "free": "3", "ate": "8", "ait": "8"
        }
        
        # Reverse mapping for validation
        self.number_words = set(self.digit_map.keys())
        
        # Statistics tracking
        self.total_predictions = 0
        self.successful_predictions = 0
        self.failed_predictions = 0
        self.average_inference_time = 0.0
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the Whisper model with optimal settings for digit recognition."""
        try:
            logger.info("Initializing Whisper model for digit recognition...")
            
            # Use Whisper tiny model for fast inference
            self.model = pipeline(
                "automatic-speech-recognition", 
                model="openai/whisper-tiny",
                device=self.device,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                return_timestamps=False  # We don't need timestamps for single digits
            )
            
            logger.info(f"Whisper model initialized successfully on device: {self.device}")
            
            # Test model with dummy input
            test_audio = np.random.randn(16000).astype(np.float32)  # 1 second of noise
            try:
                test_result = self.model(test_audio)
                logger.info("Model test successful")
            except Exception as e:
                logger.warning(f"Model test failed but model loaded: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Whisper model: {e}")
            return False
    
    def is_configured(self) -> bool:
        """Check if the processor is properly configured."""
        return self.model is not None
    
    def process_audio(self, audio_data: bytes) -> str:
        """
        Predict digit from audio data.
        
        Args:
            audio_data: Raw audio bytes (WAV format preferred)
            
        Returns:
            str: Predicted digit (0-9) or error message
        """
        if not self.is_configured():
            return "error: Model not configured"
        
        try:
            # Convert audio bytes to numpy array
            audio_array = self._convert_audio_to_array(audio_data)
            
            if audio_array is None:
                return "error: Invalid audio format"
            
            # Ensure proper sample rate and format
            audio_array = self._preprocess_audio(audio_array)
            
            # Run Whisper inference
            result = self.model(audio_array)
            text = result["text"].strip().lower()
            
            # Convert text to digit
            digit = self._text_to_digit(text)
            
            # Enhanced logging to debug transcription issues
            logger.info(f"🎤 Whisper transcription: '{text}' -> digit: '{digit}'")
            logger.info(f"📊 Audio stats: duration={len(audio_array)/16000:.2f}s, samples={len(audio_array)}, max_val={np.max(np.abs(audio_array)):.3f}")
            
            if digit in "0123456789":
                self.successful_predictions += 1
                return digit
            else:
                self.failed_predictions += 1
                return f"unclear: {text}"
                
        except Exception as e:
            logger.error(f"Whisper prediction failed: {e}")
            self.failed_predictions += 1
            return f"error: {str(e)}"
        finally:
            self.total_predictions += 1
    
    def _convert_audio_to_array(self, audio_data: bytes) -> Optional[np.ndarray]:
        """
        Convert audio bytes to numpy array.
        
        Args:
            audio_data: Raw audio bytes (could be WAV file or raw PCM from VAD)
            
        Returns:
            np.ndarray: Audio samples or None if conversion failed
        """
        # First check if this looks like raw PCM data from VAD (no file headers)
        if len(audio_data) < 100 or not audio_data.startswith(b'RIFF'):
            # This is likely raw PCM data from WebRTC VAD
            try:
                logger.debug("Processing raw PCM data from VAD segment")
                audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
                audio_array = audio_array / 32768.0  # Normalize to [-1, 1]
                self._original_sample_rate = 16000  # WebRTC VAD uses 16kHz
                return audio_array
            except Exception as e:
                logger.error(f"Failed to process raw PCM data: {e}")
                return None
        
        # This looks like a complete audio file (WAV, etc.)
        try:
            # Try to read as audio file using soundfile
            audio_buffer = io.BytesIO(audio_data)
            audio_array, sample_rate = sf.read(audio_buffer, dtype='float32')
            
            # Handle stereo to mono conversion
            if len(audio_array.shape) > 1:
                audio_array = np.mean(audio_array, axis=1)
            
            # Store original sample rate for resampling
            self._original_sample_rate = sample_rate
            
            logger.debug(f"Successfully loaded audio file: {len(audio_array)} samples at {sample_rate}Hz")
            return audio_array
            
        except Exception as e:
            logger.warning(f"Audio file conversion failed with soundfile: {e}")
            
            # Final fallback: treat as raw PCM
            try:
                logger.debug("Fallback: treating as raw PCM data")
                audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
                audio_array = audio_array / 32768.0  # Normalize to [-1, 1]
                self._original_sample_rate = 16000  # Assume 16kHz
                return audio_array
            except Exception as e2:
                logger.error(f"All audio conversion methods failed: {e2}")
                return None
    
    def _preprocess_audio(self, audio_array: np.ndarray) -> np.ndarray:
        """
        Preprocess audio for optimal Whisper performance.
        
        Args:
            audio_array: Raw audio samples
            
        Returns:
            np.ndarray: Preprocessed audio
        """
        # Resample to 16kHz if needed (Whisper's expected input)
        target_sample_rate = 16000
        
        if hasattr(self, '_original_sample_rate') and self._original_sample_rate != target_sample_rate:
            try:
                import librosa
                audio_array = librosa.resample(
                    audio_array, 
                    orig_sr=self._original_sample_rate, 
                    target_sr=target_sample_rate
                )
                logger.debug(f"Resampled audio from {self._original_sample_rate}Hz to {target_sample_rate}Hz")
            except ImportError:
                logger.warning("librosa not available for resampling, using original audio")
            except Exception as e:
                logger.warning(f"Resampling failed: {e}, using original audio")
        
        # Trim silence from edges
        audio_array = self._trim_silence(audio_array)
        
        # Ensure minimum length (Whisper works better with at least 0.1s)
        min_samples = int(0.1 * target_sample_rate)
        if len(audio_array) < min_samples:
            # Pad with silence
            padding = min_samples - len(audio_array)
            audio_array = np.pad(audio_array, (0, padding), mode='constant', constant_values=0)
        
        # Normalize audio
        max_val = np.max(np.abs(audio_array))
        if max_val > 0:
            audio_array = audio_array / max_val * 0.9  # Prevent clipping
        
        return audio_array
    
    def _trim_silence(self, audio_array: np.ndarray, silence_threshold: float = 0.01) -> np.ndarray:
        """
        Trim silence from beginning and end of audio.
        
        Args:
            audio_array: Audio samples
            silence_threshold: Threshold for silence detection
            
        Returns:
            np.ndarray: Trimmed audio
        """
        if len(audio_array) == 0:
            return audio_array
        
        # Find non-silent regions
        energy = audio_array ** 2
        non_silent = energy > silence_threshold
        
        if not np.any(non_silent):
            return audio_array  # All silence, return as is
        
        # Find first and last non-silent samples
        first_sound = np.argmax(non_silent)
        last_sound = len(non_silent) - np.argmax(non_silent[::-1]) - 1
        
        # Add small padding
        padding_samples = int(0.05 * 16000)  # 50ms padding
        first_sound = max(0, first_sound - padding_samples)
        last_sound = min(len(audio_array) - 1, last_sound + padding_samples)
        
        return audio_array[first_sound:last_sound + 1]
    
    def _text_to_digit(self, text: str) -> str:
        """
        Convert transcribed text to digit.
        
        Args:
            text: Transcribed text from Whisper
            
        Returns:
            str: Digit (0-9) or original text if no match
        """
        # Clean the text
        text = text.strip().lower()
        
        # Remove common punctuation and extra words
        text = text.replace(",", "").replace(".", "").replace("!", "").replace("?", "")
        text = text.replace("the", "").replace("number", "").replace("digit", "")
        text = text.strip()
        
        # Try direct mapping
        if text in self.digit_map:
            return self.digit_map[text]
        
        # Try word-by-word mapping for multi-word responses
        words = text.split()
        for word in words:
            if word in self.digit_map:
                return self.digit_map[word]
        
        # Check if it's already a digit
        if len(text) == 1 and text.isdigit():
            return text
        
        # Look for digits in the text
        digits_found = [char for char in text if char.isdigit()]
        if digits_found:
            return digits_found[0]  # Return first digit found
        
        # No clear digit found
        return text
    
    def predict_with_timing(self, audio_data: bytes) -> Dict[str, Any]:
        """
        Predict digit with detailed timing and confidence metrics.
        
        Args:
            audio_data: Raw audio bytes
            
        Returns:
            dict: Prediction results with timing and metadata
        """
        start_time = time.time()
        
        predicted_digit = self.process_audio(audio_data)
        
        inference_time = time.time() - start_time
        
        # Update average inference time
        if self.total_predictions > 0:
            self.average_inference_time = (
                (self.average_inference_time * (self.total_predictions - 1) + inference_time) 
                / self.total_predictions
            )
        
        # Determine success status
        is_successful = predicted_digit in "0123456789"
        confidence_score = 1.0 if is_successful else 0.0
        
        # Extract any error information
        error_info = None
        if predicted_digit.startswith("error:"):
            error_info = predicted_digit[6:].strip()
            predicted_digit = "unknown"
        elif predicted_digit.startswith("unclear:"):
            error_info = f"Transcription unclear: {predicted_digit[8:].strip()}"
            predicted_digit = "unknown"
        
        result = {
            'predicted_digit': predicted_digit,
            'confidence_score': confidence_score,
            'inference_time': round(inference_time, 4),
            'success': is_successful,
            'timestamp': time.time(),
            'model': 'openai/whisper-tiny',
            'method': 'whisper_digit'
        }
        
        if error_info:
            result['error'] = error_info
        
        return result
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            dict: Model information
        """
        return {
            'model_name': 'openai/whisper-tiny',
            'model_type': 'Speech-to-Text (ASR)',
            'specialized_for': 'Digit Recognition (0-9)',
            'device': 'GPU' if self.device >= 0 else 'CPU',
            'torch_device': self.device,
            'supports_streaming': False,
            'supported_languages': ['en'],
            'digit_mappings': len(self.digit_map)
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get processor statistics.
        
        Returns:
            dict: Performance statistics
        """
        success_rate = (
            self.successful_predictions / max(1, self.total_predictions)
        )
        
        return {
            'total_predictions': self.total_predictions,
            'successful_predictions': self.successful_predictions,
            'failed_predictions': self.failed_predictions,
            'success_rate': round(success_rate, 3),
            'average_inference_time': round(self.average_inference_time, 4),
            'model_loaded': self.is_configured()
        }
    
    def test_with_sample_audio(self) -> Dict[str, Any]:
        """
        Test the processor with generated sample audio.
        
        Returns:
            dict: Test results
        """
        if not self.is_configured():
            return {'error': 'Model not configured'}
        
        try:
            # Generate simple test audio (1 second of tone)
            sample_rate = 16000
            duration = 1.0
            frequency = 440  # A note
            
            t = np.linspace(0, duration, int(sample_rate * duration))
            test_audio = 0.3 * np.sin(2 * np.pi * frequency * t).astype(np.float32)
            
            # Run prediction
            start_time = time.time()
            result = self.model(test_audio)
            test_time = time.time() - start_time
            
            return {
                'test_successful': True,
                'test_time': round(test_time, 4),
                'transcription': result.get('text', 'No text'),
                'model_responsive': True
            }
            
        except Exception as e:
            return {
                'test_successful': False,
                'error': str(e),
                'model_responsive': False
            }