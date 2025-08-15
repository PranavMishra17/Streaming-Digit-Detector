import logging
import numpy as np
from typing import Optional
from .base_processor import AudioProcessor

logger = logging.getLogger(__name__)

class LocalWhisperProcessor(AudioProcessor):
    """
    Local Whisper model using transformers pipeline.
    Fallback when API is unavailable.
    """
    
    def __init__(self):
        super().__init__("Local Whisper (Tiny)")
        self.pipeline = None
        self.model_name = "openai/whisper-tiny"
        self.is_initialized = False
        
    def _initialize_model(self):
        """Lazy initialization of the model"""
        if self.is_initialized:
            return
            
        try:
            logger.info(f"Loading local Whisper model: {self.model_name}")
            
            from transformers import pipeline
            import torch
            
            # Use CPU for compatibility, GPU if available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            self.pipeline = pipeline(
                "automatic-speech-recognition",
                model=self.model_name,
                device=device,
                torch_dtype=torch.float32,  # Use float32 to avoid dtype issues
                return_timestamps=False  # We only need text
            )
            
            logger.info(f"Local Whisper model loaded on {device}")
            self.is_initialized = True
            
        except ImportError as e:
            logger.error("transformers library not installed. Run: pip install transformers torch")
            raise Exception("transformers library required for local processing")
        except Exception as e:
            logger.error(f"Failed to load local Whisper model: {str(e)}")
            raise Exception(f"Local model initialization failed: {str(e)}")
    
    def process_audio(self, audio_data: bytes) -> str:
        """
        Process audio using local Whisper model.
        
        Args:
            audio_data: Raw audio bytes (WAV format preferred)
            
        Returns:
            Predicted digit as string ('0'-'9')
            
        Raises:
            Exception: If processing fails
        """
        try:
            # Initialize model on first use
            self._initialize_model()
            
            # Convert audio bytes to numpy array
            from utils.audio_utils import audio_to_numpy
            audio_array, sample_rate = audio_to_numpy(audio_data)
            
            # Resample to 16kHz if needed (Whisper expects 16kHz)
            if sample_rate != 16000:
                logger.debug(f"Resampling from {sample_rate}Hz to 16kHz")
                import librosa
                audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)
            
            # Process with pipeline
            logger.debug(f"Processing audio: {len(audio_array)} samples at 16kHz")
            result = self.pipeline(audio_array)
            
            if not result or 'text' not in result:
                logger.error(f"Unexpected pipeline result: {result}")
                raise Exception("Invalid pipeline output")
            
            transcribed_text = result['text'].strip().lower()
            logger.debug(f"Local Whisper transcription: '{transcribed_text}'")
            
            # Extract digit from transcription
            predicted_digit = self._extract_digit(transcribed_text)
            
            if predicted_digit is None:
                logger.warning(f"No digit found in transcription: '{transcribed_text}'")
                return "?"
            
            return predicted_digit
            
        except Exception as e:
            logger.error(f"Local Whisper processing failed: {str(e)}")
            raise Exception(f"Local processing error: {str(e)}")
    
    def _extract_digit(self, text: str) -> Optional[str]:
        """
        Extract digit from transcribed text.
        Handles both numerical ('1', '2') and word forms ('one', 'two').
        """
        import re
        
        # Word to digit mapping
        word_to_digit = {
            'zero': '0', 'oh': '0',
            'one': '1', 'won': '1',
            'two': '2', 'to': '2', 'too': '2',
            'three': '3', 'tree': '3',
            'four': '4', 'for': '4', 'fore': '4',
            'five': '5',
            'six': '6', 'sick': '6',
            'seven': '7',
            'eight': '8', 'ate': '8',
            'nine': '9', 'niner': '9'
        }
        
        # First, try to find a direct digit
        digit_match = re.search(r'\b([0-9])\b', text)
        if digit_match:
            return digit_match.group(1)
        
        # Then try word forms
        words = text.split()
        for word in words:
            clean_word = re.sub(r'[^\w]', '', word.lower())
            if clean_word in word_to_digit:
                return word_to_digit[clean_word]
        
        # Try partial matches for robustness
        for word, digit in word_to_digit.items():
            if word in text:
                return digit
        
        return None
    
    def is_configured(self) -> bool:
        """Check if local model can be initialized."""
        try:
            import transformers
            import torch
            return True
        except ImportError:
            return False
    
    def test_connection(self) -> bool:
        """Test local model functionality."""
        try:
            self._initialize_model()
            return True
        except:
            return False