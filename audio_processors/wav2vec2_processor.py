import logging
import numpy as np
from typing import Optional
from .base_processor import AudioProcessor

logger = logging.getLogger(__name__)

class Wav2Vec2Processor(AudioProcessor):
    """
    Wav2Vec2 model processor for speech recognition.
    Lightweight alternative to Whisper.
    """
    
    def __init__(self):
        super().__init__("Wav2Vec2 (Facebook)")
        self.processor = None
        self.model = None
        self.model_name = "facebook/wav2vec2-base-960h"
        self.is_initialized = False
        
    def _initialize_model(self):
        """Lazy initialization of the model"""
        if self.is_initialized:
            return
            
        try:
            logger.info(f"Loading Wav2Vec2 model: {self.model_name}")
            
            from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
            import torch
            
            # Load processor and model
            self.processor = Wav2Vec2Processor.from_pretrained(self.model_name)
            self.model = Wav2Vec2ForCTC.from_pretrained(self.model_name)
            
            # Move to GPU if available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = self.model.to(device)
            self.device = device
            
            logger.info(f"Wav2Vec2 model loaded on {device}")
            self.is_initialized = True
            
        except ImportError as e:
            logger.error("transformers library not installed. Run: pip install transformers torch")
            raise Exception("transformers library required for Wav2Vec2 processing")
        except Exception as e:
            logger.error(f"Failed to load Wav2Vec2 model: {str(e)}")
            raise Exception(f"Wav2Vec2 model initialization failed: {str(e)}")
    
    def process_audio(self, audio_data: bytes) -> str:
        """
        Process audio using Wav2Vec2 model.
        
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
            
            # Resample to 16kHz if needed (Wav2Vec2 expects 16kHz)
            if sample_rate != 16000:
                logger.debug(f"Resampling from {sample_rate}Hz to 16kHz")
                import librosa
                audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)
            
            logger.debug(f"Processing audio: {len(audio_array)} samples at 16kHz")
            
            # Process with Wav2Vec2
            import torch
            
            # Tokenize audio
            input_values = self.processor(
                audio_array, 
                return_tensors="pt", 
                padding="longest",
                sampling_rate=16000
            ).input_values.to(self.device)
            
            # Get logits
            with torch.no_grad():
                logits = self.model(input_values).logits
            
            # Get predicted tokens
            predicted_ids = torch.argmax(logits, dim=-1)
            
            # Decode transcription
            transcription = self.processor.batch_decode(predicted_ids)[0].lower().strip()
            logger.debug(f"Wav2Vec2 transcription: '{transcription}'")
            
            # Extract digit from transcription
            predicted_digit = self._extract_digit(transcription)
            
            if predicted_digit is None:
                logger.warning(f"No digit found in transcription: '{transcription}'")
                return "?"
            
            return predicted_digit
            
        except Exception as e:
            logger.error(f"Wav2Vec2 processing failed: {str(e)}")
            raise Exception(f"Wav2Vec2 processing error: {str(e)}")
    
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
        """Check if Wav2Vec2 model can be initialized."""
        try:
            import transformers
            import torch
            return True
        except ImportError:
            return False
    
    def test_connection(self) -> bool:
        """Test Wav2Vec2 model functionality."""
        try:
            self._initialize_model()
            return True
        except:
            return False