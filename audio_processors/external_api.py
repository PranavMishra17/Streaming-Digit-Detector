import requests
import os
import re
import logging
from typing import Optional
from .base_processor import AudioProcessor

logger = logging.getLogger(__name__)

class ExternalAPIProcessor(AudioProcessor):
    """
    Hugging Face Whisper API integration for digit classification.
    Uses openai/whisper-base model for speech-to-text conversion.
    """
    
    def __init__(self):
        super().__init__("External API (Whisper)")
        self.api_url = "https://api-inference.huggingface.co/models/openai/whisper-base"
        self.token = os.getenv('HUGGING_FACE_TOKEN')
        self.headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        
        if not self.token:
            logger.warning("HUGGING_FACE_TOKEN not found in environment variables")
    
    def process_audio(self, audio_data: bytes) -> str:
        """
        Process audio using Hugging Face Whisper API.
        
        Args:
            audio_data: Raw audio bytes (WAV format preferred)
            
        Returns:
            Predicted digit as string ('0'-'9')
            
        Raises:
            Exception: If API call fails or no digit found in response
        """
        if not self.token:
            raise Exception("Hugging Face API token not configured")
        
        try:
            # Make API request
            response = requests.post(
                self.api_url,
                headers=self.headers,
                data=audio_data,
                timeout=15  # Increased timeout
            )
            
            if response.status_code == 401:
                logger.error("Hugging Face API token is invalid or expired")
                raise Exception("Invalid or expired API token - please update HUGGING_FACE_TOKEN")
            elif response.status_code == 404:
                logger.error(f"Model not found or unavailable: {self.api_url}")
                raise Exception("API model unavailable - may be loading or deprecated")
            elif response.status_code == 503:
                logger.warning("Model is loading, this may take a few moments")
                raise Exception("API model is loading - please try again in a moment")
            elif response.status_code != 200:
                logger.error(f"API request failed: {response.status_code} - {response.text}")
                raise Exception(f"API error {response.status_code}: {response.text[:100]}")
            
            # Parse response
            result = response.json()
            
            if 'text' not in result:
                logger.error(f"Unexpected API response format: {result}")
                raise Exception("Invalid API response format")
            
            transcribed_text = result['text'].strip().lower()
            logger.debug(f"Whisper transcription: '{transcribed_text}'")
            
            # Extract digit from transcription
            predicted_digit = self._extract_digit(transcribed_text)
            
            if predicted_digit is None:
                logger.warning(f"No digit found in transcription: '{transcribed_text}'")
                return "?"
            
            return predicted_digit
            
        except requests.exceptions.Timeout:
            raise Exception("API request timeout (15s) - service may be slow")
        except requests.exceptions.RequestException as e:
            raise Exception(f"API request failed: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error in external API processing: {str(e)}")
            raise
    
    def _extract_digit(self, text: str) -> Optional[str]:
        """
        Extract digit from transcribed text.
        Handles both numerical ('1', '2') and word forms ('one', 'two').
        
        Args:
            text: Transcribed text from Whisper
            
        Returns:
            Digit as string ('0'-'9') or None if not found
        """
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
        """Check if API is properly configured."""
        return bool(self.token)
    
    def test_connection(self) -> bool:
        """Test API connection with a simple request."""
        if not self.is_configured():
            return False
        
        try:
            # Test with minimal audio data
            test_response = requests.get(
                self.api_url,
                headers=self.headers,
                timeout=5
            )
            return test_response.status_code == 200
        except:
            return False