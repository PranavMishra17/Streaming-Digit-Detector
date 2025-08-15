from abc import ABC, abstractmethod
from typing import Union, Dict, Any
import time
import logging

logger = logging.getLogger(__name__)

class AudioProcessor(ABC):
    """
    Abstract base class for all audio digit classification processors.
    Provides common interface and logging functionality.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.total_predictions = 0
        self.total_inference_time = 0.0
    
    @abstractmethod
    def process_audio(self, audio_data: bytes) -> str:
        """
        Process audio data and return predicted digit as string.
        
        Args:
            audio_data: Raw audio bytes
            
        Returns:
            Predicted digit as string ('0'-'9')
        """
        pass
    
    def predict_with_timing(self, audio_data: bytes) -> Dict[str, Any]:
        """
        Process audio and return prediction with timing information.
        
        Args:
            audio_data: Raw audio bytes
            
        Returns:
            Dictionary with prediction, timing, and method info
        """
        start_time = time.time()
        
        try:
            predicted_digit = self.process_audio(audio_data)
            inference_time = time.time() - start_time
            
            self.total_predictions += 1
            self.total_inference_time += inference_time
            
            result = {
                'predicted_digit': predicted_digit,
                'inference_time': round(inference_time, 3),
                'method': self.name,
                'timestamp': time.time(),
                'average_time': round(self.total_inference_time / self.total_predictions, 3),
                'success': True
            }
            
            logger.info(f"{self.name}: Predicted '{predicted_digit}' in {inference_time:.3f}s")
            return result
            
        except Exception as e:
            inference_time = time.time() - start_time
            logger.error(f"{self.name}: Error processing audio: {str(e)}")
            
            return {
                'predicted_digit': 'ERROR',
                'inference_time': round(inference_time, 3),
                'method': self.name,
                'timestamp': time.time(),
                'success': False,
                'error': str(e)
            }
    
    def get_stats(self) -> Dict[str, float]:
        """Get processor statistics."""
        if self.total_predictions == 0:
            return {'total_predictions': 0, 'average_time': 0.0}
        
        return {
            'total_predictions': self.total_predictions,
            'total_time': round(self.total_inference_time, 3),
            'average_time': round(self.total_inference_time / self.total_predictions, 3)
        }