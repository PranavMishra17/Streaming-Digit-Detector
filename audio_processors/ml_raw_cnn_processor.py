"""
ML Raw CNN Digit Processor
Uses the trained Raw Waveform + 1D CNN model for digit classification
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union

import numpy as np
from .base_processor import AudioProcessor

# Add project root to path for ML imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

# Import ML inference
from ml_training.inference import load_classifier

logger = logging.getLogger(__name__)

class MLRawCNNProcessor(AudioProcessor):
    """
    ML-based Raw CNN digit processor using trained 1D CNN model.
    
    Performance characteristics (based on training results):
    - Test accuracy: 91.30%
    - Inference time: ~5-8ms
    - Model size: ~2.6MB
    """
    
    name = "ML Raw CNN (1D Conv)"
    
    def __init__(self, model_dir: str = "models", device: str = "auto"):
        """
        Initialize ML Raw CNN processor.
        
        Args:
            model_dir: Directory containing trained models
            device: Device to run inference on ('cpu', 'cuda', or 'auto')
        """
        super().__init__(self.name)
        
        self.model_dir = Path(model_dir)
        self.device = device if device != "auto" else None
        self.classifier = None
        self._configured = False
        
        # Performance tracking
        self.prediction_count = 0
        self.total_inference_time = 0.0
        self.last_prediction_time = None
        
        # Try to load the model
        self._initialize_classifier()
        
        logger.info(f"ML Raw CNN Processor initialized (configured: {self._configured})")
    
    def _initialize_classifier(self):
        """Initialize the ML classifier."""
        try:
            # Check if model directory exists
            if not self.model_dir.exists():
                logger.warning(f"Model directory not found: {self.model_dir}")
                return
            
            # Load the Raw CNN classifier
            self.classifier = load_classifier(
                model_dir=str(self.model_dir),
                pipeline_type='raw_cnn',
                device=self.device
            )
            
            self._configured = True
            logger.info("ML Raw CNN classifier loaded successfully")
            logger.info(f"  Model device: {self.classifier.device}")
            logger.info(f"  Parameters: {sum(p.numel() for p in self.classifier.model.parameters()):,}")
            
        except Exception as e:
            logger.error(f"Failed to load ML Raw CNN classifier: {str(e)}")
            self.classifier = None
            self._configured = False
    
    def is_configured(self) -> bool:
        """Check if the processor is properly configured."""
        return self._configured and self.classifier is not None
    
    def process_audio(self, audio_data: bytes) -> str:
        """
        Process audio and return predicted digit (required by base class).
        
        Args:
            audio_data: Raw audio data in bytes
            
        Returns:
            predicted_digit: Predicted digit as string
        """
        return self.predict(audio_data)
    
    def predict(self, audio_data: bytes) -> str:
        """
        Predict digit from audio data.
        
        Args:
            audio_data: Raw audio data in bytes
            
        Returns:
            predicted_digit: Predicted digit as string
        """
        if not self.is_configured():
            raise RuntimeError("ML Raw CNN processor not properly configured")
        
        try:
            # Convert audio with optimized format for ML models
            from utils.audio_utils import convert_for_ml_models
            optimized_audio = convert_for_ml_models(audio_data, 'raw_cnn')
            
            # Convert audio bytes to numpy array
            audio_array = self._bytes_to_audio_array(optimized_audio)
            
            # Make prediction using ML classifier
            start_time = time.time()
            result = self.classifier.predict(
                audio_array, 
                return_probabilities=True, 
                return_features=False
            )
            inference_time = time.time() - start_time
            
            # Update performance tracking
            self.prediction_count += 1
            self.total_inference_time += inference_time
            self.last_prediction_time = inference_time
            
            predicted_digit = str(result['predicted_digit'])
            confidence = result['confidence']
            
            logger.debug(f"ML Raw CNN prediction: '{predicted_digit}' "
                        f"(confidence: {confidence:.3f}, time: {inference_time*1000:.1f}ms)")
            
            return predicted_digit
            
        except Exception as e:
            logger.error(f"ML Raw CNN prediction failed: {str(e)}")
            raise
    
    def predict_with_timing(self, audio_data: bytes) -> Dict[str, Any]:
        """
        Predict digit with detailed timing and confidence information.
        
        Args:
            audio_data: Raw audio data in bytes
            
        Returns:
            result: Detailed prediction results
        """
        if not self.is_configured():
            return {
                'success': False,
                'error': 'ML Raw CNN processor not properly configured',
                'predicted_digit': None,
                'inference_time': 0.0
            }
        
        try:
            # Convert audio with optimized format for ML models
            from utils.audio_utils import convert_for_ml_models
            optimized_audio = convert_for_ml_models(audio_data, 'raw_cnn')
            
            # Convert audio bytes to numpy array
            audio_array = self._bytes_to_audio_array(optimized_audio)
            
            # Make prediction using ML classifier
            start_time = time.time()
            ml_result = self.classifier.predict(
                audio_array, 
                return_probabilities=True, 
                return_features=False
            )
            inference_time = time.time() - start_time
            
            # Update performance tracking
            self.prediction_count += 1
            self.total_inference_time += inference_time
            self.last_prediction_time = inference_time
            
            # Format result
            result = {
                'success': True,
                'predicted_digit': str(ml_result['predicted_digit']),
                'confidence': ml_result['confidence'],
                'inference_time': inference_time,
                'class_probabilities': {
                    str(k): float(v) for k, v in ml_result['class_probabilities'].items()
                },
                'top_3_predictions': [
                    {
                        'digit': str(pred['digit']),
                        'probability': pred['probability']
                    }
                    for pred in ml_result['top_3_predictions']
                ],
                'method': self.name,
                'model_type': 'ml_raw_cnn',
                'timestamp': time.time()
            }
            
            logger.debug(f"ML Raw CNN detailed prediction: '{result['predicted_digit']}' "
                        f"(confidence: {result['confidence']:.3f}, "
                        f"time: {inference_time*1000:.1f}ms)")
            
            return result
            
        except Exception as e:
            logger.error(f"ML Raw CNN prediction with timing failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'predicted_digit': None,
                'inference_time': 0.0,
                'method': self.name,
                'model_type': 'ml_raw_cnn',
                'timestamp': time.time()
            }
    
    def _bytes_to_audio_array(self, audio_data: bytes) -> np.ndarray:
        """Convert audio bytes to numpy array."""
        try:
            # Try to interpret as int16 PCM first (most common)
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # Convert to float32 and normalize
            audio_array = audio_array.astype(np.float32) / 32768.0
            
            # If the array is too short, pad it
            if len(audio_array) < 1000:  # Less than ~60ms at 16kHz
                # Pad with zeros to minimum length
                audio_array = np.pad(audio_array, (0, 1000 - len(audio_array)))
            
            return audio_array
            
        except Exception as e:
            logger.error(f"Failed to convert audio bytes to array: {str(e)}")
            # Return a small zero array as fallback
            return np.zeros(1000, dtype=np.float32)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processor performance statistics."""
        stats = super().get_stats()
        
        if self.prediction_count > 0:
            stats.update({
                'ml_predictions': self.prediction_count,
                'average_inference_time': self.total_inference_time / self.prediction_count,
                'last_inference_time': self.last_prediction_time,
                'throughput_per_second': self.prediction_count / self.total_inference_time if self.total_inference_time > 0 else 0,
                'model_configured': self.is_configured()
            })
        
        if self.classifier:
            # Get ML classifier performance stats
            ml_stats = self.classifier.get_performance_stats()
            stats['ml_classifier_stats'] = ml_stats
        
        return stats
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if not self.is_configured():
            return {'error': 'Model not loaded'}
        
        try:
            info = {
                'pipeline_type': 'raw_cnn',
                'model_class': self.classifier.model.__class__.__name__,
                'device': str(self.classifier.device),
                'parameters': sum(p.numel() for p in self.classifier.model.parameters()),
                'feature_extractor': None,  # Raw waveforms don't need feature extraction
                'has_scaler': False,
                'expected_sample_rate': 8000,
                'expected_audio_length': 8000,  # 1 second at 8kHz
                'input_shape': '(1, 1, 8000)',  # Raw waveform shape
                'model_architecture': '1D CNN'
            }
            
            if hasattr(self.classifier, 'model_path'):
                info['model_path'] = str(self.classifier.model_path)
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to get model info: {str(e)}")
            return {'error': str(e)}
    
    def benchmark_speed(self, num_samples: int = 100) -> Dict[str, Any]:
        """Benchmark inference speed."""
        if not self.is_configured():
            return {'error': 'Model not configured'}
        
        try:
            return self.classifier.benchmark_speed(num_samples)
        except Exception as e:
            logger.error(f"Benchmark failed: {str(e)}")
            return {'error': str(e)}