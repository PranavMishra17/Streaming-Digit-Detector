"""
Inference Module for Trained Digit Classification Models
Supports loading and using all three trained pipelines for predictions
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
import warnings

import torch
import torch.nn.functional as F
import numpy as np
import librosa
import soundfile as sf
from sklearn.preprocessing import StandardScaler
import pickle

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

# Import our modules
from ml_training.pipelines.mfcc_pipeline import MFCCClassifier, MFCCFeatureExtractor
from ml_training.pipelines.mel_cnn_pipeline import MelSpectrogramCNN, MelSpectrogramExtractor
from ml_training.pipelines.raw_cnn_pipeline import RawWaveformCNN
from ml_training.data.dataset_loader import DigitDatasetLoader

logger = logging.getLogger(__name__)

class DigitClassifier:
    """
    Universal digit classifier that can load and use any of the three trained pipelines.
    Provides a unified interface for inference with comprehensive error handling.
    """
    
    def __init__(self, model_path: str, pipeline_type: str, 
                 scaler_path: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize digit classifier.
        
        Args:
            model_path: Path to trained model checkpoint
            pipeline_type: Type of pipeline ('mfcc', 'mel_cnn', 'raw_cnn')
            scaler_path: Path to feature scaler (for MFCC pipeline)
            device: Device to run inference on ('cpu', 'cuda', or None for auto)
        """
        self.pipeline_type = pipeline_type.lower()
        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path) if scaler_path else None
        
        # Setup device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Initialize components
        self.model = None
        self.feature_extractor = None
        self.scaler = None
        self.data_loader = DigitDatasetLoader()
        
        # Class names
        self.class_names = [str(i) for i in range(10)]
        
        # Performance tracking
        self.inference_times = []
        self.prediction_count = 0
        
        # Load model and components
        self._load_model()
        
        logger.info(f"Digit Classifier initialized:")
        logger.info(f"  Pipeline: {self.pipeline_type}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Model: {self.model.__class__.__name__}")
        logger.info(f"  Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def _load_model(self) -> None:
        """Load model and associated components based on pipeline type."""
        try:
            # Load checkpoint
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            logger.info(f"Loading model from {self.model_path}")
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Initialize model based on pipeline type
            if self.pipeline_type == 'mfcc':
                self._setup_mfcc_pipeline(checkpoint)
            elif self.pipeline_type == 'mel_cnn':
                self._setup_mel_cnn_pipeline(checkpoint)
            elif self.pipeline_type == 'raw_cnn':
                self._setup_raw_cnn_pipeline(checkpoint)
            else:
                raise ValueError(f"Unknown pipeline type: {self.pipeline_type}")
            
            # Load model state
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def _setup_mfcc_pipeline(self, checkpoint: Dict[str, Any]) -> None:
        """Setup MFCC pipeline components."""
        # Initialize feature extractor
        self.feature_extractor = MFCCFeatureExtractor()
        
        # Initialize model with correct input dimension
        input_dim = self.feature_extractor.expected_features
        self.model = MFCCClassifier(input_dim=input_dim)
        
        # Load scaler if provided
        if self.scaler_path and self.scaler_path.exists():
            logger.info(f"Loading scaler from {self.scaler_path}")
            with open(self.scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
        else:
            logger.warning("No scaler provided for MFCC pipeline - using identity scaling")
            self.scaler = None
    
    def _setup_mel_cnn_pipeline(self, checkpoint: Dict[str, Any]) -> None:
        """Setup Mel CNN pipeline components."""
        # Initialize feature extractor
        self.feature_extractor = MelSpectrogramExtractor()
        
        # Initialize model
        self.model = MelSpectrogramCNN(input_shape=(1, 64, 51))
    
    def _setup_raw_cnn_pipeline(self, checkpoint: Dict[str, Any]) -> None:
        """Setup Raw CNN pipeline components."""
        # No feature extractor needed for raw waveforms
        self.feature_extractor = None
        
        # Initialize model
        self.model = RawWaveformCNN(input_length=8000)
    
    def predict(self, audio_input: Union[str, np.ndarray], 
                return_probabilities: bool = True, 
                return_features: bool = False) -> Dict[str, Any]:
        """
        Make prediction on audio input.
        
        Args:
            audio_input: Either path to audio file or numpy array
            return_probabilities: Whether to return class probabilities
            return_features: Whether to return extracted features
            
        Returns:
            results: Dictionary containing prediction results
        """
        start_time = time.time()
        
        try:
            # Load and preprocess audio
            audio_array = self._load_and_preprocess_audio(audio_input)
            
            # Extract features and prepare input
            model_input = self._prepare_model_input(audio_array)
            
            # Make prediction
            with torch.no_grad():
                logits = self.model(model_input)
                probabilities = F.softmax(logits, dim=1)
                predicted_class = torch.argmax(logits, dim=1).item()
                confidence = probabilities.max().item()
            
            # Prepare results
            results = {
                'predicted_digit': predicted_class,
                'predicted_class_name': self.class_names[predicted_class],
                'confidence': confidence,
                'inference_time': time.time() - start_time
            }
            
            if return_probabilities:
                probs_dict = {
                    self.class_names[i]: prob.item() 
                    for i, prob in enumerate(probabilities.squeeze())
                }
                results['class_probabilities'] = probs_dict
                results['top_3_predictions'] = self._get_top_k_predictions(probabilities, k=3)
            
            if return_features and hasattr(self, 'extracted_features'):
                results['features'] = self.extracted_features
            
            # Update performance tracking
            self.inference_times.append(results['inference_time'])
            self.prediction_count += 1
            
            logger.debug(f"Prediction: {predicted_class} (confidence: {confidence:.3f})")
            
            return results
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise
    
    def _load_and_preprocess_audio(self, audio_input: Union[str, np.ndarray]) -> np.ndarray:
        """Load and preprocess audio input."""
        if isinstance(audio_input, str):
            # Load from file
            if not Path(audio_input).exists():
                raise FileNotFoundError(f"Audio file not found: {audio_input}")
            
            try:
                audio, sr = librosa.load(audio_input, sr=None)
                logger.debug(f"Loaded audio: shape={audio.shape}, sr={sr}")
            except Exception as e:
                logger.error(f"Failed to load audio file: {str(e)}")
                raise
        else:
            # Use provided array
            audio = np.array(audio_input, dtype=np.float32)
            sr = 8000  # Assume default sample rate
            logger.debug(f"Using provided audio: shape={audio.shape}")
        
        # Preprocess using data loader
        processed_audio = self.data_loader.preprocess_audio(audio, sr)
        
        return processed_audio
    
    def _prepare_model_input(self, audio_array: np.ndarray) -> torch.Tensor:
        """Prepare input tensor for the specific model type."""
        if self.pipeline_type == 'mfcc':
            # Extract MFCC features
            features = self.feature_extractor.extract_features(audio_array)
            self.extracted_features = features.copy()
            
            # Apply scaling if available
            if self.scaler is not None:
                features = self.scaler.transform([features])[0]
            
            # Convert to tensor
            input_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            
        elif self.pipeline_type == 'mel_cnn':
            # Extract mel spectrogram
            mel_spec = self.feature_extractor.extract_features(audio_array)
            self.extracted_features = mel_spec.copy()
            
            # Add batch and channel dimensions
            input_tensor = torch.FloatTensor(mel_spec).unsqueeze(0).unsqueeze(0).to(self.device)
            
        elif self.pipeline_type == 'raw_cnn':
            # Use raw waveform
            self.extracted_features = audio_array.copy()
            
            # Add batch and channel dimensions
            input_tensor = torch.FloatTensor(audio_array).unsqueeze(0).unsqueeze(0).to(self.device)
        
        else:
            raise ValueError(f"Unknown pipeline type: {self.pipeline_type}")
        
        return input_tensor
    
    def _get_top_k_predictions(self, probabilities: torch.Tensor, k: int = 3) -> List[Dict[str, Any]]:
        """Get top-k predictions with probabilities."""
        probs = probabilities.squeeze().cpu().numpy()
        top_indices = np.argsort(probs)[-k:][::-1]  # Get top k indices in descending order
        
        top_predictions = []
        for idx in top_indices:
            top_predictions.append({
                'digit': int(idx),
                'class_name': self.class_names[idx],
                'probability': float(probs[idx])
            })
        
        return top_predictions
    
    def batch_predict(self, audio_inputs: List[Union[str, np.ndarray]], 
                     batch_size: int = 32) -> List[Dict[str, Any]]:
        """
        Make predictions on a batch of audio inputs.
        
        Args:
            audio_inputs: List of audio files or arrays
            batch_size: Processing batch size
            
        Returns:
            results: List of prediction results
        """
        all_results = []
        
        logger.info(f"Processing {len(audio_inputs)} audio samples in batches of {batch_size}")
        
        for i in range(0, len(audio_inputs), batch_size):
            batch_inputs = audio_inputs[i:i+batch_size]
            batch_results = []
            
            for audio_input in batch_inputs:
                try:
                    result = self.predict(audio_input, return_probabilities=True, return_features=False)
                    batch_results.append(result)
                except Exception as e:
                    logger.error(f"Failed to process sample {i + len(batch_results)}: {str(e)}")
                    # Add error result
                    batch_results.append({
                        'predicted_digit': -1,
                        'confidence': 0.0,
                        'error': str(e)
                    })
            
            all_results.extend(batch_results)
            
            if (i // batch_size + 1) % 10 == 0:
                logger.info(f"Processed {i + len(batch_inputs)}/{len(audio_inputs)} samples")
        
        return all_results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get inference performance statistics."""
        if not self.inference_times:
            return {'message': 'No predictions made yet'}
        
        inference_times = np.array(self.inference_times)
        
        stats = {
            'total_predictions': self.prediction_count,
            'average_inference_time': np.mean(inference_times),
            'median_inference_time': np.median(inference_times),
            'std_inference_time': np.std(inference_times),
            'min_inference_time': np.min(inference_times),
            'max_inference_time': np.max(inference_times),
            'throughput_per_second': 1.0 / np.mean(inference_times) if np.mean(inference_times) > 0 else 0
        }
        
        return stats
    
    def benchmark_speed(self, num_samples: int = 100, audio_length: int = 8000) -> Dict[str, Any]:
        """
        Benchmark inference speed with synthetic audio.
        
        Args:
            num_samples: Number of samples to test
            audio_length: Length of synthetic audio
            
        Returns:
            benchmark_results: Speed benchmark results
        """
        logger.info(f"Benchmarking inference speed with {num_samples} synthetic samples")
        
        # Generate synthetic audio samples
        synthetic_audio = []
        for _ in range(num_samples):
            # Generate random noise audio
            audio = np.random.randn(audio_length).astype(np.float32) * 0.1
            synthetic_audio.append(audio)
        
        # Warm up
        for _ in range(10):
            self.predict(synthetic_audio[0], return_probabilities=False)
        
        # Reset timing
        self.inference_times = []
        
        # Benchmark
        start_time = time.time()
        for audio in synthetic_audio:
            self.predict(audio, return_probabilities=False)
        
        total_time = time.time() - start_time
        
        # Calculate statistics
        benchmark_results = {
            'total_samples': num_samples,
            'total_time': total_time,
            'average_time_per_sample': total_time / num_samples,
            'throughput_per_second': num_samples / total_time,
            'performance_stats': self.get_performance_stats()
        }
        
        logger.info(f"Benchmark completed:")
        logger.info(f"  Average time per sample: {benchmark_results['average_time_per_sample']*1000:.2f} ms")
        logger.info(f"  Throughput: {benchmark_results['throughput_per_second']:.1f} samples/second")
        
        return benchmark_results

def load_classifier(model_dir: str, pipeline_type: str, device: str = None) -> DigitClassifier:
    """
    Convenience function to load a trained classifier.
    
    Args:
        model_dir: Directory containing model files
        pipeline_type: Type of pipeline to load
        device: Device to run on
        
    Returns:
        classifier: Loaded digit classifier
    """
    model_dir = Path(model_dir)
    
    # Find model file
    model_path = model_dir / f"{pipeline_type}_classifier" / "best_model.pt"
    if not model_path.exists():
        # Try alternative naming
        model_path = model_dir / "best_model.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"Could not find model file in {model_dir}")
    
    # Find scaler for MFCC pipeline
    scaler_path = None
    if pipeline_type == 'mfcc':
        scaler_path = model_dir / f"{pipeline_type}_classifier" / "scaler.pkl"
        if not scaler_path.exists():
            scaler_path = model_dir / "scaler.pkl"
            if not scaler_path.exists():
                logger.warning("No scaler found for MFCC pipeline")
                scaler_path = None
    
    return DigitClassifier(
        model_path=str(model_path),
        pipeline_type=pipeline_type,
        scaler_path=str(scaler_path) if scaler_path else None,
        device=device
    )

def main():
    """Demo script for inference module."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test digit classifier inference')
    parser.add_argument('--model_dir', type=str, required=True,
                       help='Directory containing trained model')
    parser.add_argument('--pipeline', type=str, required=True,
                       choices=['mfcc', 'mel_cnn', 'raw_cnn'],
                       help='Pipeline type')
    parser.add_argument('--audio', type=str, help='Path to audio file to test')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run speed benchmark')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Load classifier
        logger.info(f"Loading {args.pipeline} classifier from {args.model_dir}")
        classifier = load_classifier(args.model_dir, args.pipeline)
        
        if args.audio:
            # Test on audio file
            logger.info(f"Testing on audio file: {args.audio}")
            result = classifier.predict(args.audio, return_probabilities=True)
            
            print(f"Prediction: {result['predicted_digit']}")
            print(f"Confidence: {result['confidence']:.3f}")
            print(f"Inference time: {result['inference_time']*1000:.1f} ms")
            
            print("\\nTop 3 predictions:")
            for i, pred in enumerate(result['top_3_predictions']):
                print(f"  {i+1}. Digit {pred['digit']}: {pred['probability']:.3f}")
        
        if args.benchmark:
            # Run benchmark
            benchmark_results = classifier.benchmark_speed(num_samples=100)
            
            print(f"\\nBenchmark Results:")
            print(f"  Samples processed: {benchmark_results['total_samples']}")
            print(f"  Total time: {benchmark_results['total_time']:.2f} seconds")
            print(f"  Average per sample: {benchmark_results['average_time_per_sample']*1000:.2f} ms")
            print(f"  Throughput: {benchmark_results['throughput_per_second']:.1f} samples/second")
    
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()