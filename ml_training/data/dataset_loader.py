"""
Dataset Loading and Preprocessing for Digit Classification
Handles Free Spoken Digit Dataset (FSDD) loading and preprocessing
"""

import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from pathlib import Path
import warnings
import logging
import subprocess
import tempfile
import os
from typing import Dict, Tuple, Optional, List, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datasets import load_dataset
import torch

# Setup logging
logger = logging.getLogger(__name__)

class DigitDatasetLoader:
    """
    Comprehensive dataset loader for spoken digit classification.
    
    Data Flow:
    Raw Audio (WAV) -> Preprocessing -> Normalized Arrays
    Input: Variable length audio files (8kHz)
    Output: Fixed length arrays (8000 samples = 1 second @ 8kHz)
    """
    
    def __init__(self, sample_rate: int = 8000, max_length: int = 8000, 
                 min_length: int = 1000):
        """
        Initialize dataset loader.
        
        Args:
            sample_rate: Target sampling rate (Hz)
            max_length: Maximum audio length in samples
            min_length: Minimum audio length in samples (for validation)
        
        Data Dimensions:
            - Raw audio: (batch_size, max_length) = (N, 8000)
            - Labels: (batch_size,) = (N,)
        """
        self.sample_rate = sample_rate
        self.max_length = max_length
        self.min_length = min_length
        self.label_encoder = LabelEncoder()
        
        # Check ffmpeg availability for better audio processing
        self._ffmpeg_available = self._check_ffmpeg_available()
        if self._ffmpeg_available:
            logger.info("ffmpeg detected - will use for high-quality audio resampling")
        else:
            logger.info("ffmpeg not available - using librosa for resampling")
        
        logger.info(f"Initialized DataLoader - SR: {sample_rate}Hz, Max Length: {max_length} samples")
    
    def load_fsdd_dataset(self) -> Optional[Any]:
        """
        Load Free Spoken Digit Dataset from HuggingFace.
        
        Returns:
            dataset: HuggingFace dataset object or None if failed
            
        Data Structure:
            - 'audio': {'array': np.ndarray, 'sampling_rate': int}
            - 'label': int (0-9)
            - Total samples: ~3000
            - Speakers: 6 different
        """
        try:
            logger.info("Loading Free Spoken Digit Dataset from HuggingFace...")
            
            # Load the correct HuggingFace dataset
            dataset = load_dataset("mteb/free-spoken-digit-dataset", trust_remote_code=True)
            
            logger.info(f"Dataset loaded successfully")
            logger.info(f"Available splits: {list(dataset.keys())}")
            
            # Check which split to use
            if 'train' in dataset:
                split_size = len(dataset['train'])
                logger.info(f"Train split size: {split_size}")
            elif 'test' in dataset:
                split_size = len(dataset['test'])
                logger.info(f"Test split size: {split_size}")
            else:
                # Use first available split
                first_split = list(dataset.keys())[0]
                split_size = len(dataset[first_split])
                logger.info(f"Using '{first_split}' split with {split_size} samples")
            
            # Validate dataset structure
            sample = dataset[list(dataset.keys())[0]][0]
            logger.info(f"Dataset sample structure: {sample.keys()}")
            
            if 'audio' in sample:
                audio_info = sample['audio']
                logger.info(f"Audio info: sampling_rate={audio_info.get('sampling_rate', 'N/A')}, "
                          f"array_shape={audio_info['array'].shape if 'array' in audio_info else 'N/A'}")
            
            if 'label' in sample:
                logger.info(f"Label type: {type(sample['label'])}, value: {sample['label']}")
            
            return dataset
            
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            logger.info("Attempting fallback dataset loading...")
            
            try:
                # Fallback to manual loading
                return self._load_fsdd_fallback()
            except Exception as fallback_error:
                logger.error(f"Fallback loading failed: {str(fallback_error)}")
                return None
    
    def _load_fsdd_fallback(self):
        """
        Fallback dataset loading method with multiple strategies.
        
        Returns:
            Synthetic dataset or None if all methods fail
        """
        logger.warning("Using fallback dataset loading - attempting alternative methods")
        
        # Strategy 1: Try alternative HuggingFace dataset names
        alternative_datasets = [
            "free-spoken-digit-dataset",
            "speech_commands",  # Has similar digit data
        ]
        
        for alt_dataset in alternative_datasets:
            try:
                logger.info(f"Trying alternative dataset: {alt_dataset}")
                dataset = load_dataset(alt_dataset, trust_remote_code=True)
                logger.info(f"Successfully loaded alternative dataset: {alt_dataset}")
                return dataset
            except Exception as e:
                logger.debug(f"Alternative dataset {alt_dataset} failed: {e}")
                continue
        
        # Strategy 2: Create synthetic dataset for development/testing
        logger.info("Creating synthetic dataset for development/testing")
        return self._create_synthetic_dataset()
    
    def _create_synthetic_dataset(self):
        """
        Create a synthetic dataset with digit-like audio patterns.
        
        Returns:
            Synthetic dataset in HuggingFace format
        """
        logger.info("Generating synthetic digit dataset...")
        
        num_samples_per_digit = 50  # 50 samples per digit
        num_digits = 10
        
        synthetic_data = []
        
        for digit in range(num_digits):
            for sample_idx in range(num_samples_per_digit):
                # Generate synthetic audio with digit-specific characteristics
                audio_array = self._generate_synthetic_audio(digit)
                
                # Create sample in HuggingFace format
                sample = {
                    'audio': {
                        'array': audio_array.astype(np.float32),
                        'sampling_rate': self.sample_rate
                    },
                    'label': digit
                }
                
                synthetic_data.append(sample)
        
        logger.info(f"Created synthetic dataset with {len(synthetic_data)} samples")
        
        # Return in HuggingFace dataset-like format
        return {'train': synthetic_data}
    
    def _generate_synthetic_audio(self, digit: int) -> np.ndarray:
        """
        Generate synthetic audio for a specific digit.
        
        Args:
            digit: Digit (0-9) to generate audio for
            
        Returns:
            Synthetic audio array
        """
        duration = 1.0  # 1 second
        t = np.linspace(0, duration, self.max_length)
        
        # Create digit-specific frequency patterns
        base_freq = 200 + digit * 50  # Different base frequency for each digit
        
        # Generate harmonic series
        audio = np.zeros_like(t)
        
        # Add fundamental frequency
        audio += 0.3 * np.sin(2 * np.pi * base_freq * t)
        
        # Add harmonics
        audio += 0.2 * np.sin(2 * np.pi * base_freq * 2 * t)
        audio += 0.1 * np.sin(2 * np.pi * base_freq * 3 * t)
        
        # Add some noise for realism
        noise = np.random.normal(0, 0.05, len(t))
        audio += noise
        
        # Apply envelope (attack-decay-sustain-release)
        envelope = np.ones_like(t)
        
        # Attack (first 10%)
        attack_samples = int(0.1 * len(t))
        envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        
        # Decay (last 20%)
        decay_samples = int(0.2 * len(t))
        envelope[-decay_samples:] = np.linspace(1, 0, decay_samples)
        
        audio = audio * envelope
        
        # Normalize
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.8
        
        return audio
    
    def _check_ffmpeg_available(self) -> bool:
        """Check if ffmpeg is available on the system."""
        try:
            result = subprocess.run(['ffmpeg', '-version'], 
                                  capture_output=True, 
                                  text=True, 
                                  timeout=5)
            return result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError, subprocess.TimeoutExpired):
            return False
    
    def _convert_audio_with_ffmpeg(self, audio_array: np.ndarray, original_sr: int, target_sr: int) -> np.ndarray:
        """
        Convert audio using ffmpeg for better quality resampling.
        
        Args:
            audio_array: Input audio array
            original_sr: Original sampling rate
            target_sr: Target sampling rate
            
        Returns:
            Resampled audio array
        """
        try:
            # Create temporary files
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_input:
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_output:
                    
                    # Write input audio
                    sf.write(temp_input.name, audio_array, original_sr)
                    
                    # Run ffmpeg conversion
                    ffmpeg_cmd = [
                        'ffmpeg', 
                        '-i', temp_input.name,
                        '-ar', str(target_sr),
                        '-ac', '1',  # Mono
                        '-acodec', 'pcm_f32le',  # 32-bit float
                        '-y',  # Overwrite output
                        temp_output.name
                    ]
                    
                    result = subprocess.run(ffmpeg_cmd, 
                                          capture_output=True, 
                                          text=True,
                                          timeout=30)
                    
                    if result.returncode == 0:
                        # Read converted audio
                        converted_audio, _ = sf.read(temp_output.name, dtype='float32')
                        logger.debug(f"ffmpeg conversion successful: {original_sr}Hz -> {target_sr}Hz")
                        return converted_audio
                    else:
                        logger.warning(f"ffmpeg conversion failed: {result.stderr}")
                        return None
                        
        except Exception as e:
            logger.warning(f"ffmpeg conversion error: {e}")
            return None
        finally:
            # Clean up temporary files
            try:
                if 'temp_input' in locals():
                    os.unlink(temp_input.name)
                if 'temp_output' in locals():
                    os.unlink(temp_output.name)
            except:
                pass
        
        return None
    
    def preprocess_audio(self, audio_array: np.ndarray, sr: int) -> np.ndarray:
        """
        Standardize audio preprocessing across all pipelines.
        
        Args:
            audio_array: Input audio signal
            sr: Original sampling rate
            
        Returns:
            processed_audio: Preprocessed audio array
            
        Data Flow:
            Input: (variable_length,) -> Output: (max_length,) = (8000,)
            
        Processing Steps:
            1. Resample to target SR if needed
            2. Amplitude normalization
            3. Pad or truncate to fixed length
            4. Validate output dimensions
        """
        try:
            logger.debug(f"Preprocessing audio - Original shape: {audio_array.shape}, SR: {sr}")
            
            # Input validation
            if len(audio_array) < self.min_length:
                logger.warning(f"Audio too short ({len(audio_array)} < {self.min_length}), padding")
            
            # Step 1: Resample if necessary
            if sr != self.sample_rate:
                logger.debug(f"Resampling from {sr}Hz to {self.sample_rate}Hz")
                
                # Try ffmpeg first for better quality, then fall back to librosa
                if hasattr(self, '_ffmpeg_available') and self._ffmpeg_available:
                    ffmpeg_result = self._convert_audio_with_ffmpeg(audio_array, sr, self.sample_rate)
                    if ffmpeg_result is not None:
                        audio_array = ffmpeg_result
                        logger.debug("Used ffmpeg for resampling")
                    else:
                        # Fall back to librosa
                        audio_array = librosa.resample(
                            audio_array, 
                            orig_sr=sr, 
                            target_sr=self.sample_rate
                        )
                        logger.debug("Fell back to librosa for resampling")
                else:
                    # Use librosa
                    audio_array = librosa.resample(
                        audio_array, 
                        orig_sr=sr, 
                        target_sr=self.sample_rate
                    )
            
            # Step 2: Amplitude normalization
            max_amplitude = np.max(np.abs(audio_array))
            if max_amplitude > 1e-8:
                audio_array = audio_array / max_amplitude
            else:
                logger.warning("Audio signal has very low amplitude")
            
            # Step 3: Pad or truncate to fixed length
            if len(audio_array) > self.max_length:
                # Truncate from center to preserve important content
                start = (len(audio_array) - self.max_length) // 2
                audio_array = audio_array[start:start + self.max_length]
                logger.debug(f"Truncated audio from center")
            else:
                # Pad with zeros
                pad_length = self.max_length - len(audio_array)
                audio_array = np.pad(audio_array, (0, pad_length), mode='constant')
                logger.debug(f"Padded audio with {pad_length} zeros")
            
            # Step 4: Validate output
            assert len(audio_array) == self.max_length, f"Expected length {self.max_length}, got {len(audio_array)}"
            assert not np.isnan(audio_array).any(), "NaN values found in processed audio"
            assert not np.isinf(audio_array).any(), "Infinite values found in processed audio"
            
            logger.debug(f"Preprocessing complete - Output shape: {audio_array.shape}")
            return audio_array.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Audio preprocessing failed: {str(e)}")
            # Return zeros as fallback
            return np.zeros(self.max_length, dtype=np.float32)
    
    def create_train_test_split(self, dataset, test_size: float = 0.2, 
                               val_size: float = 0.1, random_state: int = 42) -> Dict[str, Any]:
        """
        Create stratified train/test/validation splits.
        
        Args:
            dataset: HuggingFace dataset
            test_size: Fraction for test set
            val_size: Fraction for validation set
            random_state: Random seed
            
        Returns:
            Dictionary containing splits and metadata
            
        Data Structure:
            Input: HuggingFace dataset
            Output: {
                'X_train': (n_train, max_length) = (N*0.7, 8000)
                'X_val': (n_val, max_length) = (N*0.1, 8000)  
                'X_test': (n_test, max_length) = (N*0.2, 8000)
                'y_train': (n_train,)
                'y_val': (n_val,)
                'y_test': (n_test,)
                'label_encoder': LabelEncoder object
                'dataset_info': metadata dictionary
            }
        """
        try:
            logger.info("Creating train/test/validation splits...")
            
            if dataset is None:
                raise ValueError("Dataset is None - cannot create splits")
            
            # Extract features and labels
            audio_data = []
            labels = []
            sample_rates = []
            
            # Use train split or first available split
            split_name = 'train' if 'train' in dataset else list(dataset.keys())[0]
            data_split = dataset[split_name]
            
            logger.info(f"Processing {len(data_split)} samples from '{split_name}' split")
            
            for idx, item in enumerate(data_split):
                try:
                    # Handle different data formats
                    if isinstance(item, dict):
                        # HuggingFace dataset format
                        if 'audio' in item and isinstance(item['audio'], dict):
                            audio = item['audio']['array']
                            sr = item['audio']['sampling_rate']
                        elif 'audio' in item and isinstance(item['audio'], np.ndarray):
                            # Alternative format where audio is directly an array
                            audio = item['audio']
                            sr = item.get('sampling_rate', self.sample_rate)
                        else:
                            logger.warning(f"Unexpected audio format in sample {idx}: {item.keys()}")
                            continue
                        
                        # Extract label
                        label = item.get('label', self._extract_label_from_filename(item))
                        
                        # Convert label to int if it's a string
                        if isinstance(label, str) and label.isdigit():
                            label = int(label)
                        elif not isinstance(label, (int, np.integer)):
                            logger.warning(f"Invalid label format in sample {idx}: {label}")
                            continue
                    
                    else:
                        logger.warning(f"Unexpected item format in sample {idx}: {type(item)}")
                        continue
                    
                    # Validate audio data
                    if not isinstance(audio, (np.ndarray, list)):
                        logger.warning(f"Audio data is not array-like in sample {idx}")
                        continue
                    
                    # Convert to numpy array if needed
                    if not isinstance(audio, np.ndarray):
                        audio = np.array(audio, dtype=np.float32)
                    
                    # Validate audio shape and content
                    if audio.size == 0:
                        logger.warning(f"Empty audio array in sample {idx}")
                        continue
                    
                    if np.all(audio == 0):
                        logger.warning(f"Silent audio in sample {idx}")
                        continue
                    
                    # Preprocess audio
                    processed_audio = self.preprocess_audio(audio, sr)
                    
                    # Validate preprocessing result
                    if processed_audio is None or len(processed_audio) != self.max_length:
                        logger.warning(f"Preprocessing failed for sample {idx}")
                        continue
                    
                    audio_data.append(processed_audio)
                    labels.append(label)
                    sample_rates.append(sr)
                    
                    if (idx + 1) % 100 == 0:
                        logger.info(f"Processed {idx + 1}/{len(data_split)} samples")
                    
                except Exception as e:
                    logger.warning(f"Failed to process sample {idx}: {str(e)}")
                    logger.debug(f"Sample {idx} content: {str(item)[:200]}...")
                    continue
            
            if len(audio_data) == 0:
                raise ValueError("No valid audio samples found")
            
            # Convert to numpy arrays
            audio_data = np.array(audio_data, dtype=np.float32)
            labels = np.array(labels)
            
            logger.info(f"Audio data shape: {audio_data.shape}")
            logger.info(f"Labels shape: {labels.shape}")
            logger.info(f"Unique labels: {np.unique(labels)}")
            
            # Encode labels
            labels_encoded = self.label_encoder.fit_transform(labels)
            logger.info(f"Label encoding: {dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))}")
            
            # Create stratified splits
            logger.info(f"Creating splits - Test: {test_size:.1%}, Val: {val_size:.1%}")
            
            # First split: train+val and test
            X_temp, X_test, y_temp, y_test = train_test_split(
                audio_data, labels_encoded,
                test_size=test_size,
                stratify=labels_encoded,
                random_state=random_state
            )
            
            # Second split: train and validation
            if val_size > 0:
                val_size_adjusted = val_size / (1 - test_size)
                X_train, X_val, y_train, y_val = train_test_split(
                    X_temp, y_temp,
                    test_size=val_size_adjusted,
                    stratify=y_temp,
                    random_state=random_state
                )
            else:
                X_train, X_val, y_train, y_val = X_temp, None, y_temp, None
            
            # Create dataset info
            dataset_info = {
                'total_samples': len(audio_data),
                'train_samples': len(X_train),
                'val_samples': len(X_val) if X_val is not None else 0,
                'test_samples': len(X_test),
                'sample_rate': self.sample_rate,
                'max_length': self.max_length,
                'num_classes': len(self.label_encoder.classes_),
                'class_names': self.label_encoder.classes_.tolist(),
                'audio_shape': audio_data.shape,
                'mean_sample_rate': np.mean(sample_rates),
                'std_sample_rate': np.std(sample_rates)
            }
            
            logger.info(f"Dataset splits created successfully:")
            logger.info(f"  Train: {dataset_info['train_samples']} samples")
            logger.info(f"  Val: {dataset_info['val_samples']} samples")  
            logger.info(f"  Test: {dataset_info['test_samples']} samples")
            logger.info(f"  Classes: {dataset_info['num_classes']} ({dataset_info['class_names']})")
            
            return {
                'X_train': X_train, 'y_train': y_train,
                'X_val': X_val, 'y_val': y_val,
                'X_test': X_test, 'y_test': y_test,
                'label_encoder': self.label_encoder,
                'dataset_info': dataset_info
            }
            
        except Exception as e:
            logger.error(f"Failed to create dataset splits: {str(e)}")
            raise
    
    def _extract_label_from_filename(self, item: Dict) -> int:
        """Extract digit label from filename if not in metadata."""
        try:
            # Attempt to extract from filename or path
            filename = item.get('path', item.get('filename', ''))
            # Look for digit in filename (0-9)
            for digit in range(10):
                if str(digit) in filename:
                    return digit
            return 0  # Default fallback
        except:
            return 0
    
    def validate_splits(self, data_splits: Dict[str, Any]) -> bool:
        """
        Validate dataset splits for correctness.
        
        Args:
            data_splits: Dictionary containing dataset splits
            
        Returns:
            bool: True if validation passes
        """
        try:
            logger.info("Validating dataset splits...")
            
            required_keys = ['X_train', 'y_train', 'X_test', 'y_test', 'label_encoder', 'dataset_info']
            for key in required_keys:
                if key not in data_splits:
                    logger.error(f"Missing required key: {key}")
                    return False
            
            # Check shapes
            X_train, y_train = data_splits['X_train'], data_splits['y_train']
            X_test, y_test = data_splits['X_test'], data_splits['y_test']
            
            assert X_train.shape[0] == y_train.shape[0], "Train X/y shape mismatch"
            assert X_test.shape[0] == y_test.shape[0], "Test X/y shape mismatch"
            assert X_train.shape[1] == self.max_length, f"Audio length mismatch: {X_train.shape[1]} != {self.max_length}"
            
            if data_splits['X_val'] is not None:
                X_val, y_val = data_splits['X_val'], data_splits['y_val']
                assert X_val.shape[0] == y_val.shape[0], "Val X/y shape mismatch"
                assert X_val.shape[1] == self.max_length, f"Val audio length mismatch"
            
            # Check label distribution
            unique_labels = np.unique(np.concatenate([y_train, y_test]))
            expected_labels = np.arange(len(data_splits['label_encoder'].classes_))
            assert np.array_equal(unique_labels, expected_labels), "Label distribution mismatch"
            
            logger.info("Dataset validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Dataset validation failed: {str(e)}")
            return False

def load_and_prepare_data(sample_rate: int = 8000, max_length: int = 8000,
                         test_size: float = 0.2, val_size: float = 0.1) -> Dict[str, Any]:
    """
    Convenience function to load and prepare dataset.
    
    Returns:
        Dictionary containing prepared dataset splits
    """
    logger.info("Loading and preparing digit dataset...")
    
    try:
        # Initialize loader
        data_loader = DigitDatasetLoader(sample_rate=sample_rate, max_length=max_length)
        
        # Load dataset
        dataset = data_loader.load_fsdd_dataset()
        
        if dataset is None:
            logger.error("Failed to load dataset")
            return None
        
        # Create splits
        data_splits = data_loader.create_train_test_split(
            dataset, test_size=test_size, val_size=val_size
        )
        
        # Validate
        if not data_loader.validate_splits(data_splits):
            logger.error("Dataset validation failed")
            return None
        
        logger.info("Dataset preparation completed successfully")
        return data_splits
        
    except Exception as e:
        logger.error(f"Dataset preparation failed: {str(e)}")
        return None

if __name__ == "__main__":
    # Test dataset loading
    logging.basicConfig(level=logging.INFO)
    
    logger.info("Testing dataset loader...")
    data_splits = load_and_prepare_data()
    
    if data_splits:
        info = data_splits['dataset_info']
        logger.info(f"Dataset loaded successfully: {info}")
    else:
        logger.error("Dataset loading failed")