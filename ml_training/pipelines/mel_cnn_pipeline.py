"""
Mel Spectrogram + 2D CNN Pipeline
Image-like processing approach using mel spectrograms and convolutional neural networks
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple, Any, Optional
import logging

logger = logging.getLogger(__name__)

class MelSpectrogramExtractor:
    """
    Mel Spectrogram Feature Extraction using PyTorch transforms.
    
    Data Flow:
    Raw Audio (8000,) -> Mel Spectrogram (64, 51) -> dB Scale -> Normalization
    
    Dimension Analysis:
    - Input: (8000,) samples at 8kHz = 1 second
    - STFT: n_fft=512, hop_length=160 -> frames = ceil(8000/160) = 50 frames
    - Mel bins: 64 frequency bins
    - Output: (64, 51) spectrogram (including padding)
    """
    
    def __init__(self, sample_rate: int = 8000, n_mels: int = 64, 
                 n_fft: int = 512, hop_length: int = 160):
        """
        Initialize Mel Spectrogram extractor.
        
        Args:
            sample_rate: Audio sample rate (Hz)
            n_mels: Number of mel filter banks
            n_fft: FFT window size  
            hop_length: Hop length for STFT
            
        Mel Filter Bank Design:
            - Frequency range: 0 to sample_rate/2 Hz (0 to 4000 Hz)
            - Mel scale: Perceptually uniform frequency representation
            - Filter banks: 64 triangular filters
        """
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        # Calculate expected output dimensions
        self.expected_time_frames = (8000 // hop_length) + 1  # ~51 frames
        self.expected_shape = (n_mels, self.expected_time_frames)
        
        # Setup PyTorch transforms for GPU acceleration
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Mel spectrogram transform
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=0.0,
            f_max=sample_rate / 2.0,  # Nyquist frequency
            power=2.0,  # Power spectrogram
            normalized=False
        ).to(device)
        
        # Convert to dB scale
        self.amplitude_to_db = T.AmplitudeToDB(
            stype='power',
            top_db=80.0  # Clip values at 80dB below peak
        ).to(device)
        
        self.device = device
        
        logger.info(f"Mel Spectrogram Extractor initialized:")
        logger.info(f"  Expected output shape: {self.expected_shape}")
        logger.info(f"  Device: {device}")
        logger.info(f"  Frequency range: 0-{sample_rate//2} Hz")
    
    def extract_features(self, audio_array: np.ndarray) -> np.ndarray:
        """
        Extract mel spectrogram from single audio sample.
        
        Args:
            audio_array: Input audio (8000,)
            
        Returns:
            mel_spec: Normalized mel spectrogram (64, 51)
            
        Processing Steps:
            1. Convert to tensor and move to device
            2. Extract mel spectrogram
            3. Convert to dB scale
            4. Normalize (zero mean, unit variance)
            5. Return as numpy array
        """
        try:
            # Validate input
            assert len(audio_array) > 0, "Empty audio array"
            assert not np.isnan(audio_array).any(), "NaN values in audio"
            
            # Step 1: Convert to tensor and add batch dimension
            audio_tensor = torch.FloatTensor(audio_array).unsqueeze(0).to(self.device)
            
            # Step 2: Extract mel spectrogram
            with torch.no_grad():
                mel_spec = self.mel_transform(audio_tensor)
                logger.debug(f"Raw mel spectrogram shape: {mel_spec.shape}")
                
                # Step 3: Convert to dB scale
                mel_spec_db = self.amplitude_to_db(mel_spec)
                
                # Step 4: Normalize (per-sample normalization)
                mel_spec_normalized = self._normalize_spectrogram(mel_spec_db)
                
                # Step 5: Remove batch dimension and convert to numpy
                result = mel_spec_normalized.squeeze(0).cpu().numpy()
            
            # Validate output dimensions
            expected_shape = self.expected_shape
            if result.shape != expected_shape:
                logger.warning(f"Shape mismatch: {result.shape} != {expected_shape}")
                # Pad or trim to expected shape
                result = self._resize_spectrogram(result, expected_shape)
            
            assert not np.isnan(result).any(), "NaN in mel spectrogram"
            assert not np.isinf(result).any(), "Infinite values in mel spectrogram"
            
            logger.debug(f"Final mel spectrogram shape: {result.shape}")
            
            return result.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Mel spectrogram extraction failed: {str(e)}")
            # Return zeros as fallback
            return np.zeros(self.expected_shape, dtype=np.float32)
    
    def extract_batch_features(self, audio_batch: np.ndarray) -> np.ndarray:
        """
        Extract mel spectrograms for batch of audio samples.
        
        Args:
            audio_batch: Batch of audio arrays (batch_size, 8000)
            
        Returns:
            mel_specs_batch: Batch of mel spectrograms (batch_size, 64, 51)
        """
        try:
            # Convert to tensor
            audio_tensor = torch.FloatTensor(audio_batch).to(self.device)
            
            with torch.no_grad():
                # Extract mel spectrograms
                mel_specs = self.mel_transform(audio_tensor)
                mel_specs_db = self.amplitude_to_db(mel_specs)
                
                # Normalize each spectrogram
                normalized_specs = []
                for spec in mel_specs_db:
                    normalized = self._normalize_spectrogram(spec.unsqueeze(0))
                    normalized_specs.append(normalized.squeeze(0))
                
                result = torch.stack(normalized_specs).cpu().numpy()
            
            logger.debug(f"Batch mel spectrograms shape: {result.shape}")
            
            return result.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Batch mel spectrogram extraction failed: {str(e)}")
            # Return zeros as fallback
            batch_size = len(audio_batch)
            return np.zeros((batch_size, *self.expected_shape), dtype=np.float32)
    
    def _normalize_spectrogram(self, spec: torch.Tensor) -> torch.Tensor:
        """Normalize spectrogram to zero mean and unit variance."""
        mean = torch.mean(spec)
        std = torch.std(spec)
        
        # Avoid division by zero
        std = torch.clamp(std, min=1e-8)
        
        normalized = (spec - mean) / std
        return normalized
    
    def _resize_spectrogram(self, spec: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        """Resize spectrogram to target shape by padding or cropping."""
        current_shape = spec.shape
        target_h, target_w = target_shape
        
        # Handle height (frequency bins)
        if current_shape[0] > target_h:
            spec = spec[:target_h, :]  # Crop
        elif current_shape[0] < target_h:
            pad_h = target_h - current_shape[0]
            spec = np.pad(spec, ((0, pad_h), (0, 0)), mode='constant')
        
        # Handle width (time frames)  
        if current_shape[1] > target_w:
            spec = spec[:, :target_w]  # Crop
        elif current_shape[1] < target_w:
            pad_w = target_w - current_shape[1]
            spec = np.pad(spec, ((0, 0), (0, pad_w)), mode='constant')
        
        return spec

class MelSpectrogramDataset(Dataset):
    """
    PyTorch Dataset for Mel Spectrograms.
    
    Data Flow:
    Raw Audio -> Mel Spectrogram -> Channel dimension -> Tensor
    Input: (N, 8000) -> Output: (N, 1, 64, 51) + labels (N,)
    """
    
    def __init__(self, audio_data: np.ndarray, labels: np.ndarray, 
                 feature_extractor: MelSpectrogramExtractor, precompute: bool = True):
        """
        Initialize Mel Spectrogram dataset.
        
        Args:
            audio_data: Raw audio data (N, 8000)
            labels: Labels (N,)
            feature_extractor: Mel spectrogram extractor
            precompute: Whether to precompute all spectrograms (faster training)
            
        Data Dimensions:
            audio_data: (batch_size, 8000)
            spectrograms: (batch_size, 64, 51)
            output: (batch_size, 1, 64, 51) with channel dimension
        """
        self.audio_data = audio_data
        self.labels = labels
        self.feature_extractor = feature_extractor
        self.precompute = precompute
        
        if precompute:
            logger.info(f"Precomputing mel spectrograms for {len(audio_data)} samples...")
            self.spectrograms = self._precompute_spectrograms()
            logger.info("Spectrogram precomputation completed")
        else:
            self.spectrograms = None
            logger.info("Using on-demand spectrogram computation")
    
    def _precompute_spectrograms(self) -> np.ndarray:
        """Precompute all spectrograms for faster training."""
        spectrograms = self.feature_extractor.extract_batch_features(self.audio_data)
        return spectrograms
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.audio_data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get item by index.
        
        Args:
            idx: Sample index
            
        Returns:
            spectrogram: Spectrogram tensor (1, 64, 51)
            label: Label tensor (scalar)
        """
        if self.precompute:
            # Use precomputed spectrogram
            mel_spec = self.spectrograms[idx]
        else:
            # Compute on demand
            audio = self.audio_data[idx]
            mel_spec = self.feature_extractor.extract_features(audio)
        
        # Add channel dimension for CNN (grayscale image)
        mel_spec = np.expand_dims(mel_spec, axis=0)
        
        # Convert to tensors
        spectrogram = torch.FloatTensor(mel_spec)
        label = torch.LongTensor([self.labels[idx]])[0]
        
        return spectrogram, label

class MelSpectrogramCNN(nn.Module):
    """
    2D Convolutional Neural Network for mel spectrogram classification.
    
    Architecture:
    Input (1, 64, 51) -> Conv2D blocks -> Global pooling -> Dense layers -> Output (10)
    
    Design Principles:
    - Hierarchical feature learning
    - Batch normalization for stable training
    - Dropout for regularization
    - Progressive channel increase
    - Spatial dimension reduction via pooling
    """
    
    def __init__(self, input_shape: Tuple[int, int, int] = (1, 64, 51), num_classes: int = 10):
        """
        Initialize Mel Spectrogram CNN.
        
        Args:
            input_shape: Input tensor shape (channels, height, width)
            num_classes: Number of output classes
            
        Network Architecture:
            Conv1: 1->32 channels, 3x3 kernels
            Conv2: 32->64 channels, 3x3 kernels  
            Conv3: 64->128 channels, 3x3 kernels
            Conv4: 128->256 channels, 3x3 kernels
            
            Each conv block: Conv2D -> BatchNorm -> ReLU -> MaxPool -> Dropout
            Final: Adaptive Global Average Pooling -> Dense layers
        """
        super(MelSpectrogramCNN, self).__init__()
        
        self.input_shape = input_shape
        self.num_classes = num_classes
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False)
        
        # Batch normalization layers
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Pooling and regularization
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout2d = nn.Dropout2d(0.2)
        self.dropout = nn.Dropout(0.3)
        
        # Global pooling to handle variable spatial dimensions
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
        # Initialize weights
        self._initialize_weights()
        
        # Count parameters
        self.param_count = sum(p.numel() for p in self.parameters())
        
        logger.info(f"Mel Spectrogram CNN initialized:")
        logger.info(f"  Input shape: {input_shape}")
        logger.info(f"  Architecture: Conv(32->64->128->256) -> Global Pool -> FC(512->256->{num_classes})")
        logger.info(f"  Total parameters: {self.param_count:,}")
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input spectrograms (batch_size, 1, 64, 51)
            
        Returns:
            logits: Output logits (batch_size, 10)
            
        Feature Map Evolution:
            Input: (N, 1, 64, 51)
            Conv1+Pool: (N, 32, 32, 25)  
            Conv2+Pool: (N, 64, 16, 12)
            Conv3+Pool: (N, 128, 8, 6)
            Conv4+Pool: (N, 256, 4, 3)
            Global Pool: (N, 256, 1, 1)
            Flatten: (N, 256)
            FC: (N, 10)
        """
        # Validate input dimensions
        if x.dim() != 4 or x.size(1) != self.input_shape[0]:
            raise ValueError(f"Expected input shape (batch_size, {self.input_shape[0]}, H, W), got {x.shape}")
        
        # Conv block 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout2d(x)
        
        # Conv block 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout2d(x)
        
        # Conv block 3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout2d(x)
        
        # Conv block 4
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.dropout2d(x)
        
        # Global average pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x
    
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions with confidence scores.
        
        Args:
            x: Input spectrograms (batch_size, 1, 64, 51)
            
        Returns:
            predictions: Predicted classes (batch_size,)
            probabilities: Class probabilities (batch_size, 10)
        """
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)
        
        return predictions, probabilities

def setup_mel_cnn_pipeline(data_splits: Dict[str, Any], batch_size: int = 16) -> Dict[str, Any]:
    """
    Setup complete Mel Spectrogram CNN pipeline.
    
    Args:
        data_splits: Dataset splits from data loader
        batch_size: Training batch size (smaller due to memory)
        
    Returns:
        Dictionary containing model, data loaders, and extractor
        
    Pipeline Components:
        1. Feature extraction setup
        2. Dataset creation with precomputation
        3. Data loader setup
        4. Model initialization
    """
    try:
        logger.info("Setting up Mel Spectrogram CNN pipeline...")
        
        # Step 1: Initialize feature extractor
        mel_extractor = MelSpectrogramExtractor()
        
        # Step 2: Create datasets with precomputation for speed
        train_dataset = MelSpectrogramDataset(
            data_splits['X_train'], data_splits['y_train'], 
            mel_extractor, precompute=True
        )
        
        test_dataset = MelSpectrogramDataset(
            data_splits['X_test'], data_splits['y_test'],
            mel_extractor, precompute=True
        )
        
        val_dataset = None
        if data_splits['X_val'] is not None:
            val_dataset = MelSpectrogramDataset(
                data_splits['X_val'], data_splits['y_val'],
                mel_extractor, precompute=True
            )
        
        # Step 3: Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size,
            shuffle=True, num_workers=0, pin_memory=True
        )
        
        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset, batch_size=batch_size,
                shuffle=False, num_workers=0, pin_memory=True
            )
        
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size,
            shuffle=False, num_workers=0, pin_memory=True
        )
        
        # Step 4: Initialize model
        model = MelSpectrogramCNN(input_shape=(1, 64, 51))
        
        # Move to device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        logger.info(f"Mel CNN pipeline setup completed:")
        logger.info(f"  Device: {device}")
        logger.info(f"  Train samples: {len(train_dataset)}")
        logger.info(f"  Val samples: {len(val_dataset) if val_dataset else 0}")
        logger.info(f"  Test samples: {len(test_dataset)}")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Spectrogram shape: {mel_extractor.expected_shape}")
        
        return {
            'model': model,
            'train_loader': train_loader,
            'val_loader': val_loader,
            'test_loader': test_loader,
            'feature_extractor': mel_extractor,
            'device': device,
            'input_shape': (1, 64, 51)
        }
        
    except Exception as e:
        logger.error(f"Mel CNN pipeline setup failed: {str(e)}")
        raise

if __name__ == "__main__":
    # Test Mel CNN pipeline
    import logging
    from ml_training.data.dataset_loader import load_and_prepare_data
    
    logging.basicConfig(level=logging.INFO)
    
    logger.info("Testing Mel CNN pipeline...")
    
    # Load test data
    data_splits = load_and_prepare_data()
    
    if data_splits is not None:
        # Setup pipeline
        pipeline_components = setup_mel_cnn_pipeline(data_splits, batch_size=8)
        
        # Test forward pass
        model = pipeline_components['model']
        train_loader = pipeline_components['train_loader']
        
        # Get a batch and test
        for batch_spectrograms, batch_labels in train_loader:
            logger.info(f"Batch spectrograms shape: {batch_spectrograms.shape}")
            logger.info(f"Batch labels shape: {batch_labels.shape}")
            
            # Forward pass
            with torch.no_grad():
                outputs = model(batch_spectrograms)
                logger.info(f"Model outputs shape: {outputs.shape}")
            
            break  # Test only first batch
        
        logger.info("Mel CNN pipeline test completed successfully")
    else:
        logger.error("Failed to load test data")