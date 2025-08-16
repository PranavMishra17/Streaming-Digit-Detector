"""
MFCC + Dense Neural Network Pipeline
Feature-based approach using Mel-Frequency Cepstral Coefficients
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import librosa
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple, Any, Optional
import logging

logger = logging.getLogger(__name__)

class MFCCFeatureExtractor:
    """
    MFCC Feature Extraction with statistical summarization.
    
    Data Flow:
    Raw Audio (8000 samples) -> MFCC (13 coeff) -> Delta features -> Statistical summary
    Input: (8000,) -> Output: (156,) features
    
    Feature Breakdown:
    - MFCC: 13 coefficients
    - Delta MFCC: 13 coefficients  
    - Delta-Delta MFCC: 13 coefficients
    - Statistical features per component: mean, std, max, min (4 stats)
    - Total: (13 + 13 + 13) * 4 = 156 features
    """
    
    def __init__(self, sample_rate: int = 8000, n_mfcc: int = 13, 
                 n_fft: int = 512, hop_length: int = 160):
        """
        Initialize MFCC feature extractor.
        
        Args:
            sample_rate: Audio sample rate (Hz)
            n_mfcc: Number of MFCC coefficients
            n_fft: FFT window size
            hop_length: Hop length for STFT
            
        Feature Dimensions:
            Input: (8000,) samples at 8kHz = 1 second audio
            STFT frames: ceil(8000 / 160) = 50 frames
            Output: (156,) feature vector
        """
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        # Calculate expected dimensions
        self.expected_features = n_mfcc * 3 * 4  # 3 feature types, 4 statistics each
        
        logger.debug(f"MFCC Extractor initialized - Features: {self.expected_features}")
    
    def extract_features(self, audio_array: np.ndarray) -> np.ndarray:
        """
        Extract MFCC features with delta and delta-delta coefficients.
        
        Args:
            audio_array: Input audio signal (8000,)
            
        Returns:
            feature_vector: Extracted features (156,)
            
        Processing Steps:
            1. Extract MFCC coefficients (13, n_frames)
            2. Compute delta features (first derivative)
            3. Compute delta-delta features (second derivative)
            4. Statistical summarization (mean, std, max, min)
            5. Concatenate all features
        """
        try:
            # Validate input dimensions
            assert len(audio_array) > 0, "Empty audio array"
            assert not np.isnan(audio_array).any(), "NaN values in audio"
            
            # Step 1: Basic MFCC extraction
            mfccs = librosa.feature.mfcc(
                y=audio_array,
                sr=self.sample_rate,
                n_mfcc=self.n_mfcc,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                center=True
            )
            
            logger.debug(f"MFCC shape: {mfccs.shape}")
            
            # Step 2: Delta features (first derivative)
            delta_mfccs = librosa.feature.delta(mfccs, order=1)
            
            # Step 3: Delta-delta features (second derivative) 
            delta2_mfccs = librosa.feature.delta(mfccs, order=2)
            
            # Step 4: Combine all features
            all_features = np.vstack([mfccs, delta_mfccs, delta2_mfccs])
            logger.debug(f"Combined features shape: {all_features.shape}")
            
            # Step 5: Statistical summarization across time dimension
            features_mean = np.mean(all_features, axis=1)
            features_std = np.std(all_features, axis=1)
            features_max = np.max(all_features, axis=1)
            features_min = np.min(all_features, axis=1)
            
            # Concatenate statistical features
            final_features = np.concatenate([
                features_mean, features_std, features_max, features_min
            ])
            
            # Validate output dimensions
            assert len(final_features) == self.expected_features, \
                f"Feature dimension mismatch: {len(final_features)} != {self.expected_features}"
            
            assert not np.isnan(final_features).any(), "NaN in extracted features"
            assert not np.isinf(final_features).any(), "Infinite values in features"
            
            logger.debug(f"Final feature vector shape: {final_features.shape}")
            
            return final_features.astype(np.float32)
            
        except Exception as e:
            logger.error(f"MFCC feature extraction failed: {str(e)}")
            # Return zeros as fallback
            return np.zeros(self.expected_features, dtype=np.float32)
    
    def extract_batch_features(self, audio_batch: np.ndarray) -> np.ndarray:
        """
        Extract features for batch of audio samples.
        
        Args:
            audio_batch: Batch of audio arrays (batch_size, 8000)
            
        Returns:
            features_batch: Extracted features (batch_size, 156)
        """
        batch_features = []
        
        for i, audio in enumerate(audio_batch):
            features = self.extract_features(audio)
            batch_features.append(features)
            
            if (i + 1) % 100 == 0:
                logger.debug(f"Extracted features for {i + 1}/{len(audio_batch)} samples")
        
        return np.array(batch_features, dtype=np.float32)

class MFCCDataset(Dataset):
    """
    PyTorch Dataset for MFCC features.
    
    Data Flow:
    Raw Audio -> MFCC Features -> Scaled Features -> Tensors
    Input: (N, 8000) -> Output: (N, 156) features + (N,) labels
    """
    
    def __init__(self, audio_data: np.ndarray, labels: np.ndarray, 
                 feature_extractor: MFCCFeatureExtractor, scaler: Optional[StandardScaler] = None):
        """
        Initialize MFCC dataset.
        
        Args:
            audio_data: Raw audio data (N, 8000)
            labels: Labels (N,)
            feature_extractor: MFCC feature extractor
            scaler: Optional feature scaler
            
        Data Dimensions:
            audio_data: (batch_size, 8000) 
            labels: (batch_size,)
            features: (batch_size, 156)
        """
        self.audio_data = audio_data
        self.labels = labels
        self.feature_extractor = feature_extractor
        self.scaler = scaler
        
        logger.info(f"Extracting MFCC features for {len(audio_data)} samples...")
        
        # Extract all features at initialization for efficiency
        self.features = self._extract_all_features()
        
        logger.info(f"MFCC Dataset initialized - Features shape: {self.features.shape}")
    
    def _extract_all_features(self) -> np.ndarray:
        """Extract and optionally scale all features."""
        # Extract features
        features = self.feature_extractor.extract_batch_features(self.audio_data)
        
        # Apply scaling if provided
        if self.scaler is not None:
            logger.debug("Applying feature scaling...")
            features = self.scaler.transform(features)
        
        return features
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get item by index.
        
        Args:
            idx: Sample index
            
        Returns:
            features: Feature tensor (156,)
            label: Label tensor (scalar)
        """
        features = torch.FloatTensor(self.features[idx])
        label = torch.LongTensor([self.labels[idx]])[0]
        
        return features, label

class MFCCClassifier(nn.Module):
    """
    Dense Neural Network for MFCC-based digit classification.
    
    Architecture:
    Input (156) -> Dense layers with BatchNorm + ReLU + Dropout -> Output (10)
    
    Layer Dimensions:
    - Input: 156 features
    - Hidden: [256, 128, 64] (configurable)
    - Output: 10 classes (digits 0-9)
    
    Regularization:
    - Batch Normalization
    - Dropout (0.3 default)
    - Weight decay in optimizer
    """
    
    def __init__(self, input_dim: int = 156, hidden_dims: list = [256, 128, 64], 
                 num_classes: int = 10, dropout: float = 0.3):
        """
        Initialize MFCC classifier.
        
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions  
            num_classes: Number of output classes
            dropout: Dropout probability
            
        Network Architecture:
            156 -> 256 -> 128 -> 64 -> 10
            Each hidden layer: Linear -> BatchNorm -> ReLU -> Dropout
        """
        super(MFCCClassifier, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.dropout = dropout
        
        # Build network layers
        layers = []
        prev_dim = input_dim
        
        # Hidden layers
        for i, hidden_dim in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer (no activation, handled by loss function)
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
        
        # Count parameters
        self.param_count = sum(p.numel() for p in self.parameters())
        
        logger.info(f"MFCC Classifier initialized:")
        logger.info(f"  Architecture: {input_dim} -> {' -> '.join(map(str, hidden_dims))} -> {num_classes}")
        logger.info(f"  Total parameters: {self.param_count:,}")
        logger.info(f"  Dropout: {dropout}")
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input features (batch_size, 156)
            
        Returns:
            logits: Output logits (batch_size, 10)
        """
        # Validate input dimensions
        if x.dim() != 2 or x.size(1) != self.input_dim:
            raise ValueError(f"Expected input shape (batch_size, {self.input_dim}), got {x.shape}")
        
        return self.network(x)
    
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions with confidence scores.
        
        Args:
            x: Input features (batch_size, 156)
            
        Returns:
            predictions: Predicted classes (batch_size,)
            probabilities: Class probabilities (batch_size, 10)
        """
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)
        
        return predictions, probabilities

def setup_mfcc_pipeline(data_splits: Dict[str, Any], batch_size: int = 32) -> Dict[str, Any]:
    """
    Setup complete MFCC pipeline with data loaders and model.
    
    Args:
        data_splits: Dataset splits from data loader
        batch_size: Training batch size
        
    Returns:
        Dictionary containing model, data loaders, and scaler
        
    Pipeline Components:
        1. Feature extraction and scaling
        2. Dataset creation
        3. Data loader setup
        4. Model initialization
    """
    try:
        logger.info("Setting up MFCC pipeline...")
        
        # Step 1: Initialize feature extractor
        mfcc_extractor = MFCCFeatureExtractor()
        
        # Step 2: Extract and scale training features for scaler fitting
        logger.info("Fitting feature scaler on training data...")
        train_features = mfcc_extractor.extract_batch_features(data_splits['X_train'])
        
        # Fit scaler on training data
        scaler = StandardScaler()
        scaler.fit(train_features)
        
        logger.info(f"Feature scaling statistics:")
        logger.info(f"  Mean: {scaler.mean_[:5]}... (first 5)")
        logger.info(f"  Std: {scaler.scale_[:5]}... (first 5)")
        
        # Step 3: Create datasets
        train_dataset = MFCCDataset(
            data_splits['X_train'], data_splits['y_train'], 
            mfcc_extractor, scaler
        )
        
        test_dataset = MFCCDataset(
            data_splits['X_test'], data_splits['y_test'],
            mfcc_extractor, scaler
        )
        
        val_dataset = None
        if data_splits['X_val'] is not None:
            val_dataset = MFCCDataset(
                data_splits['X_val'], data_splits['y_val'],
                mfcc_extractor, scaler
            )
        
        # Step 4: Create data loaders
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
        
        # Step 5: Initialize model
        input_dim = mfcc_extractor.expected_features
        model = MFCCClassifier(input_dim=input_dim)
        
        # Move to device if CUDA available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        logger.info(f"MFCC pipeline setup completed:")
        logger.info(f"  Device: {device}")
        logger.info(f"  Train samples: {len(train_dataset)}")
        logger.info(f"  Val samples: {len(val_dataset) if val_dataset else 0}")
        logger.info(f"  Test samples: {len(test_dataset)}")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Feature dimension: {input_dim}")
        
        return {
            'model': model,
            'train_loader': train_loader,
            'val_loader': val_loader,
            'test_loader': test_loader,
            'scaler': scaler,
            'feature_extractor': mfcc_extractor,
            'device': device,
            'input_dim': input_dim
        }
        
    except Exception as e:
        logger.error(f"MFCC pipeline setup failed: {str(e)}")
        raise

if __name__ == "__main__":
    # Test MFCC pipeline
    import logging
    from ml_training.data.dataset_loader import load_and_prepare_data
    
    logging.basicConfig(level=logging.INFO)
    
    logger.info("Testing MFCC pipeline...")
    
    # Load test data
    data_splits = load_and_prepare_data()
    
    if data_splits is not None:
        # Setup pipeline
        pipeline_components = setup_mfcc_pipeline(data_splits, batch_size=16)
        
        # Test forward pass
        model = pipeline_components['model']
        train_loader = pipeline_components['train_loader']
        
        # Get a batch and test
        for batch_features, batch_labels in train_loader:
            logger.info(f"Batch features shape: {batch_features.shape}")
            logger.info(f"Batch labels shape: {batch_labels.shape}")
            
            # Forward pass
            with torch.no_grad():
                outputs = model(batch_features)
                logger.info(f"Model outputs shape: {outputs.shape}")
            
            break  # Test only first batch
        
        logger.info("MFCC pipeline test completed successfully")
    else:
        logger.error("Failed to load test data")