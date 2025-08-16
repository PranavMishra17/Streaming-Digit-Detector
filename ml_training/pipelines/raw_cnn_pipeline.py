"""
Raw Waveform + 1D CNN Pipeline
End-to-end learning approach using 1D convolutions on raw audio signals
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple, Any, Optional
import logging

logger = logging.getLogger(__name__)

class RawWaveformDataset(Dataset):
    """
    PyTorch Dataset for Raw Waveforms with optional augmentation.
    
    Data Flow:
    Raw Audio -> Optional Augmentation -> Channel dimension -> Tensor
    Input: (N, 8000) -> Output: (N, 1, 8000) + labels (N,)
    
    Augmentation Techniques:
    - Gaussian noise injection
    - Time shifting (circular)
    - Amplitude scaling
    """
    
    def __init__(self, audio_data: np.ndarray, labels: np.ndarray, 
                 augmentation: bool = False, augmentation_prob: float = 0.5):
        """
        Initialize Raw Waveform dataset.
        
        Args:
            audio_data: Raw audio data (N, 8000)
            labels: Labels (N,)
            augmentation: Whether to apply augmentation
            augmentation_prob: Probability of applying each augmentation
            
        Data Dimensions:
            audio_data: (batch_size, 8000) raw waveforms
            output: (batch_size, 1, 8000) with channel dimension for 1D CNN
        """
        self.audio_data = audio_data.astype(np.float32)
        self.labels = labels
        self.augmentation = augmentation
        self.augmentation_prob = augmentation_prob
        
        # Augmentation parameters
        self.noise_factor = 0.005  # Small noise to avoid overfitting
        self.max_shift = 400       # Max time shift (50ms at 8kHz)
        self.amp_range = (0.8, 1.2)  # Amplitude scaling range
        
        logger.info(f"Raw Waveform Dataset initialized:")
        logger.info(f"  Samples: {len(audio_data)}")
        logger.info(f"  Audio length: {audio_data.shape[1]} samples")
        logger.info(f"  Augmentation: {augmentation}")
        if augmentation:
            logger.info(f"    Noise factor: {self.noise_factor}")
            logger.info(f"    Max shift: {self.max_shift} samples")
            logger.info(f"    Amplitude range: {self.amp_range}")
    
    def _augment_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply audio augmentation techniques.
        
        Args:
            audio: Input audio signal (8000,)
            
        Returns:
            augmented_audio: Augmented audio signal (8000,)
            
        Augmentation Steps:
            1. Add Gaussian noise (prevents overfitting)
            2. Random time shifting (temporal invariance)
            3. Amplitude scaling (volume invariance)
        """
        augmented = audio.copy()
        
        # 1. Add Gaussian noise
        if np.random.random() < self.augmentation_prob:
            noise = np.random.normal(0, self.noise_factor, audio.shape)
            augmented = augmented + noise
        
        # 2. Time shifting (circular shift)
        if np.random.random() < self.augmentation_prob:
            shift = np.random.randint(-self.max_shift, self.max_shift + 1)
            augmented = np.roll(augmented, shift)
        
        # 3. Amplitude scaling
        if np.random.random() < self.augmentation_prob:
            scale = np.random.uniform(*self.amp_range)
            augmented = augmented * scale
        
        # Ensure values stay in reasonable range
        augmented = np.clip(augmented, -1.0, 1.0)
        
        return augmented
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.audio_data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get item by index.
        
        Args:
            idx: Sample index
            
        Returns:
            audio: Audio tensor (1, 8000)
            label: Label tensor (scalar)
        """
        audio = self.audio_data[idx].copy()
        label = self.labels[idx]
        
        # Apply augmentation during training
        if self.augmentation:
            audio = self._augment_audio(audio)
        
        # Add channel dimension for 1D CNN
        audio = np.expand_dims(audio, axis=0)
        
        # Convert to tensors
        audio_tensor = torch.FloatTensor(audio)
        label_tensor = torch.LongTensor([label])[0]
        
        return audio_tensor, label_tensor

class RawWaveformCNN(nn.Module):
    """
    1D Convolutional Neural Network for raw waveform classification.
    
    Architecture:
    Input (1, 8000) -> 1D Conv blocks -> Global pooling -> Dense layers -> Output (10)
    
    Design Philosophy:
    - Multi-scale feature extraction with different kernel sizes
    - Hierarchical representation learning
    - Temporal pooling to reduce sequence length
    - End-to-end learning without manual feature engineering
    """
    
    def __init__(self, input_length: int = 8000, num_classes: int = 10):
        """
        Initialize Raw Waveform CNN.
        
        Args:
            input_length: Input audio length in samples
            num_classes: Number of output classes
            
        Network Architecture:
            Conv1: 1->64 channels, kernel=80, stride=4 (raw feature extraction)
            Conv2: 64->128 channels, kernel=3, stride=1 (pattern detection)
            Conv3: 128->128 channels, kernel=3, stride=1 (pattern refinement)
            Conv4: 128->256 channels, kernel=3, stride=1 (high-level features)
            Conv5: 256->256 channels, kernel=3, stride=1 (final features)
            
            Each block: Conv1D -> BatchNorm -> ReLU -> MaxPool -> Dropout
        """
        super(RawWaveformCNN, self).__init__()
        
        self.input_length = input_length
        self.num_classes = num_classes
        
        # 1D Convolutional layers
        # First layer: Large kernel to capture low-level features from raw audio
        self.conv1 = nn.Conv1d(1, 64, kernel_size=80, stride=4, padding=38, bias=False)
        
        # Subsequent layers: Smaller kernels for pattern detection
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv4 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv5 = nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        
        # Batch normalization layers
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(256)
        self.bn5 = nn.BatchNorm1d(256)
        
        # Pooling and regularization
        self.pool = nn.MaxPool1d(4, 4)
        self.dropout1d = nn.Dropout1d(0.2)
        self.dropout = nn.Dropout(0.3)
        
        # Global pooling to handle variable lengths
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
        # Initialize weights
        self._initialize_weights()
        
        # Count parameters
        self.param_count = sum(p.numel() for p in self.parameters())
        
        # Calculate receptive field
        self.receptive_field = self._calculate_receptive_field()
        
        logger.info(f"Raw Waveform CNN initialized:")
        logger.info(f"  Input length: {input_length} samples")
        logger.info(f"  Architecture: Conv1D(64->128->128->256->256) -> Global Pool -> FC(512->256->{num_classes})")
        logger.info(f"  Total parameters: {self.param_count:,}")
        logger.info(f"  Receptive field: {self.receptive_field} samples")
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def _calculate_receptive_field(self) -> int:
        """Calculate theoretical receptive field of the network."""
        # Conv1: kernel=80, stride=4
        # Conv2-5: kernel=3, stride=1 each, with pooling stride=4 between blocks
        # This is a simplified calculation
        rf = 80  # First layer kernel
        rf += 4 * 3 * 4  # Approximate for subsequent layers with pooling
        return min(rf, self.input_length)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input waveforms (batch_size, 1, 8000)
            
        Returns:
            logits: Output logits (batch_size, 10)
            
        Feature Map Evolution:
            Input: (N, 1, 8000)
            Conv1+Pool: (N, 64, 500)  # Large stride + pooling
            Conv2+Pool: (N, 128, 125)
            Conv3+Pool: (N, 128, 31)
            Conv4+Pool: (N, 256, 7)
            Conv5+Pool: (N, 256, 1)
            Global Pool: (N, 256, 1)
            Flatten: (N, 256)
            FC: (N, 10)
        """
        # Validate input dimensions
        if x.dim() != 3 or x.size(1) != 1:
            raise ValueError(f"Expected input shape (batch_size, 1, length), got {x.shape}")
        
        # Conv block 1: Raw feature extraction
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout1d(x)
        
        # Conv block 2: Pattern detection
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout1d(x)
        
        # Conv block 3: Pattern refinement
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout1d(x)
        
        # Conv block 4: High-level features
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.dropout1d(x)
        
        # Conv block 5: Final feature extraction
        x = self.pool(F.relu(self.bn5(self.conv5(x))))
        x = self.dropout1d(x)
        
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
            x: Input waveforms (batch_size, 1, 8000)
            
        Returns:
            predictions: Predicted classes (batch_size,)
            probabilities: Class probabilities (batch_size, 10)
        """
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)
        
        return predictions, probabilities
    
    def get_feature_maps(self, x: torch.Tensor, layer_name: str = 'conv1') -> torch.Tensor:
        """
        Extract intermediate feature maps for visualization.
        
        Args:
            x: Input waveforms (batch_size, 1, 8000)
            layer_name: Layer to extract features from
            
        Returns:
            feature_maps: Feature maps from specified layer
        """
        with torch.no_grad():
            if layer_name == 'conv1':
                return F.relu(self.bn1(self.conv1(x)))
            elif layer_name == 'conv2':
                x = self.pool(F.relu(self.bn1(self.conv1(x))))
                return F.relu(self.bn2(self.conv2(x)))
            # Add more layers as needed
            else:
                raise ValueError(f"Unknown layer name: {layer_name}")

def setup_raw_cnn_pipeline(data_splits: Dict[str, Any], batch_size: int = 16) -> Dict[str, Any]:
    """
    Setup complete Raw Waveform CNN pipeline.
    
    Args:
        data_splits: Dataset splits from data loader
        batch_size: Training batch size
        
    Returns:
        Dictionary containing model, data loaders
        
    Pipeline Components:
        1. Dataset creation with augmentation
        2. Data loader setup  
        3. Model initialization
        4. Device configuration
    """
    try:
        logger.info("Setting up Raw Waveform CNN pipeline...")
        
        # Step 1: Create datasets
        # Training set with augmentation
        train_dataset = RawWaveformDataset(
            data_splits['X_train'], data_splits['y_train'], 
            augmentation=True, augmentation_prob=0.5
        )
        
        # Test set without augmentation
        test_dataset = RawWaveformDataset(
            data_splits['X_test'], data_splits['y_test'],
            augmentation=False
        )
        
        # Validation set without augmentation
        val_dataset = None
        if data_splits['X_val'] is not None:
            val_dataset = RawWaveformDataset(
                data_splits['X_val'], data_splits['y_val'],
                augmentation=False
            )
        
        # Step 2: Create data loaders
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
        
        # Step 3: Initialize model
        audio_length = data_splits['dataset_info']['max_length']
        model = RawWaveformCNN(input_length=audio_length)
        
        # Move to device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        logger.info(f"Raw CNN pipeline setup completed:")
        logger.info(f"  Device: {device}")
        logger.info(f"  Train samples: {len(train_dataset)}")
        logger.info(f"  Val samples: {len(val_dataset) if val_dataset else 0}")
        logger.info(f"  Test samples: {len(test_dataset)}")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Audio length: {audio_length} samples")
        
        return {
            'model': model,
            'train_loader': train_loader,
            'val_loader': val_loader,
            'test_loader': test_loader,
            'device': device,
            'input_length': audio_length,
            'augmentation_enabled': True
        }
        
    except Exception as e:
        logger.error(f"Raw CNN pipeline setup failed: {str(e)}")
        raise

class WaveformAnalyzer:
    """
    Utility class for analyzing raw waveform patterns and model behavior.
    """
    
    def __init__(self, model: RawWaveformCNN, device: torch.device):
        """Initialize waveform analyzer."""
        self.model = model
        self.device = device
        self.model.eval()
    
    def analyze_learned_filters(self, save_path: str = None) -> np.ndarray:
        """
        Analyze and visualize learned convolutional filters.
        
        Args:
            save_path: Optional path to save filter visualizations
            
        Returns:
            filters: First layer filter weights
        """
        # Extract first layer filters
        first_conv = self.model.conv1
        filters = first_conv.weight.data.cpu().numpy()  # Shape: (64, 1, 80)
        
        logger.info(f"First layer filters shape: {filters.shape}")
        logger.info(f"Filter statistics:")
        logger.info(f"  Mean: {filters.mean():.4f}")
        logger.info(f"  Std: {filters.std():.4f}")
        logger.info(f"  Min: {filters.min():.4f}")
        logger.info(f"  Max: {filters.max():.4f}")
        
        return filters
    
    def compute_receptive_field_analysis(self, sample_audio: torch.Tensor) -> Dict[str, Any]:
        """
        Analyze effective receptive field using gradient-based methods.
        
        Args:
            sample_audio: Sample audio tensor (1, 1, 8000)
            
        Returns:
            analysis: Dictionary with receptive field information
        """
        self.model.eval()
        sample_audio = sample_audio.to(self.device)
        sample_audio.requires_grad_(True)
        
        # Forward pass
        output = self.model(sample_audio)
        
        # Compute gradients for max prediction
        max_class = torch.argmax(output, dim=1)
        output[0, max_class].backward()
        
        # Analyze gradients
        gradients = sample_audio.grad.data.cpu().numpy().squeeze()
        
        # Compute receptive field metrics
        grad_abs = np.abs(gradients)
        effective_length = np.sum(grad_abs > grad_abs.max() * 0.1)  # 10% threshold
        
        analysis = {
            'gradient_magnitude': grad_abs,
            'effective_receptive_field': effective_length,
            'theoretical_receptive_field': self.model.receptive_field,
            'gradient_max': grad_abs.max(),
            'gradient_mean': grad_abs.mean()
        }
        
        return analysis

if __name__ == "__main__":
    # Test Raw CNN pipeline
    import logging
    from ml_training.data.dataset_loader import load_and_prepare_data
    
    logging.basicConfig(level=logging.INFO)
    
    logger.info("Testing Raw Waveform CNN pipeline...")
    
    # Load test data
    data_splits = load_and_prepare_data()
    
    if data_splits is not None:
        # Setup pipeline
        pipeline_components = setup_raw_cnn_pipeline(data_splits, batch_size=8)
        
        # Test forward pass
        model = pipeline_components['model']
        train_loader = pipeline_components['train_loader']
        
        # Get a batch and test
        for batch_audio, batch_labels in train_loader:
            logger.info(f"Batch audio shape: {batch_audio.shape}")
            logger.info(f"Batch labels shape: {batch_labels.shape}")
            
            # Forward pass
            with torch.no_grad():
                outputs = model(batch_audio)
                logger.info(f"Model outputs shape: {outputs.shape}")
            
            break  # Test only first batch
        
        # Test waveform analyzer
        analyzer = WaveformAnalyzer(model, pipeline_components['device'])
        filters = analyzer.analyze_learned_filters()
        logger.info(f"Analyzed {len(filters)} learned filters")
        
        logger.info("Raw CNN pipeline test completed successfully")
    else:
        logger.error("Failed to load test data")