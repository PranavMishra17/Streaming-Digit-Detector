# Digit Classification ML Pipelines Report
## Three Lightweight Approaches for Spoken Digit Recognition

### Overview
This report presents three distinct machine learning pipelines for classifying spoken digits (0-9) using the Free Spoken Digit Dataset. Each approach represents different trade-offs between accuracy, speed, and computational requirements.

### Dataset Information
- **Source**: Free Spoken Digit Dataset (FSDD)
- **Format**: WAV files, 8kHz sampling rate
- **Classes**: 10 digits (0-9)
- **Total samples**: ~3,000 recordings
- **Speakers**: 6 different speakers

---

## Environment Setup

### Required Libraries
```python
# Core libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Audio processing
import librosa
import soundfile as sf
from scipy.signal import butter, filtfilt

# Machine Learning
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# HuggingFace
from datasets import load_dataset
import torchaudio
import torchaudio.transforms as T

# Visualization
from sklearn.metrics import ConfusionMatrixDisplay
import plotly.express as px
import plotly.graph_objects as go

# CUDA Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")
```

### Installation Commands
```bash
# Core ML libraries
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install librosa soundfile scipy scikit-learn

# HuggingFace and dataset
pip install datasets transformers

# Visualization
pip install matplotlib seaborn plotly

# Optional: Audio augmentation
pip install audiomentations
```

---

## Data Loading and Preprocessing

### Dataset Loading Function
```python
class DigitDatasetLoader:
    def __init__(self, sample_rate=8000, max_length=8000):
        self.sample_rate = sample_rate
        self.max_length = max_length
        self.label_encoder = LabelEncoder()
        
    def load_fsdd_dataset(self):
        """Load Free Spoken Digit Dataset from HuggingFace"""
        try:
            dataset = load_dataset("mteb/free-spoken-digit-dataset")
            return dataset
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None
    
    def preprocess_audio(self, audio_array, sr):
        """Standardize audio preprocessing across all pipelines"""
        # Resample if necessary
        if sr != self.sample_rate:
            audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=self.sample_rate)
        
        # Normalize
        audio_array = audio_array / np.max(np.abs(audio_array) + 1e-8)
        
        # Pad or truncate to fixed length
        if len(audio_array) > self.max_length:
            audio_array = audio_array[:self.max_length]
        else:
            audio_array = np.pad(audio_array, (0, self.max_length - len(audio_array)))
            
        return audio_array
    
    def create_train_test_split(self, dataset, test_size=0.2, val_size=0.1):
        """Create stratified train/test/validation splits"""
        # Extract features and labels
        audio_data = []
        labels = []
        
        for item in dataset['train']:  # FSDD typically has only 'train' split
            audio = item['audio']['array']
            label = item['label']
            
            # Preprocess audio
            processed_audio = self.preprocess_audio(audio, item['audio']['sampling_rate'])
            audio_data.append(processed_audio)
            labels.append(label)
        
        audio_data = np.array(audio_data)
        labels = np.array(labels)
        
        # Encode labels
        labels_encoded = self.label_encoder.fit_transform(labels)
        
        # First split: train+val and test
        X_temp, X_test, y_temp, y_test = train_test_split(
            audio_data, labels_encoded, 
            test_size=test_size, 
            stratify=labels_encoded, 
            random_state=42
        )
        
        # Second split: train and validation
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            stratify=y_temp,
            random_state=42
        )
        
        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test,
            'label_encoder': self.label_encoder
        }

# Initialize data loader
data_loader = DigitDatasetLoader()
dataset = data_loader.load_fsdd_dataset()
data_splits = data_loader.create_train_test_split(dataset)
```

---

## Pipeline 1: MFCC + Dense Neural Network

### Feature Extraction
```python
class MFCCFeatureExtractor:
    def __init__(self, sample_rate=8000, n_mfcc=13, n_fft=512, hop_length=160):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        
    def extract_features(self, audio_array):
        """Extract MFCC features with delta and delta-delta"""
        # Basic MFCC
        mfccs = librosa.feature.mfcc(
            y=audio_array,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        # Delta features (first derivative)
        delta_mfccs = librosa.feature.delta(mfccs)
        
        # Delta-delta features (second derivative)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        
        # Combine all features
        features = np.vstack([mfccs, delta_mfccs, delta2_mfccs])
        
        # Statistical summarization
        features_mean = np.mean(features, axis=1)
        features_std = np.std(features, axis=1)
        features_max = np.max(features, axis=1)
        features_min = np.min(features, axis=1)
        
        # Combine statistical features
        final_features = np.concatenate([features_mean, features_std, features_max, features_min])
        
        return final_features
    
    def extract_batch_features(self, audio_batch):
        """Extract features for batch of audio"""
        features = []
        for audio in audio_batch:
            feat = self.extract_features(audio)
            features.append(feat)
        return np.array(features)

# Feature extraction
mfcc_extractor = MFCCFeatureExtractor()
```

### MFCC Dataset Class
```python
class MFCCDataset(Dataset):
    def __init__(self, audio_data, labels, feature_extractor, scaler=None):
        self.audio_data = audio_data
        self.labels = labels
        self.feature_extractor = feature_extractor
        self.scaler = scaler
        
        # Extract all features
        self.features = self._extract_all_features()
        
    def _extract_all_features(self):
        features = []
        for audio in self.audio_data:
            feat = self.feature_extractor.extract_features(audio)
            features.append(feat)
        
        features = np.array(features)
        
        # Apply scaling if provided
        if self.scaler is not None:
            features = self.scaler.transform(features)
            
        return features
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.features[idx]), torch.LongTensor([self.labels[idx]])[0]
```

### MFCC Model Architecture
```python
class MFCCClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], num_classes=10, dropout=0.3):
        super(MFCCClassifier, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

# Initialize MFCC pipeline
def setup_mfcc_pipeline(data_splits):
    # Feature extraction and scaling
    scaler = StandardScaler()
    train_features = mfcc_extractor.extract_batch_features(data_splits['X_train'])
    scaler.fit(train_features)
    
    # Create datasets
    train_dataset = MFCCDataset(data_splits['X_train'], data_splits['y_train'], mfcc_extractor, scaler)
    val_dataset = MFCCDataset(data_splits['X_val'], data_splits['y_val'], mfcc_extractor, scaler)
    test_dataset = MFCCDataset(data_splits['X_test'], data_splits['y_test'], mfcc_extractor, scaler)
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Model
    input_dim = train_features.shape[1]  # 39*4 = 156 features
    model = MFCCClassifier(input_dim=input_dim).to(device)
    
    return model, train_loader, val_loader, test_loader, scaler
```

---

## Pipeline 2: Mel Spectrogram + 2D CNN

### Mel Spectrogram Feature Extraction
```python
class MelSpectrogramExtractor:
    def __init__(self, sample_rate=8000, n_mels=64, n_fft=512, hop_length=160):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        # PyTorch transforms for GPU acceleration
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=0,
            f_max=sample_rate//2
        ).to(device)
        
        self.amplitude_to_db = T.AmplitudeToDB().to(device)
        
    def extract_features(self, audio_array):
        """Extract mel spectrogram features"""
        # Convert to tensor and move to device
        audio_tensor = torch.FloatTensor(audio_array).unsqueeze(0).to(device)
        
        # Extract mel spectrogram
        mel_spec = self.mel_transform(audio_tensor)
        
        # Convert to dB scale
        mel_spec_db = self.amplitude_to_db(mel_spec)
        
        # Normalize
        mel_spec_normalized = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-8)
        
        return mel_spec_normalized.squeeze(0).cpu().numpy()
    
    def extract_batch_features(self, audio_batch):
        """Extract features for batch of audio"""
        # Convert batch to tensor
        audio_tensor = torch.FloatTensor(audio_batch).to(device)
        
        # Extract mel spectrograms
        mel_specs = self.mel_transform(audio_tensor)
        mel_specs_db = self.amplitude_to_db(mel_specs)
        
        # Normalize each spectrogram
        normalized_specs = []
        for spec in mel_specs_db:
            normalized = (spec - spec.mean()) / (spec.std() + 1e-8)
            normalized_specs.append(normalized)
        
        return torch.stack(normalized_specs).cpu().numpy()

mel_extractor = MelSpectrogramExtractor()
```

### Mel Spectrogram Dataset Class
```python
class MelSpectrogramDataset(Dataset):
    def __init__(self, audio_data, labels, feature_extractor):
        self.audio_data = audio_data
        self.labels = labels
        self.feature_extractor = feature_extractor
        
    def __len__(self):
        return len(self.audio_data)
    
    def __getitem__(self, idx):
        audio = self.audio_data[idx]
        label = self.labels[idx]
        
        # Extract mel spectrogram
        mel_spec = self.feature_extractor.extract_features(audio)
        
        # Add channel dimension for CNN
        mel_spec = np.expand_dims(mel_spec, axis=0)
        
        return torch.FloatTensor(mel_spec), torch.LongTensor([label])[0]
```

### 2D CNN Model Architecture
```python
class MelSpectrogramCNN(nn.Module):
    def __init__(self, input_shape=(1, 64, 51), num_classes=10):
        super(MelSpectrogramCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Pooling and dropout
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        self.dropout2d = nn.Dropout2d(0.2)
        
        # Calculate flattened dimension
        self._calculate_conv_output_size(input_shape)
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.conv_output_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
    def _calculate_conv_output_size(self, input_shape):
        # Calculate output size after conv layers
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            x = self._conv_forward(dummy_input)
            self.conv_output_size = x.view(1, -1).size(1)
    
    def _conv_forward(self, x):
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
        
        return x
    
    def forward(self, x):
        # Convolutional layers
        x = self._conv_forward(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

# Initialize Mel Spectrogram pipeline
def setup_mel_cnn_pipeline(data_splits):
    # Create datasets
    train_dataset = MelSpectrogramDataset(data_splits['X_train'], data_splits['y_train'], mel_extractor)
    val_dataset = MelSpectrogramDataset(data_splits['X_val'], data_splits['y_val'], mel_extractor)
    test_dataset = MelSpectrogramDataset(data_splits['X_test'], data_splits['y_test'], mel_extractor)
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Model
    model = MelSpectrogramCNN().to(device)
    
    return model, train_loader, val_loader, test_loader
```

---

## Pipeline 3: Raw Waveform + 1D CNN

### Raw Waveform Dataset Class
```python
class RawWaveformDataset(Dataset):
    def __init__(self, audio_data, labels, augmentation=False):
        self.audio_data = audio_data
        self.labels = labels
        self.augmentation = augmentation
        
    def _augment_audio(self, audio):
        """Simple audio augmentation"""
        if np.random.random() > 0.5:
            # Add small amount of noise
            noise = np.random.normal(0, 0.005, audio.shape)
            audio = audio + noise
            
        if np.random.random() > 0.5:
            # Time shifting
            shift = np.random.randint(-400, 400)
            audio = np.roll(audio, shift)
            
        return audio
    
    def __len__(self):
        return len(self.audio_data)
    
    def __getitem__(self, idx):
        audio = self.audio_data[idx].copy()
        label = self.labels[idx]
        
        # Apply augmentation during training
        if self.augmentation:
            audio = self._augment_audio(audio)
        
        # Add channel dimension
        audio = np.expand_dims(audio, axis=0)
        
        return torch.FloatTensor(audio), torch.LongTensor([label])[0]
```

### 1D CNN Model Architecture
```python
class RawWaveformCNN(nn.Module):
    def __init__(self, input_length=8000, num_classes=10):
        super(RawWaveformCNN, self).__init__()
        
        # 1D Convolutional layers
        self.conv1 = nn.Conv1d(1, 64, kernel_size=80, stride=4, padding=38)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(256)
        self.bn5 = nn.BatchNorm1d(256)
        
        # Pooling and dropout
        self.pool = nn.MaxPool1d(4, 4)
        self.dropout = nn.Dropout(0.3)
        self.dropout1d = nn.Dropout1d(0.2)
        
        # Calculate flattened dimension
        self._calculate_conv_output_size(input_length)
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.conv_output_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
    def _calculate_conv_output_size(self, input_length):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_length)
            x = self._conv_forward(dummy_input)
            self.conv_output_size = x.view(1, -1).size(1)
    
    def _conv_forward(self, x):
        # Conv block 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout1d(x)
        
        # Conv block 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout1d(x)
        
        # Conv block 3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout1d(x)
        
        # Conv block 4
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.dropout1d(x)
        
        # Conv block 5
        x = self.pool(F.relu(self.bn5(self.conv5(x))))
        x = self.dropout1d(x)
        
        return x
    
    def forward(self, x):
        # Convolutional layers
        x = self._conv_forward(x)
        
        # Global average pooling
        x = F.adaptive_avg_pool1d(x, 1)
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

# Initialize Raw Waveform pipeline
def setup_raw_waveform_pipeline(data_splits):
    # Create datasets
    train_dataset = RawWaveformDataset(data_splits['X_train'], data_splits['y_train'], augmentation=True)
    val_dataset = RawWaveformDataset(data_splits['X_val'], data_splits['y_val'], augmentation=False)
    test_dataset = RawWaveformDataset(data_splits['X_test'], data_splits['y_test'], augmentation=False)
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Model
    model = RawWaveformCNN().to(device)
    
    return model, train_loader, val_loader, test_loader
```

---

## Training and Evaluation Framework

### Training Function
```python
class ModelTrainer:
    def __init__(self, model, train_loader, val_loader, test_loader, num_epochs=50):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.num_epochs = num_epochs
        
        # Training setup
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # History tracking
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(device), target.to(device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(device), target.to(device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def train(self):
        best_val_acc = 0
        
        for epoch in range(self.num_epochs):
            # Training
            train_loss, train_acc = self.train_epoch()
            
            # Validation
            val_loss, val_acc = self.validate()
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), 'best_model.pth')
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Print progress
            if epoch % 5 == 0:
                print(f'Epoch {epoch}/{self.num_epochs}:')
                print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
                print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
                print()
        
        return self.history
    
    def evaluate_test(self):
        """Evaluate on test set"""
        self.model.load_state_dict(torch.load('best_model.pth'))
        self.model.eval()
        
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(device), target.to(device)
                output = self.model(data)
                pred = output.argmax(dim=1)
                
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_preds)
        report = classification_report(all_targets, all_preds)
        
        return accuracy, report, all_targets, all_preds
```

### Evaluation and Visualization
```python
def plot_training_history(history, title="Training History"):
    """Plot training and validation metrics"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(history['train_loss'], label='Training Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_title(f'{title} - Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(history['train_acc'], label='Training Accuracy')
    ax2.plot(history['val_acc'], label='Validation Accuracy')
    ax2.set_title(f'{title} - Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix')
    plt.show()

def compare_models(results_dict):
    """Compare multiple model results"""
    models = list(results_dict.keys())
    accuracies = [results_dict[model]['accuracy'] for model in models]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, accuracies, color=['skyblue', 'lightgreen', 'lightcoral'])
    plt.title('Model Comparison - Test Accuracy')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 100)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{acc:.2f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
```

---

## Model Training and Comparison

### Training All Pipelines
```python
def train_all_pipelines(data_splits):
    """Train and evaluate all three pipelines"""
    results = {}
    
    # Pipeline 1: MFCC + Dense NN
    print("Training MFCC + Dense Neural Network...")
    model1, train_loader1, val_loader1, test_loader1, scaler = setup_mfcc_pipeline(data_splits)
    trainer1 = ModelTrainer(model1, train_loader1, val_loader1, test_loader1, num_epochs=30)
    history1 = trainer1.train()
    acc1, report1, y_true1, y_pred1 = trainer1.evaluate_test()
    
    results['MFCC + Dense NN'] = {
        'model': model1,
        'history': history1,
        'accuracy': acc1 * 100,
        'report': report1,
        'predictions': (y_true1, y_pred1)
    }
    
    # Pipeline 2: Mel Spectrogram + 2D CNN
    print("Training Mel Spectrogram + 2D CNN...")
    model2, train_loader2, val_loader2, test_loader2 = setup_mel_cnn_pipeline(data_splits)
    trainer2 = ModelTrainer(model2, train_loader2, val_loader2, test_loader2, num_epochs=40)
    history2 = trainer2.train()
    acc2, report2, y_true2, y_pred2 = trainer2.evaluate_test()
    
    results['Mel Spectrogram + 2D CNN'] = {
        'model': model2,
        'history': history2,
        'accuracy': acc2 * 100,
        'report': report2,
        'predictions': (y_true2, y_pred2)
    }
    
    # Pipeline 3: Raw Waveform + 1D CNN
    print("Training Raw Waveform + 1D CNN...")
    model3, train_loader3, val_loader3, test_loader3 = setup_raw_waveform_pipeline(data_splits)
    trainer3 = ModelTrainer(model3, train_loader3, val_loader3, test_loader3, num_epochs=50)
    history3 = trainer3.train()
    acc3, report3, y_true3, y_pred3 = trainer3.evaluate_test()
    
    results['Raw Waveform + 1D CNN'] = {
        'model': model3,
        'history': history3,
        'accuracy': acc3 * 100,
        'report': report3,
        'predictions': (y_true3, y_pred3)
    }
    
    return results

# Run training
if __name__ == "__main__":
    # Load and prepare data
    data_loader = DigitDatasetLoader()
    dataset = data_loader.load_fsdd_dataset()
    data_splits = data_loader.create_train_test_split(dataset)
    
    # Train all models
    results = train_all_pipelines(data_splits)
    
    # Display results
    print("\n" + "="*50)
    print("FINAL RESULTS SUMMARY")
    print("="*50)
    
    for model_name, result in results.items():
        print(f"\n{model_name}:")
        print(f"  Test Accuracy: {result['accuracy']:.2f}%")
        print(f"  Classification Report:")
        print(result['report'])
    
    # Plot comparisons
    compare_models(results)
    
    # Plot individual training histories
    for model_name, result in results.items():
        plot_training_history(result['history'], model_name)
        
        # Plot confusion matrix
        y_true, y_pred = result['predictions']
        class_names = [str(i) for i in range(10)]
        plot_confusion_matrix(y_true, y_pred, class_names)
```

---

## Inference and Real-time Prediction

### Real-time Inference Class
```python
class DigitClassifier:
    def __init__(self, model_path, pipeline_type, scaler=None):
        self.pipeline_type = pipeline_type
        self.scaler = scaler
        
        # Load appropriate model
        if pipeline_type == 'MFCC':
            self.model = MFCCClassifier(input_dim=156)
            self.feature_extractor = MFCCFeatureExtractor()
        elif pipeline_type == 'MelSpectrogram':
            self.model = MelSpectrogramCNN()
            self.feature_extractor = MelSpectrogramExtractor()
        elif pipeline_type == 'RawWaveform':
            self.model = RawWaveformCNN()
            self.feature_extractor = None
        
        # Load trained weights
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()
        
        self.data_loader = DigitDatasetLoader()
        
    def predict(self, audio_path_or_array):
        """Predict digit from audio file or array"""
        # Load audio if path provided
        if isinstance(audio_path_or_array, str):
            audio, sr = librosa.load(audio_path_or_array, sr=8000)
        else:
            audio = audio_path_or_array
            
        # Preprocess audio
        processed_audio = self.data_loader.preprocess_audio(audio, 8000)
        
        with torch.no_grad():
            if self.pipeline_type == 'MFCC':
                # Extract MFCC features
                features = self.feature_extractor.extract_features(processed_audio)
                if self.scaler:
                    features = self.scaler.transform([features])[0]
                
                # Predict
                input_tensor = torch.FloatTensor(features).unsqueeze(0).to(device)
                output = self.model(input_tensor)
                
            elif self.pipeline_type == 'MelSpectrogram':
                # Extract mel spectrogram
                mel_spec = self.feature_extractor.extract_features(processed_audio)
                input_tensor = torch.FloatTensor(mel_spec).unsqueeze(0).unsqueeze(0).to(device)
                output = self.model(input_tensor)
                
            elif self.pipeline_type == 'RawWaveform':
                # Use raw waveform
                input_tensor = torch.FloatTensor(processed_audio).unsqueeze(0).unsqueeze(0).to(device)
                output = self.model(input_tensor)
            
            # Get prediction
            probabilities = F.softmax(output, dim=1)
            predicted_class = output.argmax(dim=1).item()
            confidence = probabilities.max().item()
            
        return predicted_class, confidence, probabilities.squeeze().cpu().numpy()

# Example usage
# classifier = DigitClassifier('best_model.pth', 'MelSpectrogram')
# prediction, confidence, probs = classifier.predict('test_audio.wav')
# print(f"Predicted digit: {prediction} (confidence: {confidence:.3f})")
```

### Performance Benchmarking
```python
import time

def benchmark_inference_speed(classifier, test_audio_paths, num_runs=100):
    """Benchmark inference speed"""
    times = []
    
    for _ in range(num_runs):
        audio_path = np.random.choice(test_audio_paths)
        
        start_time = time.time()
        prediction, confidence, _ = classifier.predict(audio_path)
        end_time = time.time()
        
        times.append(end_time - start_time)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    print(f"Average inference time: {avg_time*1000:.2f} ± {std_time*1000:.2f} ms")
    print(f"Throughput: {1/avg_time:.1f} predictions/second")
    
    return avg_time, std_time

# Benchmark all models
def benchmark_all_models(test_audio_paths):
    """Benchmark all trained models"""
    model_types = ['MFCC', 'MelSpectrogram', 'RawWaveform']
    
    for model_type in model_types:
        print(f"\nBenchmarking {model_type} model:")
        classifier = DigitClassifier(f'best_model_{model_type.lower()}.pth', model_type)
        benchmark_inference_speed(classifier, test_audio_paths)
```

---

## Expected Performance Summary

| Pipeline | Expected Accuracy | Inference Speed | Model Size | Memory Usage |
|----------|------------------|----------------|------------|--------------|
| MFCC + Dense NN | 93-95% | ~1-2ms | <1MB | Low |
| Mel Spectrogram + 2D CNN | 95-97% | ~3-5ms | ~5MB | Medium |
| Raw Waveform + 1D CNN | 94-96% | ~5-8ms | ~10MB | Medium-High |

### Key Implementation Notes

1. **MFCC Pipeline**: Fastest inference, most interpretable features, lowest memory usage
2. **Mel Spectrogram Pipeline**: Best balance of accuracy and speed, proven architecture  
3. **Raw Waveform Pipeline**: Most modern approach, automatic feature learning, highest complexity

### Next Steps for Deployment

1. **Model Optimization**: Quantization, pruning, and ONNX conversion for production
2. **Real-time Integration**: Microphone input with streaming audio processing
3. **Web Interface**: Flask/FastAPI backend with JavaScript frontend
4. **Mobile Deployment**: TensorFlow Lite or PyTorch Mobile conversion

This template provides a comprehensive foundation for building and comparing all three audio classification pipelines, with production-ready training, evaluation, and inference capabilities.