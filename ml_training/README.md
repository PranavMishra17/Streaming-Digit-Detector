# ML Training Pipeline for Digit Classification

Comprehensive machine learning training system for spoken digit recognition (0-9) with three different approaches: MFCC+Dense Neural Network, Mel Spectrogram+2D CNN, and Raw Waveform+1D CNN.

## 🏗️ Architecture Overview

```
ml_training/
├── data/
│   ├── dataset_loader.py          # Dataset loading and preprocessing
│   └── __init__.py
├── pipelines/
│   ├── mfcc_pipeline.py           # MFCC + Dense NN pipeline
│   ├── mel_cnn_pipeline.py        # Mel Spectrogram + 2D CNN pipeline
│   ├── raw_cnn_pipeline.py        # Raw Waveform + 1D CNN pipeline
│   └── __init__.py
├── utils/
│   ├── logging_utils.py           # Comprehensive logging system
│   ├── visualization.py          # Training visualization and reports
│   └── __init__.py
├── train.py                       # Main training orchestration
├── inference.py                   # Model inference and prediction
└── README.md                      # This file
```

## 📊 Pipeline Comparison

| Pipeline | Features | Model Type | Expected Accuracy | Speed |
|----------|----------|------------|-------------------|-------|
| MFCC + Dense NN | Hand-crafted (156D) | Fully Connected | 93-95% | ~1-2ms |
| Mel Spectrogram + 2D CNN | Mel Spectrogram (64×51) | 2D Convolution | 95-97% | ~3-5ms |
| Raw Waveform + 1D CNN | Raw Audio (8000 samples) | 1D Convolution | 94-96% | ~5-8ms |

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install torch torchvision torchaudio librosa soundfile scikit-learn
pip install datasets transformers matplotlib seaborn plotly pandas
```

### 2. Train All Models

```bash
# Train all three pipelines with comparison
python ml_training/train.py --pipeline all --epochs 50 --batch_size 32

# Train specific pipeline
python ml_training/train.py --pipeline mfcc --epochs 30 --batch_size 32
```

### 3. Use Trained Models

```python
from ml_training.inference import load_classifier

# Load trained model
classifier = load_classifier("models", "mfcc")

# Make prediction
result = classifier.predict("path/to/audio.wav")
print(f"Predicted digit: {result['predicted_digit']}")
print(f"Confidence: {result['confidence']:.3f}")
```

## 📚 Detailed Usage

### Training Configuration

Create a training configuration file:

```json
{
    "pipelines": ["mfcc", "mel_cnn", "raw_cnn"],
    "num_epochs": 50,
    "batch_size": 32,
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "scheduler_type": "plateau",
    "early_stopping_patience": 10,
    "gradient_clipping": 1.0,
    "save_frequency": 10,
    "sample_rate": 8000,
    "max_length": 8000
}
```

Run with configuration:
```bash
python ml_training/train.py --config training_config.json
```

### Data Flow and Dimensions

#### MFCC Pipeline
```
Raw Audio (8000,) → MFCC (13,T) → Delta (13,T) → Delta² (13,T) 
→ Statistics (156,) → Dense NN → Predictions (10,)
```

#### Mel CNN Pipeline  
```
Raw Audio (8000,) → Mel Spectrogram (64,51) → 2D CNN → Predictions (10,)
```

#### Raw CNN Pipeline
```
Raw Audio (8000,) → 1D CNN → Predictions (10,)
```

### Advanced Training Features

- **Automatic Mixed Precision (AMP)**: Faster training on modern GPUs
- **Early Stopping**: Prevents overfitting with configurable patience
- **Learning Rate Scheduling**: Plateau, step, and cosine annealing schedulers
- **Comprehensive Checkpointing**: Save/resume training at any point
- **Extensive Logging**: Structured logging with performance metrics
- **Real-time Visualization**: Training curves, confusion matrices, model comparison

### Model Architectures

#### MFCC Classifier
```
Input (156) → Dense(256) → BatchNorm → ReLU → Dropout(0.3)
           → Dense(128) → BatchNorm → ReLU → Dropout(0.3)
           → Dense(64)  → BatchNorm → ReLU → Dropout(0.3)
           → Dense(10)
```

#### Mel CNN
```
Input (1,64,51) → Conv2D(32) → BN → ReLU → MaxPool → Dropout2D
                → Conv2D(64) → BN → ReLU → MaxPool → Dropout2D
                → Conv2D(128) → BN → ReLU → MaxPool → Dropout2D
                → Conv2D(256) → BN → ReLU → MaxPool → Dropout2D
                → GlobalAvgPool → Dense(512) → Dense(256) → Dense(10)
```

#### Raw CNN  
```
Input (1,8000) → Conv1D(64,k=80,s=4) → BN → ReLU → MaxPool → Dropout1D
               → Conv1D(128,k=3) → BN → ReLU → MaxPool → Dropout1D
               → Conv1D(128,k=3) → BN → ReLU → MaxPool → Dropout1D
               → Conv1D(256,k=3) → BN → ReLU → MaxPool → Dropout1D
               → Conv1D(256,k=3) → BN → ReLU → MaxPool → Dropout1D
               → GlobalAvgPool → Dense(512) → Dense(256) → Dense(10)
```

## 🔧 Advanced Usage

### Custom Pipeline Development

```python
from ml_training.pipelines.mfcc_pipeline import MFCCFeatureExtractor, MFCCDataset, MFCCClassifier
from ml_training.utils.logging_utils import setup_training_logger

# Custom feature extraction
extractor = MFCCFeatureExtractor(n_mfcc=20, n_fft=1024)

# Custom model architecture
model = MFCCClassifier(
    input_dim=240,  # 20 MFCC * 3 types * 4 stats
    hidden_dims=[512, 256, 128],
    dropout=0.4
)
```

### Batch Inference

```python
classifier = load_classifier("models", "mel_cnn")

# Process multiple files
audio_files = ["digit1.wav", "digit2.wav", "digit3.wav"]
results = classifier.batch_predict(audio_files, batch_size=16)

for i, result in enumerate(results):
    print(f"File {audio_files[i]}: {result['predicted_digit']} ({result['confidence']:.3f})")
```

### Performance Benchmarking

```python
# Speed benchmark
classifier = load_classifier("models", "raw_cnn")
benchmark_results = classifier.benchmark_speed(num_samples=1000)

print(f"Average inference time: {benchmark_results['average_time_per_sample']*1000:.2f} ms")
print(f"Throughput: {benchmark_results['throughput_per_second']:.1f} samples/second")
```

## 📈 Output Structure

### Training Outputs
```
models/
├── mfcc_classifier/
│   ├── best_model.pt              # Best model checkpoint
│   ├── checkpoint_epoch_*.pt      # Regular checkpoints
│   └── scaler.pkl                 # Feature scaler (MFCC only)
├── mel_cnn_classifier/
└── raw_cnn_classifier/

train_logs/
├── mfcc_classifier_*.log          # Training logs
├── mel_cnn_classifier_*.log
├── raw_cnn_classifier_*.log
├── metrics_summary_*.json         # Performance metrics
├── config_*.json                  # Training configurations
└── plots/                         # Visualization outputs
    ├── training_history_*.png
    ├── confusion_matrix_*.png
    ├── model_comparison.png
    └── training_report.html
```

### Log Structure
```json
{
    "experiment_timestamp": "2023-12-01T10:30:00",
    "training_metrics": {
        "epoch_1": {
            "train_loss": 0.8234,
            "train_acc": 72.45,
            "val_loss": 0.7891,
            "val_acc": 75.23,
            "lr": 0.001
        }
    },
    "timers": {
        "train_epoch_1": 45.67,
        "val_epoch_1": 12.34
    },
    "total_experiment_time": 1234.56
}
```

## 🎯 Key Features

- **Modular Design**: Easy to extend with new architectures
- **Comprehensive Logging**: Every aspect of training is logged
- **Error Recovery**: Graceful error handling and fallback mechanisms  
- **Memory Efficient**: Optimized data loading and batch processing
- **GPU Acceleration**: Full CUDA support with mixed precision
- **Reproducible**: Deterministic training with seed control
- **Production Ready**: Inference optimization and deployment utilities

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size
   python ml_training/train.py --batch_size 16
   ```

2. **Dataset Loading Issues**
   ```python
   # Check dataset availability
   from ml_training.data.dataset_loader import load_and_prepare_data
   data = load_and_prepare_data()
   print(data['dataset_info'] if data else "Failed to load")
   ```

3. **Model Loading Errors**
   ```python
   # Verify model files
   from pathlib import Path
   model_dir = Path("models/mfcc_classifier")
   print(f"Model exists: {(model_dir / 'best_model.pt').exists()}")
   ```

### Performance Optimization

- **Mixed Precision**: Automatically enabled on compatible GPUs
- **Data Loading**: Use `num_workers > 0` for faster data loading
- **Batch Size**: Optimize based on available GPU memory
- **Feature Precomputation**: Enabled by default for faster training

## 📋 Requirements

- Python 3.8+
- PyTorch 1.12+
- librosa 0.9+
- scikit-learn 1.1+
- datasets (HuggingFace)
- matplotlib, seaborn, plotly
- torchaudio

## 🔄 Integration with Main App

The trained models can be integrated with the main Flask application by replacing the existing audio processors:

```python
# In your Flask app
from ml_training.inference import load_classifier

class MLDigitProcessor(AudioProcessor):
    def __init__(self, pipeline_type="mfcc"):
        self.classifier = load_classifier("models", pipeline_type)
    
    def process_audio(self, audio_data):
        result = self.classifier.predict(audio_data)
        return str(result['predicted_digit'])
```

This ML training pipeline provides a complete, production-ready system for training and deploying high-performance digit classification models with comprehensive monitoring, visualization, and error handling capabilities.