# Railway Deployment Guide

This guide covers deploying the Audio Digit Classifier to Railway with only the necessary components for production.

## Deployment Configuration

### Files Configured for Production

1. **Procfile** - Railway deployment configuration
2. **requirements.txt** - Streamlined dependencies (ML models + VAD only)
3. **.railwayignore** - Excludes unnecessary files and training modules

### What's Included in Production

✅ **Core Components:**
- Flask web application (`app.py`)
- 3 trained ML models (MFCC, Mel CNN, Raw CNN)
- VAD (Voice Activity Detection) functionality
- Essential audio processing utilities

✅ **ML Models & Inference:**
- `models/mfcc_classifier/` - Best performing model (98.52% accuracy)
- `models/mel_cnn_classifier/` - Good performance (97.22% accuracy) 
- `models/raw_cnn_classifier/` - Fair performance (91.30% accuracy)
- `ml_training/inference.py` - Model loading and prediction
- `ml_training/pipelines/` - Model architectures
- `ml_training/data/dataset_loader.py` - Audio preprocessing

✅ **Audio Processors:**
- `audio_processors/ml_*.py` - ML model processors
- `audio_processors/base_processor.py` - Base class
- Legacy processors for comparison (MFCC, Mel Spectrogram, Raw Spectrogram)

✅ **Utilities:**
- `utils/webrtc_vad.py` - WebRTC VAD implementation
- `utils/enhanced_vad.py` - Enhanced VAD with ffmpeg support
- `utils/audio_utils.py` - Audio format conversion
- `utils/logging_utils.py` - Performance logging
- `utils/session_manager.py` - Audio session management
- `utils/noise_utils.py` - Noise injection for testing

### What's Excluded from Production

❌ **Excluded Components:**
- Wav2Vec2 processor (requires heavy transformers library)
- Whisper processors (external API dependencies)
- External API processors (require API keys)
- ML training scripts (`ml_training/train.py`, `ml_training/demo.py`)
- Training logs and development files
- Documentation files (README.md, docs/)

## Railway Deployment Steps

### 1. Connect Repository
```bash
# Connect your GitHub repository to Railway
# Railway will automatically detect the Procfile and requirements.txt
```

### 2. Environment Variables
Set these environment variables in Railway dashboard:
```bash
FLASK_ENV=production
SECRET_KEY=your-secret-key-here
FLASK_PORT=5000  # Railway will override with $PORT
MAX_AUDIO_DURATION=10
MAX_FILE_SIZE=10485760  # 10MB
```

### 3. Model Files
Ensure your trained model files are present in the repository:
```
models/
├── mfcc_classifier/
│   ├── best_model.pt
│   └── scaler.pkl
├── mel_cnn_classifier/
│   └── best_model.pt
└── raw_cnn_classifier/
    └── best_model.pt
```

### 4. Deploy
```bash
# Railway will automatically:
# 1. Install dependencies from requirements.txt
# 2. Run the Procfile command
# 3. Start the application on the assigned port
```

## Performance Optimizations

### Resource Usage
- **CPU**: Models run efficiently on CPU (1-8ms inference time)
- **Memory**: ~200-500MB for all 3 models loaded
- **Storage**: ~26MB total deployment (reduced from 340MB+)
  - 7.5MB - Mel CNN model
  - 7.4MB - Raw CNN model  
  - 1.0MB - MFCC model + scaler
  - 10MB - Dependencies and code

### Production Features
- **Gunicorn** WSGI server with 2 workers
- **Request timeout**: 120 seconds
- **Max requests per worker**: 1000 (auto-restart)
- **Preload application** for faster startup
- **Access and error logging** to stdout

## Testing the Deployment

### Health Check
```bash
curl https://your-app.railway.app/health
```

### API Endpoints
```bash
# Process audio file
curl -X POST https://your-app.railway.app/process_audio \
  -F "audio=@test_audio.wav" \
  -F "method=ml_mfcc"

# Get processor status
curl https://your-app.railway.app/processor_info/ml_mfcc
```

### Available Methods
- `ml_mfcc` - Best accuracy (98.52%)
- `ml_mel_cnn` - Good accuracy (97.22%)
- `ml_raw_cnn` - Fair accuracy (91.30%)
- `mfcc` - Legacy MFCC processor
- `mel_spectrogram` - Legacy Mel spectrogram processor
- `raw_spectrogram` - Legacy raw spectrogram processor

## Troubleshooting

### Common Issues

1. **Model files not found**
   - Ensure model files are committed to git
   - Check .railwayignore doesn't exclude model files

2. **Memory issues**
   - Railway free tier has 512MB memory limit
   - Consider reducing to 1-2 models if needed

3. **Slow startup**
   - Models take 10-30 seconds to load on cold start
   - Preload option in Procfile helps reduce this

### Logs
```bash
# View Railway logs
railway logs --tail
```

## Scaling Considerations

For production workloads:
- Increase worker count in Procfile
- Add Redis for session storage
- Consider model optimization (quantization)
- Implement request queuing for high loads

## Security Notes

- Set proper SECRET_KEY in production
- Enable HTTPS (Railway provides this automatically)
- Consider rate limiting for public APIs
- Validate file uploads thoroughly
