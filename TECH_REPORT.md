# Technical Code Report - Audio Digit Classification

## Project Overview
Real-time spoken digit recognition (0-9) web application with retro gaming aesthetic. Implements multiple ML approaches for performance comparison.

## Architecture

### Core Components
- **Flask Backend** (`app.py`): Main server with processor management and API endpoints
- **Web Frontend**: Real-time audio recording and visualization interface
- **Audio Processors**: Modular ML pipeline supporting multiple classification methods
- **Session Management**: Organized audio chunk storage and session tracking

## File Structure

### Backend Core
```
app.py                      # Flask server, processor initialization, API routes
requirements.txt           # Python dependencies
```

### Audio Processing Pipeline
```
audio_processors/
├── base_processor.py       # Abstract base class for all processors
├── external_api.py         # HuggingFace Whisper API integration
├── faster_whisper_processor.py  # Optimized Whisper with VAD
├── whisper_digit_processor.py   # Local Whisper implementation  
├── wav2vec2_processor.py   # Facebook Wav2Vec2 model
├── local_whisper.py        # Direct Whisper model usage
├── raw_spectrogram.py      # STFT-based classification
├── mel_spectrogram.py      # Mel-scale feature extraction
└── mfcc_processor.py       # MFCC feature classification
```

### Utilities
```
utils/
├── audio_utils.py          # Audio format conversion, validation
├── session_manager.py      # Session tracking, chunk storage
├── webrtc_vad.py          # Voice Activity Detection
├── vad_feature_integration.py  # VAD integration utilities
├── logging_utils.py        # Performance logging, Flask setup
├── noise_utils.py          # Audio noise injection for testing
└── webm_converter.py       # WebM audio format handling
```

### Frontend
```
static/
├── js/
│   ├── main.js             # Application controller, UI coordination
│   ├── audio-recorder.js   # Microphone access, WebRTC recording
│   ├── audio-visualizer.js # Real-time waveform visualization
│   └── noise-generator.js  # Client-side noise generation
├── css/retro.css           # Custom retro pixel styling
└── fonts/                  # Pixel fonts for retro aesthetic
```

### Templates & Testing
```
templates/
├── index.html              # Main application UI
└── error.html              # Error handling page

tests/
├── test_processors.py      # Audio processor unit tests
└── test_audio_utils.py     # Audio utility function tests
```

## Data Flow

### Input Processing
1. **Audio Capture**: Browser microphone → WebRTC recording → WAV chunks
2. **Session Management**: Unique session ID → organized folder structure → chunk storage
3. **VAD Processing**: Voice activity detection → audio segmentation → noise filtering

### ML Pipeline
1. **Format Validation**: Audio format checking → conversion if needed
2. **Processor Selection**: Dynamic fallback system → best available model
3. **Classification**: Audio bytes → feature extraction → digit prediction (0-9)
4. **Performance Logging**: Inference time tracking → accuracy metrics

### Output Structure
```
output/
└── session{timestamp}_{uuid}/
    ├── session_info.json   # Session metadata, processor info
    └── chunks/
        ├── 001.wav         # Individual audio recordings
        ├── 002.wav
        └── ...
```

## Key Features

### Processor Management
- **Dynamic Initialization**: Automatic fallback if preferred processors unavailable
- **Performance Tracking**: Per-processor inference time and accuracy logging
- **Modular Design**: Easy addition of new classification methods

### Real-time Processing
- **Streaming Audio**: Continuous microphone input with visualization
- **VAD Integration**: Automatic speech detection and segmentation
- **WebRTC Support**: Cross-browser audio capture compatibility

### Session Management
- **Organized Storage**: Systematic file organization with unique session IDs
- **Metadata Tracking**: JSON session info with processor details and timing
- **Chunk Management**: Individual audio segment storage for analysis

## Dependencies

### Python Backend
- **Flask**: Web framework and API endpoints
- **librosa**: Audio feature extraction and processing
- **transformers**: HuggingFace model integration
- **webrtcvad**: Voice activity detection
- **faster-whisper**: Optimized Whisper implementation

### JavaScript Frontend
- **Web Audio API**: Native browser audio processing
- **WebRTC**: Real-time audio capture
- **Canvas API**: Audio visualization rendering
- **No external JS frameworks**: Vanilla JavaScript implementation

## Performance Characteristics
- **External API**: ~2s inference (network dependent)
- **Faster-Whisper**: ~100-500ms local inference
- **MFCC**: <100ms feature-based classification
- **Real-time Visualization**: 60fps audio waveform rendering