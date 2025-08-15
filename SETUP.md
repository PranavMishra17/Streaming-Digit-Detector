# 🎮 Audio Digit Classifier - Setup Guide

## Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment (Optional)**
   ```bash
   cp .env.example .env
   # Edit .env and add your Hugging Face token for External API
   ```

3. **Run the Application**
   ```bash
   python app.py
   ```

4. **Open in Browser**
   Navigate to: http://localhost:5000

## Features Overview

### 🎯 4 Processing Methods
- **External API**: Hugging Face Whisper (working - needs API token)
- **Raw Spectrogram**: STFT-based processing (placeholder)
- **Mel Spectrogram**: Mel-scale analysis (placeholder)
- **MFCC Features**: Cepstral coefficients (placeholder)

### 🎮 Retro UI Features
- **8-bit pixel art styling** with Press Start 2P font
- **Real-time audio visualization** with retro scanlines
- **Cabinet-style method selection** like arcade games
- **Performance monitoring** and statistics tracking
- **Noise injection testing** for robustness evaluation

### 🎤 Audio Controls
- **Space bar** or button to start/stop recording
- **Auto-silence detection** for hands-free operation
- **Real-time waveform** with level meters
- **Noise injection** (White, Pink, Brown, Background)

## Usage Instructions

### Recording Audio
1. Click "🎤 Start Recording" or press **SPACE**
2. Clearly say a digit (0-9) into your microphone
3. Recording will auto-stop after 1.5 seconds of silence
4. Or manually click "⏹️ Stop Recording"

### Processing Methods
1. Select a processing method from the cabinet-style UI
2. Click "🧠 Analyze Audio" to process your recording
3. View results: predicted digit, inference time, accuracy stats

### Noise Testing
1. Adjust noise type (White, Pink, Brown, Background)
2. Set noise level using the slider (0.0 to 0.5)
3. Record and process audio to test robustness

### Performance Monitoring
- View real-time statistics in the Performance Monitor cabinet
- Click "📊 View Stats" for detailed method comparison
- Check Activity Log for detailed processing information

## Technical Architecture

### Backend (Flask)
- **Audio Processing Pipeline**: Modular processor system
- **Real-time Performance Logging**: Tracks all inference metrics
- **RESTful API**: Clean endpoints for audio processing
- **Error Handling**: Comprehensive error management

### Frontend (Vanilla JavaScript)
- **Web Audio API**: Microphone access and real-time analysis
- **Canvas Visualization**: Retro-style audio waveforms
- **Client-side Noise Generation**: Testing without server load
- **Responsive Design**: Works on desktop and mobile

### Audio Processing
- **Format Standardization**: Converts all audio to mono 16kHz WAV
- **Silence Detection**: Smart auto-stop functionality
- **Noise Injection**: Multiple noise types for robustness testing
- **Performance Metrics**: Detailed timing and accuracy tracking

## API Endpoints

### Main Routes
- `GET /` - Main application interface
- `POST /process_audio` - Process audio with selected method
- `GET /health` - Application health check
- `GET /stats` - Get performance statistics

### Processor Information
- `GET /processor_info/<method>` - Get detailed processor info
- `GET /stats/<method>` - Get method-specific statistics

### Testing Utilities
- `GET /test_audio/<digit>` - Generate test audio for a digit

## Configuration

### Environment Variables (.env)
```bash
# Hugging Face API (for External API processor)
HUGGING_FACE_TOKEN=your_token_here

# Flask Configuration
FLASK_ENV=development
FLASK_DEBUG=True
FLASK_PORT=5000

# Audio Settings
MAX_AUDIO_DURATION=10
DEFAULT_SAMPLE_RATE=16000
```

### Browser Requirements
- **Modern browser** with Web Audio API support
- **HTTPS required** for microphone access (localhost exempt)
- **Microphone permission** must be granted
- **Minimum 2GB RAM** for optimal performance

## Development

### Running Tests
```bash
python -m pytest tests/ -v
```

### Project Structure
```
├── app.py                    # Main Flask application
├── static/
│   ├── css/retro.css        # 8-bit styling
│   └── js/                  # Client-side modules
├── templates/               # HTML templates
├── audio_processors/        # Processing methods
├── utils/                   # Utilities and logging
└── tests/                   # Unit tests
```

### Adding New Processors
1. Create new processor in `audio_processors/`
2. Extend `AudioProcessor` base class
3. Implement `process_audio()` method
4. Register in `app.py` processors dict
5. Add UI cabinet in `templates/index.html`

## Troubleshooting

### Common Issues
1. **Microphone not working**: Check browser permissions
2. **External API errors**: Verify Hugging Face token in .env
3. **Audio quality issues**: Ensure quiet environment
4. **Performance problems**: Close other audio applications

### Browser Compatibility
- ✅ **Chrome 66+** (Recommended)
- ✅ **Firefox 60+** 
- ✅ **Safari 12+**
- ⚠️ **Edge 79+** (Some audio issues possible)

### HTTPS Requirement
For microphone access, most browsers require HTTPS. Development servers on localhost are exempt from this requirement.

## Performance Targets

- **External API**: <2s inference time
- **MFCC**: <100ms inference time (when implemented)
- **Mel Spectrogram**: <500ms inference time (when implemented)
- **Raw Spectrogram**: <1s inference time (when implemented)
- **Real-time Visualization**: 60fps

## Future Enhancements

### Planned Features
1. **Local ML Models**: Implement placeholder processors with actual models
2. **Audio Dataset Collection**: Save recordings for training
3. **Model Training Interface**: Web-based training pipeline
4. **Export Functionality**: Download results as CSV/JSON
5. **Multi-language Support**: Extend beyond English digits

### Technical Improvements
1. **WebRTC Integration**: Better audio quality
2. **Progressive Web App**: Offline functionality
3. **WebGL Visualization**: Enhanced graphics
4. **WebAssembly**: Faster client-side processing

---

## 🎮 Ready to Play?

Start the app and begin testing spoken digit recognition with retro gaming style!

```bash
python app.py
# Open http://localhost:5000
# Click 🎤 and say a digit!
```