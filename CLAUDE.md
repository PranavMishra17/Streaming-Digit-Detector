# Audio Digit Classification Web App

## Project Overview
Lightweight prototype for spoken digit recognition (0-9) with retro gaming aesthetic. Compares 4 ML approaches: External API, Raw Spectrogram, Mel Spectrogram, and MFCC. Built with Flask backend, HTML/JS frontend, real-time audio processing.

## Architecture
- **Backend**: Flask Python server with audio processing endpoints
- **Frontend**: Retro pixelated UI using NES.css framework
- **Audio**: Web Audio API for microphone input and real-time visualization
- **ML Pipeline**: Modular cabinet system for different classification approaches

## Development Guidelines

### Code Standards
- Use ES6+ modules (import/export), not CommonJS (require)
- Python: PEP8 compliance, type hints where appropriate
- Destructure imports: `import { getUserMedia } from 'mediaDevices'`
- Async/await for all asynchronous operations
- Clear variable names: `audioContext`, `melSpectrogram`, not `ctx`, `ms`

### File Structure
```
/
├── app.py                    # Flask main application
├── static/
│   ├── css/
│   │   ├── retro.css        # Custom retro styling
│   │   └── nes.min.css      # NES.css framework
│   ├── js/
│   │   ├── audio-recorder.js # Microphone handling
│   │   ├── audio-visualizer.js # Real-time waveform
│   │   ├── noise-generator.js  # Audio noise injection
│   │   └── main.js          # App coordination
│   └── fonts/               # Pixel fonts
├── templates/
│   └── index.html           # Main UI template
├── audio_processors/
│   ├── external_api.py      # Hugging Face Whisper integration
│   ├── raw_spectrogram.py   # STFT-based processing
│   ├── mel_spectrogram.py   # Mel-scale processing  
│   ├── mfcc_processor.py    # MFCC feature extraction
│   └── base_processor.py    # Abstract base class
├── utils/
│   ├── audio_utils.py       # Audio format conversion utilities
│   ├── logging_utils.py     # Performance logging
│   └── noise_utils.py       # Noise injection utilities
└── requirements.txt
```

### Error Handling
- Graceful microphone permission failures with user-friendly messages
- Network timeout handling for external APIs
- Audio format validation before processing
- Clear error states in UI with retry mechanisms
- Comprehensive logging for debugging

### Debugging & Logging
- Use structured logging with timestamps and inference metrics
- Log all audio processing times and model predictions
- Console debugging for audio visualization issues
- Performance monitoring for each classification method
- Error tracking with stack traces in development

### Testing Strategy
- Unit tests for each audio processor module
- Mock audio data for testing without microphone
- Cross-browser compatibility testing (Chrome, Firefox, Safari)
- Mobile device testing for touch interactions
- Performance benchmarks for inference times

## Important Commands

### Development
- `python app.py` - Start Flask development server (port 5000)
- `python -m pytest tests/` - Run test suite
- `python -m flask run --debug` - Debug mode with auto-reload

### Audio Processing
- Test individual processors: `python -m audio_processors.mfcc_processor test_audio.wav`
- Validate audio format: `python -m utils.audio_utils validate input.wav`
- Generate test noise: `python -m utils.noise_utils --type white --duration 5`

### Frontend
- Serve static files through Flask, no separate build process
- Use browser dev tools for audio context debugging
- Test microphone access with HTTPS (required for getUserMedia)

## Implementation Priority
1. **Phase 1**: Basic UI with retro styling and microphone visualization
2. **Phase 2**: External API integration (Hugging Face Whisper)
3. **Phase 3**: Local processing methods (Raw, Mel, MFCC) 
4. **Phase 4**: Performance comparison and noise robustness testing

## Browser Requirements
- Modern browsers with Web Audio API support
- HTTPS required for microphone access
- Minimum 2GB RAM for local audio processing
- Tested primarily on Chrome/Firefox desktop

## Performance Targets
- External API: <2s inference time
- MFCC: <100ms inference time
- Mel Spectrogram: <500ms inference time  
- Raw Spectrogram: <1s inference time
- Real-time audio visualization: 60fps

## Security Considerations
- No audio data stored on server
- Client-side processing preferred when possible
- API keys stored in environment variables
- CORS policy for external API calls
- User consent for microphone access

## Dependencies
- **Python**: Flask, numpy, librosa, scipy, requests, transformers
- **JavaScript**: No external dependencies (vanilla JS + Web Audio API)
- **CSS**: NES.css framework for retro styling
- **Fonts**: Press Start 2P, Pixel, other 8-bit fonts