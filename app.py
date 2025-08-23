"""
Audio Digit Classification API for Hugging Face Spaces
Backend API for spoken digit recognition (0-9) - HF Spaces deployment
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import time
import logging
from typing import Dict, Any, Optional
from dotenv import load_dotenv
import numpy as np

# Import audio processors (only essential ones for deployment)
from audio_processors.external_api import ExternalAPIProcessor
from audio_processors.whisper_digit_processor import WhisperDigitProcessor
from audio_processors.ml_mfcc_processor import MLMFCCProcessor
from audio_processors.ml_mel_cnn_processor import MLMelCNNProcessor
from audio_processors.ml_raw_cnn_processor import MLRawCNNProcessor

# Import utilities
from utils.audio_utils import validate_audio_format, convert_audio_format, get_audio_duration, convert_for_ml_models
from utils.logging_utils import performance_logger, setup_flask_logging

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'hf_spaces_deployment_key')

# Enable CORS for frontend requests from Vercel
CORS(app, origins=['*'])  # In production, specify your Vercel domain

# Setup logging
setup_flask_logging(app)

# Configuration for HF Spaces
MAX_AUDIO_DURATION = 10  # seconds
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'm4a', 'webm'}

def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Global processor cache for model persistence
_processor_cache = {}

def initialize_processors():
    """Initialize audio processors optimized for HF Spaces deployment with caching."""
    global _processor_cache
    
    # Return cached processors if already initialized
    if _processor_cache:
        app.logger.info(f"Using cached processors: {len(_processor_cache)} available")
        return _processor_cache
    
    procs = {}
    
    # ML-trained processors (high priority - use best models only)
    ml_processors = [
        ('ml_mfcc', MLMFCCProcessor, 'ML MFCC + Dense NN (Best - 98.52%)'),
        ('ml_mel_cnn', MLMelCNNProcessor, 'ML Mel CNN (Good - 97.22%)'),
        ('ml_raw_cnn', MLRawCNNProcessor, 'ML Raw CNN (Fair - 91.30%)')
    ]
    
    ml_working_count = 0
    for proc_key, proc_class, proc_name in ml_processors:
        try:
            # Initialize once and cache
            app.logger.info(f"Loading {proc_name}...")
            processor = proc_class()
            if processor.is_configured():
                procs[proc_key] = processor
                ml_working_count += 1
                app.logger.info(f"[OK] {proc_name} loaded successfully (cached)")
            else:
                app.logger.warning(f"[WARN] {proc_name} not configured (model files missing)")
        except Exception as e:
            app.logger.error(f"[FAIL] Failed to initialize {proc_name}: {str(e)}")
    
    # External API processor as fallback
    try:
        external_processor = ExternalAPIProcessor()
        if external_processor.is_configured():
            procs['external_api'] = external_processor
            app.logger.info("[OK] External API processor initialized (cached)")
        else:
            app.logger.warning("[WARN] External API not configured")
    except Exception as e:
        app.logger.error(f"[FAIL] Failed to initialize External API: {str(e)}")
    
    # Whisper digit processor as another fallback
    try:
        whisper_processor = WhisperDigitProcessor()
        if whisper_processor.is_configured():
            procs['whisper_digit'] = whisper_processor
            app.logger.info("[OK] Whisper digit processor initialized (cached)")
    except Exception as e:
        app.logger.error(f"[FAIL] Failed to initialize Whisper: {str(e)}")
    
    # Cache the processors globally
    _processor_cache = procs
    
    app.logger.info(f"Processor initialization complete:")
    app.logger.info(f"  ML Models loaded: {ml_working_count}/3")
    app.logger.info(f"  Total processors cached: {len(procs)}")
    
    return procs

# Initialize processors on startup (cached globally)
processors = initialize_processors()

@app.route('/')
def index():
    """API status endpoint."""
    return jsonify({
        'message': 'Streaming Digit Classifier API',
        'status': 'running',
        'version': '1.0.0',
        'available_processors': list(processors.keys()),
        'documentation': 'Frontend at Vercel, Backend API at HF Spaces'
    })

@app.route('/api/process_audio', methods=['POST'])
def process_audio():
    """
    Process audio file with selected method and return digit prediction.
    Expects multipart form data with 'audio' file and 'method' selection.
    """
    try:
        # Validate request
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        if 'method' not in request.form:
            return jsonify({'error': 'No processing method specified'}), 400
        
        audio_file = request.files['audio']
        method = request.form['method']
        
        # Validate audio file
        if audio_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(audio_file.filename):
            return jsonify({'error': 'Unsupported file format'}), 400
        
        # Validate method
        if method not in processors:
            return jsonify({'error': f'Unknown processing method: {method}'}), 400
        
        # Read audio data
        audio_data = audio_file.read()
        
        # Check file size
        if len(audio_data) > MAX_FILE_SIZE:
            return jsonify({'error': 'Audio file too large'}), 400
        
        # Convert to standard format
        try:
            app.logger.debug(f"Converting audio format. Original size: {len(audio_data)} bytes")
            standardized_audio = convert_audio_format(audio_data)
            app.logger.debug(f"Converted audio size: {len(standardized_audio)} bytes")
        except Exception as e:
            app.logger.error(f"Audio conversion failed: {str(e)}")
            return jsonify({'error': 'Failed to process audio format - unsupported format or corrupted file'}), 400
        
        # Check audio duration
        duration = get_audio_duration(standardized_audio)
        if duration > MAX_AUDIO_DURATION:
            return jsonify({
                'error': f'Audio too long: {duration:.1f}s (max: {MAX_AUDIO_DURATION}s)'
            }), 400
        
        if duration < 0.1:
            return jsonify({'error': 'Audio too short (minimum: 0.1s)'}), 400
        
        # Log audio input info
        performance_logger.log_audio_info(duration, {
            'filename': audio_file.filename,
            'size_bytes': len(audio_data),
            'converted_size': len(standardized_audio),
            'method': method
        })
        
        # Process with selected method
        processor = processors[method]
        result = processor.predict_with_timing(standardized_audio)
        
        # Log performance
        performance_logger.log_prediction(method, result)
        
        # Add additional metadata
        result.update({
            'audio_duration': round(duration, 3),
            'file_size': len(audio_data),
            'api_version': '1.0.0'
        })
        
        app.logger.info(f"Processed audio with {method}: '{result['predicted_digit']}' in {result['inference_time']}s")
        
        return jsonify(result)
    
    except Exception as e:
        app.logger.error(f"Audio processing error: {str(e)}")
        return jsonify({
            'error': 'Internal processing error',
            'success': False,
            'timestamp': time.time()
        }), 500

@app.route('/api/process_audio_chunk', methods=['POST'])
def process_audio_chunk():
    """
    Process streaming audio chunk for real-time digit recognition.
    """
    try:
        # Validate request
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio chunk provided'}), 400
        
        audio_file = request.files['audio']
        method = request.form.get('method', 'ml_mfcc')  # Default to best ML model
        
        # Validate method
        if method not in processors:
            return jsonify({'error': f'Unknown processing method: {method}'}), 400
        
        # Read audio data
        audio_data = audio_file.read()
        
        # Check chunk size
        if len(audio_data) > MAX_FILE_SIZE:
            return jsonify({'error': 'Audio chunk too large'}), 400
        
        if len(audio_data) < 100:
            return jsonify({'error': 'Audio chunk too small'}), 400
        
        # Convert to standardized format
        try:
            standardized_audio = convert_for_ml_models(audio_data, 'streaming')
        except Exception as e:
            app.logger.error(f"Audio conversion failed for chunk: {str(e)}")
            return jsonify({'error': 'Failed to process audio chunk format'}), 400
        
        # Process audio chunk
        processor = processors[method]
        result = processor.predict_with_timing(standardized_audio)
        
        # Add streaming metadata
        result.update({
            'segment_index': 0,
            'segment_size': len(standardized_audio),
            'is_streaming': True,
            'api_version': '1.0.0'
        })
        
        app.logger.info(f"Streaming prediction: '{result['predicted_digit']}' "
                      f"(Inference: {result['inference_time']}s)")
        
        return jsonify({
            'success': True,
            'segments_detected': 1,
            'total_results': 1,
            'results': [result],
            'timestamp': time.time(),
            'has_fallback': False
        })
    
    except Exception as e:
        app.logger.error(f"Streaming audio processing error: {str(e)}")
        return jsonify({
            'error': 'Internal streaming processing error',
            'success': False,
            'timestamp': time.time()
        }), 500

@app.route('/api/processors')
def get_processors():
    """Get information about available processors."""
    try:
        processor_info = {}
        for name, processor in processors.items():
            info = {
                'name': processor.name,
                'method': name,
                'configured': getattr(processor, 'is_configured', lambda: True)()
            }
            
            # Add model-specific info if available
            if hasattr(processor, 'get_model_info'):
                info.update(processor.get_model_info())
            
            processor_info[name] = info
        
        return jsonify(processor_info)
    
    except Exception as e:
        app.logger.error(f"Error getting processors: {str(e)}")
        return jsonify({'error': 'Failed to retrieve processor information'}), 500

@app.route('/api/health')
def health_check():
    """Health check endpoint."""
    try:
        # Check processor availability
        processor_health = {}
        for name, processor in processors.items():
            processor_health[name] = {
                'available': True,
                'configured': getattr(processor, 'is_configured', lambda: True)()
            }
        
        return jsonify({
            'status': 'healthy',
            'timestamp': time.time(),
            'processors': processor_health,
            'version': '1.0.0',
            'deployment': 'huggingface-spaces'
        })
    
    except Exception as e:
        app.logger.error(f"Health check failed: {str(e)}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': time.time()
        }), 500

@app.errorhandler(404)
def not_found_error(error):
    """Handle 404 errors."""
    return jsonify({'error': 'Endpoint not found', 'status': 404}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    app.logger.error(f"Internal error: {str(error)}")
    return jsonify({'error': 'Internal server error', 'status': 500}), 500

@app.errorhandler(413)
def too_large_error(error):
    """Handle file too large errors."""
    return jsonify({'error': 'File too large', 'status': 413}), 413

if __name__ == '__main__':
    # Log startup information
    try:
        import importlib.metadata
        flask_version = importlib.metadata.version('flask')
    except:
        flask_version = 'unknown'
        
    performance_logger.log_system_info({
        'python_version': os.sys.version,
        'flask_version': flask_version,
        'processors_loaded': list(processors.keys()),
        'max_audio_duration': MAX_AUDIO_DURATION,
        'max_file_size': MAX_FILE_SIZE,
        'deployment': 'huggingface-spaces'
    })
    
    # Run server (HF Spaces requires port 7860)
    port = int(os.getenv('PORT', 7860))
    
    app.logger.info(f"Starting Audio Digit Classifier API on port {port}")
    app.logger.info("Deployment: Hugging Face Spaces")
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=False,  # Disable debug in production
        threaded=True
    )