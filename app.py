"""
Audio Digit Classification Web Application
Retro game-inspired Flask app for spoken digit recognition (0-9)
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import time
import logging
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Import audio processors
from audio_processors.external_api import ExternalAPIProcessor
from audio_processors.raw_spectrogram import RawSpectrogramProcessor  
from audio_processors.mel_spectrogram import MelSpectrogramProcessor
from audio_processors.mfcc_processor import MFCCProcessor

# Import utilities
from utils.audio_utils import validate_audio_format, convert_to_mono_16khz, get_audio_duration
from utils.logging_utils import performance_logger, setup_flask_logging
from utils.noise_utils import noise_generator

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'dev_key_change_in_production')

# Setup logging
setup_flask_logging(app)

# Initialize audio processors
processors = {
    'external_api': ExternalAPIProcessor(),
    'raw_spectrogram': RawSpectrogramProcessor(),
    'mel_spectrogram': MelSpectrogramProcessor(),
    'mfcc': MFCCProcessor()
}

# Configuration
MAX_AUDIO_DURATION = int(os.getenv('MAX_AUDIO_DURATION', 10))  # seconds
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'm4a'}

def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Main application page."""
    try:
        # Get processor status for UI
        processor_status = {}
        for name, processor in processors.items():
            if hasattr(processor, 'is_configured'):
                processor_status[name] = {
                    'configured': processor.is_configured(),
                    'method': processor.name
                }
            else:
                processor_status[name] = {
                    'configured': True,  # Placeholder processors are always "configured"
                    'method': processor.name
                }
        
        return render_template('index.html', processor_status=processor_status)
    
    except Exception as e:
        app.logger.error(f"Error loading index page: {str(e)}")
        return render_template('error.html', error="Failed to load application"), 500

@app.route('/process_audio', methods=['POST'])
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
        
        # Validate audio format
        if not validate_audio_format(audio_data):
            return jsonify({'error': 'Invalid or corrupted audio file'}), 400
        
        # Check audio duration
        duration = get_audio_duration(audio_data)
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
            'method': method
        })
        
        # Convert to standard format for processing
        try:
            standardized_audio = convert_to_mono_16khz(audio_data)
        except Exception as e:
            app.logger.error(f"Audio conversion failed: {str(e)}")
            return jsonify({'error': 'Failed to process audio format'}), 400
        
        # Apply noise injection if requested
        noise_type = request.form.get('noise_type')
        noise_level = request.form.get('noise_level', '0.0')
        
        if noise_type and noise_type != 'none':
            try:
                noise_level_float = float(noise_level)
                if 0.0 < noise_level_float <= 1.0:
                    standardized_audio = noise_generator.inject_noise(
                        standardized_audio, 
                        noise_type, 
                        noise_level_float
                    )
                    app.logger.debug(f"Applied {noise_type} noise at level {noise_level_float}")
            except (ValueError, Exception) as e:
                app.logger.warning(f"Noise injection failed: {str(e)}")
                # Continue without noise injection
        
        # Process with selected method
        processor = processors[method]
        result = processor.predict_with_timing(standardized_audio)
        
        # Log performance
        performance_logger.log_prediction(method, result)
        
        # Add additional metadata
        result.update({
            'audio_duration': round(duration, 3),
            'file_size': len(audio_data),
            'noise_applied': noise_type if noise_type and noise_type != 'none' else None
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

@app.route('/stats')
def get_stats():
    """Get performance statistics for all processing methods."""
    try:
        stats = performance_logger.get_all_stats()
        return jsonify(stats)
    except Exception as e:
        app.logger.error(f"Error getting stats: {str(e)}")
        return jsonify({'error': 'Failed to retrieve statistics'}), 500

@app.route('/stats/<method>')
def get_method_stats(method: str):
    """Get detailed statistics for a specific processing method."""
    try:
        if method not in processors:
            return jsonify({'error': f'Unknown method: {method}'}), 400
        
        stats = performance_logger.get_method_stats(method)
        return jsonify(stats)
    except Exception as e:
        app.logger.error(f"Error getting method stats: {str(e)}")
        return jsonify({'error': 'Failed to retrieve method statistics'}), 500

@app.route('/processor_info/<method>')
def get_processor_info(method: str):
    """Get information about a specific processor."""
    try:
        if method not in processors:
            return jsonify({'error': f'Unknown method: {method}'}), 400
        
        processor = processors[method]
        info = {
            'name': processor.name,
            'method': method
        }
        
        # Add model-specific info if available
        if hasattr(processor, 'get_model_info'):
            info.update(processor.get_model_info())
        
        # Add configuration status
        if hasattr(processor, 'is_configured'):
            info['configured'] = processor.is_configured()
            if method == 'external_api' and hasattr(processor, 'test_connection'):
                info['connection_test'] = processor.test_connection()
        
        # Add processor stats
        stats = processor.get_stats()
        info['stats'] = stats
        
        return jsonify(info)
    
    except Exception as e:
        app.logger.error(f"Error getting processor info: {str(e)}")
        return jsonify({'error': 'Failed to retrieve processor information'}), 500

@app.route('/test_audio/<digit>')
def generate_test_audio(digit: str):
    """Generate test audio for a specific digit."""
    try:
        if digit not in '0123456789':
            return jsonify({'error': 'Digit must be 0-9'}), 400
        
        from utils.audio_utils import create_test_audio
        
        duration = float(request.args.get('duration', 1.0))
        if not 0.1 <= duration <= 5.0:
            return jsonify({'error': 'Duration must be between 0.1 and 5.0 seconds'}), 400
        
        test_audio = create_test_audio(digit, duration)
        
        # Return as downloadable file
        from flask import Response
        return Response(
            test_audio,
            mimetype='audio/wav',
            headers={'Content-Disposition': f'attachment; filename=test_digit_{digit}.wav'}
        )
    
    except Exception as e:
        app.logger.error(f"Error generating test audio: {str(e)}")
        return jsonify({'error': 'Failed to generate test audio'}), 500

@app.route('/health')
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
            'version': '1.0.0'
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
    return render_template('error.html', 
                         error="Page not found", 
                         error_code=404), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    app.logger.error(f"Internal error: {str(error)}")
    return render_template('error.html', 
                         error="Internal server error", 
                         error_code=500), 500

@app.errorhandler(413)
def too_large_error(error):
    """Handle file too large errors."""
    return jsonify({'error': 'File too large'}), 413

# Static file serving for development
@app.route('/static/<path:filename>')
def static_files(filename):
    """Serve static files."""
    return send_from_directory(app.static_folder, filename)

if __name__ == '__main__':
    # Log startup information
    try:
        import flask
        flask_version = getattr(flask, '__version__', 'unknown')
    except:
        flask_version = 'unknown'
        
    performance_logger.log_system_info({
        'python_version': os.sys.version,
        'flask_version': flask_version,
        'processors_loaded': list(processors.keys()),
        'max_audio_duration': MAX_AUDIO_DURATION,
        'max_file_size': MAX_FILE_SIZE
    })
    
    # Run development server
    port = int(os.getenv('FLASK_PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'
    
    app.logger.info(f"Starting Audio Digit Classifier on port {port}")
    app.logger.info(f"Debug mode: {debug}")
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug,
        threaded=True
    )