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
import numpy as np

# Import audio processors
from audio_processors.external_api import ExternalAPIProcessor
from audio_processors.local_whisper import LocalWhisperProcessor
from audio_processors.wav2vec2_processor import Wav2Vec2Processor
from audio_processors.raw_spectrogram import RawSpectrogramProcessor  
from audio_processors.mel_spectrogram import MelSpectrogramProcessor
from audio_processors.mfcc_processor import MFCCProcessor
from audio_processors.whisper_digit_processor import WhisperDigitProcessor
from audio_processors.faster_whisper_processor import FasterWhisperDigitProcessor

# Import new ML-trained processors
from audio_processors.ml_mfcc_processor import MLMFCCProcessor
from audio_processors.ml_mel_cnn_processor import MLMelCNNProcessor
from audio_processors.ml_raw_cnn_processor import MLRawCNNProcessor

# Import utilities
from utils.audio_utils import validate_audio_format, convert_audio_format, get_audio_duration, convert_for_ml_models
from utils.logging_utils import performance_logger, setup_flask_logging
from utils.noise_utils import noise_generator
from utils.webrtc_vad import WebRTCVADProcessor, StreamingAudioBuffer
from utils.enhanced_vad import create_enhanced_vad
from utils.session_manager import session_manager

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'dev_key_change_in_production')

# Setup logging
setup_flask_logging(app)

# Initialize audio processors with dynamic fallback
def initialize_processors():
    """Initialize audio processors with intelligent fallback system."""
    procs = {}
    
    # First, initialize ML-trained processors (high priority)
    ml_processors = [
        ('ml_mfcc', MLMFCCProcessor, 'ML MFCC + Dense NN (Best - 98.52%)'),
        ('ml_mel_cnn', MLMelCNNProcessor, 'ML Mel CNN (Good - 97.22%)'),
        ('ml_raw_cnn', MLRawCNNProcessor, 'ML Raw CNN (Fair - 91.30%)')
    ]
    
    ml_working_count = 0
    for proc_key, proc_class, proc_name in ml_processors:
        try:
            processor = proc_class()
            if processor.is_configured():
                procs[proc_key] = processor
                ml_working_count += 1
                app.logger.info(f"[OK] {proc_name} loaded successfully")
            else:
                app.logger.warning(f"[WARN] {proc_name} not configured (model files missing)")
        except Exception as e:
            app.logger.error(f"[FAIL] Failed to initialize {proc_name}: {str(e)}")
    
    # Try to initialize speech recognition processors in order of preference
    speech_processors = [
        ('wav2vec2', Wav2Vec2Processor, 'Wav2Vec2 (Facebook) - External API Default'),
        ('faster_whisper', FasterWhisperDigitProcessor, 'Faster-Whisper with VAD'),
        ('whisper_digit', WhisperDigitProcessor, 'Whisper Digit Recognition'),
        ('external_api', ExternalAPIProcessor, 'External API (Whisper)'),
        ('local_whisper', LocalWhisperProcessor, 'Local Whisper (Tiny)')
    ]
    
    working_speech_processor = None
    
    for proc_key, proc_class, proc_name in speech_processors:
        try:
            processor = proc_class()
            if processor.is_configured():
                procs[proc_key] = processor
                if working_speech_processor is None:
                    working_speech_processor = proc_key
                app.logger.info(f"[OK] {proc_name} initialized and configured")
            else:
                app.logger.warning(f"[WARN] {proc_name} not configured (missing dependencies/tokens)")
        except Exception as e:
            app.logger.error(f"[FAIL] Failed to initialize {proc_name}: {str(e)}")
    
    # Set primary speech processor (prefer wav2vec2 as default)
    if 'wav2vec2' in procs:
        procs['primary_speech'] = procs['wav2vec2']
        app.logger.info("Primary speech processor: wav2vec2")
    elif working_speech_processor:
        procs['primary_speech'] = procs[working_speech_processor]
        app.logger.info(f"Primary speech processor: {working_speech_processor}")
    
    # Add legacy processors for comparison (lower priority)
    legacy_processors = [
        ('raw_spectrogram', RawSpectrogramProcessor),
        ('mel_spectrogram', MelSpectrogramProcessor),
        ('mfcc', MFCCProcessor)
    ]
    
    for proc_key, proc_class in legacy_processors:
        try:
            procs[proc_key] = proc_class()
        except Exception as e:
            app.logger.error(f"Failed to initialize legacy {proc_key}: {str(e)}")
    
    app.logger.info(f"Processor initialization complete:")
    app.logger.info(f"  ML Models loaded: {ml_working_count}/3")
    app.logger.info(f"  Total processors: {len(procs)}")
    
    return procs

processors = initialize_processors()

# Initialize Enhanced VAD processor for streaming
try:
    # Create enhanced VAD with optimized settings for digit recognition
    vad_config = {
        'sample_rate': 16000,
        'frame_duration_ms': 30,
        'aggressiveness': 1,
        'min_speech_duration': 0.4,  # Minimum 400ms for digit recognition
        'max_speech_duration': 3.0,  # Maximum 3s for longer digits
        'silence_threshold': 0.01    # Low threshold for sensitivity
    }
    
    enhanced_vad = create_enhanced_vad(vad_config)
    
    # Also keep legacy VAD for compatibility
    vad_processor = WebRTCVADProcessor(aggressiveness=1, sample_rate=16000, frame_duration=30)
    vad_processor.silence_threshold = 0.5
    vad_processor.speech_threshold = 0.2
    vad_processor.min_speech_duration = 0.4
    vad_processor.max_speech_duration = 3.0
    vad_processor.max_silence_duration = 2.0
    
    audio_buffer = StreamingAudioBuffer(sample_rate=16000, max_duration=30)
    
    app.logger.info("Enhanced VAD processor initialized successfully")
    app.logger.info(f"  Enhanced VAD features: ffmpeg={enhanced_vad.ffmpeg_available}, webrtc={enhanced_vad.webrtc_vad is not None}")
    
except Exception as e:
    app.logger.error(f"Failed to initialize VAD processors: {e}")
    enhanced_vad = None
    vad_processor = None
    audio_buffer = None

# Configuration
MAX_AUDIO_DURATION = int(os.getenv('MAX_AUDIO_DURATION', 10))  # seconds
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'm4a', 'webm'}

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

# Session Management Endpoints
@app.route('/session/create', methods=['POST'])
def create_session():
    """Create a new audio recording session."""
    try:
        data = request.get_json() or {}
        custom_session_id = data.get('session_id')
        
        session_id = session_manager.create_session(custom_session_id)
        session = session_manager.get_session(session_id)
        
        if not session:
            return jsonify({'error': 'Failed to create session'}), 500
        
        # Cleanup old sessions (optional)
        cleanup_hours = data.get('cleanup_old_sessions', 24)
        if cleanup_hours > 0:
            cleaned = session_manager.cleanup_old_sessions(cleanup_hours)
            if cleaned > 0:
                app.logger.info(f"Cleaned up {cleaned} old sessions")
        
        response = {
            'success': True,
            'session_id': session_id,
            'session_dir': str(session.session_dir),
            'chunks_dir': str(session.chunks_dir),
            'created_at': session.created_at,
            'message': f'Session created: {session_id}'
        }
        
        app.logger.info(f"📁 Created new recording session: {session_id}")
        return jsonify(response)
        
    except Exception as e:
        app.logger.error(f"Error creating session: {str(e)}")
        return jsonify({'error': 'Failed to create session'}), 500

@app.route('/session/<session_id>/info')
def get_session_info(session_id: str):
    """Get information about a specific session."""
    try:
        session = session_manager.get_session(session_id)
        if not session:
            return jsonify({'error': 'Session not found'}), 404
        
        return jsonify(session.get_session_summary())
        
    except Exception as e:
        app.logger.error(f"Error getting session info: {str(e)}")
        return jsonify({'error': 'Failed to get session info'}), 500

@app.route('/session/<session_id>/close', methods=['POST'])
def close_session(session_id: str):
    """Close and finalize a session."""
    try:
        success = session_manager.close_session(session_id)
        
        if not success:
            return jsonify({'error': 'Session not found or already closed'}), 404
        
        return jsonify({
            'success': True,
            'message': f'Session {session_id} closed successfully'
        })
        
    except Exception as e:
        app.logger.error(f"Error closing session: {str(e)}")
        return jsonify({'error': 'Failed to close session'}), 500

@app.route('/sessions')
def get_all_sessions():
    """Get statistics and information about all sessions."""
    try:
        stats = session_manager.get_session_stats()
        return jsonify(stats)
        
    except Exception as e:
        app.logger.error(f"Error getting session stats: {str(e)}")
        return jsonify({'error': 'Failed to get session statistics'}), 500

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
        
        # Try to validate and convert audio format
        # For streaming chunks, we may get WebM/OGG format from MediaRecorder
        audio_is_valid = validate_audio_format(audio_data)
        
        if not audio_is_valid:
            # Check if it might be a different format (WebM, OGG)
            try:
                # Try to convert using audio utilities
                app.logger.debug(f"Audio format validation failed, attempting conversion. File type from header: {audio_data[:8]}")
                if len(audio_data) < 8:
                    return jsonify({'error': 'Audio file too small or corrupted'}), 400
                
                # For WebM/OGG files, we'll need special handling
                # Let's try the conversion anyway and see if it works
                pass  # Will be handled in conversion step
                
            except Exception as e:
                app.logger.error(f"Audio format detection failed: {str(e)}")
                return jsonify({'error': 'Invalid or corrupted audio file'}), 400
        
        # Convert to standard format for processing first
        try:
            app.logger.debug(f"Converting audio format. Original size: {len(audio_data)} bytes")
            standardized_audio = convert_audio_format(audio_data)
            app.logger.debug(f"Converted audio size: {len(standardized_audio)} bytes")
        except Exception as e:
            app.logger.error(f"Audio conversion failed: {str(e)}")
            return jsonify({'error': 'Failed to process audio format - unsupported format or corrupted file'}), 400
        
        # Check audio duration using the converted audio
        duration = get_audio_duration(standardized_audio)
        if duration > MAX_AUDIO_DURATION:
            return jsonify({
                'error': f'Audio too long: {duration:.1f}s (max: {MAX_AUDIO_DURATION}s)'
            }), 400
        
        if duration < 0.1:
            # For streaming chunks, be more lenient with minimum duration
            min_duration = 0.05 if 'streaming' in audio_file.filename.lower() else 0.1
            if duration < min_duration:
                return jsonify({'error': f'Audio too short (minimum: {min_duration}s)'}), 400
        
        # Log audio input info
        performance_logger.log_audio_info(duration, {
            'filename': audio_file.filename,
            'size_bytes': len(audio_data),
            'converted_size': len(standardized_audio),
            'method': method
        })
        
        # Apply noise injection if requested (verify it actually works)
        noise_type = request.form.get('noise_type')
        noise_level = request.form.get('noise_level', '0.0')
        noise_applied = False
        
        if noise_type and noise_type != 'none':
            try:
                noise_level_float = float(noise_level)
                if 0.0 < noise_level_float <= 1.0:
                    # Store original audio for comparison
                    original_energy = np.sqrt(np.mean(np.frombuffer(standardized_audio, dtype=np.int16).astype(np.float32) ** 2))
                    
                    # Apply noise
                    noisy_audio = noise_generator.inject_noise(
                        standardized_audio, 
                        noise_type, 
                        noise_level_float
                    )
                    
                    # Verify noise was actually applied
                    noisy_energy = np.sqrt(np.mean(np.frombuffer(noisy_audio, dtype=np.int16).astype(np.float32) ** 2))
                    
                    if abs(noisy_energy - original_energy) > 0.01:  # Detectable change
                        standardized_audio = noisy_audio
                        noise_applied = True
                        app.logger.info(f"✓ Applied {noise_type} noise: {noise_level_float} (energy: {original_energy:.3f} -> {noisy_energy:.3f})")
                    else:
                        app.logger.warning(f"Noise injection had no detectable effect: {noise_type} at {noise_level_float}")
                        
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
            'noise_applied': noise_type if noise_applied else None,
            'noise_level': noise_level_float if noise_applied else 0.0,
            'noise_verification': 'applied' if noise_applied else 'none' if not noise_type or noise_type == 'none' else 'failed'
        })
        
        app.logger.info(f"Processed audio with {method}: '{result['predicted_digit']}' in {result['inference_time']}s")
        
        # Save audio chunk to session if session_id provided
        session_id = request.form.get('session_id')
        if session_id:
            try:
                session = session_manager.get_session(session_id)
                if session:
                    # Use the standardized audio for saving (better quality)
                    saved_path = session.save_audio_chunk(
                        standardized_audio,
                        prediction_result=result,
                        chunk_type="full_audio"
                    )
                    result['saved_to'] = saved_path
                    result['session_id'] = session_id
                    app.logger.info(f"💾 Audio saved to session {session_id}: {saved_path}")
                else:
                    app.logger.warning(f"Session {session_id} not found, audio not saved")
            except Exception as e:
                app.logger.error(f"Failed to save audio to session {session_id}: {e}")
                # Don't fail the request if saving fails
        
        return jsonify(result)
    
    except Exception as e:
        app.logger.error(f"Audio processing error: {str(e)}")
        return jsonify({
            'error': 'Internal processing error',
            'success': False,
            'timestamp': time.time()
        }), 500

@app.route('/process_audio_chunk', methods=['POST'])
def process_audio_chunk():
    """
    Process streaming audio chunk with Enhanced VAD for real-time digit recognition.
    Expects raw audio data and returns VAD results and/or digit predictions.
    """
    try:
        if not enhanced_vad and not vad_processor:
            return jsonify({'error': 'VAD processor not available'}), 500
        
        # Validate request
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio chunk provided'}), 400
        
        audio_file = request.files['audio']
        method = request.form.get('method', 'whisper_digit')
        
        # Validate method
        if method not in processors:
            return jsonify({'error': f'Unknown processing method: {method}'}), 400
        
        # Read audio data
        audio_data = audio_file.read()
        
        # Check chunk size (be more lenient for streaming)
        if len(audio_data) > MAX_FILE_SIZE:
            return jsonify({'error': 'Audio chunk too large'}), 400
        
        if len(audio_data) < 100:  # Very small chunks
            return jsonify({'error': 'Audio chunk too small'}), 400
        
        # Process with Enhanced VAD
        start_time = time.time()
        
        # Convert to standardized format with enhanced validation
        try:
            # Use ML-optimized conversion for better quality
            standardized_audio = convert_for_ml_models(audio_data, 'streaming')
        except Exception as e:
            app.logger.error(f"Audio conversion failed for chunk: {str(e)}")
            return jsonify({'error': 'Failed to process audio chunk format'}), 400
        
        # Run Enhanced VAD processing
        speech_segments = []
        vad_debug_info = {}
        
        if enhanced_vad:
            # Use enhanced VAD with comprehensive debugging
            try:
                vad_results = enhanced_vad.detect_speech_segments(standardized_audio)
                speech_segments = [segment[0] for segment in vad_results]  # Extract audio data
                vad_debug_info = {
                    'method': 'enhanced_vad',
                    'segments_found': len(vad_results),
                    'validation_info': enhanced_vad.validate_and_debug_audio(standardized_audio),
                    'vad_stats': enhanced_vad.get_stats()
                }
                
                app.logger.debug(f"Enhanced VAD: {len(vad_results)} segments detected")
                
                # Log any audio issues detected
                if vad_debug_info['validation_info']['issues']:
                    app.logger.warning(f"Audio issues detected: {vad_debug_info['validation_info']['issues']}")
                
            except Exception as e:
                app.logger.error(f"Enhanced VAD processing failed: {str(e)}")
                # Fall back to legacy VAD
                if vad_processor:
                    speech_segments = vad_processor.process_audio_chunk(standardized_audio)
                    vad_debug_info = {'method': 'legacy_vad_fallback', 'error': str(e)}
        elif vad_processor:
            # Use legacy VAD
            speech_segments = vad_processor.process_audio_chunk(standardized_audio)
            vad_debug_info = {'method': 'legacy_vad_only'}
        
        vad_time = time.time() - start_time
        
        # Debug logging
        app.logger.debug(f"Audio chunk: {len(audio_data)} bytes, converted: {len(standardized_audio)} bytes")
        app.logger.debug(f"VAD processing: {len(speech_segments)} segments detected in {vad_time:.4f}s")
        
        results = []
        
        # If no VAD segments detected but we have audio, try processing directly (fallback)
        if len(speech_segments) == 0 and len(standardized_audio) > 8000:  # At least 8KB for meaningful audio
            app.logger.info("No VAD segments detected, trying direct processing as fallback")
            
            try:
                processor = processors[method]
                # Use the base AudioProcessor interface
                fallback_result = processor.predict_with_timing(standardized_audio)
                
                fallback_result.update({
                    'segment_index': 0,
                    'segment_size': len(standardized_audio),
                    'vad_processing_time': round(vad_time, 4),
                    'is_streaming': True,
                    'fallback_processing': True
                })
                
                results.append(fallback_result)
                
                app.logger.info(f"Fallback prediction: '{fallback_result['predicted_digit']}' "
                              f"(VAD: {vad_time:.3f}s, Inference: {fallback_result['inference_time']}s)")
                              
            except Exception as e:
                app.logger.error(f"Error in fallback processing: {str(e)}")
                results.append({
                    'segment_index': 0,
                    'error': 'Fallback processing failed',
                    'success': False,
                    'fallback_processing': True
                })
        
        # Process each detected speech segment
        for i, segment in enumerate(speech_segments):
            try:
                # Get processor and run prediction
                processor = processors[method]
                segment_result = processor.predict_with_timing(segment)
                
                segment_result.update({
                    'segment_index': i,
                    'segment_size': len(segment),
                    'vad_processing_time': round(vad_time, 4),
                    'is_streaming': True
                })
                
                results.append(segment_result)
                
                # Log the prediction
                app.logger.info(f"Streaming prediction {i}: '{segment_result['predicted_digit']}' "
                              f"(VAD: {vad_time:.3f}s, Inference: {segment_result['inference_time']}s)")
                
            except Exception as e:
                app.logger.error(f"Error processing speech segment {i}: {str(e)}")
                results.append({
                    'segment_index': i,
                    'error': 'Processing failed',
                    'success': False
                })
        
        # Get VAD status
        vad_stats = vad_processor.get_stats()
        
        # Save audio chunks to session if session_id provided
        session_id = request.form.get('session_id')
        if session_id and (len(speech_segments) > 0 or len(standardized_audio) > 8000):
            try:
                session = session_manager.get_session(session_id)
                if session:
                    # Save each detected speech segment
                    saved_chunks = []
                    for i, segment in enumerate(speech_segments):
                        try:
                            # Get the corresponding prediction result
                            segment_result = results[i] if i < len(results) else None
                            
                            saved_path = session.save_audio_chunk(
                                segment,
                                prediction_result=segment_result,
                                chunk_type="vad_segment"
                            )
                            saved_chunks.append(saved_path)
                            
                        except Exception as e:
                            app.logger.error(f"Failed to save speech segment {i}: {e}")
                    
                    # If no VAD segments but we have fallback results, save the original audio
                    if len(speech_segments) == 0 and any(r.get('fallback_processing', False) for r in results):
                        try:
                            fallback_result = next(r for r in results if r.get('fallback_processing', False))
                            saved_path = session.save_audio_chunk(
                                standardized_audio,
                                prediction_result=fallback_result,
                                chunk_type="fallback_audio"
                            )
                            saved_chunks.append(saved_path)
                        except Exception as e:
                            app.logger.error(f"Failed to save fallback audio: {e}")
                    
                    # Add saved chunks info to response
                    response_extra = {
                        'saved_chunks': saved_chunks,
                        'session_id': session_id,
                        'chunks_saved': len(saved_chunks)
                    }
                    
                    if saved_chunks:
                        app.logger.info(f"💾 Saved {len(saved_chunks)} audio chunks to session {session_id}")
                        
                else:
                    app.logger.warning(f"Session {session_id} not found, chunks not saved")
                    response_extra = {'session_error': 'Session not found'}
                    
            except Exception as e:
                app.logger.error(f"Failed to save chunks to session {session_id}: {e}")
                response_extra = {'session_error': str(e)}
        else:
            response_extra = {}

        response = {
            'success': True,
            'segments_detected': len(speech_segments),
            'total_results': len(results),
            'results': results,
            'vad_processing_time': round(vad_time, 4),
            'timestamp': time.time(),
            'vad_status': {
                'is_recording': vad_stats['is_recording'],
                'triggered': vad_stats['triggered'],
                'speech_ratio': vad_stats['speech_ratio']
            },
            'has_fallback': any(r.get('fallback_processing', False) for r in results),
            **response_extra
        }
        
        return jsonify(response)
    
    except Exception as e:
        app.logger.error(f"Streaming audio processing error: {str(e)}")
        return jsonify({
            'error': 'Internal streaming processing error',
            'success': False,
            'timestamp': time.time()
        }), 500

@app.route('/vad_status')
def get_vad_status():
    """Get current VAD processor status and comprehensive statistics."""
    try:
        response = {
            'timestamp': time.time(),
            'enhanced_vad_available': enhanced_vad is not None,
            'legacy_vad_available': vad_processor is not None,
            'buffer_available': audio_buffer is not None
        }
        
        # Enhanced VAD stats
        if enhanced_vad:
            response['enhanced_vad_stats'] = enhanced_vad.get_stats()
            response['primary_vad'] = 'enhanced'
        
        # Legacy VAD stats
        if vad_processor:
            response['legacy_vad_stats'] = vad_processor.get_stats()
            if not enhanced_vad:
                response['primary_vad'] = 'legacy'
        
        # Audio buffer stats
        if audio_buffer:
            response['buffer_stats'] = audio_buffer.get_stats()
        
        if not enhanced_vad and not vad_processor:
            return jsonify({'error': 'No VAD processor available'}), 500
        
        return jsonify(response)
    
    except Exception as e:
        app.logger.error(f"Error getting VAD status: {str(e)}")
        return jsonify({'error': 'Failed to get VAD status'}), 500

@app.route('/reset_vad', methods=['POST'])
def reset_vad_state():
    """Reset VAD processor state for new session."""
    try:
        reset_count = 0
        
        # Reset enhanced VAD
        if enhanced_vad:
            # Enhanced VAD doesn't need explicit reset as it's stateless per request
            app.logger.info("Enhanced VAD: stateless, no reset needed")
            reset_count += 1
        
        # Reset legacy VAD
        if vad_processor:
            vad_processor.reset_state()
            reset_count += 1
        
        # Reset audio buffer
        if audio_buffer:
            audio_buffer.clear()
            reset_count += 1
        
        if reset_count == 0:
            return jsonify({'error': 'No VAD processor available'}), 500
        
        app.logger.info(f"VAD state reset successfully ({reset_count} components)")
        
        return jsonify({
            'success': True,
            'message': f'VAD state reset ({reset_count} components)',
            'components_reset': reset_count,
            'timestamp': time.time()
        })
    
    except Exception as e:
        app.logger.error(f"Error resetting VAD state: {str(e)}")
        return jsonify({'error': 'Failed to reset VAD state'}), 500

@app.route('/debug_transcription', methods=['POST'])
def debug_transcription():
    """Debug endpoint to test transcription directly."""
    try:
        data = request.get_json()
        test_phrase = data.get('phrase', 'five')
        
        # Generate synthetic speech-like audio for the phrase
        import numpy as np
        sample_rate = 16000
        duration = 1.0
        
        # Create a more complex signal that might be recognized
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Generate different frequencies for different digits
        digit_frequencies = {
            'zero': [200, 300], 'one': [250, 350], 'two': [300, 400], 
            'three': [350, 450], 'four': [400, 500], 'five': [450, 550],
            'six': [500, 600], 'seven': [550, 650], 'eight': [600, 700], 'nine': [650, 750]
        }
        
        freqs = digit_frequencies.get(test_phrase, [440, 880])
        audio_signal = (0.3 * np.sin(2 * np.pi * freqs[0] * t) + 
                       0.2 * np.sin(2 * np.pi * freqs[1] * t) +
                       0.05 * np.random.randn(len(t)))
        
        # Convert to bytes
        audio_int16 = (audio_signal * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()
        
        # Test with different processors
        results = {}
        
        for method_name, processor in [('whisper_digit', processors.get('whisper_digit')),
                                     ('faster_whisper', processors.get('faster_whisper'))]:
            if processor and hasattr(processor, 'predict_with_timing'):
                try:
                    result = processor.predict_with_timing(audio_bytes)
                    results[method_name] = result
                except Exception as e:
                    results[method_name] = {'error': str(e)}
        
        return jsonify({
            'test_phrase': test_phrase,
            'audio_duration': duration,
            'audio_samples': len(audio_bytes),
            'processors_tested': list(results.keys()),
            'results': results
        })
        
    except Exception as e:
        app.logger.error(f"Debug transcription error: {str(e)}")
        return jsonify({'error': str(e)}), 500

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

@app.route('/favicon.ico')
def favicon():
    """Serve favicon."""
    return send_from_directory(app.static_folder, 'favicon.ico')

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