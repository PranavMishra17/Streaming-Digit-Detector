"""
Enhanced VAD Implementation with ffmpeg support and comprehensive debugging
"""

import numpy as np
import logging
import subprocess
import tempfile
import os
import time
import wave
import io
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from threading import Thread, Lock
import asyncio
import concurrent.futures

# Try to import WebRTC VAD
try:
    import webrtcvad
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False
    logging.warning("webrtcvad not available - using fallback VAD implementation")

logger = logging.getLogger(__name__)

class EnhancedVAD:
    """
    Enhanced Voice Activity Detection with ffmpeg integration and comprehensive debugging.
    
    Features:
    - ffmpeg-based audio preprocessing
    - Multiple VAD implementations (WebRTC, simple energy-based)
    - Comprehensive audio validation and debugging
    - Async audio chunk saving
    - Real-time performance monitoring
    """
    
    def __init__(self, 
                 sample_rate: int = 16000,
                 frame_duration_ms: int = 30,
                 aggressiveness: int = 1,
                 min_speech_duration: float = 0.4,
                 max_speech_duration: float = 3.0,
                 silence_threshold: float = 0.01):
        """
        Initialize Enhanced VAD.
        
        Args:
            sample_rate: Target sample rate (Hz)
            frame_duration_ms: Frame duration in milliseconds
            aggressiveness: VAD aggressiveness (0-3)
            min_speech_duration: Minimum speech segment duration (seconds)
            max_speech_duration: Maximum speech segment duration (seconds)
            silence_threshold: Energy threshold for silence detection
        """
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)
        self.aggressiveness = aggressiveness
        self.min_speech_duration = min_speech_duration
        self.max_speech_duration = max_speech_duration
        self.silence_threshold = silence_threshold
        
        # Initialize WebRTC VAD if available
        self.webrtc_vad = None
        if WEBRTC_AVAILABLE:
            try:
                self.webrtc_vad = webrtcvad.Vad(aggressiveness)
                logger.info(f"WebRTC VAD initialized (aggressiveness: {aggressiveness})")
            except Exception as e:
                logger.error(f"Failed to initialize WebRTC VAD: {e}")
                self.webrtc_vad = None
        
        # Check ffmpeg availability
        self.ffmpeg_available = self._check_ffmpeg_available()
        
        # Performance tracking
        self.stats = {
            'total_chunks_processed': 0,
            'speech_segments_detected': 0,
            'processing_time_total': 0.0,
            'last_processing_time': 0.0,
            'ffmpeg_conversions': 0,
            'audio_validation_failures': 0,
            'webrtc_available': WEBRTC_AVAILABLE and self.webrtc_vad is not None,
            'ffmpeg_available': self.ffmpeg_available
        }
        
        # Async processing
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        self.save_lock = Lock()
        
        logger.info(f"Enhanced VAD initialized:")
        logger.info(f"  Sample rate: {sample_rate} Hz")
        logger.info(f"  Frame duration: {frame_duration_ms} ms")
        logger.info(f"  WebRTC VAD: {'Available' if self.webrtc_vad else 'Not available'}")
        logger.info(f"  ffmpeg: {'Available' if self.ffmpeg_available else 'Not available'}")
    
    def _check_ffmpeg_available(self) -> bool:
        """Check if ffmpeg is available."""
        try:
            result = subprocess.run(['ffmpeg', '-version'], 
                                  capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except Exception:
            return False
    
    def preprocess_audio_with_ffmpeg(self, audio_data: bytes) -> Optional[bytes]:
        """
        Preprocess audio using ffmpeg for optimal VAD performance.
        
        Args:
            audio_data: Raw audio bytes
            
        Returns:
            Preprocessed audio bytes or None if processing fails
        """
        if not self.ffmpeg_available:
            logger.debug("ffmpeg not available for audio preprocessing")
            return None
        
        temp_input = None
        temp_output = None
        
        try:
            # Create temporary files
            with tempfile.NamedTemporaryFile(suffix='.input', delete=False) as temp_input:
                temp_input.write(audio_data)
                temp_input.flush()
                
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_output:
                pass
            
            # ffmpeg command for VAD-optimized preprocessing
            ffmpeg_cmd = [
                'ffmpeg',
                '-i', temp_input.name,
                '-ar', str(self.sample_rate),    # Resample to target rate
                '-ac', '1',                      # Convert to mono
                '-acodec', 'pcm_s16le',          # 16-bit PCM
                '-af', 'highpass=f=80,lowpass=f=8000,dynaudnorm=f=10:g=3',  # Audio filters for speech
                '-f', 'wav',
                '-loglevel', 'error',
                '-y',
                temp_output.name
            ]
            
            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                with open(temp_output.name, 'rb') as f:
                    preprocessed_audio = f.read()
                
                self.stats['ffmpeg_conversions'] += 1
                logger.debug(f"ffmpeg preprocessing: {len(audio_data)} -> {len(preprocessed_audio)} bytes")
                return preprocessed_audio
            else:
                logger.error(f"ffmpeg preprocessing failed: {result.stderr}")
                return None
                
        except Exception as e:
            logger.error(f"ffmpeg preprocessing error: {e}")
            return None
            
        finally:
            # Cleanup
            try:
                if temp_input and os.path.exists(temp_input.name):
                    os.unlink(temp_input.name)
                if temp_output and os.path.exists(temp_output.name):
                    os.unlink(temp_output.name)
            except Exception:
                pass
    
    def validate_and_debug_audio(self, audio_data: bytes) -> Dict[str, Any]:
        """
        Comprehensive audio validation and debugging.
        
        Args:
            audio_data: Audio data to validate
            
        Returns:
            Validation results and debugging information
        """
        debug_info = {
            'size_bytes': len(audio_data),
            'valid_wav': False,
            'sample_rate': None,
            'channels': None,
            'duration': 0.0,
            'energy_level': 0.0,
            'is_silent': True,
            'format_detected': 'unknown',
            'issues': []
        }
        
        try:
            # Check minimum size
            if len(audio_data) < 44:
                debug_info['issues'].append(f"Too small: {len(audio_data)} bytes (need ≥44 for WAV)")
                return debug_info
            
            # Detect format by header
            if audio_data.startswith(b'RIFF') and b'WAVE' in audio_data[:20]:
                debug_info['format_detected'] = 'wav'
            elif audio_data.startswith(b'OggS'):
                debug_info['format_detected'] = 'ogg'
            elif audio_data.startswith(b'\x1a\x45\xdf\xa3'):
                debug_info['format_detected'] = 'webm'
            
            # Try to parse as WAV
            try:
                with wave.open(io.BytesIO(audio_data), 'rb') as wav:
                    debug_info['valid_wav'] = True
                    debug_info['sample_rate'] = wav.getframerate()
                    debug_info['channels'] = wav.getnchannels()
                    debug_info['duration'] = wav.getnframes() / wav.getframerate()
                    
                    # Read audio samples for analysis
                    wav.rewind()
                    frames = wav.readframes(wav.getnframes())
                    
                    if len(frames) > 0:
                        # Convert to numpy for analysis
                        audio_array = np.frombuffer(frames, dtype=np.int16)
                        
                        # Calculate energy level
                        energy = np.sqrt(np.mean(audio_array.astype(np.float32) ** 2))
                        debug_info['energy_level'] = float(energy)
                        debug_info['is_silent'] = energy < (self.silence_threshold * 32768)
                        
                        # Check for constant beep (common issue)
                        if len(audio_array) > 100:
                            # Check if audio is a constant tone (beep)
                            diff = np.diff(audio_array)
                            if np.std(diff) < 100:  # Very low variation
                                debug_info['issues'].append("Constant tone/beep detected")
                        
                        # Check dynamic range
                        if np.max(audio_array) - np.min(audio_array) < 1000:
                            debug_info['issues'].append("Very low dynamic range")
                            
            except Exception as wav_error:
                debug_info['issues'].append(f"WAV parsing failed: {wav_error}")
            
            # Additional format-specific checks
            if debug_info['format_detected'] in ['ogg', 'webm'] and not debug_info['valid_wav']:
                debug_info['issues'].append("Non-WAV format detected - requires conversion")
            
            logger.debug(f"Audio validation: {debug_info}")
            
            if debug_info['issues']:
                self.stats['audio_validation_failures'] += 1
                logger.warning(f"Audio validation issues: {debug_info['issues']}")
            
            return debug_info
            
        except Exception as e:
            debug_info['issues'].append(f"Validation error: {str(e)}")
            logger.error(f"Audio validation failed: {e}")
            return debug_info
    
    def detect_speech_segments(self, audio_data: bytes) -> List[Tuple[bytes, Dict[str, Any]]]:
        """
        Detect speech segments using multiple methods.
        
        Args:
            audio_data: Input audio data
            
        Returns:
            List of (segment_audio, segment_info) tuples
        """
        start_time = time.time()
        
        # Validate and debug audio
        debug_info = self.validate_and_debug_audio(audio_data)
        
        segments = []
        
        try:
            # Preprocess with ffmpeg if available
            processed_audio = self.preprocess_audio_with_ffmpeg(audio_data)
            if processed_audio:
                working_audio = processed_audio
                logger.debug("Using ffmpeg-preprocessed audio for VAD")
            else:
                working_audio = audio_data
                logger.debug("Using original audio for VAD")
            
            # Re-validate processed audio
            if processed_audio:
                processed_debug = self.validate_and_debug_audio(processed_audio)
                logger.debug(f"Processed audio validation: {processed_debug}")
            
            # Method 1: WebRTC VAD (if available)
            if self.webrtc_vad and debug_info['valid_wav']:
                webrtc_segments = self._webrtc_vad_detection(working_audio)
                segments.extend(webrtc_segments)
                logger.debug(f"WebRTC VAD found {len(webrtc_segments)} segments")
            
            # Method 2: Energy-based VAD (fallback)
            if not segments or debug_info['issues']:
                energy_segments = self._energy_based_vad(working_audio)
                segments.extend(energy_segments)
                logger.debug(f"Energy VAD found {len(energy_segments)} segments")
            
            # Method 3: Simple duration-based segmentation (last resort)
            if not segments and len(audio_data) > 8000:  # > 8KB
                fallback_segment = self._create_fallback_segment(working_audio)
                if fallback_segment:
                    segments.append(fallback_segment)
                    logger.debug("Used fallback segmentation")
            
            processing_time = time.time() - start_time
            self.stats['total_chunks_processed'] += 1
            self.stats['speech_segments_detected'] += len(segments)
            self.stats['processing_time_total'] += processing_time
            self.stats['last_processing_time'] = processing_time
            
            logger.debug(f"VAD processing complete: {len(segments)} segments in {processing_time:.3f}s")
            
            return segments
            
        except Exception as e:
            logger.error(f"Speech segment detection failed: {e}")
            return []
    
    def _webrtc_vad_detection(self, audio_data: bytes) -> List[Tuple[bytes, Dict[str, Any]]]:
        """WebRTC-based speech detection."""
        segments = []
        
        try:
            frame_size_bytes = self.frame_size * 2  # 16-bit = 2 bytes per sample
            frames = []
            
            # Extract frames
            for i in range(0, len(audio_data) - frame_size_bytes + 1, frame_size_bytes):
                frame = audio_data[i:i + frame_size_bytes]
                if len(frame) == frame_size_bytes:
                    frames.append(frame)
            
            if len(frames) < 5:  # Need minimum frames
                return segments
            
            # VAD processing
            speech_frames = []
            for frame in frames:
                try:
                    is_speech = self.webrtc_vad.is_speech(frame, self.sample_rate)
                    speech_frames.append((frame, is_speech))
                except Exception as e:
                    logger.debug(f"WebRTC VAD frame processing failed: {e}")
                    speech_frames.append((frame, False))
            
            # Group consecutive speech frames
            current_segment = []
            for frame, is_speech in speech_frames:
                if is_speech:
                    current_segment.append(frame)
                else:
                    if len(current_segment) > 0:
                        # End of speech segment
                        segment_audio = b''.join(current_segment)
                        segment_duration = len(current_segment) * self.frame_duration_ms / 1000
                        
                        if segment_duration >= self.min_speech_duration:
                            segments.append((segment_audio, {
                                'duration': segment_duration,
                                'method': 'webrtc_vad',
                                'frames': len(current_segment)
                            }))
                        
                        current_segment = []
            
            # Handle final segment
            if current_segment:
                segment_audio = b''.join(current_segment)
                segment_duration = len(current_segment) * self.frame_duration_ms / 1000
                
                if segment_duration >= self.min_speech_duration:
                    segments.append((segment_audio, {
                        'duration': segment_duration,
                        'method': 'webrtc_vad',
                        'frames': len(current_segment)
                    }))
            
            return segments
            
        except Exception as e:
            logger.error(f"WebRTC VAD detection failed: {e}")
            return []
    
    def _energy_based_vad(self, audio_data: bytes) -> List[Tuple[bytes, Dict[str, Any]]]:
        """Energy-based speech detection."""
        segments = []
        
        try:
            # Try to parse as WAV or raw PCM
            try:
                with wave.open(io.BytesIO(audio_data), 'rb') as wav:
                    frames = wav.readframes(wav.getnframes())
                    sample_rate = wav.getframerate()
            except:
                # Assume raw 16-bit PCM
                frames = audio_data
                sample_rate = self.sample_rate
            
            if len(frames) < 1000:  # Too short
                return segments
            
            # Convert to numpy array
            audio_samples = np.frombuffer(frames, dtype=np.int16)
            audio_float = audio_samples.astype(np.float32) / 32768.0
            
            # Calculate energy in overlapping windows
            window_size = int(sample_rate * 0.1)  # 100ms windows
            hop_size = window_size // 2
            
            energies = []
            for i in range(0, len(audio_float) - window_size, hop_size):
                window = audio_float[i:i + window_size]
                energy = np.sqrt(np.mean(window ** 2))
                energies.append(energy)
            
            if len(energies) < 3:
                return segments
            
            # Adaptive threshold
            mean_energy = np.mean(energies)
            threshold = max(self.silence_threshold, mean_energy * 0.3)
            
            # Find speech segments
            speech_windows = energies > threshold
            
            # Group consecutive speech windows
            speech_start = None
            for i, is_speech in enumerate(speech_windows):
                if is_speech and speech_start is None:
                    speech_start = i
                elif not is_speech and speech_start is not None:
                    # End of speech
                    start_sample = speech_start * hop_size
                    end_sample = min(i * hop_size + window_size, len(audio_samples))
                    
                    segment_samples = audio_samples[start_sample:end_sample]
                    segment_duration = len(segment_samples) / sample_rate
                    
                    if segment_duration >= self.min_speech_duration:
                        # Convert back to bytes
                        segment_audio = segment_samples.tobytes()
                        
                        segments.append((segment_audio, {
                            'duration': segment_duration,
                            'method': 'energy_based',
                            'start_time': start_sample / sample_rate,
                            'energy_threshold': threshold,
                            'mean_energy': mean_energy
                        }))
                    
                    speech_start = None
            
            return segments
            
        except Exception as e:
            logger.error(f"Energy-based VAD failed: {e}")
            return []
    
    def _create_fallback_segment(self, audio_data: bytes) -> Optional[Tuple[bytes, Dict[str, Any]]]:
        """Create a fallback segment when VAD methods fail."""
        try:
            # Use the entire audio as a segment if it's reasonable length
            debug_info = self.validate_and_debug_audio(audio_data)
            
            if debug_info['duration'] > 0:
                duration = debug_info['duration']
            else:
                # Estimate duration based on size (assume 16-bit, mono, 16kHz)
                estimated_samples = len(audio_data) // 2
                duration = estimated_samples / self.sample_rate
            
            if self.min_speech_duration <= duration <= self.max_speech_duration:
                return (audio_data, {
                    'duration': duration,
                    'method': 'fallback',
                    'estimated': True,
                    'issues': debug_info['issues']
                })
            
            return None
            
        except Exception as e:
            logger.error(f"Fallback segment creation failed: {e}")
            return None
    
    async def save_audio_chunk_async(self, audio_data: bytes, session_id: str, 
                                   chunk_type: str = "vad_chunk") -> Optional[str]:
        """
        Asynchronously save audio chunk to file.
        
        Args:
            audio_data: Audio data to save
            session_id: Session identifier
            chunk_type: Type of chunk (for filename)
            
        Returns:
            Path to saved file or None if failed
        """
        def _save_chunk():
            try:
                with self.save_lock:
                    timestamp = int(time.time() * 1000)
                    filename = f"{chunk_type}_{session_id}_{timestamp}.wav"
                    filepath = Path("output") / filename
                    
                    # Ensure output directory exists
                    filepath.parent.mkdir(exist_ok=True)
                    
                    # Save as WAV file
                    with open(filepath, 'wb') as f:
                        f.write(audio_data)
                    
                    logger.debug(f"Saved audio chunk: {filepath}")
                    return str(filepath)
                    
            except Exception as e:
                logger.error(f"Failed to save audio chunk: {e}")
                return None
        
        # Run in executor to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(self.executor, _save_chunk)
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive VAD statistics."""
        stats = self.stats.copy()
        
        if stats['total_chunks_processed'] > 0:
            stats['average_processing_time'] = stats['processing_time_total'] / stats['total_chunks_processed']
            stats['segments_per_chunk'] = stats['speech_segments_detected'] / stats['total_chunks_processed']
        else:
            stats['average_processing_time'] = 0.0
            stats['segments_per_chunk'] = 0.0
        
        return stats
    
    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
        logger.info("Enhanced VAD cleaned up")

# Convenience function for creating enhanced VAD
def create_enhanced_vad(config: Optional[Dict[str, Any]] = None) -> EnhancedVAD:
    """Create enhanced VAD with optional configuration."""
    if config is None:
        config = {}
    
    return EnhancedVAD(
        sample_rate=config.get('sample_rate', 16000),
        frame_duration_ms=config.get('frame_duration_ms', 30),
        aggressiveness=config.get('aggressiveness', 1),
        min_speech_duration=config.get('min_speech_duration', 0.4),
        max_speech_duration=config.get('max_speech_duration', 3.0),
        silence_threshold=config.get('silence_threshold', 0.01)
    )