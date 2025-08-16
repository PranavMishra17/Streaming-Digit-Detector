"""
Integration module for WebRTC VAD with MFCC and Spectrogram processors
Combines voice activity detection with real-time feature extraction
"""

import numpy as np
import librosa
import logging
from typing import Dict, List, Optional, Tuple
import time
from collections import deque
import threading
import queue

from utils.webrtc_vad import WebRTCVADProcessor
from audio_processors.mfcc_processor import MFCCProcessor
from audio_processors.mel_spectrogram import MelSpectrogramProcessor
from audio_processors.raw_spectrogram import RawSpectrogramProcessor

logger = logging.getLogger(__name__)

class StreamingFeatureExtractor:
    """
    Real-time feature extraction with VAD integration.
    Combines WebRTC VAD with MFCC, Mel Spectrogram, and Raw Spectrogram processing.
    """
    
    def __init__(self, sample_rate=16000, n_mfcc=13, n_fft=2048, hop_length=512):
        """
        Initialize streaming feature extractor.
        
        Args:
            sample_rate: Audio sample rate
            n_mfcc: Number of MFCC coefficients
            n_fft: FFT window size
            hop_length: Hop length for STFT
        """
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        # Initialize VAD processor
        self.vad_processor = WebRTCVADProcessor(
            aggressiveness=2,
            sample_rate=sample_rate,
            frame_duration=30
        )
        
        # Initialize feature processors
        self.mfcc_processor = MFCCProcessor()
        self.mel_processor = MelSpectrogramProcessor()
        self.raw_spec_processor = RawSpectrogramProcessor()
        
        # Buffers for overlapped processing
        self.audio_buffer = deque(maxlen=sample_rate * 2)  # 2 second buffer
        self.feature_buffer = deque(maxlen=100)  # Store recent feature vectors
        
        # Threading for real-time processing
        self.processing_queue = queue.Queue()
        self.feature_queue = queue.Queue()
        self.is_processing = False
        self.processing_thread = None
        
        # Statistics
        self.total_chunks_processed = 0
        self.features_extracted = 0
        self.speech_segments_processed = 0
        
        logger.info("Streaming Feature Extractor initialized")
    
    def extract_features_realtime(self, audio_chunk: bytes) -> Dict[str, np.ndarray]:
        """
        Extract features from streaming audio chunk with VAD.
        
        Args:
            audio_chunk: Raw audio bytes
            
        Returns:
            dict: Extracted features for detected speech segments
        """
        # Process with VAD first
        speech_segments = self.vad_processor.process_audio_chunk(audio_chunk)
        
        features_list = []
        
        for segment in speech_segments:
            # Convert bytes to numpy array
            audio_array = np.frombuffer(segment, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Extract comprehensive features
            features = self._compute_streaming_features(audio_array)
            
            if features:
                features_list.append(features)
                self.features_extracted += 1
        
        self.total_chunks_processed += 1
        
        if speech_segments:
            self.speech_segments_processed += len(speech_segments)
            logger.debug(f"Extracted features from {len(speech_segments)} speech segments")
        
        return features_list
    
    def _compute_streaming_features(self, audio_data: np.ndarray) -> Optional[Dict[str, np.ndarray]]:
        """
        Compute comprehensive feature set optimized for streaming.
        
        Args:
            audio_data: Audio samples as numpy array
            
        Returns:
            dict: Feature dictionary or None if extraction fails
        """
        try:
            if len(audio_data) < self.n_fft:
                logger.debug("Audio segment too short for feature extraction")
                return None
            
            features = {}
            
            # Core MFCC features
            mfccs = librosa.feature.mfcc(
                y=audio_data, 
                sr=self.sample_rate, 
                n_mfcc=self.n_mfcc,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            
            # Statistical summaries for streaming
            features['mfcc_mean'] = np.mean(mfccs, axis=1)
            features['mfcc_std'] = np.std(mfccs, axis=1)
            features['mfcc_delta'] = np.mean(librosa.feature.delta(mfccs), axis=1)
            features['mfcc_delta2'] = np.mean(librosa.feature.delta(mfccs, order=2), axis=1)
            
            # Spectral features
            features['spectral_centroid'] = np.mean(
                librosa.feature.spectral_centroid(y=audio_data, sr=self.sample_rate)
            )
            features['spectral_bandwidth'] = np.mean(
                librosa.feature.spectral_bandwidth(y=audio_data, sr=self.sample_rate)
            )
            features['spectral_rolloff'] = np.mean(
                librosa.feature.spectral_rolloff(y=audio_data, sr=self.sample_rate)
            )
            features['zero_crossing_rate'] = np.mean(
                librosa.feature.zero_crossing_rate(audio_data)
            )
            
            # Energy features
            features['rms_energy'] = np.mean(librosa.feature.rms(y=audio_data))
            
            # Mel spectrogram features
            mel_spec = librosa.feature.melspectrogram(
                y=audio_data, 
                sr=self.sample_rate,
                n_mels=40,  # Reduced for streaming
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            features['mel_spec_mean'] = np.mean(mel_spec, axis=1)
            features['mel_spec_std'] = np.std(mel_spec, axis=1)
            
            # Raw spectrogram features
            stft = librosa.stft(audio_data, n_fft=self.n_fft, hop_length=self.hop_length)
            magnitude_spec = np.abs(stft)
            features['raw_spec_mean'] = np.mean(magnitude_spec, axis=1)
            features['raw_spec_std'] = np.std(magnitude_spec, axis=1)
            
            # Harmonic and percussive components
            harmonic, percussive = librosa.effects.hpss(audio_data)
            features['harmonic_ratio'] = np.mean(harmonic ** 2) / (np.mean(audio_data ** 2) + 1e-8)
            features['percussive_ratio'] = np.mean(percussive ** 2) / (np.mean(audio_data ** 2) + 1e-8)
            
            # Tempo and rhythm features (simplified for streaming)
            tempo, _ = librosa.beat.beat_track(y=audio_data, sr=self.sample_rate)
            features['tempo'] = tempo
            
            # Add metadata
            features['_metadata'] = {
                'duration': len(audio_data) / self.sample_rate,
                'sample_rate': self.sample_rate,
                'n_samples': len(audio_data),
                'extraction_timestamp': time.time()
            }
            
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            return None
    
    def extract_mfcc_features(self, audio_data: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract only MFCC features for lightweight processing.
        
        Args:
            audio_data: Audio samples
            
        Returns:
            np.ndarray: MFCC feature vector
        """
        try:
            mfccs = librosa.feature.mfcc(
                y=audio_data,
                sr=self.sample_rate,
                n_mfcc=self.n_mfcc,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            return np.mean(mfccs, axis=1)
        except Exception as e:
            logger.error(f"MFCC extraction error: {e}")
            return None
    
    def extract_spectrogram_features(self, audio_data: np.ndarray) -> Optional[Dict[str, np.ndarray]]:
        """
        Extract spectrogram-based features.
        
        Args:
            audio_data: Audio samples
            
        Returns:
            dict: Spectrogram features
        """
        try:
            # Mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio_data,
                sr=self.sample_rate,
                n_mels=80,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            
            # Raw spectrogram
            stft = librosa.stft(audio_data, n_fft=self.n_fft, hop_length=self.hop_length)
            magnitude_spec = np.abs(stft)
            
            return {
                'mel_spectrogram': mel_spec,
                'mel_spec_db': librosa.power_to_db(mel_spec),
                'raw_spectrogram': magnitude_spec,
                'raw_spec_db': librosa.amplitude_to_db(magnitude_spec)
            }
        except Exception as e:
            logger.error(f"Spectrogram extraction error: {e}")
            return None
    
    def process_with_vad_and_features(self, audio_chunk: bytes, feature_type: str = 'all') -> List[Dict]:
        """
        Process audio chunk with VAD and extract specified features.
        
        Args:
            audio_chunk: Raw audio bytes
            feature_type: Type of features to extract ('mfcc', 'spectrogram', 'all')
            
        Returns:
            List[dict]: Feature results for each speech segment
        """
        # Get speech segments from VAD
        speech_segments = self.vad_processor.process_audio_chunk(audio_chunk)
        
        results = []
        
        for i, segment in enumerate(speech_segments):
            # Convert to numpy array
            audio_array = np.frombuffer(segment, dtype=np.int16).astype(np.float32) / 32768.0
            
            segment_result = {
                'segment_index': i,
                'segment_duration': len(audio_array) / self.sample_rate,
                'segment_samples': len(audio_array)
            }
            
            # Extract requested features
            if feature_type == 'mfcc':
                mfcc_features = self.extract_mfcc_features(audio_array)
                if mfcc_features is not None:
                    segment_result['mfcc'] = mfcc_features
            
            elif feature_type == 'spectrogram':
                spec_features = self.extract_spectrogram_features(audio_array)
                if spec_features is not None:
                    segment_result.update(spec_features)
            
            elif feature_type == 'all':
                comprehensive_features = self._compute_streaming_features(audio_array)
                if comprehensive_features is not None:
                    segment_result.update(comprehensive_features)
            
            results.append(segment_result)
        
        return results
    
    def start_streaming_processing(self):
        """Start background thread for streaming processing."""
        if self.is_processing:
            return
        
        self.is_processing = True
        self.processing_thread = threading.Thread(target=self._streaming_worker, daemon=True)
        self.processing_thread.start()
        logger.info("Started streaming feature processing")
    
    def stop_streaming_processing(self):
        """Stop background streaming processing."""
        self.is_processing = False
        if self.processing_thread:
            self.processing_thread.join(timeout=1.0)
        logger.info("Stopped streaming feature processing")
    
    def add_audio_chunk(self, audio_chunk: bytes, feature_type: str = 'all'):
        """
        Add audio chunk to processing queue.
        
        Args:
            audio_chunk: Raw audio bytes
            feature_type: Type of features to extract
        """
        if self.is_processing:
            try:
                self.processing_queue.put_nowait((audio_chunk, feature_type))
            except queue.Full:
                logger.warning("Processing queue full, dropping chunk")
    
    def get_feature_results(self) -> List[Dict]:
        """
        Get all available feature extraction results.
        
        Returns:
            List[dict]: Available feature results
        """
        results = []
        try:
            while True:
                result = self.feature_queue.get_nowait()
                results.append(result)
        except queue.Empty:
            pass
        return results
    
    def _streaming_worker(self):
        """Background worker for streaming feature processing."""
        while self.is_processing:
            try:
                # Get audio chunk with timeout
                audio_chunk, feature_type = self.processing_queue.get(timeout=0.1)
                
                # Process chunk
                start_time = time.time()
                results = self.process_with_vad_and_features(audio_chunk, feature_type)
                processing_time = time.time() - start_time
                
                # Add processing metadata
                for result in results:
                    result['processing_time'] = processing_time
                    result['timestamp'] = time.time()
                
                # Add results to output queue
                for result in results:
                    try:
                        self.feature_queue.put_nowait(result)
                    except queue.Full:
                        logger.warning("Feature queue full, dropping result")
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Streaming feature processing error: {e}")
    
    def get_stats(self) -> Dict:
        """
        Get feature extraction statistics.
        
        Returns:
            dict: Processing statistics
        """
        vad_stats = self.vad_processor.get_stats()
        
        return {
            'total_chunks_processed': self.total_chunks_processed,
            'features_extracted': self.features_extracted,
            'speech_segments_processed': self.speech_segments_processed,
            'vad_stats': vad_stats,
            'is_processing': self.is_processing,
            'queue_sizes': {
                'processing_queue': self.processing_queue.qsize(),
                'feature_queue': self.feature_queue.qsize()
            }
        }
    
    def reset_state(self):
        """Reset all processing state."""
        self.vad_processor.reset_state()
        self.audio_buffer.clear()
        self.feature_buffer.clear()
        
        # Clear queues
        while not self.processing_queue.empty():
            try:
                self.processing_queue.get_nowait()
            except queue.Empty:
                break
        
        while not self.feature_queue.empty():
            try:
                self.feature_queue.get_nowait()
            except queue.Empty:
                break
        
        logger.info("Feature extractor state reset")

class VADMFCCProcessor:
    """
    Simplified VAD + MFCC processor for digit recognition.
    Optimized for low-latency real-time processing.
    """
    
    def __init__(self, sample_rate=16000, n_mfcc=13):
        """Initialize VAD + MFCC processor."""
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        
        self.vad_processor = WebRTCVADProcessor(
            aggressiveness=1,  # Less aggressive for better digit detection
            sample_rate=sample_rate,
            frame_duration=30
        )
        
        self.features_extracted = 0
        
        logger.info("VAD-MFCC processor initialized")
    
    def process_audio_for_digit_recognition(self, audio_chunk: bytes) -> List[np.ndarray]:
        """
        Process audio chunk and extract MFCC features from speech segments.
        
        Args:
            audio_chunk: Raw audio bytes
            
        Returns:
            List[np.ndarray]: MFCC feature vectors for each speech segment
        """
        # Get speech segments
        speech_segments = self.vad_processor.process_audio_chunk(audio_chunk)
        
        mfcc_features = []
        
        for segment in speech_segments:
            # Convert to numpy array
            audio_array = np.frombuffer(segment, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Extract MFCC features
            try:
                mfccs = librosa.feature.mfcc(
                    y=audio_array,
                    sr=self.sample_rate,
                    n_mfcc=self.n_mfcc,
                    n_fft=1024,  # Smaller FFT for faster processing
                    hop_length=256
                )
                
                # Use mean across time for simplicity
                mfcc_mean = np.mean(mfccs, axis=1)
                mfcc_features.append(mfcc_mean)
                self.features_extracted += 1
                
            except Exception as e:
                logger.error(f"MFCC extraction failed: {e}")
        
        return mfcc_features
    
    def get_stats(self) -> Dict:
        """Get processing statistics."""
        vad_stats = self.vad_processor.get_stats()
        
        return {
            'features_extracted': self.features_extracted,
            'vad_stats': vad_stats
        }