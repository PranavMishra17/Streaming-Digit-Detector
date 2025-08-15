import numpy as np
import wave
import io
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

def validate_audio_format(audio_data: bytes) -> bool:
    """
    Validate that audio data is in a supported format.
    
    Args:
        audio_data: Raw audio bytes
        
    Returns:
        True if format is supported, False otherwise
    """
    try:
        with wave.open(io.BytesIO(audio_data), 'rb') as wav_file:
            # Check basic WAV properties
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            frame_rate = wav_file.getframerate()
            
            logger.debug(f"Audio format: {channels} channels, {sample_width} bytes/sample, {frame_rate} Hz")
            
            # Support mono or stereo, 16-bit, common sample rates
            if channels not in [1, 2]:
                return False
            if sample_width not in [2]:  # 16-bit
                return False
            if frame_rate not in [16000, 22050, 44100, 48000]:
                return False
                
            return True
    except Exception as e:
        logger.error(f"Audio validation failed: {str(e)}")
        return False

def convert_to_mono_16khz(audio_data: bytes) -> bytes:
    """
    Convert audio to mono, 16kHz format suitable for speech recognition.
    
    Args:
        audio_data: Input audio bytes (WAV format)
        
    Returns:
        Converted audio bytes in mono 16kHz WAV format
        
    Raises:
        Exception: If conversion fails
    """
    try:
        with wave.open(io.BytesIO(audio_data), 'rb') as input_wav:
            frames = input_wav.readframes(input_wav.getnframes())
            channels = input_wav.getnchannels()
            sample_width = input_wav.getsampwidth()
            frame_rate = input_wav.getframerate()
            
            # Convert to numpy array
            if sample_width == 2:
                audio_array = np.frombuffer(frames, dtype=np.int16)
            else:
                raise Exception(f"Unsupported sample width: {sample_width}")
            
            # Convert stereo to mono if needed
            if channels == 2:
                audio_array = audio_array.reshape(-1, 2)
                audio_array = np.mean(audio_array, axis=1).astype(np.int16)
            
            # Resample to 16kHz if needed
            if frame_rate != 16000:
                # Simple downsampling (for production, use proper resampling)
                ratio = frame_rate / 16000
                if ratio > 1:
                    # Downsample by taking every nth sample
                    indices = np.arange(0, len(audio_array), ratio).astype(int)
                    audio_array = audio_array[indices]
                else:
                    # Upsample by repeating samples (basic interpolation)
                    audio_array = np.repeat(audio_array, int(1/ratio))
            
            # Create output WAV
            output = io.BytesIO()
            with wave.open(output, 'wb') as output_wav:
                output_wav.setnchannels(1)  # Mono
                output_wav.setsampwidth(2)  # 16-bit
                output_wav.setframerate(16000)  # 16kHz
                output_wav.writeframes(audio_array.tobytes())
            
            return output.getvalue()
            
    except Exception as e:
        logger.error(f"Audio conversion failed: {str(e)}")
        raise Exception(f"Failed to convert audio: {str(e)}")

def get_audio_duration(audio_data: bytes) -> float:
    """
    Get duration of audio in seconds.
    
    Args:
        audio_data: WAV audio bytes
        
    Returns:
        Duration in seconds
    """
    try:
        with wave.open(io.BytesIO(audio_data), 'rb') as wav_file:
            frames = wav_file.getnframes()
            frame_rate = wav_file.getframerate()
            duration = frames / frame_rate
            return duration
    except Exception as e:
        logger.error(f"Failed to get audio duration: {str(e)}")
        return 0.0

def audio_to_numpy(audio_data: bytes) -> Tuple[np.ndarray, int]:
    """
    Convert WAV audio bytes to numpy array.
    
    Args:
        audio_data: WAV audio bytes
        
    Returns:
        Tuple of (audio_array, sample_rate)
        
    Raises:
        Exception: If conversion fails
    """
    try:
        with wave.open(io.BytesIO(audio_data), 'rb') as wav_file:
            frames = wav_file.readframes(wav_file.getnframes())
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            
            if sample_width == 2:
                audio_array = np.frombuffer(frames, dtype=np.int16)
            else:
                raise Exception(f"Unsupported sample width: {sample_width}")
            
            # Convert to float32 and normalize
            audio_array = audio_array.astype(np.float32) / 32767.0
            
            # Handle stereo
            if channels == 2:
                audio_array = audio_array.reshape(-1, 2)
                audio_array = np.mean(audio_array, axis=1)
            
            return audio_array, sample_rate
            
    except Exception as e:
        logger.error(f"Failed to convert audio to numpy: {str(e)}")
        raise Exception(f"Audio conversion failed: {str(e)}")

def create_test_audio(digit: str, duration: float = 1.0, sample_rate: int = 16000) -> bytes:
    """
    Create test audio data for development purposes.
    
    Args:
        digit: Digit to simulate ('0'-'9')
        duration: Audio duration in seconds
        sample_rate: Sample rate in Hz
        
    Returns:
        WAV audio bytes
    """
    try:
        # Create simple tone pattern based on digit
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        
        # Different frequency patterns for each digit
        freq_map = {
            '0': [400, 600],     # Low frequencies
            '1': [800, 1000],    # Higher frequencies
            '2': [600, 800],
            '3': [700, 900],
            '4': [500, 700],
            '5': [900, 1100],
            '6': [450, 650],
            '7': [750, 950],
            '8': [550, 750],
            '9': [850, 1050]
        }
        
        freqs = freq_map.get(digit, [440, 880])
        
        # Generate tone
        signal = np.sin(freqs[0] * 2.0 * np.pi * t) * 0.3 + np.sin(freqs[1] * 2.0 * np.pi * t) * 0.3
        
        # Add some envelope
        envelope = np.exp(-3 * t)
        signal = signal * envelope
        
        # Convert to int16
        signal = (signal * 32767).astype(np.int16)
        
        # Create WAV
        output = io.BytesIO()
        with wave.open(output, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(signal.tobytes())
        
        return output.getvalue()
        
    except Exception as e:
        logger.error(f"Failed to create test audio: {str(e)}")
        raise Exception(f"Test audio creation failed: {str(e)}")