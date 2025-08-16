"""
WebM to WAV converter without FFmpeg dependency
Uses basic audio processing for WebM/OGG streams
"""

import logging
import io
import struct
from typing import Optional

logger = logging.getLogger(__name__)

def convert_webm_to_wav(webm_data: bytes) -> Optional[bytes]:
    """
    Convert WebM audio data to WAV format.
    This is a simplified converter for basic WebM streams.
    
    Args:
        webm_data: Raw WebM audio bytes
        
    Returns:
        WAV audio bytes or None if conversion fails
    """
    try:
        return create_fallback_wav(webm_data)
        
    except Exception as e:
        logger.error(f"WebM conversion failed: {str(e)}")
        return None

def create_fallback_wav(webm_data):
    """Properly convert WebM to WAV using subprocess"""
    import subprocess
    import tempfile
    import os
    
    webm_path = None
    wav_path = None
    
    try:
        # Write WebM data to temp file
        with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as webm_file:
            webm_file.write(webm_data)
            webm_path = webm_file.name
        
        # Output WAV path
        wav_path = webm_path.replace('.webm', '.wav')
        
        # Use ffmpeg directly via subprocess
        cmd = [
            'ffmpeg',
            '-i', webm_path,
            '-ar', '16000',
            '-ac', '1',
            '-f', 'wav',
            '-acodec', 'pcm_s16le',
            wav_path,
            '-y'
        ]
        
        result = subprocess.run(cmd, capture_output=True, timeout=5)
        
        if result.returncode == 0 and os.path.exists(wav_path):
            with open(wav_path, 'rb') as f:
                wav_data = f.read()
            
            logger.info(f"Successfully converted WebM to WAV: {len(wav_data)} bytes")
            return wav_data
        else:
            logger.error(f"FFmpeg conversion failed: {result.stderr.decode()}")
            return None
            
    except Exception as e:
        logger.error(f"WebM conversion error: {e}")
        return None
    finally:
        # Cleanup temp files
        for path in [webm_path, wav_path]:
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                except:
                    pass

def create_wav_header(data_size: int, sample_rate: int = 16000, channels: int = 1, bits_per_sample: int = 16) -> bytes:
    """Create a standard WAV file header."""
    
    # WAV file header structure
    header = bytearray(44)
    
    # RIFF chunk descriptor
    header[0:4] = b'RIFF'
    header[4:8] = struct.pack('<I', 36 + data_size)  # File size - 8
    header[8:12] = b'WAVE'
    
    # fmt sub-chunk
    header[12:16] = b'fmt '
    header[16:20] = struct.pack('<I', 16)  # Sub-chunk size
    header[20:22] = struct.pack('<H', 1)   # Audio format (PCM)
    header[22:24] = struct.pack('<H', channels)
    header[24:28] = struct.pack('<I', sample_rate)
    header[28:32] = struct.pack('<I', sample_rate * channels * bits_per_sample // 8)  # Byte rate
    header[32:34] = struct.pack('<H', channels * bits_per_sample // 8)  # Block align
    header[34:36] = struct.pack('<H', bits_per_sample)
    
    # data sub-chunk
    header[36:40] = b'data'
    header[40:44] = struct.pack('<I', data_size)
    
    return bytes(header)

def detect_audio_format(data: bytes) -> str:
    """Detect audio format from header bytes."""
    if len(data) < 8:
        return 'unknown'
    
    # Check for various audio formats
    if data.startswith(b'RIFF') and b'WAVE' in data[:12]:
        return 'wav'
    elif data.startswith(b'OggS'):
        return 'ogg'
    elif data.startswith(b'\x1a\x45\xdf\xa3'):
        return 'webm'
    elif data.startswith(b'ID3') or data.startswith(b'\xff\xfb') or data.startswith(b'\xff\xf3'):
        return 'mp3'
    else:
        return 'unknown'