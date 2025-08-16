"""
Session Management for Audio Chunk Storage
Handles session creation, audio chunk saving, and folder organization
"""

import os
import time
import uuid
import logging
import wave
import numpy as np
from typing import Dict, Optional, List
from pathlib import Path
import json
import threading

logger = logging.getLogger(__name__)

class SessionManager:
    """
    Manages audio recording sessions with systematic file storage.
    Each session gets a unique ID and folder for organized chunk storage.
    """
    
    def __init__(self, base_output_dir: str = "output"):
        """
        Initialize session manager.
        
        Args:
            base_output_dir: Base directory for all session outputs
        """
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(exist_ok=True)
        
        # Active sessions tracking
        self.active_sessions: Dict[str, 'AudioSession'] = {}
        self.lock = threading.Lock()
        
        logger.info(f"Session manager initialized with output directory: {self.base_output_dir}")
    
    def create_session(self, session_id: Optional[str] = None) -> str:
        """
        Create a new audio recording session.
        
        Args:
            session_id: Optional custom session ID, otherwise auto-generated
            
        Returns:
            str: Session ID
        """
        if not session_id:
            # Generate session ID with timestamp and short UUID
            timestamp = int(time.time())
            short_uuid = str(uuid.uuid4())[:8]
            session_id = f"session{timestamp}_{short_uuid}"
        
        with self.lock:
            if session_id in self.active_sessions:
                logger.warning(f"Session {session_id} already exists, returning existing session")
                return session_id
            
            # Create session object
            session = AudioSession(session_id, self.base_output_dir)
            self.active_sessions[session_id] = session
            
            logger.info(f"✅ Created new session: {session_id}")
            return session_id
    
    def get_session(self, session_id: str) -> Optional['AudioSession']:
        """Get an existing session by ID."""
        with self.lock:
            return self.active_sessions.get(session_id)
    
    def close_session(self, session_id: str) -> bool:
        """
        Close and finalize a session.
        
        Args:
            session_id: Session to close
            
        Returns:
            bool: True if session was closed successfully
        """
        with self.lock:
            if session_id not in self.active_sessions:
                logger.warning(f"Session {session_id} not found")
                return False
            
            session = self.active_sessions[session_id]
            session.finalize()
            del self.active_sessions[session_id]
            
            logger.info(f"✅ Closed session: {session_id} ({session.chunk_count} chunks saved)")
            return True
    
    def cleanup_old_sessions(self, max_age_hours: int = 24) -> int:
        """
        Clean up sessions older than specified hours.
        
        Args:
            max_age_hours: Maximum age in hours before cleanup
            
        Returns:
            int: Number of sessions cleaned up
        """
        cutoff_time = time.time() - (max_age_hours * 3600)
        cleaned_count = 0
        
        # Find old session folders
        for session_dir in self.base_output_dir.iterdir():
            if not session_dir.is_dir() or not session_dir.name.startswith('session'):
                continue
            
            try:
                # Check if session has a metadata file with creation time
                metadata_file = session_dir / "session_info.json"
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                        if metadata.get('created_at', 0) < cutoff_time:
                            import shutil
                            shutil.rmtree(session_dir)
                            cleaned_count += 1
                            logger.info(f"Cleaned up old session: {session_dir.name}")
                else:
                    # Fallback to directory modification time
                    if session_dir.stat().st_mtime < cutoff_time:
                        import shutil
                        shutil.rmtree(session_dir)
                        cleaned_count += 1
                        logger.info(f"Cleaned up old session: {session_dir.name}")
                        
            except Exception as e:
                logger.error(f"Error cleaning up session {session_dir.name}: {e}")
        
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} old sessions")
        
        return cleaned_count
    
    def get_session_stats(self) -> Dict:
        """Get statistics about all sessions."""
        with self.lock:
            stats = {
                'active_sessions': len(self.active_sessions),
                'total_chunks_active': sum(s.chunk_count for s in self.active_sessions.values()),
                'session_details': {
                    sid: {
                        'chunk_count': session.chunk_count,
                        'created_at': session.created_at,
                        'folder_path': str(session.session_dir)
                    }
                    for sid, session in self.active_sessions.items()
                }
            }
        
        # Count total session folders
        total_session_dirs = len([
            d for d in self.base_output_dir.iterdir() 
            if d.is_dir() and d.name.startswith('session')
        ])
        stats['total_session_folders'] = total_session_dirs
        
        return stats


class AudioSession:
    """
    Represents a single audio recording session with systematic chunk storage.
    """
    
    def __init__(self, session_id: str, base_output_dir: Path):
        """
        Initialize audio session.
        
        Args:
            session_id: Unique session identifier
            base_output_dir: Base directory for output
        """
        self.session_id = session_id
        self.created_at = time.time()
        self.chunk_count = 0
        
        # Create session directory
        self.session_dir = base_output_dir / session_id
        self.session_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        self.chunks_dir = self.session_dir / "chunks"
        self.chunks_dir.mkdir(exist_ok=True)
        
        # Session metadata
        self.metadata = {
            'session_id': session_id,
            'created_at': self.created_at,
            'created_at_human': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.created_at)),
            'chunk_count': 0,
            'chunks': []
        }
        
        self._save_metadata()
        logger.info(f"📁 Session folder created: {self.session_dir}")
    
    def save_audio_chunk(self, audio_data: bytes, prediction_result: Optional[Dict] = None, 
                        chunk_type: str = "speech") -> str:
        """
        Save an audio chunk to the session folder.
        
        Args:
            audio_data: Raw audio bytes (WAV format preferred)
            prediction_result: Optional prediction results to save alongside
            chunk_type: Type of chunk ("speech", "vad_segment", "raw", etc.)
            
        Returns:
            str: Path to saved chunk file
        """
        self.chunk_count += 1
        
        # Generate chunk filename
        chunk_filename = f"{self.chunk_count:03d}.wav"
        chunk_path = self.chunks_dir / chunk_filename
        
        try:
            # Save audio data
            if self._is_wav_format(audio_data):
                # Already WAV format, save directly
                with open(chunk_path, 'wb') as f:
                    f.write(audio_data)
                logger.debug(f"Saved WAV chunk: {chunk_path}")
            else:
                # Convert raw PCM to WAV
                self._save_pcm_as_wav(audio_data, chunk_path)
                logger.debug(f"Converted and saved PCM chunk: {chunk_path}")
            
            # Update metadata
            chunk_info = {
                'chunk_id': self.chunk_count,
                'filename': chunk_filename,
                'chunk_type': chunk_type,
                'size_bytes': len(audio_data),
                'saved_at': time.time(),
                'saved_at_human': time.strftime('%Y-%m-%d %H:%M:%S'),
                'audio_format': 'wav' if self._is_wav_format(audio_data) else 'pcm_converted'
            }
            
            # Add prediction results if provided
            if prediction_result:
                chunk_info['prediction'] = prediction_result
            
            self.metadata['chunks'].append(chunk_info)
            self.metadata['chunk_count'] = self.chunk_count
            self._save_metadata()
            
            logger.info(f"💾 Saved audio chunk {self.chunk_count}: {chunk_path}")
            return str(chunk_path)
            
        except Exception as e:
            logger.error(f"Failed to save audio chunk {self.chunk_count}: {e}")
            # Rollback chunk count on failure
            self.chunk_count -= 1
            raise
    
    def _is_wav_format(self, audio_data: bytes) -> bool:
        """Check if audio data is in WAV format."""
        return audio_data.startswith(b'RIFF') and b'WAVE' in audio_data[:12]
    
    def _save_pcm_as_wav(self, pcm_data: bytes, output_path: Path, 
                        sample_rate: int = 16000, channels: int = 1, sample_width: int = 2):
        """
        Convert raw PCM data to WAV format and save.
        
        Args:
            pcm_data: Raw PCM bytes
            output_path: Output WAV file path
            sample_rate: Sample rate (default 16kHz for speech)
            channels: Number of channels (default mono)
            sample_width: Sample width in bytes (default 16-bit)
        """
        try:
            with wave.open(str(output_path), 'wb') as wav_file:
                wav_file.setnchannels(channels)
                wav_file.setsampwidth(sample_width)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(pcm_data)
                
        except Exception as e:
            logger.error(f"PCM to WAV conversion failed: {e}")
            # Fallback: save as raw PCM with .pcm extension
            raw_path = output_path.with_suffix('.pcm')
            with open(raw_path, 'wb') as f:
                f.write(pcm_data)
            logger.warning(f"Saved as raw PCM instead: {raw_path}")
    
    def _save_metadata(self):
        """Save session metadata to JSON file."""
        try:
            metadata_path = self.session_dir / "session_info.json"
            with open(metadata_path, 'w') as f:
                json.dump(self.metadata, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save session metadata: {e}")
    
    def finalize(self):
        """Finalize the session and save final metadata."""
        self.metadata['finalized_at'] = time.time()
        self.metadata['finalized_at_human'] = time.strftime('%Y-%m-%d %H:%M:%S')
        self.metadata['final_chunk_count'] = self.chunk_count
        self._save_metadata()
        
        logger.info(f"📋 Finalized session {self.session_id}: {self.chunk_count} chunks saved")
    
    def get_chunk_list(self) -> List[str]:
        """Get list of all chunk files in order."""
        chunk_files = []
        for i in range(1, self.chunk_count + 1):
            chunk_file = self.chunks_dir / f"{i:03d}.wav"
            if chunk_file.exists():
                chunk_files.append(str(chunk_file))
            else:
                # Check for .pcm fallback
                pcm_file = self.chunks_dir / f"{i:03d}.pcm"
                if pcm_file.exists():
                    chunk_files.append(str(pcm_file))
        return chunk_files
    
    def get_session_summary(self) -> Dict:
        """Get comprehensive session summary."""
        return {
            'session_id': self.session_id,
            'created_at': self.created_at,
            'chunk_count': self.chunk_count,
            'session_dir': str(self.session_dir),
            'chunks_dir': str(self.chunks_dir),
            'chunk_files': self.get_chunk_list(),
            'metadata': self.metadata
        }


# Global session manager instance
session_manager = SessionManager()