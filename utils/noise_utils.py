import numpy as np
import wave
import io
import logging
from typing import Literal, Optional

logger = logging.getLogger(__name__)

NoiseType = Literal['white', 'pink', 'brown', 'background', 'speech']

class NoiseGenerator:
    """
    Audio noise generator for robustness testing.
    Supports various types of noise injection for testing digit recognition.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def generate_white_noise(self, duration: float, sample_rate: int = 16000, 
                           amplitude: float = 0.1) -> np.ndarray:
        """
        Generate white noise signal.
        
        Args:
            duration: Duration in seconds
            sample_rate: Sample rate in Hz
            amplitude: Noise amplitude (0.0 to 1.0)
            
        Returns:
            Numpy array of white noise
        """
        samples = int(duration * sample_rate)
        noise = np.random.normal(0, amplitude, samples)
        return noise.astype(np.float32)
    
    def generate_pink_noise(self, duration: float, sample_rate: int = 16000, 
                          amplitude: float = 0.1) -> np.ndarray:
        """
        Generate pink noise (1/f noise).
        
        Args:
            duration: Duration in seconds  
            sample_rate: Sample rate in Hz
            amplitude: Noise amplitude
            
        Returns:
            Numpy array of pink noise
        """
        samples = int(duration * sample_rate)
        
        # Generate white noise
        white = np.random.randn(samples)
        
        # Apply 1/f filter in frequency domain
        freqs = np.fft.fftfreq(samples, 1/sample_rate)
        freqs[0] = 1  # Avoid division by zero
        
        # 1/f filter
        filter_response = 1.0 / np.sqrt(np.abs(freqs))
        filter_response[0] = 0
        
        # Apply filter
        white_fft = np.fft.fft(white)
        pink_fft = white_fft * filter_response
        pink = np.real(np.fft.ifft(pink_fft))
        
        # Normalize and scale
        pink = pink / np.std(pink) * amplitude
        return pink.astype(np.float32)
    
    def generate_brown_noise(self, duration: float, sample_rate: int = 16000, 
                           amplitude: float = 0.1) -> np.ndarray:
        """
        Generate brown noise (1/f^2 noise).
        
        Args:
            duration: Duration in seconds
            sample_rate: Sample rate in Hz
            amplitude: Noise amplitude
            
        Returns:
            Numpy array of brown noise
        """
        samples = int(duration * sample_rate)
        
        # Generate white noise and integrate (cumulative sum)
        white = np.random.randn(samples)
        brown = np.cumsum(white)
        
        # Normalize and scale
        brown = brown / np.std(brown) * amplitude
        return brown.astype(np.float32)
    
    def generate_background_noise(self, duration: float, sample_rate: int = 16000, 
                                amplitude: float = 0.05) -> np.ndarray:
        """
        Generate realistic background noise (mixture of different noise types).
        
        Args:
            duration: Duration in seconds
            sample_rate: Sample rate in Hz
            amplitude: Noise amplitude
            
        Returns:
            Numpy array of background noise
        """
        # Mix different types of noise
        white = self.generate_white_noise(duration, sample_rate, amplitude * 0.3)
        pink = self.generate_pink_noise(duration, sample_rate, amplitude * 0.5)
        
        # Add some low-frequency rumble
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        rumble = amplitude * 0.2 * np.sin(2 * np.pi * 60 * t)  # 60 Hz hum
        
        background = white + pink + rumble
        return background.astype(np.float32)
    
    def inject_noise(self, audio_data: bytes, noise_type: NoiseType, 
                    noise_level: float = 0.1) -> bytes:
        """
        Inject noise into existing audio data.
        
        Args:
            audio_data: Original audio bytes (WAV format)
            noise_type: Type of noise to inject
            noise_level: Noise level relative to signal (0.0 to 1.0)
            
        Returns:
            Audio bytes with noise injected
            
        Raises:
            Exception: If noise injection fails
        """
        try:
            # Convert input audio to numpy
            with wave.open(io.BytesIO(audio_data), 'rb') as wav_file:
                frames = wav_file.readframes(wav_file.getnframes())
                sample_rate = wav_file.getframerate()
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                
                if sample_width != 2:
                    raise Exception(f"Unsupported sample width: {sample_width}")
                
                audio_array = np.frombuffer(frames, dtype=np.int16)
                
                # Convert to float
                audio_float = audio_array.astype(np.float32) / 32767.0
                
                # Handle stereo
                if channels == 2:
                    audio_float = audio_float.reshape(-1, 2)
                    # Process each channel separately
                    for ch in range(2):
                        channel_data = audio_float[:, ch]
                        duration = len(channel_data) / sample_rate
                        
                        # Generate appropriate noise
                        if noise_type == 'white':
                            noise = self.generate_white_noise(duration, sample_rate, noise_level)
                        elif noise_type == 'pink':
                            noise = self.generate_pink_noise(duration, sample_rate, noise_level)
                        elif noise_type == 'brown':
                            noise = self.generate_brown_noise(duration, sample_rate, noise_level)
                        elif noise_type == 'background':
                            noise = self.generate_background_noise(duration, sample_rate, noise_level)
                        else:
                            raise Exception(f"Unsupported noise type: {noise_type}")
                        
                        # Ensure same length
                        if len(noise) != len(channel_data):
                            noise = noise[:len(channel_data)]
                        
                        # Add noise
                        audio_float[:, ch] = channel_data + noise
                    
                    # Flatten back
                    audio_float = audio_float.flatten()
                else:
                    # Mono processing
                    duration = len(audio_float) / sample_rate
                    
                    # Generate noise
                    if noise_type == 'white':
                        noise = self.generate_white_noise(duration, sample_rate, noise_level)
                    elif noise_type == 'pink':
                        noise = self.generate_pink_noise(duration, sample_rate, noise_level)
                    elif noise_type == 'brown':
                        noise = self.generate_brown_noise(duration, sample_rate, noise_level)
                    elif noise_type == 'background':
                        noise = self.generate_background_noise(duration, sample_rate, noise_level)
                    else:
                        raise Exception(f"Unsupported noise type: {noise_type}")
                    
                    # Ensure same length
                    if len(noise) != len(audio_float):
                        noise = noise[:len(audio_float)]
                    
                    # Add noise
                    audio_float = audio_float + noise
                
                # Clip to prevent overflow
                audio_float = np.clip(audio_float, -1.0, 1.0)
                
                # Convert back to int16
                audio_int16 = (audio_float * 32767).astype(np.int16)
                
                # Create output WAV
                output = io.BytesIO()
                with wave.open(output, 'wb') as output_wav:
                    output_wav.setnchannels(channels)
                    output_wav.setsampwidth(sample_width)
                    output_wav.setframerate(sample_rate)
                    output_wav.writeframes(audio_int16.tobytes())
                
                self.logger.debug(f"Injected {noise_type} noise at level {noise_level}")
                return output.getvalue()
                
        except Exception as e:
            self.logger.error(f"Noise injection failed: {str(e)}")
            raise Exception(f"Failed to inject noise: {str(e)}")
    
    def create_pure_noise(self, noise_type: NoiseType, duration: float = 1.0, 
                         sample_rate: int = 16000, amplitude: float = 0.3) -> bytes:
        """
        Create pure noise audio file for testing.
        
        Args:
            noise_type: Type of noise to generate
            duration: Duration in seconds
            sample_rate: Sample rate in Hz
            amplitude: Noise amplitude
            
        Returns:
            WAV audio bytes containing pure noise
        """
        try:
            # Generate noise
            if noise_type == 'white':
                noise = self.generate_white_noise(duration, sample_rate, amplitude)
            elif noise_type == 'pink':
                noise = self.generate_pink_noise(duration, sample_rate, amplitude)
            elif noise_type == 'brown':
                noise = self.generate_brown_noise(duration, sample_rate, amplitude)
            elif noise_type == 'background':
                noise = self.generate_background_noise(duration, sample_rate, amplitude)
            else:
                raise Exception(f"Unsupported noise type: {noise_type}")
            
            # Convert to int16
            noise_int16 = (np.clip(noise, -1.0, 1.0) * 32767).astype(np.int16)
            
            # Create WAV
            output = io.BytesIO()
            with wave.open(output, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(noise_int16.tobytes())
            
            return output.getvalue()
            
        except Exception as e:
            self.logger.error(f"Pure noise generation failed: {str(e)}")
            raise Exception(f"Failed to create pure noise: {str(e)}")

# Global noise generator instance
noise_generator = NoiseGenerator()