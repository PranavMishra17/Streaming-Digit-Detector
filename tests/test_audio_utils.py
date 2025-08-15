import unittest
import sys
import os
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.audio_utils import (
    validate_audio_format, 
    convert_to_mono_16khz,
    get_audio_duration,
    audio_to_numpy,
    create_test_audio
)
from utils.noise_utils import NoiseGenerator

class TestAudioUtils(unittest.TestCase):
    """
    Unit tests for audio utility functions.
    Tests audio format validation, conversion, and test data generation.
    """
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_audio = create_test_audio('3', duration=0.5)
        self.noise_generator = NoiseGenerator()
    
    def test_create_test_audio(self):
        """Test test audio generation."""
        # Test different digits
        for digit in '0123456789':
            audio = create_test_audio(digit, duration=0.5)
            self.assertIsInstance(audio, bytes)
            self.assertGreater(len(audio), 100)  # Should have reasonable size
    
    def test_validate_audio_format(self):
        """Test audio format validation."""
        # Test valid audio
        is_valid = validate_audio_format(self.test_audio)
        self.assertTrue(is_valid)
        
        # Test invalid audio
        invalid_audio = b"not_audio_data"
        is_valid = validate_audio_format(invalid_audio)
        self.assertFalse(is_valid)
    
    def test_get_audio_duration(self):
        """Test audio duration calculation."""
        duration = get_audio_duration(self.test_audio)
        self.assertGreater(duration, 0)
        self.assertAlmostEqual(duration, 0.5, delta=0.1)  # Should be close to 0.5s
    
    def test_audio_to_numpy(self):
        """Test audio to numpy conversion."""
        audio_array, sample_rate = audio_to_numpy(self.test_audio)
        
        self.assertIsInstance(audio_array, np.ndarray)
        self.assertIsInstance(sample_rate, int)
        self.assertGreater(len(audio_array), 0)
        self.assertGreater(sample_rate, 0)
        
        # Audio should be normalized to [-1, 1]
        self.assertLessEqual(np.max(np.abs(audio_array)), 1.0)
    
    def test_convert_to_mono_16khz(self):
        """Test audio format conversion."""
        converted_audio = convert_to_mono_16khz(self.test_audio)
        
        self.assertIsInstance(converted_audio, bytes)
        self.assertGreater(len(converted_audio), 0)
        
        # Validate converted audio
        is_valid = validate_audio_format(converted_audio)
        self.assertTrue(is_valid)
        
        # Check properties of converted audio
        audio_array, sample_rate = audio_to_numpy(converted_audio)
        self.assertEqual(sample_rate, 16000)
    
    def test_noise_generation(self):
        """Test noise generation functions."""
        duration = 0.1  # Short duration for testing
        
        # White noise
        white_noise = self.noise_generator.generate_white_noise(duration)
        self.assertIsInstance(white_noise, np.ndarray)
        self.assertGreater(len(white_noise), 0)
        
        # Pink noise  
        pink_noise = self.noise_generator.generate_pink_noise(duration)
        self.assertIsInstance(pink_noise, np.ndarray)
        self.assertGreater(len(pink_noise), 0)
        
        # Brown noise
        brown_noise = self.noise_generator.generate_brown_noise(duration)
        self.assertIsInstance(brown_noise, np.ndarray)
        self.assertGreater(len(brown_noise), 0)
        
        # Background noise
        bg_noise = self.noise_generator.generate_background_noise(duration)
        self.assertIsInstance(bg_noise, np.ndarray)
        self.assertGreater(len(bg_noise), 0)
    
    def test_noise_injection(self):
        """Test noise injection into audio."""
        # Test with different noise types
        noise_types = ['white', 'pink', 'brown', 'background']
        
        for noise_type in noise_types:
            with self.subTest(noise_type=noise_type):
                noisy_audio = self.noise_generator.inject_noise(
                    self.test_audio, 
                    noise_type, 
                    noise_level=0.1
                )
                
                self.assertIsInstance(noisy_audio, bytes)
                self.assertGreater(len(noisy_audio), 0)
                
                # Should still be valid audio
                is_valid = validate_audio_format(noisy_audio)
                self.assertTrue(is_valid)
    
    def test_create_pure_noise(self):
        """Test pure noise audio creation."""
        noise_types = ['white', 'pink', 'brown', 'background']
        
        for noise_type in noise_types:
            with self.subTest(noise_type=noise_type):
                pure_noise = self.noise_generator.create_pure_noise(
                    noise_type, 
                    duration=0.1
                )
                
                self.assertIsInstance(pure_noise, bytes)
                self.assertGreater(len(pure_noise), 0)
                
                # Should be valid audio
                is_valid = validate_audio_format(pure_noise)
                self.assertTrue(is_valid)
    
    def test_error_handling(self):
        """Test error handling in audio utilities."""
        # Test with invalid audio data
        invalid_audio = b"definitely_not_audio"
        
        # Should handle gracefully
        with self.assertRaises(Exception):
            convert_to_mono_16khz(invalid_audio)
        
        with self.assertRaises(Exception):
            audio_to_numpy(invalid_audio)
        
        # Duration should return 0 for invalid audio
        duration = get_audio_duration(invalid_audio)
        self.assertEqual(duration, 0.0)
    
    def test_audio_properties_consistency(self):
        """Test consistency of audio properties across operations."""
        # Create audio with specific properties
        original_duration = 0.8
        test_audio = create_test_audio('7', duration=original_duration)
        
        # Check original properties
        duration = get_audio_duration(test_audio)
        self.assertAlmostEqual(duration, original_duration, delta=0.1)
        
        # Convert and check properties preserved reasonably
        converted = convert_to_mono_16khz(test_audio)
        converted_duration = get_audio_duration(converted)
        
        # Duration should be approximately preserved
        self.assertAlmostEqual(converted_duration, original_duration, delta=0.2)
        
        # Add noise and check still valid
        noisy = self.noise_generator.inject_noise(converted, 'white', 0.05)
        noisy_duration = get_audio_duration(noisy)
        self.assertAlmostEqual(noisy_duration, original_duration, delta=0.3)

if __name__ == '__main__':
    unittest.main()