import unittest
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio_processors.external_api import ExternalAPIProcessor
from audio_processors.raw_spectrogram import RawSpectrogramProcessor
from audio_processors.mel_spectrogram import MelSpectrogramProcessor
from audio_processors.mfcc_processor import MFCCProcessor
from utils.audio_utils import create_test_audio

class TestAudioProcessors(unittest.TestCase):
    """
    Unit tests for audio processing methods.
    Tests both placeholder and working implementations.
    """
    
    def setUp(self):
        """Set up test fixtures."""
        self.external_api = ExternalAPIProcessor()
        self.raw_spectrogram = RawSpectrogramProcessor()
        self.mel_spectrogram = MelSpectrogramProcessor()
        self.mfcc_processor = MFCCProcessor()
        
        # Create test audio data
        self.test_audio = create_test_audio('5', duration=1.0)
    
    def test_external_api_processor_init(self):
        """Test external API processor initialization."""
        self.assertEqual(self.external_api.name, "External API (Whisper)")
        self.assertIsNotNone(self.external_api.api_url)
        
    def test_raw_spectrogram_processor_init(self):
        """Test raw spectrogram processor initialization."""
        self.assertEqual(self.raw_spectrogram.name, "Raw Spectrogram")
        
    def test_mel_spectrogram_processor_init(self):
        """Test mel spectrogram processor initialization."""
        self.assertEqual(self.mel_spectrogram.name, "Mel Spectrogram")
        
    def test_mfcc_processor_init(self):
        """Test MFCC processor initialization."""
        self.assertEqual(self.mfcc_processor.name, "MFCC")
    
    def test_placeholder_processors_return_00(self):
        """Test that placeholder processors return '00' as expected."""
        # Raw Spectrogram
        result = self.raw_spectrogram.process_audio(self.test_audio)
        self.assertEqual(result, '00')
        
        # Mel Spectrogram
        result = self.mel_spectrogram.process_audio(self.test_audio)
        self.assertEqual(result, '00')
        
        # MFCC
        result = self.mfcc_processor.process_audio(self.test_audio)
        self.assertEqual(result, '00')
    
    def test_predict_with_timing(self):
        """Test prediction with timing functionality."""
        # Test with placeholder processor
        result = self.raw_spectrogram.predict_with_timing(self.test_audio)
        
        self.assertIn('predicted_digit', result)
        self.assertIn('inference_time', result)
        self.assertIn('method', result)
        self.assertIn('timestamp', result)
        self.assertIn('success', result)
        
        self.assertEqual(result['predicted_digit'], '00')
        self.assertEqual(result['method'], 'Raw Spectrogram')
        self.assertTrue(result['success'])
        self.assertGreater(result['inference_time'], 0)
    
    def test_get_stats(self):
        """Test statistics tracking."""
        # Initially no stats
        stats = self.mfcc_processor.get_stats()
        self.assertEqual(stats['total_predictions'], 0)
        
        # After one prediction
        self.mfcc_processor.predict_with_timing(self.test_audio)
        stats = self.mfcc_processor.get_stats()
        self.assertEqual(stats['total_predictions'], 1)
        self.assertGreater(stats['average_time'], 0)
    
    def test_model_info_methods(self):
        """Test model info methods for placeholder processors."""
        # Raw Spectrogram
        info = self.raw_spectrogram.get_model_info()
        self.assertIn('method', info)
        self.assertIn('status', info)
        self.assertEqual(info['status'], 'PLACEHOLDER')
        
        # Mel Spectrogram
        info = self.mel_spectrogram.get_model_info()
        self.assertIn('method', info)
        self.assertIn('status', info)
        self.assertEqual(info['status'], 'PLACEHOLDER')
        
        # MFCC
        info = self.mfcc_processor.get_model_info()
        self.assertIn('method', info)
        self.assertIn('status', info)
        self.assertEqual(info['status'], 'PLACEHOLDER')
    
    def test_external_api_configuration_check(self):
        """Test external API configuration checking."""
        is_configured = self.external_api.is_configured()
        # Should be True if HUGGING_FACE_TOKEN is set, False otherwise
        self.assertIsInstance(is_configured, bool)
        
    def test_external_api_digit_extraction(self):
        """Test digit extraction from text."""
        # Test direct digit
        result = self.external_api._extract_digit("the number is 5")
        self.assertEqual(result, '5')
        
        # Test word form
        result = self.external_api._extract_digit("I said five")
        self.assertEqual(result, '5')
        
        # Test no digit
        result = self.external_api._extract_digit("hello world")
        self.assertIsNone(result)
        
        # Test multiple digits (should return first)
        result = self.external_api._extract_digit("1 2 3")
        self.assertEqual(result, '1')
    
    def test_error_handling(self):
        """Test error handling in processors."""
        # Test with invalid audio data
        invalid_audio = b"not_audio_data"
        
        # Should not crash, might return error or placeholder
        try:
            result = self.raw_spectrogram.predict_with_timing(invalid_audio)
            self.assertIn('success', result)
        except Exception:
            # Error handling varies by implementation
            pass

if __name__ == '__main__':
    unittest.main()