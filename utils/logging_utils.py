import logging
import time
from typing import Dict, List, Any
from collections import defaultdict, deque
import json

class PerformanceLogger:
    """
    Performance logger for tracking audio processing metrics.
    Provides detailed logging and statistics for each processing method.
    """
    
    def __init__(self, max_history: int = 100):
        self.max_history = max_history
        self.method_stats = defaultdict(lambda: {
            'predictions': deque(maxlen=max_history),
            'inference_times': deque(maxlen=max_history),
            'errors': deque(maxlen=max_history),
            'total_calls': 0,
            'total_errors': 0
        })
        
        # Setup structured logging
        self.setup_logging()
    
    def setup_logging(self):
        """Setup structured logging with proper formatting."""
        # Create custom formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Setup console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        # Setup file handler
        file_handler = logging.FileHandler('audio_digit_classifier.log')
        file_handler.setFormatter(formatter)
        
        # Configure root logger
        logging.basicConfig(
            level=logging.INFO,
            handlers=[console_handler, file_handler]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def log_prediction(self, method: str, result: Dict[str, Any]):
        """
        Log a prediction result with performance metrics.
        
        Args:
            method: Processing method name
            result: Prediction result dictionary
        """
        stats = self.method_stats[method]
        stats['total_calls'] += 1
        
        if result.get('success', True):
            stats['predictions'].append({
                'digit': result.get('predicted_digit'),
                'timestamp': result.get('timestamp', time.time()),
                'inference_time': result.get('inference_time', 0)
            })
            stats['inference_times'].append(result.get('inference_time', 0))
            
            self.logger.info(json.dumps({
                'event': 'prediction',
                'method': method,
                'digit': result.get('predicted_digit'),
                'inference_time': result.get('inference_time'),
                'timestamp': result.get('timestamp')
            }))
        else:
            stats['total_errors'] += 1
            stats['errors'].append({
                'error': result.get('error'),
                'timestamp': result.get('timestamp', time.time()),
                'inference_time': result.get('inference_time', 0)
            })
            
            self.logger.error(json.dumps({
                'event': 'error',
                'method': method,
                'error': result.get('error'),
                'timestamp': result.get('timestamp')
            }))
    
    def get_method_stats(self, method: str) -> Dict[str, Any]:
        """
        Get performance statistics for a specific method.
        
        Args:
            method: Processing method name
            
        Returns:
            Dictionary with performance statistics
        """
        stats = self.method_stats[method]
        inference_times = list(stats['inference_times'])
        
        if not inference_times:
            return {
                'method': method,
                'total_calls': stats['total_calls'],
                'successful_predictions': 0,
                'error_rate': 0.0,
                'avg_inference_time': 0.0,
                'min_inference_time': 0.0,
                'max_inference_time': 0.0
            }
        
        successful_predictions = len(inference_times)
        error_rate = stats['total_errors'] / stats['total_calls'] if stats['total_calls'] > 0 else 0
        
        return {
            'method': method,
            'total_calls': stats['total_calls'],
            'successful_predictions': successful_predictions,
            'error_rate': round(error_rate * 100, 2),
            'avg_inference_time': round(sum(inference_times) / len(inference_times), 3),
            'min_inference_time': round(min(inference_times), 3),
            'max_inference_time': round(max(inference_times), 3),
            'recent_predictions': list(stats['predictions'])[-10:]  # Last 10 predictions
        }
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics for all processing methods."""
        all_stats = {}
        for method in self.method_stats.keys():
            all_stats[method] = self.get_method_stats(method)
        
        return all_stats
    
    def get_comparison_report(self) -> str:
        """
        Generate a comparison report of all processing methods.
        
        Returns:
            Formatted string with method comparison
        """
        all_stats = self.get_all_stats()
        
        if not all_stats:
            return "No statistics available yet."
        
        report = "\n=== Audio Processing Method Comparison ===\n\n"
        
        for method, stats in all_stats.items():
            report += f"Method: {method}\n"
            report += f"  Total Calls: {stats['total_calls']}\n"
            report += f"  Successful: {stats['successful_predictions']}\n"
            report += f"  Error Rate: {stats['error_rate']}%\n"
            report += f"  Avg Time: {stats['avg_inference_time']}s\n"
            report += f"  Min/Max: {stats['min_inference_time']}s / {stats['max_inference_time']}s\n"
            report += "\n"
        
        # Find best performing method
        if len(all_stats) > 1:
            best_speed = min(all_stats.items(), key=lambda x: x[1]['avg_inference_time'])
            best_accuracy = min(all_stats.items(), key=lambda x: x[1]['error_rate'])
            
            report += f"Fastest Method: {best_speed[0]} ({best_speed[1]['avg_inference_time']}s avg)\n"
            report += f"Most Accurate: {best_accuracy[0]} ({best_accuracy[1]['error_rate']}% error rate)\n"
        
        return report
    
    def log_system_info(self, info: Dict[str, Any]):
        """Log system information for debugging."""
        self.logger.info(json.dumps({
            'event': 'system_info',
            'timestamp': time.time(),
            **info
        }))
    
    def log_audio_info(self, duration: float, format_info: Dict[str, Any]):
        """Log audio input information."""
        self.logger.debug(json.dumps({
            'event': 'audio_input',
            'duration': duration,
            'format': format_info,
            'timestamp': time.time()
        }))

# Global performance logger instance
performance_logger = PerformanceLogger()

def setup_flask_logging(app):
    """Setup logging configuration for Flask application."""
    if not app.debug:
        # Production logging
        file_handler = logging.FileHandler('flask_app.log')
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)s %(name)s %(message)s'
        ))
        file_handler.setLevel(logging.INFO)
        app.logger.addHandler(file_handler)
        app.logger.setLevel(logging.INFO)
    
    app.logger.info('Audio Digit Classifier startup')