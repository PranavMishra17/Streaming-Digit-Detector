"""
Comprehensive Logging and Error Handling Utilities for ML Training
"""

import logging
import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import json
import traceback
from contextlib import contextmanager
import time

class MLTrainingLogger:
    """
    Specialized logger for ML training with structured logging,
    performance metrics, and comprehensive error handling.
    """
    
    def __init__(self, log_dir: str = "train_logs", experiment_name: str = None):
        """
        Initialize ML training logger.
        
        Args:
            log_dir: Directory for log files
            experiment_name: Name for this training experiment
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True, parents=True)
        
        # Create experiment-specific log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = experiment_name or "ml_training"
        self.log_file = self.log_dir / f"{exp_name}_{timestamp}.log"
        
        # Setup logger
        self.logger = self._setup_logger()
        
        # Metrics tracking
        self.metrics = {}
        self.timers = {}
        
        self.logger.info(f"ML Training Logger initialized - Log file: {self.log_file}")
    
    def _setup_logger(self) -> logging.Logger:
        """Setup structured logger with file and console handlers."""
        logger = logging.getLogger(f"ml_training_{id(self)}")
        logger.setLevel(logging.DEBUG)
        
        # Avoid duplicate handlers
        if logger.handlers:
            logger.handlers.clear()
        
        # Custom formatter
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # File handler
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger
    
    def log_system_info(self):
        """Log system and environment information."""
        import torch
        import numpy as np
        import librosa
        
        self.logger.info("=== SYSTEM INFORMATION ===")
        self.logger.info(f"Python Version: {sys.version}")
        self.logger.info(f"PyTorch Version: {torch.__version__}")
        self.logger.info(f"NumPy Version: {np.__version__}")
        self.logger.info(f"Librosa Version: {librosa.__version__}")
        
        if torch.cuda.is_available():
            self.logger.info(f"CUDA Available: Yes")
            self.logger.info(f"CUDA Version: {torch.version.cuda}")
            self.logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            self.logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            self.logger.info("CUDA Available: No")
        
        self.logger.info("=== END SYSTEM INFO ===")
    
    def log_experiment_config(self, config: Dict[str, Any]):
        """Log experiment configuration."""
        self.logger.info("=== EXPERIMENT CONFIGURATION ===")
        for key, value in config.items():
            self.logger.info(f"{key}: {value}")
        
        # Save config to JSON
        config_file = self.log_dir / f"config_{self.log_file.stem}.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        
        self.logger.info(f"Configuration saved to: {config_file}")
        self.logger.info("=== END CONFIGURATION ===")
    
    def log_dataset_info(self, dataset_info: Dict[str, Any]):
        """Log dataset information and statistics."""
        self.logger.info("=== DATASET INFORMATION ===")
        for key, value in dataset_info.items():
            self.logger.info(f"{key}: {value}")
        self.logger.info("=== END DATASET INFO ===")
    
    def log_model_info(self, model, model_name: str = "Model"):
        """Log model architecture and parameter count."""
        try:
            import torch
            import torch.nn as nn
            
            self.logger.info(f"=== {model_name.upper()} ARCHITECTURE ===")
            
            if isinstance(model, nn.Module):
                # Count parameters
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                
                self.logger.info(f"Total Parameters: {total_params:,}")
                self.logger.info(f"Trainable Parameters: {trainable_params:,}")
                self.logger.info(f"Non-trainable Parameters: {total_params - trainable_params:,}")
                
                # Model architecture
                self.logger.info("Model Architecture:")
                for name, module in model.named_modules():
                    if len(list(module.children())) == 0:  # Leaf modules only
                        self.logger.info(f"  {name}: {module}")
            
            self.logger.info(f"=== END {model_name.upper()} ARCHITECTURE ===")
            
        except Exception as e:
            self.logger.error(f"Failed to log model info: {str(e)}")
    
    @contextmanager
    def timer(self, name: str):
        """Context manager for timing operations."""
        start_time = time.time()
        self.logger.debug(f"Timer started: {name}")
        
        try:
            yield
        finally:
            elapsed = time.time() - start_time
            self.timers[name] = elapsed
            self.logger.info(f"Timer '{name}': {elapsed:.4f} seconds")
    
    def log_training_epoch(self, epoch: int, train_loss: float, train_acc: float,
                          val_loss: float, val_acc: float, lr: float = None):
        """Log training epoch results."""
        log_msg = (f"Epoch {epoch:3d} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        if lr is not None:
            log_msg += f" | LR: {lr:.2e}"
        
        self.logger.info(log_msg)
        
        # Store metrics
        epoch_metrics = {
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        }
        
        if lr is not None:
            epoch_metrics['lr'] = lr
            
        self.metrics[f'epoch_{epoch}'] = epoch_metrics
    
    def log_test_results(self, test_acc: float, test_loss: float = None, 
                        classification_report: str = None):
        """Log final test results."""
        self.logger.info("=== TEST RESULTS ===")
        self.logger.info(f"Test Accuracy: {test_acc:.4f}")
        
        if test_loss is not None:
            self.logger.info(f"Test Loss: {test_loss:.4f}")
        
        if classification_report:
            self.logger.info("Classification Report:")
            for line in classification_report.split('\n'):
                if line.strip():
                    self.logger.info(f"  {line}")
        
        self.logger.info("=== END TEST RESULTS ===")
    
    def log_error(self, error: Exception, context: str = ""):
        """Log error with full traceback and context."""
        error_msg = f"ERROR in {context}: {str(error)}"
        self.logger.error(error_msg)
        self.logger.error(f"Error Type: {type(error).__name__}")
        self.logger.error("Full Traceback:")
        
        for line in traceback.format_exc().split('\n'):
            if line.strip():
                self.logger.error(f"  {line}")
    
    def log_checkpoint(self, epoch: int, model_path: str, metrics: Dict[str, float]):
        """Log checkpoint save information."""
        self.logger.info(f"Checkpoint saved at epoch {epoch}")
        self.logger.info(f"Model path: {model_path}")
        self.logger.info(f"Checkpoint metrics: {metrics}")
    
    def save_metrics_summary(self):
        """Save comprehensive metrics summary to JSON."""
        try:
            # Create metrics summary
            summary = {
                'experiment_timestamp': datetime.now().isoformat(),
                'log_file': str(self.log_file),
                'timers': self.timers,
                'training_metrics': self.metrics,
                'total_experiment_time': sum(self.timers.values())
            }
            
            # Save to JSON
            metrics_file = self.log_dir / f"metrics_summary_{self.log_file.stem}.json"
            with open(metrics_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            self.logger.info(f"Metrics summary saved to: {metrics_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save metrics summary: {str(e)}")
    
    def close(self):
        """Close logger and save final metrics."""
        self.logger.info("Closing ML Training Logger")
        self.save_metrics_summary()
        
        # Close handlers
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)

class SafeExecutor:
    """
    Wrapper for safe execution of ML operations with error handling.
    """
    
    def __init__(self, logger: MLTrainingLogger):
        self.logger = logger
    
    def safe_execute(self, func, *args, context: str = "", **kwargs):
        """
        Safely execute function with error handling and logging.
        
        Args:
            func: Function to execute
            *args: Function arguments
            context: Context description for error logging
            **kwargs: Function keyword arguments
            
        Returns:
            Result of function or None if error
        """
        try:
            with self.logger.timer(f"execute_{func.__name__}"):
                result = func(*args, **kwargs)
            return result
            
        except Exception as e:
            self.logger.log_error(e, context or f"executing {func.__name__}")
            return None
    
    def safe_model_operation(self, model, operation: str, *args, **kwargs):
        """
        Safely execute model operations (forward pass, training step, etc.).
        
        Args:
            model: PyTorch model
            operation: Operation name (forward, train_step, etc.)
            *args: Operation arguments
            **kwargs: Operation keyword arguments
            
        Returns:
            Operation result or None if error
        """
        try:
            if operation == "forward":
                with self.logger.timer("model_forward"):
                    return model(*args, **kwargs)
            elif operation == "train_step":
                return self._safe_train_step(model, *args, **kwargs)
            else:
                return getattr(model, operation)(*args, **kwargs)
                
        except Exception as e:
            self.logger.log_error(e, f"model {operation}")
            return None
    
    def _safe_train_step(self, model, data, target, optimizer, criterion):
        """Safely execute training step."""
        try:
            model.train()
            optimizer.zero_grad()
            
            with self.logger.timer("forward_pass"):
                output = model(data)
            
            with self.logger.timer("loss_computation"):
                loss = criterion(output, target)
            
            with self.logger.timer("backward_pass"):
                loss.backward()
            
            with self.logger.timer("optimizer_step"):
                optimizer.step()
            
            return output, loss
            
        except Exception as e:
            self.logger.log_error(e, "training step")
            return None, None

def setup_training_logger(experiment_name: str, log_dir: str = "train_logs") -> MLTrainingLogger:
    """
    Convenience function to setup training logger.
    
    Args:
        experiment_name: Name for the experiment
        log_dir: Directory for log files
        
    Returns:
        MLTrainingLogger instance
    """
    logger = MLTrainingLogger(log_dir=log_dir, experiment_name=experiment_name)
    logger.log_system_info()
    return logger

if __name__ == "__main__":
    # Test logger
    logger = setup_training_logger("test_experiment")
    
    # Test configuration logging
    config = {
        "model": "MFCC_CNN",
        "batch_size": 32,
        "learning_rate": 0.001,
        "epochs": 50
    }
    logger.log_experiment_config(config)
    
    # Test timer
    with logger.timer("test_operation"):
        time.sleep(1)
    
    # Test error logging
    try:
        raise ValueError("Test error")
    except Exception as e:
        logger.log_error(e, "testing error handling")
    
    logger.close()