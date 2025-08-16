"""
Comprehensive Training Script for Digit Classification Models
Supports all three pipelines: MFCC+Dense, MelCNN, RawCNN
Includes checkpoints, early stopping, extensive logging, and visualization
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

# Import our modules
from ml_training.data.dataset_loader import load_and_prepare_data
from ml_training.pipelines.mfcc_pipeline import setup_mfcc_pipeline
from ml_training.pipelines.mel_cnn_pipeline import setup_mel_cnn_pipeline
from ml_training.pipelines.raw_cnn_pipeline import setup_raw_cnn_pipeline
from ml_training.utils.logging_utils import setup_training_logger, SafeExecutor
from ml_training.utils.visualization import create_visualizer

logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Universal model trainer with comprehensive features:
    - Checkpointing and model saving
    - Early stopping with patience
    - Learning rate scheduling
    - Comprehensive logging and metrics
    - Automatic mixed precision (AMP) support
    - Visualization and reporting
    """
    
    def __init__(self, model: nn.Module, train_loader, val_loader, test_loader,
                 device: torch.device, experiment_name: str, 
                 output_dir: str = "models", log_dir: str = "train_logs"):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model to train
            train_loader: Training data loader
            val_loader: Validation data loader  
            test_loader: Test data loader
            device: Training device
            experiment_name: Name for this training run
            output_dir: Directory to save models
            log_dir: Directory for logs
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.experiment_name = experiment_name
        
        # Setup directories
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Setup logger
        self.ml_logger = setup_training_logger(experiment_name, log_dir)
        self.safe_executor = SafeExecutor(self.ml_logger)
        
        # Training state
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')
        self.training_history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'lr': []
        }
        
        # Early stopping
        self.early_stopping_patience = 0
        self.early_stopping_counter = 0
        
        # Checkpointing
        self.checkpoint_dir = self.output_dir / experiment_name
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        
        # Setup visualization
        viz_dir = Path(log_dir) / "plots" / experiment_name
        self.visualizer = create_visualizer(str(viz_dir))
        
        self.ml_logger.logger.info(f"Trainer initialized for {experiment_name}")
        self.ml_logger.log_model_info(model, experiment_name)
    
    def setup_training(self, learning_rate: float = 0.001, weight_decay: float = 1e-4,
                      scheduler_type: str = 'plateau', early_stopping_patience: int = 10,
                      gradient_clipping: float = None) -> None:
        """
        Setup training components.
        
        Args:
            learning_rate: Initial learning rate
            weight_decay: L2 regularization weight
            scheduler_type: Type of learning rate scheduler
            early_stopping_patience: Patience for early stopping
            gradient_clipping: Max gradient norm for clipping
        """
        # Loss function
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Learning rate scheduler
        if scheduler_type == 'plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, 
                patience=5, verbose=True, min_lr=1e-6
            )
        elif scheduler_type == 'step':
            self.scheduler = StepLR(self.optimizer, step_size=10, gamma=0.7)
        elif scheduler_type == 'cosine':
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=50, eta_min=1e-6)
        else:
            self.scheduler = None
        
        # Early stopping
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_counter = 0
        
        # Gradient clipping
        self.gradient_clipping = gradient_clipping
        
        # Mixed precision training (if available)
        self.scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
        
        # Log configuration
        config = {
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'scheduler_type': scheduler_type,
            'early_stopping_patience': early_stopping_patience,
            'gradient_clipping': gradient_clipping,
            'mixed_precision': self.scaler is not None,
            'optimizer': 'Adam',
            'loss_function': 'CrossEntropyLoss with label smoothing',
            'device': str(self.device)
        }
        self.ml_logger.log_experiment_config(config)
    
    def train_epoch(self) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Returns:
            avg_loss: Average training loss
            accuracy: Training accuracy
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with self.ml_logger.timer(f"train_epoch_{self.current_epoch}"):
            for batch_idx, (data, target) in enumerate(self.train_loader):
                # Move to device
                data, target = data.to(self.device), target.to(self.device)
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass with mixed precision
                if self.scaler is not None:
                    with torch.cuda.amp.autocast():
                        output = self.model(data)
                        loss = self.criterion(output, target)
                else:
                    output = self.model(data)
                    loss = self.criterion(output, target)
                
                # Backward pass
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                    
                    # Gradient clipping
                    if self.gradient_clipping is not None:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.gradient_clipping
                        )
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    
                    # Gradient clipping
                    if self.gradient_clipping is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.gradient_clipping
                        )
                    
                    self.optimizer.step()
                
                # Statistics
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
                # Log batch progress
                if batch_idx % 50 == 0:
                    self.ml_logger.logger.debug(
                        f"Batch {batch_idx}/{len(self.train_loader)}: "
                        f"Loss={loss.item():.4f}, "
                        f"Acc={100.*correct/total:.2f}%"
                    )
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self) -> Tuple[float, float]:
        """
        Validate model.
        
        Returns:
            avg_loss: Average validation loss
            accuracy: Validation accuracy
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            with self.ml_logger.timer(f"val_epoch_{self.current_epoch}"):
                for data, target in self.val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    
                    # Forward pass
                    if self.scaler is not None:
                        with torch.cuda.amp.autocast():
                            output = self.model(data)
                            loss = self.criterion(output, target)
                    else:
                        output = self.model(data)
                        loss = self.criterion(output, target)
                    
                    # Statistics
                    total_loss += loss.item()
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
                    total += target.size(0)
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def test(self) -> Dict[str, Any]:
        """
        Evaluate on test set.
        
        Returns:
            results: Dictionary with test metrics and predictions
        """
        self.model.eval()
        all_preds = []
        all_targets = []
        total_loss = 0.0
        
        with torch.no_grad():
            with self.ml_logger.timer("test_evaluation"):
                for data, target in self.test_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    
                    # Forward pass
                    if self.scaler is not None:
                        with torch.cuda.amp.autocast():
                            output = self.model(data)
                            loss = self.criterion(output, target)
                    else:
                        output = self.model(data)
                        loss = self.criterion(output, target)
                    
                    total_loss += loss.item()
                    pred = output.argmax(dim=1)
                    
                    all_preds.extend(pred.cpu().numpy())
                    all_targets.extend(target.cpu().numpy())
        
        # Calculate metrics
        test_loss = total_loss / len(self.test_loader)
        test_acc = accuracy_score(all_targets, all_preds)
        class_report = classification_report(all_targets, all_preds, digits=4)
        conf_matrix = confusion_matrix(all_targets, all_preds)
        
        results = {
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix,
            'predictions': all_preds,
            'targets': all_targets
        }
        
        return results
    
    def save_checkpoint(self, is_best: bool = False) -> str:
        """
        Save model checkpoint.
        
        Args:
            is_best: Whether this is the best model so far
            
        Returns:
            checkpoint_path: Path to saved checkpoint
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_acc': self.best_val_acc,
            'best_val_loss': self.best_val_loss,
            'training_history': self.training_history,
            'model_config': {
                'model_class': self.model.__class__.__name__,
                'param_count': sum(p.numel() for p in self.model.parameters())
            }
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{self.current_epoch:03d}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Best model
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            self.ml_logger.log_checkpoint(
                self.current_epoch, str(best_path),
                {'val_acc': self.best_val_acc, 'val_loss': self.best_val_loss}
            )
        
        return str(checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.scaler and checkpoint.get('scaler_state_dict'):
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_val_acc = checkpoint['best_val_acc']
        self.best_val_loss = checkpoint['best_val_loss']
        self.training_history = checkpoint['training_history']
        
        self.ml_logger.logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def train(self, num_epochs: int, save_frequency: int = 10) -> Dict[str, Any]:
        """
        Main training loop.
        
        Args:
            num_epochs: Number of epochs to train
            save_frequency: Frequency of checkpoint saving
            
        Returns:
            results: Training results and final test evaluation
        """
        self.ml_logger.logger.info(f"Starting training for {num_epochs} epochs")
        
        try:
            for epoch in range(1, num_epochs + 1):
                self.current_epoch = epoch
                
                # Training epoch
                train_loss, train_acc = self.train_epoch()
                
                # Validation epoch
                if self.val_loader is not None:
                    val_loss, val_acc = self.validate()
                else:
                    val_loss, val_acc = train_loss, train_acc  # Fallback
                
                # Learning rate scheduling
                current_lr = self.optimizer.param_groups[0]['lr']
                if self.scheduler is not None:
                    if isinstance(self.scheduler, ReduceLROnPlateau):
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()
                
                # Update history
                self.training_history['train_loss'].append(train_loss)
                self.training_history['train_acc'].append(train_acc)
                self.training_history['val_loss'].append(val_loss)
                self.training_history['val_acc'].append(val_acc)
                self.training_history['lr'].append(current_lr)
                
                # Log epoch results
                self.ml_logger.log_training_epoch(
                    epoch, train_loss, train_acc, val_loss, val_acc, current_lr
                )
                
                # Check for best model
                is_best = val_acc > self.best_val_acc
                if is_best:
                    self.best_val_acc = val_acc
                    self.best_val_loss = val_loss
                    self.early_stopping_counter = 0
                else:
                    self.early_stopping_counter += 1
                
                # Save checkpoint
                if epoch % save_frequency == 0 or is_best:
                    self.save_checkpoint(is_best)
                
                # Early stopping check
                if self.early_stopping_counter >= self.early_stopping_patience:
                    self.ml_logger.logger.info(
                        f"Early stopping triggered at epoch {epoch} "
                        f"(patience: {self.early_stopping_patience})"
                    )
                    break
            
            # Final evaluation
            self.ml_logger.logger.info("Training completed. Evaluating on test set...")
            
            # Load best model for testing
            best_model_path = self.checkpoint_dir / "best_model.pt"
            if best_model_path.exists():
                self.load_checkpoint(str(best_model_path))
            
            test_results = self.test()
            
            # Log test results
            self.ml_logger.log_test_results(
                test_results['test_accuracy'],
                test_results['test_loss'],
                test_results['classification_report']
            )
            
            # Generate visualizations
            self._generate_visualizations(test_results)
            
            # Compile final results
            final_results = {
                'experiment_name': self.experiment_name,
                'total_epochs': self.current_epoch,
                'best_val_accuracy': self.best_val_acc,
                'best_val_loss': self.best_val_loss,
                'test_results': test_results,
                'training_history': self.training_history,
                'model_path': str(best_model_path)
            }
            
            return final_results
            
        except Exception as e:
            self.ml_logger.log_error(e, "training loop")
            raise
        finally:
            self.ml_logger.close()
    
    def _generate_visualizations(self, test_results: Dict[str, Any]) -> None:
        """Generate comprehensive visualizations."""
        try:
            # Training history
            self.visualizer.plot_training_history(
                self.training_history, 
                title=f"{self.experiment_name} Training History"
            )
            
            # Confusion matrix
            class_names = [str(i) for i in range(10)]
            self.visualizer.plot_confusion_matrix(
                test_results['targets'], 
                test_results['predictions'],
                class_names,
                title=f"{self.experiment_name} Confusion Matrix"
            )
            
            self.ml_logger.logger.info("Visualizations generated successfully")
            
        except Exception as e:
            self.ml_logger.log_error(e, "visualization generation")

def train_pipeline(pipeline_type: str, data_splits: Dict[str, Any], 
                  config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Train a specific pipeline.
    
    Args:
        pipeline_type: Type of pipeline ('mfcc', 'mel_cnn', 'raw_cnn')
        data_splits: Dataset splits
        config: Training configuration
        
    Returns:
        results: Training results
    """
    logger.info(f"Training {pipeline_type} pipeline...")
    
    # Setup pipeline
    if pipeline_type == 'mfcc':
        pipeline_components = setup_mfcc_pipeline(
            data_splits, batch_size=config['batch_size']
        )
    elif pipeline_type == 'mel_cnn':
        pipeline_components = setup_mel_cnn_pipeline(
            data_splits, batch_size=config['batch_size'] // 2  # Smaller batch for CNN
        )
    elif pipeline_type == 'raw_cnn':
        pipeline_components = setup_raw_cnn_pipeline(
            data_splits, batch_size=config['batch_size'] // 2  # Smaller batch for CNN
        )
    else:
        raise ValueError(f"Unknown pipeline type: {pipeline_type}")
    
    # Setup trainer
    trainer = ModelTrainer(
        model=pipeline_components['model'],
        train_loader=pipeline_components['train_loader'],
        val_loader=pipeline_components['val_loader'],
        test_loader=pipeline_components['test_loader'],
        device=pipeline_components['device'],
        experiment_name=f"{pipeline_type}_classifier"
    )
    
    # Setup training
    trainer.setup_training(
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        scheduler_type=config['scheduler_type'],
        early_stopping_patience=config['early_stopping_patience'],
        gradient_clipping=config['gradient_clipping']
    )
    
    # Train
    results = trainer.train(
        num_epochs=config['num_epochs'],
        save_frequency=config['save_frequency']
    )
    
    return results

def train_all_pipelines(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Train all three pipelines and compare results.
    
    Args:
        config: Training configuration
        
    Returns:
        all_results: Results from all pipelines
    """
    logger.info("Loading dataset...")
    data_splits = load_and_prepare_data(
        sample_rate=config['sample_rate'],
        max_length=config['max_length']
    )
    
    if data_splits is None:
        raise RuntimeError("Failed to load dataset")
    
    # Log dataset info
    ml_logger = setup_training_logger("comparison_study")
    ml_logger.log_dataset_info(data_splits['dataset_info'])
    ml_logger.close()
    
    all_results = {}
    pipelines = config['pipelines']
    
    for pipeline_type in pipelines:
        try:
            results = train_pipeline(pipeline_type, data_splits, config)
            all_results[pipeline_type] = results
            
            logger.info(f"{pipeline_type} training completed:")
            logger.info(f"  Best val accuracy: {results['best_val_accuracy']:.4f}")
            logger.info(f"  Test accuracy: {results['test_results']['test_accuracy']:.4f}")
            
        except Exception as e:
            logger.error(f"Failed to train {pipeline_type}: {str(e)}")
            continue
    
    # Generate comparison visualizations
    if len(all_results) > 1:
        try:
            logger.info("Generating comparison visualizations...")
            viz_dir = Path("train_logs/plots/comparison")
            viz_dir.mkdir(exist_ok=True, parents=True)
            
            visualizer = create_visualizer(str(viz_dir))
            
            # Format results for comparison
            comparison_data = {}
            for pipeline, results in all_results.items():
                comparison_data[pipeline] = {
                    'accuracy': results['test_results']['test_accuracy'] * 100,
                    'training_time': sum(results.get('training_times', [0])),
                    'param_count': sum(p.numel() for p in torch.load(results['model_path'])['model_state_dict'].values() if isinstance(p, torch.Tensor))
                }
            
            visualizer.compare_models(comparison_data)
            
        except Exception as e:
            logger.error(f"Failed to generate comparisons: {str(e)}")
    
    return all_results

def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description='Train digit classification models')
    parser.add_argument('--pipeline', type=str, choices=['mfcc', 'mel_cnn', 'raw_cnn', 'all'],
                       default='all', help='Pipeline to train')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--config', type=str, help='Path to config JSON file')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('train_logs/training.log'),
            logging.StreamHandler()
        ]
    )
    
    # Load or create config
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = {
            'pipelines': ['mfcc', 'mel_cnn', 'raw_cnn'] if args.pipeline == 'all' else [args.pipeline],
            'num_epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'weight_decay': 1e-4,
            'scheduler_type': 'plateau',
            'early_stopping_patience': 10,
            'gradient_clipping': 1.0,
            'save_frequency': 10,
            'sample_rate': 8000,
            'max_length': 8000
        }
    
    logger.info(f"Training configuration: {config}")
    
    try:
        if len(config['pipelines']) == 1:
            # Single pipeline training
            data_splits = load_and_prepare_data()
            results = train_pipeline(config['pipelines'][0], data_splits, config)
            logger.info("Single pipeline training completed successfully")
        else:
            # Multi-pipeline comparison
            results = train_all_pipelines(config)
            logger.info("Multi-pipeline training completed successfully")
        
        logger.info("Training script completed successfully")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()