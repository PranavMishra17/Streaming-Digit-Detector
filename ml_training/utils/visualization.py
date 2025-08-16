"""
Visualization Utilities for ML Training
Generate comprehensive graphs, plots, and reports
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json

logger = logging.getLogger(__name__)

class TrainingVisualizer:
    """
    Comprehensive visualization suite for ML training analysis.
    Generates training curves, confusion matrices, model comparisons, and reports.
    """
    
    def __init__(self, output_dir: str = "train_logs/plots"):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save plots and reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Set style
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        sns.set_palette("husl")
        
        logger.info(f"Visualizer initialized - Output directory: {self.output_dir}")
    
    def plot_training_history(self, history: Dict[str, List[float]], 
                            title: str = "Training History",
                            save_name: Optional[str] = None) -> str:
        """
        Plot training and validation curves.
        
        Args:
            history: Dictionary with 'train_loss', 'val_loss', 'train_acc', 'val_acc'
            title: Plot title
            save_name: Optional filename for saving
            
        Returns:
            Path to saved plot
        """
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            epochs = range(1, len(history['train_loss']) + 1)
            
            # Loss plots
            ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
            ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
            ax1.set_title('Model Loss', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Accuracy plots
            ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
            ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
            ax2.set_title('Model Accuracy', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy (%)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Loss smoothed (moving average)
            if len(history['train_loss']) > 5:
                window = min(5, len(history['train_loss']) // 4)
                train_smooth = pd.Series(history['train_loss']).rolling(window).mean()
                val_smooth = pd.Series(history['val_loss']).rolling(window).mean()
                
                ax3.plot(epochs, train_smooth, 'b-', label=f'Train Loss (MA-{window})', linewidth=2)
                ax3.plot(epochs, val_smooth, 'r-', label=f'Val Loss (MA-{window})', linewidth=2)
                ax3.set_title('Smoothed Loss Curves', fontsize=14, fontweight='bold')
                ax3.set_xlabel('Epoch')
                ax3.set_ylabel('Loss')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
            
            # Learning curves analysis
            train_val_gap = np.array(history['val_loss']) - np.array(history['train_loss'])
            ax4.plot(epochs, train_val_gap, 'g-', label='Validation - Training Loss', linewidth=2)
            ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            ax4.set_title('Overfitting Analysis', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Loss Difference')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            plt.suptitle(title, fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            # Save plot
            save_path = self._save_plot(fig, save_name or f"training_history_{title.lower().replace(' ', '_')}")
            
            return save_path
            
        except Exception as e:
            logger.error(f"Failed to plot training history: {str(e)}")
            return None
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                             class_names: List[str], title: str = "Confusion Matrix",
                             save_name: Optional[str] = None) -> str:
        """
        Plot confusion matrix with detailed analysis.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: List of class names
            title: Plot title
            save_name: Optional filename
            
        Returns:
            Path to saved plot
        """
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # Raw confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=class_names, yticklabels=class_names, ax=ax1)
            ax1.set_title('Raw Confusion Matrix', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Predicted')
            ax1.set_ylabel('Actual')
            
            # Normalized confusion matrix
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                       xticklabels=class_names, yticklabels=class_names, ax=ax2)
            ax2.set_title('Normalized Confusion Matrix', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Predicted')
            ax2.set_ylabel('Actual')
            
            # Per-class accuracy
            per_class_acc = cm.diagonal() / cm.sum(axis=1)
            bars = ax3.bar(class_names, per_class_acc, color='skyblue', alpha=0.8)
            ax3.set_title('Per-Class Accuracy', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Class')
            ax3.set_ylabel('Accuracy')
            ax3.set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, acc in zip(bars, per_class_acc):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
            
            # Class distribution
            unique, counts = np.unique(y_true, return_counts=True)
            ax4.pie(counts, labels=[class_names[i] for i in unique], autopct='%1.1f%%',
                   startangle=90, colors=plt.cm.Set3(np.linspace(0, 1, len(unique))))
            ax4.set_title('True Class Distribution', fontsize=14, fontweight='bold')
            
            plt.suptitle(title, fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            save_path = self._save_plot(fig, save_name or f"confusion_matrix_{title.lower().replace(' ', '_')}")
            
            return save_path
            
        except Exception as e:
            logger.error(f"Failed to plot confusion matrix: {str(e)}")
            return None
    
    def compare_models(self, results_dict: Dict[str, Dict[str, Any]],
                      save_name: Optional[str] = None) -> str:
        """
        Compare multiple model results with comprehensive analysis.
        
        Args:
            results_dict: Dictionary of model results
            save_name: Optional filename
            
        Returns:
            Path to saved plot
        """
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            models = list(results_dict.keys())
            accuracies = [results_dict[model]['accuracy'] for model in models]
            
            # Accuracy comparison
            colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
            bars = ax1.bar(models, accuracies, color=colors, alpha=0.8)
            ax1.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Test Accuracy (%)')
            ax1.set_ylim(0, 100)
            
            # Add value labels
            for bar, acc in zip(bars, accuracies):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')
            
            # Training time comparison (if available)
            training_times = []
            for model in models:
                if 'training_time' in results_dict[model]:
                    training_times.append(results_dict[model]['training_time'])
                else:
                    training_times.append(0)  # Placeholder
            
            if any(training_times):
                ax2.bar(models, training_times, color=colors, alpha=0.8)
                ax2.set_title('Training Time Comparison', fontsize=14, fontweight='bold')
                ax2.set_ylabel('Training Time (seconds)')
            
            # Model size comparison (parameters)
            param_counts = []
            for model in models:
                if 'param_count' in results_dict[model]:
                    param_counts.append(results_dict[model]['param_count'] / 1e6)  # Convert to millions
                else:
                    param_counts.append(0)
            
            if any(param_counts):
                ax3.bar(models, param_counts, color=colors, alpha=0.8)
                ax3.set_title('Model Size Comparison', fontsize=14, fontweight='bold')
                ax3.set_ylabel('Parameters (Millions)')
            
            # Accuracy vs Size scatter
            if any(param_counts):
                scatter = ax4.scatter(param_counts, accuracies, c=colors[:len(models)], 
                                    s=100, alpha=0.8)
                
                # Add model labels
                for i, model in enumerate(models):
                    ax4.annotate(model, (param_counts[i], accuracies[i]),
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=10, fontweight='bold')
                
                ax4.set_xlabel('Parameters (Millions)')
                ax4.set_ylabel('Accuracy (%)')
                ax4.set_title('Accuracy vs Model Size', fontsize=14, fontweight='bold')
                ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            save_path = self._save_plot(fig, save_name or "model_comparison")
            
            return save_path
            
        except Exception as e:
            logger.error(f"Failed to plot model comparison: {str(e)}")
            return None
    
    def plot_feature_analysis(self, audio_data: np.ndarray, labels: np.ndarray,
                             feature_extractor, title: str = "Feature Analysis",
                             save_name: Optional[str] = None) -> str:
        """
        Analyze and visualize extracted features.
        
        Args:
            audio_data: Raw audio data
            labels: Corresponding labels
            feature_extractor: Feature extraction object
            title: Plot title
            save_name: Optional filename
            
        Returns:
            Path to saved plot
        """
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # Sample audio waveforms
            unique_labels = np.unique(labels)
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
            
            for i, label in enumerate(unique_labels[:5]):  # Show first 5 classes
                indices = np.where(labels == label)[0]
                if len(indices) > 0:
                    sample_audio = audio_data[indices[0]]
                    time_axis = np.linspace(0, len(sample_audio)/8000, len(sample_audio))
                    ax1.plot(time_axis, sample_audio, label=f'Digit {label}', 
                           color=colors[i], alpha=0.7, linewidth=1)
            
            ax1.set_title('Sample Audio Waveforms', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Time (seconds)')
            ax1.set_ylabel('Amplitude')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Feature extraction example
            if hasattr(feature_extractor, 'extract_features'):
                sample_features = feature_extractor.extract_features(audio_data[0])
                
                if len(sample_features.shape) == 2:  # 2D features (spectrograms)
                    im = ax2.imshow(sample_features, aspect='auto', origin='lower', 
                                   cmap='viridis', interpolation='nearest')
                    ax2.set_title('Sample Feature Representation', fontsize=14, fontweight='bold')
                    ax2.set_xlabel('Time Frames')
                    ax2.set_ylabel('Feature Bins')
                    plt.colorbar(im, ax=ax2)
                else:  # 1D features (MFCC statistical features)
                    ax2.bar(range(len(sample_features)), sample_features, alpha=0.8)
                    ax2.set_title('Sample Feature Vector', fontsize=14, fontweight='bold')
                    ax2.set_xlabel('Feature Index')
                    ax2.set_ylabel('Feature Value')
            
            # Audio length distribution
            lengths = [len(audio) for audio in audio_data[:100]]  # Sample for speed
            ax3.hist(lengths, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax3.set_title('Audio Length Distribution', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Length (samples)')
            ax3.set_ylabel('Frequency')
            ax3.grid(True, alpha=0.3)
            
            # Label distribution
            unique, counts = np.unique(labels, return_counts=True)
            ax4.bar(unique, counts, alpha=0.8, color='lightgreen', edgecolor='black')
            ax4.set_title('Label Distribution', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Digit Class')
            ax4.set_ylabel('Count')
            ax4.set_xticks(unique)
            ax4.grid(True, alpha=0.3)
            
            plt.suptitle(title, fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            save_path = self._save_plot(fig, save_name or f"feature_analysis_{title.lower().replace(' ', '_')}")
            
            return save_path
            
        except Exception as e:
            logger.error(f"Failed to plot feature analysis: {str(e)}")
            return None
    
    def generate_training_report(self, results_dict: Dict[str, Dict[str, Any]],
                               dataset_info: Dict[str, Any], 
                               save_name: Optional[str] = None) -> str:
        """
        Generate comprehensive HTML training report.
        
        Args:
            results_dict: Dictionary of model results
            dataset_info: Dataset information
            save_name: Optional filename
            
        Returns:
            Path to saved HTML report
        """
        try:
            # Create HTML report
            html_content = self._generate_html_report(results_dict, dataset_info)
            
            # Save report
            report_path = self.output_dir / (save_name or "training_report.html")
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"Training report saved to: {report_path}")
            return str(report_path)
            
        except Exception as e:
            logger.error(f"Failed to generate training report: {str(e)}")
            return None
    
    def _generate_html_report(self, results_dict: Dict[str, Dict[str, Any]],
                             dataset_info: Dict[str, Any]) -> str:
        """Generate HTML content for training report."""
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>ML Training Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #2c3e50; }}
                h2 {{ color: #34495e; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .metric {{ background-color: #e8f5e8; }}
                .best {{ background-color: #d4edda; font-weight: bold; }}
            </style>
        </head>
        <body>
            <h1>ML Training Report</h1>
            <p><strong>Generated:</strong> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Dataset Information</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Total Samples</td><td>{dataset_info.get('total_samples', 'N/A')}</td></tr>
                <tr><td>Train Samples</td><td>{dataset_info.get('train_samples', 'N/A')}</td></tr>
                <tr><td>Validation Samples</td><td>{dataset_info.get('val_samples', 'N/A')}</td></tr>
                <tr><td>Test Samples</td><td>{dataset_info.get('test_samples', 'N/A')}</td></tr>
                <tr><td>Number of Classes</td><td>{dataset_info.get('num_classes', 'N/A')}</td></tr>
                <tr><td>Sample Rate</td><td>{dataset_info.get('sample_rate', 'N/A')} Hz</td></tr>
                <tr><td>Audio Length</td><td>{dataset_info.get('max_length', 'N/A')} samples</td></tr>
            </table>
            
            <h2>Model Comparison</h2>
            <table>
                <tr><th>Model</th><th>Test Accuracy (%)</th><th>Best Val Accuracy (%)</th></tr>
        """
        
        # Find best model
        best_accuracy = 0
        best_model = ""
        for model_name, results in results_dict.items():
            if results['accuracy'] > best_accuracy:
                best_accuracy = results['accuracy']
                best_model = model_name
        
        # Add model rows
        for model_name, results in results_dict.items():
            css_class = "best" if model_name == best_model else ""
            html += f"""
                <tr class="{css_class}">
                    <td>{model_name}</td>
                    <td>{results['accuracy']:.2f}</td>
                    <td>{max(results.get('history', {}).get('val_acc', [0])):.2f}</td>
                </tr>
            """
        
        html += """
            </table>
            
            <h2>Detailed Results</h2>
        """
        
        # Add detailed results for each model
        for model_name, results in results_dict.items():
            html += f"""
            <h3>{model_name}</h3>
            <h4>Classification Report:</h4>
            <pre>{results.get('report', 'No report available')}</pre>
            """
        
        html += """
        </body>
        </html>
        """
        
        return html
    
    def _save_plot(self, fig, filename: str) -> str:
        """Save matplotlib figure with proper naming and format."""
        save_path = self.output_dir / f"{filename}.png"
        fig.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close(fig)
        
        logger.info(f"Plot saved to: {save_path}")
        return str(save_path)

def create_visualizer(output_dir: str = "train_logs/plots") -> TrainingVisualizer:
    """
    Convenience function to create visualizer.
    
    Args:
        output_dir: Output directory for plots
        
    Returns:
        TrainingVisualizer instance
    """
    return TrainingVisualizer(output_dir=output_dir)

if __name__ == "__main__":
    # Test visualizer
    import logging
    logging.basicConfig(level=logging.INFO)
    
    viz = create_visualizer()
    
    # Test training history plot
    dummy_history = {
        'train_loss': [1.0, 0.8, 0.6, 0.4, 0.3, 0.2],
        'val_loss': [1.1, 0.9, 0.7, 0.5, 0.4, 0.3],
        'train_acc': [30, 45, 60, 75, 85, 90],
        'val_acc': [28, 42, 58, 72, 82, 88]
    }
    
    viz.plot_training_history(dummy_history, "Test Training History")
    logger.info("Visualization test completed")