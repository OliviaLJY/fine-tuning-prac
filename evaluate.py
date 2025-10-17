"""
Evaluation script for autonomous driving model
"""

import os
import yaml
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.models.driving_model import create_model
from src.data.dataset import create_data_loaders
from src.utils.preprocessing import DrivingImagePreprocessor, denormalize_steering_angle


class Evaluator:
    """Evaluation manager for driving model"""
    
    def __init__(self, config, checkpoint_path, device='cuda'):
        """
        Initialize evaluator
        
        Args:
            config: Configuration dictionary
            checkpoint_path: Path to model checkpoint
            device: Device to evaluate on
        """
        self.config = config
        self.device = device if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Load model
        self.model = create_model(config['model']).to(self.device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print(f"Model loaded from: {checkpoint_path}")
        print(f"Checkpoint epoch: {checkpoint['epoch']}")
        print(f"Best validation loss: {checkpoint['best_val_loss']:.6f}")
    
    def evaluate(self, data_loader, dataset_name='Test'):
        """
        Evaluate model on dataset
        
        Args:
            data_loader: DataLoader for evaluation
            dataset_name: Name of dataset for logging
            
        Returns:
            Dictionary of metrics
        """
        print(f"\nEvaluating on {dataset_name} set...")
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for images, targets in tqdm(data_loader, desc=f"Evaluating {dataset_name}"):
                images = images.to(self.device)
                targets = targets.cpu().numpy()
                
                # Get predictions
                outputs = self.model(images)
                predictions = outputs.cpu().numpy()
                
                all_predictions.extend(predictions.flatten())
                all_targets.extend(targets.flatten())
        
        # Convert to numpy arrays
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)
        
        # Calculate metrics
        mse = mean_squared_error(targets, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(targets, predictions)
        r2 = r2_score(targets, predictions)
        
        # Calculate denormalized errors (in degrees)
        max_angle = 25.0  # Assuming steering angles are normalized by this value
        denorm_predictions = denormalize_steering_angle(predictions, max_angle)
        denorm_targets = denormalize_steering_angle(targets, max_angle)
        mae_degrees = mean_absolute_error(denorm_targets, denorm_predictions)
        rmse_degrees = np.sqrt(mean_squared_error(denorm_targets, denorm_predictions))
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mae_degrees': mae_degrees,
            'rmse_degrees': rmse_degrees,
            'predictions': predictions,
            'targets': targets
        }
        
        # Print metrics
        print(f"\n{dataset_name} Set Metrics:")
        print(f"  MSE: {mse:.6f}")
        print(f"  RMSE: {rmse:.6f}")
        print(f"  MAE: {mae:.6f}")
        print(f"  R² Score: {r2:.6f}")
        print(f"  MAE (degrees): {mae_degrees:.4f}°")
        print(f"  RMSE (degrees): {rmse_degrees:.4f}°")
        
        return metrics
    
    def plot_predictions(self, metrics, output_path='evaluation_plots.png'):
        """
        Plot prediction results
        
        Args:
            metrics: Dictionary of metrics from evaluate()
            output_path: Path to save plot
        """
        predictions = metrics['predictions']
        targets = metrics['targets']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Scatter plot
        axes[0, 0].scatter(targets, predictions, alpha=0.5, s=10)
        axes[0, 0].plot([targets.min(), targets.max()], 
                       [targets.min(), targets.max()], 
                       'r--', lw=2, label='Perfect prediction')
        axes[0, 0].set_xlabel('True Steering Angle')
        axes[0, 0].set_ylabel('Predicted Steering Angle')
        axes[0, 0].set_title(f'Predictions vs True Values (R² = {metrics["r2"]:.4f})')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Error distribution
        errors = predictions - targets
        axes[0, 1].hist(errors, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 1].axvline(x=0, color='r', linestyle='--', lw=2)
        axes[0, 1].set_xlabel('Prediction Error')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title(f'Error Distribution (MAE = {metrics["mae"]:.4f})')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Predictions over samples
        sample_indices = np.arange(min(500, len(targets)))
        axes[1, 0].plot(sample_indices, targets[sample_indices], 
                       label='True', alpha=0.7, linewidth=1.5)
        axes[1, 0].plot(sample_indices, predictions[sample_indices], 
                       label='Predicted', alpha=0.7, linewidth=1.5)
        axes[1, 0].set_xlabel('Sample Index')
        axes[1, 0].set_ylabel('Steering Angle')
        axes[1, 0].set_title('Predictions Over Samples (first 500)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Absolute errors
        abs_errors = np.abs(errors)
        axes[1, 1].plot(sample_indices, abs_errors[sample_indices], 
                       alpha=0.7, linewidth=1)
        axes[1, 1].axhline(y=metrics['mae'], color='r', linestyle='--', 
                          lw=2, label=f'MAE = {metrics["mae"]:.4f}')
        axes[1, 1].set_xlabel('Sample Index')
        axes[1, 1].set_ylabel('Absolute Error')
        axes[1, 1].set_title('Absolute Errors (first 500)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nPlots saved to: {output_path}")
        plt.close()


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Evaluate autonomous driving model')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'], help='Device to evaluate on')
    parser.add_argument('--output', type=str, default='evaluation_results',
                       help='Output directory for results')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Create transforms (no augmentation for evaluation)
    preprocessor = DrivingImagePreprocessor(
        image_width=config['data']['image_width'],
        image_height=config['data']['image_height'],
        augment=False
    )
    
    # Create data loaders
    train_csv = os.path.join(config['data']['processed_data_path'], 'train.csv')
    val_csv = os.path.join(config['data']['processed_data_path'], 'val.csv')
    test_csv = os.path.join(config['data']['processed_data_path'], 'test.csv')
    
    print("Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        train_csv=train_csv,
        val_csv=val_csv,
        test_csv=test_csv,
        root_dir=config['data']['raw_data_path'],
        train_transform=preprocessor,
        val_transform=preprocessor,
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers']
    )
    
    # Create evaluator
    evaluator = Evaluator(config, args.checkpoint, device=args.device)
    
    # Evaluate on all splits
    test_metrics = evaluator.evaluate(test_loader, 'Test')
    val_metrics = evaluator.evaluate(val_loader, 'Validation')
    
    # Generate plots
    evaluator.plot_predictions(
        test_metrics, 
        os.path.join(args.output, 'test_predictions.png')
    )
    evaluator.plot_predictions(
        val_metrics, 
        os.path.join(args.output, 'val_predictions.png')
    )
    
    print(f"\nEvaluation complete! Results saved to: {args.output}")


if __name__ == '__main__':
    main()

