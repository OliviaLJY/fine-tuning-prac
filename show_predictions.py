"""
Show multiple predictions with visualizations
"""

import os
import yaml
import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

from src.models.driving_model import create_model
from src.utils.preprocessing import DrivingImagePreprocessor, denormalize_steering_angle


def show_multiple_predictions(checkpoint_path, image_dir, num_samples=6):
    """Show predictions for multiple images"""
    
    # Load config
    with open('config_demo.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load model
    device = 'cpu'
    model = create_model(config['model']).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create preprocessor
    preprocessor = DrivingImagePreprocessor(
        image_width=config['data']['image_width'],
        image_height=config['data']['image_height'],
        augment=False
    )
    
    # Get sample images
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])[:num_samples]
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    print(f"\nPredicting on {len(image_files)} sample images...\n")
    
    for idx, img_file in enumerate(image_files):
        img_path = os.path.join(image_dir, img_file)
        
        # Load and predict
        image = Image.open(img_path).convert('RGB')
        input_tensor = preprocessor(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(input_tensor)
            normalized_angle = output.cpu().numpy()[0, 0]
            angle_degrees = denormalize_steering_angle(normalized_angle, max_angle=25.0)
        
        # Display
        axes[idx].imshow(image)
        axes[idx].axis('off')
        
        # Determine direction
        if angle_degrees < -2:
            direction = "LEFT"
            color = 'red'
        elif angle_degrees > 2:
            direction = "RIGHT"
            color = 'blue'
        else:
            direction = "STRAIGHT"
            color = 'green'
        
        title = f"{img_file}\nSteering: {angle_degrees:.2f}° ({direction})"
        axes[idx].set_title(title, fontsize=10, color=color, weight='bold')
        
        print(f"{img_file}:")
        print(f"  Steering: {angle_degrees:.2f}° ({direction})")
        print(f"  Normalized: {normalized_angle:.4f}\n")
    
    plt.tight_layout()
    output_path = 'evaluation_results/multiple_predictions.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Visualization saved to: {output_path}")
    plt.close()


if __name__ == '__main__':
    checkpoint = 'checkpoints/best_model.pth'
    image_dir = 'data/raw/images'
    
    print("="*60)
    print("  AUTONOMOUS DRIVING - MULTIPLE PREDICTIONS")
    print("="*60)
    
    show_multiple_predictions(checkpoint, image_dir, num_samples=6)
    
    print("\n" + "="*60)
    print("  Prediction complete!")
    print("="*60)
    print("\nCheck evaluation_results/ folder for visualizations:")
    print("  - test_predictions.png (scatter plots and error analysis)")
    print("  - val_predictions.png (validation results)")
    print("  - sample_prediction.png (single image with steering arrow)")
    print("  - multiple_predictions.png (6 sample predictions)")


