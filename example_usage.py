"""
Example usage script demonstrating the fine-tuning pipeline
This is a simple example showing how to use the main components
"""

import torch
import yaml
from src.models.driving_model import create_model, DrivingModel
from src.utils.preprocessing import DrivingImagePreprocessor
from PIL import Image
import numpy as np


def example_model_creation():
    """Example: Creating and inspecting a model"""
    print("=" * 50)
    print("Example 1: Creating a Model")
    print("=" * 50)
    
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create model
    model = create_model(config['model'])
    
    print(f"\nModel architecture:")
    print(model)
    
    # Check trainable parameters
    total, trainable = model.count_parameters()
    print(f"\nTotal parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Percentage trainable: {100 * trainable / total:.2f}%")


def example_preprocessing():
    """Example: Image preprocessing"""
    print("\n" + "=" * 50)
    print("Example 2: Image Preprocessing")
    print("=" * 50)
    
    # Create preprocessor with augmentation
    train_preprocessor = DrivingImagePreprocessor(
        image_width=224,
        image_height=224,
        augment=True,
        config={
            'horizontal_flip': True,
            'brightness': 0.2,
            'rotation': 5
        }
    )
    
    # Create a dummy image
    dummy_image = Image.new('RGB', (640, 480), color='blue')
    
    # Preprocess
    preprocessed = train_preprocessor(dummy_image)
    
    print(f"\nOriginal image size: {dummy_image.size}")
    print(f"Preprocessed tensor shape: {preprocessed.shape}")
    print(f"Tensor range: [{preprocessed.min():.3f}, {preprocessed.max():.3f}]")


def example_model_inference():
    """Example: Single image inference (without loading checkpoint)"""
    print("\n" + "=" * 50)
    print("Example 3: Model Inference")
    print("=" * 50)
    
    # Create model
    model = DrivingModel(
        model_name='resnet34',
        pretrained=True,
        num_outputs=1
    )
    model.eval()
    
    # Create dummy input
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 224, 224)
    
    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"\nPredicted steering angles (normalized):")
    for i, angle in enumerate(output.squeeze()):
        angle_degrees = angle.item() * 25.0  # Denormalize (assuming max 25°)
        print(f"  Sample {i+1}: {angle.item():.4f} ({angle_degrees:.2f}°)")


def example_fine_tuning_strategies():
    """Example: Different fine-tuning strategies"""
    print("\n" + "=" * 50)
    print("Example 4: Fine-Tuning Strategies")
    print("=" * 50)
    
    # Strategy 1: Full fine-tuning
    print("\nStrategy 1: Full Fine-Tuning")
    model1 = DrivingModel('resnet34', pretrained=True, freeze_backbone=False)
    total1, trainable1 = model1.count_parameters()
    print(f"  Trainable: {trainable1:,} / {total1:,} ({100*trainable1/total1:.1f}%)")
    
    # Strategy 2: Freeze backbone
    print("\nStrategy 2: Feature Extraction (Frozen Backbone)")
    model2 = DrivingModel('resnet34', pretrained=True, freeze_backbone=True)
    total2, trainable2 = model2.count_parameters()
    print(f"  Trainable: {trainable2:,} / {total2:,} ({100*trainable2/total2:.1f}%)")
    
    # Strategy 3: Freeze some layers
    print("\nStrategy 3: Partial Freezing (First 5 Layers)")
    model3 = DrivingModel('resnet34', pretrained=True, freeze_layers=5)
    total3, trainable3 = model3.count_parameters()
    print(f"  Trainable: {trainable3:,} / {total3:,} ({100*trainable3/total3:.1f}%)")
    
    # Strategy 4: Progressive unfreezing
    print("\nStrategy 4: Progressive Unfreezing")
    model4 = DrivingModel('resnet34', pretrained=True, freeze_backbone=True)
    print("  Initial - frozen backbone")
    _, trainable = model4.count_parameters()
    print(f"    Trainable: {trainable:,}")
    
    model4.unfreeze_layers(3)
    print("  After unfreezing last 3 layers:")
    _, trainable = model4.count_parameters()
    print(f"    Trainable: {trainable:,}")


def example_training_workflow():
    """Example: Typical training workflow"""
    print("\n" + "=" * 50)
    print("Example 5: Training Workflow")
    print("=" * 50)
    
    workflow = """
    Step-by-step training workflow:
    
    1. Prepare Your Data:
       python prepare_data.py \\
           --csv data/raw/driving_data.csv \\
           --root_dir data/raw \\
           --output data/processed \\
           --verify
    
    2. Configure Training:
       Edit config.yaml to set:
       - Model architecture (resnet34 recommended to start)
       - Training parameters (learning rate, epochs, etc.)
       - Data augmentation settings
    
    3. Start Training:
       python train.py --config config.yaml
    
    4. Monitor Progress:
       tensorboard --logdir logs
       Open: http://localhost:6006
    
    5. Evaluate Model:
       python evaluate.py \\
           --config config.yaml \\
           --checkpoint checkpoints/best_model.pth \\
           --output evaluation_results
    
    6. Run Inference:
       # Single image
       python inference.py \\
           --checkpoint checkpoints/best_model.pth \\
           --image test_image.jpg \\
           --output prediction.png
       
       # Video
       python inference.py \\
           --checkpoint checkpoints/best_model.pth \\
           --video test_video.mp4 \\
           --output output_video.mp4
    """
    
    print(workflow)


def example_configuration():
    """Example: Understanding configuration"""
    print("\n" + "=" * 50)
    print("Example 6: Configuration Tips")
    print("=" * 50)
    
    tips = """
    Key configuration parameters:
    
    1. Model Selection:
       - resnet18: Fast, 11M params, good for quick experiments
       - resnet34: Balanced, 21M params, recommended default
       - resnet50: Powerful, 23M params, best accuracy
       - resnet101: Heavy, 42M params, for complex scenarios
    
    2. Learning Rate:
       - Start with 1e-4 for fine-tuning pre-trained models
       - Use 1e-3 for training from scratch
       - Enable scheduler for automatic adjustment
    
    3. Data Augmentation:
       - Essential for small datasets (<10k images)
       - Horizontal flip: YES (but flip steering angle too!)
       - Brightness/contrast: 0.2 is a good start
       - Rotation: Keep small (5-10 degrees max)
       - Avoid: Vertical flip (unrealistic for driving)
    
    4. Early Stopping:
       - patience=10: Stop if no improvement for 10 epochs
       - Prevents overfitting
       - Saves training time
    
    5. Batch Size:
       - GPU memory limited: Use 16-32
       - High memory GPU: Can use 64-128
       - Affects training speed and stability
    """
    
    print(tips)


def main():
    """Run all examples"""
    print("\n")
    print("🚗 " * 20)
    print(" " * 15 + "AUTONOMOUS DRIVING FINE-TUNING")
    print(" " * 20 + "EXAMPLE USAGE")
    print("🚗 " * 20)
    
    try:
        example_model_creation()
        example_preprocessing()
        example_model_inference()
        example_fine_tuning_strategies()
        example_training_workflow()
        example_configuration()
        
        print("\n" + "=" * 50)
        print("All examples completed successfully! ✅")
        print("=" * 50)
        print("\nNext steps:")
        print("1. Prepare your dataset")
        print("2. Review config.yaml")
        print("3. Run: python train.py")
        print("\nFor more information, see README.md")
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        print("Make sure you have installed all requirements:")
        print("  pip install -r requirements.txt")


if __name__ == '__main__':
    main()

