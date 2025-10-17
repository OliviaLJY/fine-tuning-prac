"""
Data preparation script for autonomous driving dataset
This script helps prepare and organize your dataset for training
"""

import os
import argparse
import pandas as pd
from src.data.dataset import split_dataset


def create_sample_dataset(output_dir, num_samples=100):
    """
    Create a sample dataset structure for demonstration
    This shows the expected format for your actual data
    
    Args:
        output_dir: Directory to create sample data
        num_samples: Number of sample entries to create
    """
    print("Creating sample dataset structure...")
    
    # Create directories
    images_dir = os.path.join(output_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)
    
    # Create sample CSV
    data = {
        'image_path': [f'images/sample_{i:04d}.jpg' for i in range(num_samples)],
        'steering_angle': [0.0] * num_samples  # Placeholder values
    }
    
    df = pd.DataFrame(data)
    csv_path = os.path.join(output_dir, 'driving_data.csv')
    df.to_csv(csv_path, index=False)
    
    print(f"Sample CSV created at: {csv_path}")
    print(f"\nExpected CSV format:")
    print("  - Column 1: 'image_path' - Relative path to image")
    print("  - Column 2: 'steering_angle' - Steering angle (normalized to [-1, 1])")
    print(f"\nYour dataset should have images in: {images_dir}")
    print(f"Example: {images_dir}/sample_0000.jpg")
    
    return csv_path


def verify_dataset(csv_file, root_dir):
    """
    Verify dataset integrity
    
    Args:
        csv_file: Path to CSV file
        root_dir: Root directory containing images
    """
    print("\nVerifying dataset...")
    
    df = pd.read_csv(csv_file)
    print(f"Total entries in CSV: {len(df)}")
    
    # Check required columns
    required_cols = ['image_path', 'steering_angle']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"ERROR: Missing required columns: {missing_cols}")
        return False
    
    # Check for missing images
    missing_images = []
    for idx, row in df.iterrows():
        img_path = os.path.join(root_dir, row['image_path'])
        if not os.path.exists(img_path):
            missing_images.append(img_path)
            if len(missing_images) <= 5:  # Only print first 5
                print(f"  Missing: {img_path}")
    
    if missing_images:
        print(f"\nWARNING: {len(missing_images)} images not found!")
        return False
    
    # Check steering angle range
    min_angle = df['steering_angle'].min()
    max_angle = df['steering_angle'].max()
    print(f"Steering angle range: [{min_angle:.4f}, {max_angle:.4f}]")
    
    if min_angle < -1.0 or max_angle > 1.0:
        print("WARNING: Steering angles should be normalized to [-1, 1]")
        print("You may need to normalize your angles using:")
        print("  normalized_angle = angle / max_angle")
    
    print("\nDataset verification complete!")
    return True


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Prepare autonomous driving dataset')
    parser.add_argument('--csv', type=str, required=True,
                       help='Path to CSV file with all data')
    parser.add_argument('--root_dir', type=str, required=True,
                       help='Root directory containing images')
    parser.add_argument('--output', type=str, default='data/processed',
                       help='Output directory for split datasets')
    parser.add_argument('--train_split', type=float, default=0.8,
                       help='Training set split ratio')
    parser.add_argument('--val_split', type=float, default=0.1,
                       help='Validation set split ratio')
    parser.add_argument('--test_split', type=float, default=0.1,
                       help='Test set split ratio')
    parser.add_argument('--verify', action='store_true',
                       help='Verify dataset before splitting')
    parser.add_argument('--create_sample', action='store_true',
                       help='Create sample dataset structure')
    args = parser.parse_args()
    
    if args.create_sample:
        # Create sample dataset
        csv_path = create_sample_dataset(args.root_dir)
        print("\nSample dataset created!")
        print("Replace this with your actual driving data.")
        return
    
    # Verify dataset if requested
    if args.verify:
        is_valid = verify_dataset(args.csv, args.root_dir)
        if not is_valid:
            print("\nDataset verification failed! Please fix the issues above.")
            return
    
    # Split dataset
    print("\nSplitting dataset...")
    train_csv, val_csv, test_csv = split_dataset(
        csv_file=args.csv,
        output_dir=args.output,
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=args.test_split
    )
    
    print("\nDataset preparation complete!")
    print(f"Training CSV: {train_csv}")
    print(f"Validation CSV: {val_csv}")
    print(f"Test CSV: {test_csv}")
    print("\nYou can now run training with:")
    print("  python train.py --config config.yaml")


if __name__ == '__main__':
    main()

