"""
Generate sample training data for demonstration purposes
Creates synthetic images and corresponding steering angles
"""

import os
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import random

def generate_road_image(width=640, height=480, steering_angle=0.0):
    """
    Generate a synthetic road image with a road path
    
    Args:
        width: Image width
        height: Image height
        steering_angle: Steering angle (-1 to 1)
        
    Returns:
        PIL Image
    """
    # Create base image (sky + road)
    img = Image.new('RGB', (width, height))
    draw = ImageDraw.Draw(img)
    
    # Sky (top half)
    sky_color = (135 + random.randint(-20, 20), 
                 206 + random.randint(-20, 20), 
                 235 + random.randint(-20, 20))
    draw.rectangle([0, 0, width, height//2], fill=sky_color)
    
    # Road (bottom half)
    road_color = (70 + random.randint(-10, 10), 
                  70 + random.randint(-10, 10), 
                  70 + random.randint(-10, 10))
    draw.rectangle([0, height//2, width, height], fill=road_color)
    
    # Draw road lanes based on steering angle
    # Steering angle affects the perspective of the road
    center_x = width // 2
    
    # Vanishing point shifts based on steering
    vanish_x = center_x + int(steering_angle * 200)
    vanish_y = height // 3
    
    # Left lane
    left_bottom = center_x - 200
    left_top = vanish_x - 30
    draw.line([(left_bottom, height), (left_top, vanish_y)], 
              fill=(255, 255, 0), width=8)
    
    # Right lane
    right_bottom = center_x + 200
    right_top = vanish_x + 30
    draw.line([(right_bottom, height), (right_top, vanish_y)], 
              fill=(255, 255, 0), width=8)
    
    # Center line (dashed)
    for i in range(10):
        y_start = height - (i * height // 10)
        y_end = y_start - height // 20
        x_start = center_x + int((y_start - vanish_y) * steering_angle * 0.3)
        x_end = center_x + int((y_end - vanish_y) * steering_angle * 0.3)
        if y_end > vanish_y:
            draw.line([(x_start, y_start), (x_end, y_end)], 
                     fill=(255, 255, 255), width=4)
    
    # Add some random noise for variety
    pixels = np.array(img)
    noise = np.random.randint(-15, 15, pixels.shape, dtype=np.int16)
    pixels = np.clip(pixels.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(pixels)
    
    return img


def generate_dataset(num_samples=1000, output_dir='data/raw'):
    """
    Generate a synthetic dataset of road images
    
    Args:
        num_samples: Number of images to generate
        output_dir: Output directory
    """
    print(f"Generating {num_samples} sample images...")
    
    # Create directories
    images_dir = os.path.join(output_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)
    
    # Generate data
    data = []
    
    # Different driving scenarios with corresponding angles
    scenarios = [
        ('straight', 0.0, 0.05),      # Mostly straight
        ('slight_left', -0.2, 0.1),   # Slight left
        ('slight_right', 0.2, 0.1),   # Slight right
        ('left_turn', -0.5, 0.15),    # Left turn
        ('right_turn', 0.5, 0.15),    # Right turn
        ('sharp_left', -0.8, 0.1),    # Sharp left
        ('sharp_right', 0.8, 0.1),    # Sharp right
    ]
    
    # Distribute samples across scenarios
    samples_per_scenario = num_samples // len(scenarios)
    
    sample_idx = 0
    for scenario_name, mean_angle, std_angle in scenarios:
        for i in range(samples_per_scenario):
            # Generate steering angle with some variation
            steering_angle = np.clip(
                np.random.normal(mean_angle, std_angle), 
                -1.0, 1.0
            )
            
            # Generate image
            img = generate_road_image(steering_angle=steering_angle)
            
            # Save image
            img_filename = f'img_{sample_idx:05d}.jpg'
            img_path = os.path.join(images_dir, img_filename)
            img.save(img_path, quality=85)
            
            # Add to data
            data.append({
                'image_path': f'images/{img_filename}',
                'steering_angle': float(steering_angle)
            })
            
            sample_idx += 1
            
            if (sample_idx + 1) % 100 == 0:
                print(f"  Generated {sample_idx + 1}/{num_samples} images...")
    
    # Create CSV
    df = pd.DataFrame(data)
    csv_path = os.path.join(output_dir, 'driving_data.csv')
    df.to_csv(csv_path, index=False)
    
    print(f"\n✅ Dataset generated successfully!")
    print(f"  Images: {len(data)}")
    print(f"  CSV file: {csv_path}")
    print(f"  Images directory: {images_dir}")
    print(f"\nSteering angle statistics:")
    print(f"  Mean: {df['steering_angle'].mean():.4f}")
    print(f"  Std: {df['steering_angle'].std():.4f}")
    print(f"  Min: {df['steering_angle'].min():.4f}")
    print(f"  Max: {df['steering_angle'].max():.4f}")
    
    return csv_path


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate sample training data')
    parser.add_argument('--num_samples', type=int, default=1000,
                       help='Number of samples to generate')
    parser.add_argument('--output_dir', type=str, default='data/raw',
                       help='Output directory')
    args = parser.parse_args()
    
    # Generate dataset
    csv_path = generate_dataset(args.num_samples, args.output_dir)
    
    print("\n" + "="*60)
    print("Next steps:")
    print("="*60)
    print(f"1. Verify and split the dataset:")
    print(f"   python prepare_data.py --csv {csv_path} --root_dir {args.output_dir} --verify")
    print(f"\n2. Train the model:")
    print(f"   python train.py --config config.yaml")

