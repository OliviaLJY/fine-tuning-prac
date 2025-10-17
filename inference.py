"""
Inference script for autonomous driving model
"""

import os
import yaml
import argparse
import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from src.models.driving_model import create_model
from src.utils.preprocessing import DrivingImagePreprocessor, denormalize_steering_angle


class DrivingInference:
    """Inference engine for driving model"""
    
    def __init__(self, config, checkpoint_path, device='cuda'):
        """
        Initialize inference engine
        
        Args:
            config: Configuration dictionary
            checkpoint_path: Path to model checkpoint
            device: Device to run inference on
        """
        self.config = config
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        # Load model
        self.model = create_model(config['model']).to(self.device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Create preprocessor
        self.preprocessor = DrivingImagePreprocessor(
            image_width=config['data']['image_width'],
            image_height=config['data']['image_height'],
            augment=False
        )
        
        print(f"Inference engine initialized on {self.device}")
        print(f"Model loaded from: {checkpoint_path}")
    
    def predict_steering(self, image):
        """
        Predict steering angle for a single image
        
        Args:
            image: PIL Image, numpy array, or path to image file
            
        Returns:
            Predicted steering angle (normalized and in degrees)
        """
        # Load image if path is provided
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Preprocess
        input_tensor = self.preprocessor(image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            output = self.model(input_tensor)
            normalized_angle = output.cpu().numpy()[0, 0]
        
        # Convert to degrees
        angle_degrees = denormalize_steering_angle(normalized_angle, max_angle=25.0)
        
        return normalized_angle, angle_degrees
    
    def predict_batch(self, image_paths):
        """
        Predict steering angles for multiple images
        
        Args:
            image_paths: List of image paths
            
        Returns:
            List of (normalized_angle, angle_degrees) tuples
        """
        results = []
        for path in image_paths:
            normalized, degrees = self.predict_steering(path)
            results.append((normalized, degrees))
        return results
    
    def visualize_prediction(self, image_path, output_path=None):
        """
        Visualize prediction with steering angle overlay
        
        Args:
            image_path: Path to input image
            output_path: Path to save visualization (if None, display only)
        """
        # Load original image
        original_image = cv2.imread(image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        
        # Get prediction
        normalized_angle, angle_degrees = self.predict_steering(image_path)
        
        # Create visualization
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(original_image)
        ax.axis('off')
        
        # Draw steering indicator
        height, width = original_image.shape[:2]
        center_x, center_y = width // 2, int(height * 0.8)
        
        # Calculate arrow endpoint based on steering angle
        arrow_length = 100
        # Negative angle = turn left, positive = turn right
        end_x = center_x + int(arrow_length * np.sin(np.radians(angle_degrees)))
        end_y = center_y - int(arrow_length * np.cos(np.radians(angle_degrees)))
        
        # Draw arrow
        ax.arrow(center_x, center_y, end_x - center_x, end_y - center_y,
                head_width=30, head_length=30, fc='red', ec='red', linewidth=3)
        
        # Add text
        direction = "LEFT" if angle_degrees < 0 else "RIGHT" if angle_degrees > 0 else "STRAIGHT"
        text = f"Steering: {angle_degrees:.2f}° ({direction})"
        ax.text(width // 2, 50, text, fontsize=20, color='white',
               ha='center', va='top', 
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to: {output_path}")
            plt.close()
        else:
            plt.show()
    
    def process_video(self, video_path, output_path):
        """
        Process video and generate predictions for each frame
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video
        """
        # Open video
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"Processing video: {video_path}")
        print(f"Total frames: {total_frames}")
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Get prediction
            normalized_angle, angle_degrees = self.predict_steering(frame_rgb)
            
            # Draw steering indicator
            center_x, center_y = width // 2, int(height * 0.8)
            arrow_length = 100
            end_x = center_x + int(arrow_length * np.sin(np.radians(angle_degrees)))
            end_y = center_y - int(arrow_length * np.cos(np.radians(angle_degrees)))
            
            cv2.arrowedLine(frame, (center_x, center_y), (end_x, end_y),
                          (0, 0, 255), 5, tipLength=0.3)
            
            # Add text
            direction = "LEFT" if angle_degrees < 0 else "RIGHT" if angle_degrees > 0 else "STRAIGHT"
            text = f"Steering: {angle_degrees:.2f} ({direction})"
            cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                       1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Write frame
            out.write(frame)
            
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Processed {frame_count}/{total_frames} frames")
        
        cap.release()
        out.release()
        print(f"Video processing complete! Output saved to: {output_path}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Run inference with autonomous driving model')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--image', type=str, default=None,
                       help='Path to single image for inference')
    parser.add_argument('--video', type=str, default=None,
                       help='Path to video file for inference')
    parser.add_argument('--output', type=str, default='inference_output',
                       help='Output directory or file path')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'], help='Device to run inference on')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create inference engine
    inference = DrivingInference(config, args.checkpoint, device=args.device)
    
    # Run inference
    if args.image:
        # Single image inference
        output_path = args.output if args.output.endswith('.png') else \
                     os.path.join(args.output, 'prediction.png')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        normalized, degrees = inference.predict_steering(args.image)
        print(f"\nPrediction for {args.image}:")
        print(f"  Normalized angle: {normalized:.6f}")
        print(f"  Steering angle: {degrees:.4f}°")
        
        inference.visualize_prediction(args.image, output_path)
        
    elif args.video:
        # Video inference
        output_path = args.output if args.output.endswith('.mp4') else \
                     os.path.join(args.output, 'output.mp4')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        inference.process_video(args.video, output_path)
        
    else:
        print("Please provide either --image or --video for inference")


if __name__ == '__main__':
    main()

