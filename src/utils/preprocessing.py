"""
Data preprocessing utilities for autonomous driving images
"""

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms


class DrivingImagePreprocessor:
    """Preprocessor for autonomous driving images"""
    
    def __init__(self, image_width=224, image_height=224, augment=False, config=None):
        """
        Initialize preprocessor
        
        Args:
            image_width: Target image width
            image_height: Target image height
            augment: Whether to apply data augmentation
            config: Configuration dictionary for augmentation parameters
        """
        self.image_width = image_width
        self.image_height = image_height
        self.augment = augment
        self.config = config or {}
        
        # Base transforms (always applied)
        self.base_transform = transforms.Compose([
            transforms.Resize((image_height, image_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Augmentation transforms (applied during training)
        if augment:
            aug_list = [transforms.Resize((image_height, image_width))]
            
            if config.get('random_crop', False):
                aug_list.append(transforms.RandomResizedCrop(
                    (image_height, image_width), 
                    scale=(0.8, 1.0)
                ))
            
            if config.get('horizontal_flip', False):
                aug_list.append(transforms.RandomHorizontalFlip(p=0.5))
            
            if config.get('brightness', 0) or config.get('contrast', 0) or \
               config.get('saturation', 0) or config.get('hue', 0):
                aug_list.append(transforms.ColorJitter(
                    brightness=config.get('brightness', 0),
                    contrast=config.get('contrast', 0),
                    saturation=config.get('saturation', 0),
                    hue=config.get('hue', 0)
                ))
            
            if config.get('rotation', 0):
                aug_list.append(transforms.RandomRotation(config.get('rotation', 0)))
            
            aug_list.extend([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            
            self.augment_transform = transforms.Compose(aug_list)
    
    def __call__(self, image):
        """
        Preprocess an image
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            Preprocessed image tensor
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        if self.augment:
            return self.augment_transform(image)
        else:
            return self.base_transform(image)


def crop_roi(image, top_crop=0.35):
    """
    Crop region of interest (removes sky/hood)
    
    Args:
        image: Input image (numpy array)
        top_crop: Fraction of image to crop from top
        
    Returns:
        Cropped image
    """
    height = image.shape[0]
    crop_height = int(height * top_crop)
    return image[crop_height:, :, :]


def adjust_steering_for_flip(steering_angle):
    """
    Adjust steering angle when image is horizontally flipped
    
    Args:
        steering_angle: Original steering angle
        
    Returns:
        Adjusted steering angle
    """
    return -steering_angle


def normalize_steering_angle(angle, max_angle=25.0):
    """
    Normalize steering angle to [-1, 1] range
    
    Args:
        angle: Steering angle in degrees
        max_angle: Maximum steering angle
        
    Returns:
        Normalized angle
    """
    return np.clip(angle / max_angle, -1.0, 1.0)


def denormalize_steering_angle(normalized_angle, max_angle=25.0):
    """
    Denormalize steering angle from [-1, 1] to degrees
    
    Args:
        normalized_angle: Normalized angle
        max_angle: Maximum steering angle
        
    Returns:
        Angle in degrees
    """
    return normalized_angle * max_angle

