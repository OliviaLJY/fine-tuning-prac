"""
Dataset class for autonomous driving data
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class DrivingDataset(Dataset):
    """
    PyTorch Dataset for autonomous driving data
    Expects CSV file with columns: image_path, steering_angle
    """
    
    def __init__(self, csv_file, root_dir, transform=None, target_transform=None):
        """
        Initialize dataset
        
        Args:
            csv_file: Path to CSV file with annotations
            root_dir: Root directory containing images
            transform: Optional transform to be applied on images
            target_transform: Optional transform to be applied on labels
        """
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Get image path and load image
        img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        
        # Get steering angle
        steering_angle = self.data_frame.iloc[idx, 1]
        steering_angle = np.array([steering_angle], dtype=np.float32)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            steering_angle = self.target_transform(steering_angle)
        
        return image, torch.tensor(steering_angle, dtype=torch.float32)


class DrivingDatasetMemory(Dataset):
    """
    In-memory version of driving dataset for faster training
    Loads all images into memory at initialization
    """
    
    def __init__(self, csv_file, root_dir, transform=None, target_transform=None):
        """
        Initialize dataset and load all images into memory
        
        Args:
            csv_file: Path to CSV file with annotations
            root_dir: Root directory containing images
            transform: Optional transform to be applied on images
            target_transform: Optional transform to be applied on labels
        """
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        
        # Load all images into memory
        print(f"Loading {len(self.data_frame)} images into memory...")
        self.images = []
        for idx in range(len(self.data_frame)):
            img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx, 0])
            image = Image.open(img_name).convert('RGB')
            self.images.append(image)
        print("All images loaded!")
        
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Get preloaded image
        image = self.images[idx]
        
        # Get steering angle
        steering_angle = self.data_frame.iloc[idx, 1]
        steering_angle = np.array([steering_angle], dtype=np.float32)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            steering_angle = self.target_transform(steering_angle)
        
        return image, torch.tensor(steering_angle, dtype=torch.float32)


def create_data_loaders(train_csv, val_csv, test_csv, root_dir, 
                       train_transform, val_transform, batch_size, 
                       num_workers=4, use_memory=False):
    """
    Create train, validation, and test data loaders
    
    Args:
        train_csv: Path to training CSV
        val_csv: Path to validation CSV
        test_csv: Path to test CSV
        root_dir: Root directory for images
        train_transform: Transform for training data
        val_transform: Transform for validation/test data
        batch_size: Batch size
        num_workers: Number of workers for data loading
        use_memory: Whether to use in-memory dataset
        
    Returns:
        train_loader, val_loader, test_loader
    """
    DatasetClass = DrivingDatasetMemory if use_memory else DrivingDataset
    
    train_dataset = DatasetClass(train_csv, root_dir, transform=train_transform)
    val_dataset = DatasetClass(val_csv, root_dir, transform=val_transform)
    test_dataset = DatasetClass(test_csv, root_dir, transform=val_transform)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def split_dataset(csv_file, output_dir, train_split=0.8, val_split=0.1, test_split=0.1, random_state=42):
    """
    Split dataset into train, validation, and test sets
    
    Args:
        csv_file: Path to CSV file with all data
        output_dir: Directory to save split CSV files
        train_split: Fraction for training set
        val_split: Fraction for validation set
        test_split: Fraction for test set
        random_state: Random seed for reproducibility
        
    Returns:
        Paths to train, val, and test CSV files
    """
    assert abs(train_split + val_split + test_split - 1.0) < 1e-6, \
        "Splits must sum to 1.0"
    
    # Read full dataset
    df = pd.read_csv(csv_file)
    
    # First split: separate train from (val + test)
    train_df, temp_df = train_test_split(
        df, 
        test_size=(1 - train_split), 
        random_state=random_state
    )
    
    # Second split: separate val from test
    val_size = val_split / (val_split + test_split)
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=(1 - val_size), 
        random_state=random_state
    )
    
    # Save splits
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, 'train.csv')
    val_path = os.path.join(output_dir, 'val.csv')
    test_path = os.path.join(output_dir, 'test.csv')
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Dataset split complete:")
    print(f"  Training: {len(train_df)} samples")
    print(f"  Validation: {len(val_df)} samples")
    print(f"  Test: {len(test_df)} samples")
    
    return train_path, val_path, test_path

