"""
Training script for autonomous driving model
"""

import os
import yaml
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

from src.models.driving_model import create_model
from src.data.dataset import create_data_loaders
from src.utils.preprocessing import DrivingImagePreprocessor


class Trainer:
    """Training manager for driving model"""
    
    def __init__(self, config, device='cuda'):
        """
        Initialize trainer
        
        Args:
            config: Configuration dictionary
            device: Device to train on ('cuda' or 'cpu')
        """
        self.config = config
        self.device = device if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Create model
        self.model = create_model(config['model']).to(self.device)
        
        # Setup loss function
        loss_type = config['loss']['type']
        if loss_type == 'mse':
            self.criterion = nn.MSELoss()
        elif loss_type == 'mae':
            self.criterion = nn.L1Loss()
        elif loss_type == 'huber':
            self.criterion = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        # Setup optimizer
        optimizer_type = config['training']['optimizer']
        lr = config['training']['learning_rate']
        weight_decay = config['training']['weight_decay']
        
        if optimizer_type == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(), 
                lr=lr, 
                weight_decay=weight_decay
            )
        elif optimizer_type == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(), 
                lr=lr, 
                weight_decay=weight_decay
            )
        elif optimizer_type == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(), 
                lr=lr, 
                momentum=0.9, 
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
        
        # Setup learning rate scheduler
        scheduler_type = config['training']['scheduler']
        if scheduler_type == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, 
                step_size=10, 
                gamma=0.1
            )
        elif scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=config['training']['num_epochs']
            )
        elif scheduler_type == 'reduce_on_plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, 
                mode='min', 
                patience=5, 
                factor=0.5
            )
        else:
            self.scheduler = None
        
        # Setup directories
        self.checkpoint_dir = config['checkpoint']['save_dir']
        self.log_dir = config['logging']['log_dir']
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Setup tensorboard
        if config['logging']['tensorboard']:
            self.writer = SummaryWriter(self.log_dir)
        else:
            self.writer = None
        
        # Training state
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.current_epoch = 0
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = len(train_loader)
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, (images, targets) in enumerate(progress_bar):
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config['training'].get('gradient_clip'):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['training']['gradient_clip']
                )
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f'{avg_loss:.6f}'})
            
            # Log to tensorboard
            if self.writer and batch_idx % self.config['logging']['print_frequency'] == 0:
                global_step = self.current_epoch * num_batches + batch_idx
                self.writer.add_scalar('Train/Batch_Loss', loss.item(), global_step)
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        num_batches = len(val_loader)
        
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc="Validating"):
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def save_checkpoint(self, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
        }
        
        # Save regular checkpoint
        if self.current_epoch % self.config['checkpoint']['save_frequency'] == 0:
            checkpoint_path = os.path.join(
                self.checkpoint_dir, 
                f'checkpoint_epoch_{self.current_epoch}.pth'
            )
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"Best model saved: {best_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        print(f"Checkpoint loaded from epoch {self.current_epoch}")
    
    def train(self, train_loader, val_loader, num_epochs):
        """Main training loop"""
        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}\n")
        
        for epoch in range(1, num_epochs + 1):
            self.current_epoch = epoch
            
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss = self.validate(val_loader)
            
            # Log metrics
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  Val Loss: {val_loss:.6f}")
            
            if self.writer:
                self.writer.add_scalar('Train/Epoch_Loss', train_loss, epoch)
                self.writer.add_scalar('Val/Loss', val_loss, epoch)
                self.writer.add_scalar('Learning_Rate', 
                                      self.optimizer.param_groups[0]['lr'], epoch)
            
            # Update learning rate
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Check for improvement
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                print(f"  New best validation loss: {val_loss:.6f}")
            else:
                self.epochs_without_improvement += 1
            
            # Save checkpoint
            self.save_checkpoint(is_best)
            
            # Early stopping
            early_stopping_patience = self.config['training']['early_stopping_patience']
            if self.epochs_without_improvement >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                break
        
        print(f"\nTraining complete! Best validation loss: {self.best_val_loss:.6f}")
        
        if self.writer:
            self.writer.close()


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Train autonomous driving model')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'], help='Device to train on')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create transforms
    train_preprocessor = DrivingImagePreprocessor(
        image_width=config['data']['image_width'],
        image_height=config['data']['image_height'],
        augment=True,
        config=config['augmentation']
    )
    
    val_preprocessor = DrivingImagePreprocessor(
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
        train_transform=train_preprocessor,
        val_transform=val_preprocessor,
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers']
    )
    
    # Create trainer
    trainer = Trainer(config, device=args.device)
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train model
    trainer.train(
        train_loader, 
        val_loader, 
        num_epochs=config['training']['num_epochs']
    )


if __name__ == '__main__':
    main()

