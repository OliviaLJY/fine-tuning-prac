"""
Model architecture for autonomous driving with fine-tuning support
"""

import torch
import torch.nn as nn
import torchvision.models as models


class DrivingModel(nn.Module):
    """
    Fine-tuned ResNet model for steering angle prediction
    """
    
    def __init__(self, model_name='resnet34', pretrained=True, num_outputs=1, 
                 freeze_backbone=False, freeze_layers=0):
        """
        Initialize the model
        
        Args:
            model_name: Name of the base model (resnet18, resnet34, resnet50, resnet101)
            pretrained: Whether to use pretrained weights
            num_outputs: Number of output values (1 for steering angle)
            freeze_backbone: Whether to freeze all backbone layers
            freeze_layers: Number of initial layers to freeze (0 = none)
        """
        super(DrivingModel, self).__init__()
        
        self.model_name = model_name
        
        # Load pretrained model
        if model_name == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            feature_dim = 512
        elif model_name == 'resnet34':
            self.backbone = models.resnet34(pretrained=pretrained)
            feature_dim = 512
        elif model_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            feature_dim = 2048
        elif model_name == 'resnet101':
            self.backbone = models.resnet101(pretrained=pretrained)
            feature_dim = 2048
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Remove the final fully connected layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Freeze layers if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        elif freeze_layers > 0:
            # Freeze specified number of initial layers
            layers = list(self.backbone.children())
            for i in range(min(freeze_layers, len(layers))):
                for param in layers[i].parameters():
                    param.requires_grad = False
        
        # Add custom head for steering angle prediction
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_outputs),
            nn.Tanh()  # Output in range [-1, 1]
        )
        
    def forward(self, x):
        """Forward pass"""
        features = self.backbone(x)
        output = self.head(features)
        return output
    
    def unfreeze_layers(self, num_layers=None):
        """
        Unfreeze layers for fine-tuning
        
        Args:
            num_layers: Number of layers to unfreeze from the end (None = all)
        """
        layers = list(self.backbone.children())
        if num_layers is None:
            # Unfreeze all layers
            for param in self.backbone.parameters():
                param.requires_grad = True
        else:
            # Unfreeze last num_layers
            for i in range(max(0, len(layers) - num_layers), len(layers)):
                for param in layers[i].parameters():
                    param.requires_grad = True
    
    def get_trainable_parameters(self):
        """Get list of trainable parameters"""
        return [p for p in self.parameters() if p.requires_grad]
    
    def count_parameters(self):
        """Count total and trainable parameters"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


class LightweightDrivingModel(nn.Module):
    """
    Lightweight CNN model for autonomous driving (train from scratch)
    Useful for resource-constrained environments or custom architectures
    """
    
    def __init__(self, num_outputs=1):
        """
        Initialize lightweight model
        
        Args:
            num_outputs: Number of output values
        """
        super(LightweightDrivingModel, self).__init__()
        
        self.conv_layers = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Conv Block 2
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Conv Block 3
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Conv Block 4
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256 * 3 * 3, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_outputs),
            nn.Tanh()
        )
    
    def forward(self, x):
        """Forward pass"""
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


def create_model(config):
    """
    Factory function to create model based on configuration
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        Model instance
    """
    model_name = config.get('name', 'resnet34')
    pretrained = config.get('pretrained', True)
    num_outputs = config.get('num_outputs', 1)
    freeze_backbone = config.get('freeze_backbone', False)
    freeze_layers = config.get('freeze_layers', 0)
    
    if model_name == 'lightweight':
        model = LightweightDrivingModel(num_outputs=num_outputs)
    else:
        model = DrivingModel(
            model_name=model_name,
            pretrained=pretrained,
            num_outputs=num_outputs,
            freeze_backbone=freeze_backbone,
            freeze_layers=freeze_layers
        )
    
    total_params, trainable_params = model.count_parameters() if hasattr(model, 'count_parameters') else (0, 0)
    print(f"Model: {model_name}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return model

