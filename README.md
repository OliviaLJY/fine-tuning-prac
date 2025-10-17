# Autonomous Driving Fine-Tuning Project

A comprehensive deep learning project for fine-tuning pre-trained models for autonomous vehicle steering angle prediction. This project uses PyTorch and transfer learning with ResNet architectures to predict steering angles from camera images.

## 🚗 Project Overview

This project implements an end-to-end pipeline for training a deep learning model to predict steering angles for autonomous driving. The model uses a fine-tuned ResNet backbone (pre-trained on ImageNet) with a custom head for regression tasks.

### Key Features

- **Transfer Learning**: Fine-tune pre-trained ResNet models (ResNet18, ResNet34, ResNet50, ResNet101)
- **Flexible Training**: Configurable hyperparameters, data augmentation, and training strategies
- **Comprehensive Evaluation**: Detailed metrics and visualization tools
- **Real-time Inference**: Support for single images and video processing
- **TensorBoard Integration**: Real-time training monitoring
- **Early Stopping**: Automatic training termination to prevent overfitting
- **Data Augmentation**: Built-in image augmentation for improved generalization

## 📁 Project Structure

```
fine tuning/
├── config.yaml                 # Configuration file
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── .gitignore                  # Git ignore rules
│
├── train.py                    # Main training script
├── evaluate.py                 # Model evaluation script
├── inference.py                # Inference on images/videos
├── prepare_data.py             # Data preparation utilities
│
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset.py          # Dataset and data loader classes
│   ├── models/
│   │   ├── __init__.py
│   │   └── driving_model.py    # Model architecture
│   └── utils/
│       ├── __init__.py
│       └── preprocessing.py    # Data preprocessing utilities
│
├── data/
│   ├── raw/                    # Raw images directory
│   └── processed/              # Processed CSV splits
│
├── checkpoints/                # Model checkpoints
├── logs/                       # Training logs and TensorBoard
└── models/                     # Final trained models
```

## 🔧 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for training)
- pip or conda package manager

### Setup

1. **Clone or navigate to the project directory:**
```bash
cd "/Users/lijiayu/Desktop/fine tuning"
```

2. **Create a virtual environment (recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## 📊 Dataset Preparation

### Dataset Format

Your dataset should consist of:
1. **Images**: Driving camera images (JPG, PNG)
2. **CSV File**: Annotations with image paths and steering angles

Expected CSV format:
```csv
image_path,steering_angle
images/img_0001.jpg,0.15
images/img_0002.jpg,-0.23
images/img_0003.jpg,0.0
```

**Important Notes:**
- `image_path`: Relative path from the root data directory
- `steering_angle`: Normalized to range [-1, 1] where:
  - `-1.0` = maximum left turn
  - `0.0` = straight
  - `1.0` = maximum right turn

### Preparing Your Dataset

#### Option 1: Use Your Own Dataset

1. Organize your images in the `data/raw/` directory
2. Create a CSV file with image paths and steering angles
3. Verify and split your dataset:

```bash
python prepare_data.py \
    --csv data/raw/driving_data.csv \
    --root_dir data/raw \
    --output data/processed \
    --verify
```

#### Option 2: Create Sample Dataset Structure

To understand the expected format:

```bash
python prepare_data.py \
    --create_sample \
    --root_dir data/raw
```

### Popular Datasets for Autonomous Driving

- **Udacity Self-Driving Car Dataset**: Simulator-based driving data
- **Comma.ai Dataset**: Real-world highway driving
- **CARLA Simulator**: Synthetic driving data with full control
- **BDD100K**: Large-scale diverse driving video dataset

## 🚀 Training

### Quick Start

1. **Configure training parameters** (edit `config.yaml`)
2. **Prepare your dataset** (see Dataset Preparation)
3. **Start training:**

```bash
python train.py --config config.yaml
```

### Training Options

```bash
# Train with default configuration
python train.py

# Train with custom config
python train.py --config my_config.yaml

# Resume training from checkpoint
python train.py --resume checkpoints/checkpoint_epoch_10.pth

# Train on CPU (not recommended)
python train.py --device cpu
```

### Monitoring Training

#### TensorBoard

Launch TensorBoard to monitor training in real-time:

```bash
tensorboard --logdir logs
```

Then open your browser to `http://localhost:6006`

Metrics tracked:
- Training loss per batch and epoch
- Validation loss
- Learning rate schedule
- Model gradients (optional)

## ⚙️ Configuration

Edit `config.yaml` to customize training:

### Key Configuration Options

```yaml
# Model Selection
model:
  name: "resnet34"          # resnet18, resnet34, resnet50, resnet101
  pretrained: true          # Use ImageNet pre-trained weights
  freeze_backbone: false    # Freeze backbone layers
  freeze_layers: 0          # Number of layers to freeze (0 = none)

# Training Parameters
training:
  num_epochs: 50
  learning_rate: 0.0001
  optimizer: "adam"         # adam, sgd, adamw
  scheduler: "reduce_on_plateau"
  early_stopping_patience: 10

# Data Augmentation
augmentation:
  horizontal_flip: true
  brightness: 0.2
  rotation: 5
```

### Fine-Tuning Strategies

1. **Full Fine-Tuning** (Default):
   - All layers trainable
   - Best for sufficient data
   ```yaml
   freeze_backbone: false
   freeze_layers: 0
   ```

2. **Feature Extraction**:
   - Freeze backbone, train only head
   - Good for limited data
   ```yaml
   freeze_backbone: true
   ```

3. **Gradual Unfreezing**:
   - Freeze initial layers only
   - Balance between 1 and 2
   ```yaml
   freeze_backbone: false
   freeze_layers: 5
   ```

## 📈 Evaluation

Evaluate your trained model on the test set:

```bash
python evaluate.py \
    --config config.yaml \
    --checkpoint checkpoints/best_model.pth \
    --output evaluation_results
```

### Metrics Computed

- **MSE**: Mean Squared Error
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **R² Score**: Coefficient of determination
- **MAE (degrees)**: Error in actual steering degrees

### Visualization Outputs

The evaluation script generates:
1. **Scatter plot**: Predictions vs. true values
2. **Error distribution**: Histogram of prediction errors
3. **Time series**: Predictions over sample sequence
4. **Absolute errors**: Error magnitude visualization

## 🎯 Inference

### Single Image Inference

```bash
python inference.py \
    --config config.yaml \
    --checkpoint checkpoints/best_model.pth \
    --image path/to/image.jpg \
    --output predictions/image_prediction.png
```

### Video Processing

```bash
python inference.py \
    --config config.yaml \
    --checkpoint checkpoints/best_model.pth \
    --video path/to/video.mp4 \
    --output predictions/video_output.mp4
```

The output video will display:
- Original frames with steering angle overlay
- Visual arrow indicating steering direction
- Numerical steering angle value

## 🎓 Training Tips

### For Best Results

1. **Data Quality**:
   - Use high-quality, diverse driving images
   - Balance dataset (equal left, right, straight samples)
   - Minimum 10,000 images recommended

2. **Preprocessing**:
   - Crop unnecessary regions (sky, car hood)
   - Normalize steering angles properly
   - Consider temporal sequences for better predictions

3. **Augmentation**:
   - Use horizontal flipping (remember to flip angle too!)
   - Brightness/contrast variations for different lighting
   - Avoid excessive rotation or cropping

4. **Model Selection**:
   - ResNet34: Good balance of speed and accuracy
   - ResNet50: Better for complex scenarios
   - ResNet18: Faster training, less accuracy

5. **Hyperparameters**:
   - Start with learning rate 1e-4
   - Use learning rate scheduling
   - Enable early stopping to prevent overfitting

### Common Issues

**Overfitting:**
- Increase dropout rates
- Add more data augmentation
- Reduce model complexity
- Enable early stopping

**Underfitting:**
- Increase model capacity (try ResNet50/101)
- Train for more epochs
- Reduce regularization
- Lower dropout rates

**Poor Generalization:**
- Collect more diverse data
- Increase augmentation
- Check for data leakage
- Ensure proper train/val/test split

## 📚 Model Architecture

### DrivingModel

The main model consists of:

1. **Backbone**: Pre-trained ResNet (conv layers)
   - Extracts visual features from images
   - Optionally frozen for transfer learning

2. **Custom Head**: Fully connected layers
   - FC: 512/2048 → 256 (ReLU, Dropout 0.5)
   - FC: 256 → 64 (ReLU, Dropout 0.3)
   - FC: 64 → 1 (Tanh activation)

3. **Output**: Single value in range [-1, 1]
   - Represents normalized steering angle
   - Converted to degrees for interpretation

### Parameter Counts

| Model | Total Parameters | Trainable (full) |
|-------|-----------------|------------------|
| ResNet18 | ~11M | ~11M |
| ResNet34 | ~21M | ~21M |
| ResNet50 | ~23M | ~23M |
| ResNet101 | ~42M | ~42M |

## 🔬 Advanced Usage

### Custom Model Architecture

Modify `src/models/driving_model.py` to create custom architectures:

```python
from src.models.driving_model import DrivingModel

# Create custom model
model = DrivingModel(
    model_name='resnet50',
    pretrained=True,
    freeze_layers=7  # Freeze first 7 layers
)

# Unfreeze layers later for progressive training
model.unfreeze_layers(num_layers=3)  # Unfreeze last 3 layers
```

### Custom Data Augmentation

Edit `src/utils/preprocessing.py` to add custom augmentations:

```python
from torchvision import transforms

custom_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])
```

## 🤝 Contributing

To extend this project:

1. Add new model architectures in `src/models/`
2. Implement custom loss functions in `train.py`
3. Add new augmentation techniques in `src/utils/preprocessing.py`
4. Create visualization tools in `evaluate.py`

## 📝 Citation

If you use this project in your research, please cite:

```bibtex
@misc{autonomous_driving_finetuning,
  title={Autonomous Driving Fine-Tuning Project},
  author={Your Name},
  year={2024},
  publisher={GitHub},
  howpublished={\url{https://github.com/yourusername/autonomous-driving-finetuning}}
}
```

## 📄 License

This project is provided as-is for educational and research purposes.

## 🙏 Acknowledgments

- PyTorch team for the deep learning framework
- torchvision for pre-trained models
- Udacity, Comma.ai, and CARLA for autonomous driving datasets
- Open source community for various tools and libraries

## 📧 Support

For questions or issues:
1. Check the documentation above
2. Review closed issues on GitHub
3. Open a new issue with detailed information

## 🚦 Next Steps

After setting up the project:

1. ✅ Install dependencies
2. ✅ Prepare your dataset
3. ✅ Configure training parameters
4. ✅ Start training
5. ✅ Monitor with TensorBoard
6. ✅ Evaluate on test set
7. ✅ Run inference on new images/videos

Happy training! 🚗💨

