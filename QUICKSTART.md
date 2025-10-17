# Quick Start Guide

Get started with the autonomous driving fine-tuning project in 5 minutes!

## 🚀 Fast Setup

### 1. Install Dependencies (2 minutes)

```bash
# Navigate to project directory
cd "/Users/lijiayu/Desktop/fine tuning"

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On macOS/Linux

# Install requirements
pip install -r requirements.txt
```

### 2. Run Examples (1 minute)

See the project in action without any data:

```bash
python example_usage.py
```

This will demonstrate:
- Model creation and architecture
- Image preprocessing pipeline
- Different fine-tuning strategies
- Inference workflow

## 📊 Working with Your Data

### Option A: Use Sample Dataset Structure

Create a sample structure to understand the format:

```bash
python prepare_data.py --create_sample --root_dir data/raw
```

This creates:
- `data/raw/driving_data.csv` - Sample CSV template
- `data/raw/images/` - Directory for images

### Option B: Use Real Dataset

1. **Organize your data:**
   ```
   data/raw/
   ├── images/
   │   ├── img_0001.jpg
   │   ├── img_0002.jpg
   │   └── ...
   └── driving_data.csv
   ```

2. **CSV format:**
   ```csv
   image_path,steering_angle
   images/img_0001.jpg,0.15
   images/img_0002.jpg,-0.23
   ```

3. **Prepare and split dataset:**
   ```bash
   python prepare_data.py \
       --csv data/raw/driving_data.csv \
       --root_dir data/raw \
       --output data/processed \
       --verify
   ```

## 🎓 Training in 3 Steps

### Step 1: Configure (Optional)

Edit `config.yaml` to customize:

```yaml
model:
  name: "resnet34"  # Choose: resnet18, resnet34, resnet50, resnet101

training:
  num_epochs: 50
  learning_rate: 0.0001
```

### Step 2: Train

```bash
python train.py --config config.yaml
```

Training will:
- ✅ Load and preprocess data
- ✅ Initialize model with pre-trained weights
- ✅ Train with validation
- ✅ Save checkpoints automatically
- ✅ Apply early stopping

### Step 3: Monitor

In another terminal:

```bash
tensorboard --logdir logs
```

Open: http://localhost:6006

## 📈 Evaluation

Evaluate your trained model:

```bash
python evaluate.py \
    --checkpoint checkpoints/best_model.pth \
    --output evaluation_results
```

Results include:
- Performance metrics (MSE, MAE, R²)
- Visualization plots
- Error analysis

## 🎯 Inference

### Single Image

```bash
python inference.py \
    --checkpoint checkpoints/best_model.pth \
    --image path/to/image.jpg \
    --output prediction.png
```

### Video

```bash
python inference.py \
    --checkpoint checkpoints/best_model.pth \
    --video path/to/video.mp4 \
    --output output_video.mp4
```

## 📚 Recommended Datasets

### For Beginners

**Udacity Self-Driving Car Dataset**
- Simulator-based, easy to use
- Download: https://github.com/udacity/self-driving-car
- Size: ~4GB
- Images: ~8,000

### For Advanced Users

**Comma.ai Dataset**
- Real-world highway driving
- Download: https://github.com/commaai/research
- Size: ~80GB
- Images: ~100,000+

## 🔧 Common Commands Cheat Sheet

```bash
# Setup
pip install -r requirements.txt

# Prepare data
python prepare_data.py --csv DATA.csv --root_dir data/raw --verify

# Train
python train.py

# Resume training
python train.py --resume checkpoints/checkpoint_epoch_10.pth

# Evaluate
python evaluate.py --checkpoint checkpoints/best_model.pth

# Inference (image)
python inference.py --checkpoint MODEL.pth --image IMAGE.jpg

# Inference (video)
python inference.py --checkpoint MODEL.pth --video VIDEO.mp4

# Monitor training
tensorboard --logdir logs
```

## ⚡ Quick Tips

1. **Start Small**: Use ResNet34 with default settings
2. **Check Data**: Always use `--verify` flag when preparing data
3. **Monitor**: Keep TensorBoard running during training
4. **Early Stop**: Enable it to save time and prevent overfitting
5. **GPU**: Training on GPU is 10-50x faster than CPU

## 🆘 Troubleshooting

### "CUDA out of memory"
- Reduce batch_size in `config.yaml` (try 16 or 8)
- Use smaller model (resnet18 instead of resnet50)

### "No such file or directory"
- Check image paths in CSV are relative to root_dir
- Verify with: `python prepare_data.py --verify`

### Training loss not decreasing
- Check learning rate (try 1e-4 or 1e-5)
- Verify data normalization
- Ensure steering angles are in [-1, 1]

### Poor validation performance
- Add more data augmentation
- Increase dataset size
- Enable dropout
- Try different model architecture

## 📖 Next Steps

1. ✅ Complete this quick start
2. 📚 Read full [README.md](README.md) for details
3. 🔬 Experiment with different configurations
4. 🎓 Try advanced fine-tuning strategies
5. 🚗 Deploy your model!

## 💡 Learning Resources

- PyTorch tutorials: https://pytorch.org/tutorials/
- Transfer learning guide: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
- Autonomous driving datasets: https://github.com/autonomousvision/awesome-autonomous-driving

---

Happy training! If you encounter any issues, refer to the full README.md or check the example_usage.py for code examples.

