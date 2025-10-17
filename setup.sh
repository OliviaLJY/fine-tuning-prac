#!/bin/bash

# Setup script for Autonomous Driving Fine-Tuning Project

echo "=================================="
echo "  Autonomous Driving Fine-Tuning"
echo "  Setup Script"
echo "=================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo ""
echo "Installing requirements..."
pip install -r requirements.txt

# Check installation
echo ""
echo "Verifying installation..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torchvision; print(f'torchvision: {torchvision.__version__}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"

echo ""
echo "=================================="
echo "  Setup Complete! ✅"
echo "=================================="
echo ""
echo "Next steps:"
echo "1. Activate virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Run example:"
echo "   python example_usage.py"
echo ""
echo "3. Prepare your data:"
echo "   python prepare_data.py --create_sample --root_dir data/raw"
echo ""
echo "4. Start training:"
echo "   python train.py --config config.yaml"
echo ""
echo "For more information, see README.md or QUICKSTART.md"
echo ""

