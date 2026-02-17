#!/bin/bash
# =============================================================================
# EC2 Setup Script for AI-Generated Image Detection
#
# Run this on a fresh EC2 instance with Deep Learning AMI (Ubuntu)
# Instance type: g4dn.xlarge (T4 GPU, 16GB VRAM)
#
# Usage:
#   chmod +x setup_ec2.sh
#   ./setup_ec2.sh
# =============================================================================

set -e  # Exit on error

echo "============================================"
echo "  EC2 Setup — AI Image Detection"
echo "============================================"

# ------------------------------------------
# 1. System updates
# ------------------------------------------
echo "[1/6] Updating system..."
sudo apt-get update -y
sudo apt-get install -y git unzip htop

# ------------------------------------------
# 2. Clone the repo
# ------------------------------------------
echo "[2/6] Cloning repository..."
if [ ! -d "Re-Image" ]; then
    git clone https://github.com/Nitish-Vemuri/Re-Image.git
fi
cd Re-Image

# ------------------------------------------
# 3. Python environment
# ------------------------------------------
echo "[3/6] Setting up Python environment..."
python3 -m venv .venv
source .venv/bin/activate

# ------------------------------------------
# 4. Install dependencies (GPU version of PyTorch)
# ------------------------------------------
echo "[4/6] Installing dependencies..."
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# ------------------------------------------
# 5. Create .env file
# ------------------------------------------
echo "[5/6] Setting up environment..."
if [ ! -f ".env" ]; then
    echo "KAGGLE_API_TOKEN=your_token_here" > .env
    echo "⚠️  Edit .env and add your KAGGLE_API_TOKEN"
fi

# Create output dirs
python3 -c "from src.config import create_dirs; create_dirs()"

# ------------------------------------------
# 6. Verify GPU
# ------------------------------------------
echo "[6/6] Verifying setup..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available:  {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU:             {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory:      {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')
else:
    print('WARNING: No GPU detected!')
"

echo ""
echo "============================================"
echo "  Setup complete!"
echo "============================================"
echo ""
echo "  Next steps:"
echo "    1. Edit .env → add your KAGGLE_API_TOKEN"
echo "    2. cd src"
echo "    3. python dataset.py          # Download dataset"
echo "    4. python train.py            # Start training"
echo "    5. python evaluate.py         # Evaluate model"
echo "    6. python inference.py --image test.jpg  # Predict"
echo ""
echo "  To stop the instance when done:"
echo "    sudo shutdown -h now"
echo ""
