# Re-Image: AI-Generated Image Detection

Detect whether an image is **Real** or **AI-Generated** using a fine-tuned Vision Transformer (ViT).

## Overview

- **Model**: [ViT-B/16](https://huggingface.co/google/vit-base-patch16-224-in21k) (pre-trained on ImageNet-21K)
- **Dataset**: [CIFAKE](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images) (120K images — 60K real + 60K AI-generated)
- **Task**: Binary classification — Real vs AI-Generated
- **Training**: Fine-tuning on AWS EC2 (GPU)

## Project Structure

```
Re-Image/
├── src/
│   ├── config.py       # All hyperparameters, paths, settings
│   ├── dataset.py      # Download CIFAKE, transforms, DataLoaders
│   ├── model.py        # ViT model setup (create & load)
│   ├── train.py        # Training loop with early stopping
│   ├── evaluate.py     # Test metrics, confusion matrix, plots
│   └── inference.py    # Predict on single image or folder
├── data/               # Dataset (auto-downloaded, not in git)
├── outputs/
│   ├── best_model.pth  # Saved model weights
│   ├── logs/           # Training history, metrics
│   └── plots/          # Confusion matrix, accuracy curves
├── .env                # Kaggle API token (not in git)
├── .gitignore
├── requirements.txt
├── setup_ec2.sh        # One-command EC2 setup
└── README.md
```

---

## AWS EC2 Setup (Step-by-Step)

### Step 1: Create an AWS Account

1. Go to [aws.amazon.com](https://aws.amazon.com/) → **Create an AWS Account**
2. Add payment method (credit/debit card)
3. Select the **Free Tier** support plan

### Step 2: Request GPU Instance Quota (Important!)

New AWS accounts have **0 GPU quota** by default. You need to request it:

1. Go to [Service Quotas Console](https://console.aws.amazon.com/servicequotas/)
2. Search for **Amazon EC2**
3. Search for **"Running On-Demand G and VT instances"**
4. Click it → **Request quota increase**
5. Set **New quota value** to `4` (for 1x `g4dn.xlarge` which uses 4 vCPUs)
6. Submit — approval usually takes **a few minutes to 24 hours**

> ⚠️ You cannot launch a GPU instance until this is approved.

### Step 3: Create a Key Pair (for SSH)

1. Go to [EC2 Console](https://console.aws.amazon.com/ec2/)
2. Left sidebar → **Network & Security** → **Key Pairs**
3. Click **Create key pair**
   - Name: `re-image-key`
   - Key pair type: **RSA**
   - File format: **.pem** (for Mac/Linux) or **.ppk** (for PuTTY on Windows)
4. Download the file — keep it safe, you can't download it again
5. Move it somewhere safe:
   ```
   # Windows
   move re-image-key.pem C:\Users\<you>\.ssh\

   # Mac/Linux
   mv re-image-key.pem ~/.ssh/
   chmod 400 ~/.ssh/re-image-key.pem
   ```

### Step 4: Launch EC2 Instance

1. Go to [EC2 Console](https://console.aws.amazon.com/ec2/) → **Launch Instance**

2. **Name**: `re-image-training`

3. **AMI (OS Image)**:
   - Click **Browse more AMIs**
   - Search for: `Deep Learning OSS Nvidia Driver AMI GPU PyTorch`
   - Select the **Ubuntu** version
   - This AMI comes with NVIDIA drivers, CUDA, and cuDNN pre-installed

4. **Instance Type**: `g4dn.xlarge`
   | Spec | Value |
   |------|-------|
   | GPU | 1x NVIDIA T4 (16GB VRAM) |
   | vCPUs | 4 |
   | RAM | 16 GB |
   | Cost | ~$0.53/hr on-demand |

5. **Key Pair**: Select `re-image-key` (created in Step 3)

6. **Network Settings**:
   - Allow **SSH traffic** from **My IP** (or Anywhere if your IP changes)
   
7. **Storage**: Change root volume to **50 GB** (default 8GB is too small for dataset + model)

8. Click **Launch Instance**

### Step 5: Connect to EC2

Wait 1-2 minutes for the instance to start, then:

#### Option A: SSH from Terminal
```bash
# Find your instance's Public IP in the EC2 console
ssh -i ~/.ssh/re-image-key.pem ubuntu@<your-ec2-public-ip>
```

#### Option B: VS Code Remote SSH (Recommended — keeps Copilot!)

1. Install the **Remote - SSH** extension in VS Code
2. Press `Ctrl+Shift+P` → **Remote-SSH: Connect to Host**
3. Enter: `ssh -i C:\Users\<you>\.ssh\re-image-key.pem ubuntu@<ec2-public-ip>`
4. Or add to `~/.ssh/config`:
   ```
   Host re-image
       HostName <your-ec2-public-ip>
       User ubuntu
       IdentityFile C:\Users\<you>\.ssh\re-image-key.pem
   ```
   Then just connect to `re-image`

5. VS Code opens on the EC2 instance — full Copilot support!

### Step 6: Setup the Project on EC2

```bash
# Download and run setup script
git clone https://github.com/Nitish-Vemuri/Re-Image.git
cd Re-Image
chmod +x setup_ec2.sh
./setup_ec2.sh
```

Then edit `.env` with your Kaggle token:
```bash
nano .env
# Change: KAGGLE_API_TOKEN=your_actual_token_here
# Save: Ctrl+O, Enter, Ctrl+X
```

### Step 7: Train the Model

```bash
cd src
source ../.venv/bin/activate

# Download dataset
python dataset.py

# Train (full fine-tuning, ~30-60 min on T4)
python train.py

# Or frozen backbone (faster, ~10-15 min)
python train.py --freeze
```

### Step 8: Evaluate & Test

```bash
# Evaluate on test set (20K images)
python evaluate.py

# Predict on a single image
python inference.py --image /path/to/image.jpg

# Predict on a folder
python inference.py --dir /path/to/images/
```

### Step 9: STOP the Instance When Done!

> ⚠️ **CRITICAL**: GPU instances cost ~$0.53/hr. Always stop when not using.

**From EC2 Console:**
1. Go to EC2 → Instances
2. Select your instance
3. **Instance State** → **Stop Instance**

**From terminal:**
```bash
sudo shutdown -h now
```

**Stop vs Terminate:**
| Action | Your files | Billing | Can restart? |
|--------|-----------|---------|-------------|
| **Stop** | Kept ✓ | Storage only (~$0.08/GB/mo) | Yes |
| **Terminate** | Deleted ✗ | Nothing | No |

### Step 10: Restart Later

1. EC2 Console → Select instance → **Start Instance**
2. **Note**: The public IP changes! Check the new IP in the console.
3. Reconnect via SSH / VS Code with the new IP

---

## Local Setup (CPU — for code development only)

```bash
git clone https://github.com/Nitish-Vemuri/Re-Image.git
cd Re-Image

# Create venv (use python.org Python, not MinGW)
py -3.9 -m venv .venv
.venv\Scripts\Activate.ps1   # Windows
# source .venv/bin/activate   # Mac/Linux

# Install (CPU version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# Create .env
echo "KAGGLE_API_TOKEN=your_token" > .env
```

---

## Model Details

| Setting | Value |
|---------|-------|
| Architecture | ViT-B/16 (Vision Transformer) |
| Pre-training | ImageNet-21K (14M images) |
| Fine-tuning data | CIFAKE (100K train, 20K test) |
| Image size | 224 × 224 |
| Batch size | 32 |
| Learning rate | 2e-5 (AdamW) |
| Scheduler | OneCycleLR (cosine, 10% warmup) |
| Early stopping | Patience = 3 epochs |
| Parameters | 85.8M total |

---

## Usage Summary

```bash
# Train
python src/train.py
python src/train.py --freeze          # Faster (frozen backbone)
python src/train.py --epochs 5        # Custom epochs
python src/train.py --resume model.pth  # Resume training

# Evaluate
python src/evaluate.py
python src/evaluate.py --model path/to/model.pth

# Inference
python src/inference.py --image photo.jpg
python src/inference.py --dir folder/
```

---

## Cost Estimate

| What | Cost |
|------|------|
| EC2 g4dn.xlarge (training ~1hr) | ~$0.53 |
| EC2 spot instance (same, 70% off) | ~$0.16 |
| EBS storage 50GB (when stopped) | ~$4/month |
| Total for one training run | **< $1** |
