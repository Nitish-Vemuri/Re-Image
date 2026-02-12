"""
Configuration for AI-Generated Image Detection (ViT Fine-tuning)
All hyperparameters, paths, and settings in one place.
"""

import os
import torch

# =============================================================================
# PATHS
# =============================================================================
# Project root (one level up from src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dataset paths
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")

# Output paths
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "best_model.pth")
LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")

# =============================================================================
# DATASET
# =============================================================================
# Kaggle dataset identifier
KAGGLE_DATASET = "birdy654/cifake-real-and-ai-generated-synthetic-images"

# Class labels
CLASSES = ["REAL", "FAKE"]
NUM_CLASSES = len(CLASSES)

# Class-to-index mapping
CLASS_TO_IDX = {"REAL": 0, "FAKE": 1}
IDX_TO_CLASS = {0: "REAL", 1: "FAKE"}

# =============================================================================
# MODEL
# =============================================================================
# Pre-trained ViT model from Hugging Face
MODEL_NAME = "google/vit-base-patch16-224-in21k"

# Image size expected by ViT-B/16
IMAGE_SIZE = 224

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 2e-5           # Low LR for fine-tuning pre-trained model
WEIGHT_DECAY = 0.01            # AdamW regularization
WARMUP_STEPS = 500             # Learning rate warmup

# Validation split (from training set if needed)
VAL_SPLIT = 0.1

# Early stopping
EARLY_STOPPING_PATIENCE = 3   # Stop if val loss doesn't improve for N epochs

# =============================================================================
# DATA AUGMENTATION
# =============================================================================
# Training augmentations
RANDOM_HORIZONTAL_FLIP = True
RANDOM_ROTATION_DEGREES = 10
COLOR_JITTER = True

# =============================================================================
# DEVICE
# =============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# MISC
# =============================================================================
SEED = 42
NUM_WORKERS = 4                # DataLoader workers (adjust based on CPU cores)
PIN_MEMORY = True              # Speed up data transfer to GPU
LOG_INTERVAL = 50              # Print training loss every N batches


def create_dirs():
    """Create output directories if they don't exist."""
    for dir_path in [OUTPUT_DIR, LOGS_DIR, PLOTS_DIR, DATA_DIR]:
        os.makedirs(dir_path, exist_ok=True)


if __name__ == "__main__":
    # Print config for verification
    print(f"Project Root:  {PROJECT_ROOT}")
    print(f"Data Dir:      {DATA_DIR}")
    print(f"Output Dir:    {OUTPUT_DIR}")
    print(f"Device:        {DEVICE}")
    print(f"Model:         {MODEL_NAME}")
    print(f"Image Size:    {IMAGE_SIZE}")
    print(f"Batch Size:    {BATCH_SIZE}")
    print(f"Epochs:        {NUM_EPOCHS}")
    print(f"Learning Rate: {LEARNING_RATE}")
