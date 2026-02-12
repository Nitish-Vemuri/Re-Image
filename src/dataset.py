"""
Dataset loading, preprocessing, and augmentation for CIFAKE dataset.

CIFAKE structure after download:
    data/
    ├── train/
    │   ├── REAL/    (50,000 images)
    │   └── FAKE/    (50,000 images)
    └── test/
        ├── REAL/    (10,000 images)
        └── FAKE/    (10,000 images)
"""

import os
from dotenv import load_dotenv

# Load .env file (contains KAGGLE_API_TOKEN)
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from config import (
    BATCH_SIZE,
    COLOR_JITTER,
    DATA_DIR,
    IMAGE_SIZE,
    KAGGLE_DATASET,
    NUM_WORKERS,
    PIN_MEMORY,
    RANDOM_HORIZONTAL_FLIP,
    RANDOM_ROTATION_DEGREES,
    SEED,
    TEST_DIR,
    TRAIN_DIR,
    VAL_SPLIT,
)

import torch


# =============================================================================
# DOWNLOAD DATASET FROM KAGGLE
# =============================================================================
def download_dataset():
    """
    Download CIFAKE dataset from Kaggle.

    Uses KAGGLE_API_TOKEN from .env file to authenticate via HTTP API.
    Falls back to kaggle.json if available.
    """
    import zipfile
    import requests

    # Check if dataset already exists
    if os.path.exists(TRAIN_DIR) and os.path.exists(TEST_DIR):
        train_count = sum(len(files) for _, _, files in os.walk(TRAIN_DIR))
        test_count = sum(len(files) for _, _, files in os.walk(TEST_DIR))
        print(f"Dataset already exists: {train_count} train, {test_count} test images.")
        return

    print(f"Downloading dataset: {KAGGLE_DATASET}")
    print(f"Destination: {DATA_DIR}")

    os.makedirs(DATA_DIR, exist_ok=True)

    token = os.environ.get("KAGGLE_API_TOKEN")
    if not token:
        raise ValueError(
            "KAGGLE_API_TOKEN not found. "
            "Set it in the .env file at the project root."
        )

    # Download zip via Kaggle REST API
    url = f"https://www.kaggle.com/api/v1/datasets/download/{KAGGLE_DATASET}"
    headers = {"Authorization": f"Bearer {token}"}
    zip_path = os.path.join(DATA_DIR, "cifake.zip")

    print("Downloading (this may take a few minutes)...")
    response = requests.get(url, headers=headers, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    downloaded = 0

    with open(zip_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total_size:
                pct = downloaded * 100 // total_size
                print(f"\r  Progress: {pct}% ({downloaded // (1024*1024)}MB)", end="")
    print()

    # Extract
    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(DATA_DIR)
    os.remove(zip_path)
    print("Dataset downloaded and extracted successfully.")

    # Verify download
    if os.path.exists(TRAIN_DIR) and os.path.exists(TEST_DIR):
        train_count = sum(len(files) for _, _, files in os.walk(TRAIN_DIR))
        test_count = sum(len(files) for _, _, files in os.walk(TEST_DIR))
        print(f"Verified: {train_count} train images, {test_count} test images.")
    else:
        print(f"WARNING: Expected directories not found at {TRAIN_DIR} and {TEST_DIR}")
        print(f"Contents of {DATA_DIR}:")
        for item in os.listdir(DATA_DIR):
            print(f"  {item}")
        print(f"Contents of {DATA_DIR}:")
        for item in os.listdir(DATA_DIR):
            print(f"  {item}")


# =============================================================================
# TRANSFORMS
# =============================================================================
def get_train_transforms():
    """
    Training transforms with data augmentation.
    Augmentation helps the model generalize and reduces overfitting.
    """
    transform_list = [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    ]

    # Data augmentation (only for training)
    if RANDOM_HORIZONTAL_FLIP:
        transform_list.append(transforms.RandomHorizontalFlip(p=0.5))

    if RANDOM_ROTATION_DEGREES > 0:
        transform_list.append(transforms.RandomRotation(RANDOM_ROTATION_DEGREES))

    if COLOR_JITTER:
        transform_list.append(
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
            )
        )

    # Convert to tensor and normalize (ImageNet stats, standard for ViT)
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return transforms.Compose(transform_list)


def get_eval_transforms():
    """
    Evaluation/test transforms — no augmentation, just resize + normalize.
    Used for validation, testing, and inference.
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


# =============================================================================
# DATALOADERS
# =============================================================================
def get_dataloaders():
    """
    Create train, validation, and test DataLoaders.

    - Training set: 90% of train/ (with augmentation)
    - Validation set: 10% of train/ (no augmentation)
    - Test set: test/ (no augmentation)

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Load full training dataset (with augmentation transforms)
    full_train_dataset = datasets.ImageFolder(
        root=TRAIN_DIR,
        transform=get_train_transforms(),
    )

    # Split into train and validation
    total_size = len(full_train_dataset)
    val_size = int(total_size * VAL_SPLIT)
    train_size = total_size - val_size

    generator = torch.Generator().manual_seed(SEED)
    train_dataset, val_dataset = random_split(
        full_train_dataset, [train_size, val_size], generator=generator
    )

    # Override transforms for validation subset (no augmentation)
    # We create a separate dataset for validation with eval transforms
    val_dataset_eval = datasets.ImageFolder(
        root=TRAIN_DIR,
        transform=get_eval_transforms(),
    )
    # Use the same indices as the validation split
    val_dataset_eval = torch.utils.data.Subset(val_dataset_eval, val_dataset.indices)

    # Test dataset (no augmentation)
    test_dataset = datasets.ImageFolder(
        root=TEST_DIR,
        transform=get_eval_transforms(),
    )

    # Print dataset info
    print(f"Training samples:   {train_size}")
    print(f"Validation samples: {val_size}")
    print(f"Test samples:       {len(test_dataset)}")
    print(f"Classes:            {full_train_dataset.classes}")
    print(f"Class-to-idx:       {full_train_dataset.class_to_idx}")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=True,  # Drop incomplete last batch for stable batch norm
    )

    val_loader = DataLoader(
        val_dataset_eval,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    return train_loader, val_loader, test_loader


# =============================================================================
# MAIN (for standalone testing)
# =============================================================================
if __name__ == "__main__":
    # Step 1: Download dataset
    download_dataset()

    # Step 2: Create data loaders and verify
    print("\nCreating DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders()

    # Step 3: Check a sample batch
    images, labels = next(iter(train_loader))
    print(f"\nSample batch:")
    print(f"  Images shape: {images.shape}")   # Expected: [32, 3, 224, 224]
    print(f"  Labels shape: {labels.shape}")   # Expected: [32]
    print(f"  Labels:       {labels[:10]}")     # 0=REAL, 1=FAKE
