"""
Vision Transformer (ViT) model for AI-Generated Image Detection.

Architecture:
    Pre-trained ViT-B/16 (google/vit-base-patch16-224-in21k)
        ↓
    Replace classification head (1000 → 2 classes)
        ↓
    Output: [FAKE, REAL] logits

The pre-trained ViT was trained on ImageNet-21K (14M images, 21K classes).
We fine-tune it for binary classification: Real vs AI-Generated.
"""

import torch
import torch.nn as nn
from transformers import ViTForImageClassification, ViTConfig

from config import DEVICE, MODEL_NAME, NUM_CLASSES


def create_model(freeze_backbone=False):
    """
    Load pre-trained ViT and replace the classification head.

    Args:
        freeze_backbone: If True, freeze all ViT layers except the
            classification head. Useful for:
            - Quick experiments (trains much faster)
            - Small datasets (prevents overfitting)
            Set to False for best accuracy (fine-tune everything).

    Returns:
        model: ViTForImageClassification with 2-class head, on DEVICE.
    """
    print(f"Loading pre-trained model: {MODEL_NAME}")

    # Load ViT with a new 2-class classification head
    model = ViTForImageClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True,  # Head size changes from 21K → 2
    )

    if freeze_backbone:
        # Freeze all parameters except the classification head
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False
        print("Backbone frozen — only training classification head.")

    model = model.to(DEVICE)

    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Device:               {DEVICE}")

    return model


def load_model(checkpoint_path):
    """
    Load a trained model from a checkpoint file.

    Args:
        checkpoint_path: Path to the saved .pth file.

    Returns:
        model: Loaded model in eval mode on DEVICE.
    """
    print(f"Loading model from: {checkpoint_path}")

    # Create model architecture
    model = ViTForImageClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True,
    )

    # Load trained weights
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

    # Handle both full checkpoint dict and raw state_dict
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded from epoch {checkpoint.get('epoch', '?')}, "
              f"val_acc: {checkpoint.get('val_acc', '?')}")
    else:
        model.load_state_dict(checkpoint)

    model = model.to(DEVICE)
    model.eval()
    print("Model loaded and set to eval mode.")

    return model


# =============================================================================
# MAIN (for standalone testing)
# =============================================================================
if __name__ == "__main__":
    # Test 1: Create model (full fine-tuning)
    print("=" * 50)
    print("Test 1: Full fine-tuning model")
    print("=" * 50)
    model = create_model(freeze_backbone=False)

    # Test 2: Create model (frozen backbone)
    print("\n" + "=" * 50)
    print("Test 2: Frozen backbone model")
    print("=" * 50)
    model_frozen = create_model(freeze_backbone=True)

    # Test 3: Forward pass with dummy input
    print("\n" + "=" * 50)
    print("Test 3: Forward pass")
    print("=" * 50)
    dummy_input = torch.randn(4, 3, 224, 224).to(DEVICE)  # Batch of 4 images
    with torch.no_grad():
        outputs = model(pixel_values=dummy_input)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=1)
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {logits.shape}")     # Expected: [4, 2]
    print(f"Probabilities:\n{probs}")
    print(f"Predictions:  {torch.argmax(probs, dim=1)}")  # 0=FAKE, 1=REAL
