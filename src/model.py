"""
CLIP-based model for AI-Generated Image Detection.

Architecture:
    Pre-trained CLIP Vision Encoder (openai/clip-vit-base-patch16)
        ↓
    Pooled [CLS] token features (768-dim)
        ↓
    Linear classification head (768 → 2 classes)
        ↓
    Output: [REAL, FAKE] logits

CLIP was trained on 400M image-text pairs — it has a rich understanding
of natural image structure, making it strong at spotting AI artifacts.

We fine-tune only the linear head by default (freeze_backbone=True):
    - Trainable params: ~1,538  (just the head)
    - Frozen params:    ~86M    (CLIP vision encoder)
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import CLIPVisionModel

from config import DEVICE, MODEL_NAME, NUM_CLASSES


@dataclass
class ModelOutput:
    """Simple output wrapper so train/inference code can use .logits as before."""
    logits: torch.Tensor


class CLIPClassifier(nn.Module):
    """
    CLIP Vision Encoder + linear classification head.

    The CLIP encoder extracts a 768-dim pooled representation per image.
    A single linear layer maps that to NUM_CLASSES (2: REAL / FAKE).
    """

    def __init__(self, freeze_backbone=True):
        super().__init__()

        # Load pre-trained CLIP vision encoder (no text encoder needed)
        self.vision_model = CLIPVisionModel.from_pretrained(MODEL_NAME)
        hidden_size = self.vision_model.config.hidden_size  # 768 for base-patch16

        # Classification head: 768 → 2
        self.classifier = nn.Linear(hidden_size, NUM_CLASSES)

        if freeze_backbone:
            # Freeze the entire CLIP encoder; only train the linear head
            for param in self.vision_model.parameters():
                param.requires_grad = False
            print("Backbone frozen — only training classification head.")

    def forward(self, pixel_values):
        # Extract visual features from CLIP encoder
        vision_outputs = self.vision_model(pixel_values=pixel_values)

        # pooler_output: [CLS] token → shape [batch, 768]
        pooled = vision_outputs.pooler_output

        # Map to class scores
        logits = self.classifier(pooled)

        return ModelOutput(logits=logits)


def create_model(freeze_backbone=True):
    """
    Create a CLIP-based classifier for REAL vs FAKE image detection.

    Args:
        freeze_backbone: If True (default), freeze the CLIP vision encoder
            and only train the linear head (~1.5K params).
            Set to False to unfreeze all ~86M params for full fine-tuning.

    Returns:
        model: CLIPClassifier on DEVICE.
    """
    print(f"Loading pre-trained CLIP model: {MODEL_NAME}")

    model = CLIPClassifier(freeze_backbone=freeze_backbone)
    model = model.to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Device:               {DEVICE}")

    return model


def load_model(checkpoint_path):
    """
    Load a trained CLIPClassifier from a checkpoint file.

    Args:
        checkpoint_path: Path to the saved .pth file.

    Returns:
        model: Loaded model in eval mode on DEVICE.
    """
    print(f"Loading model from: {checkpoint_path}")

    # Re-create the model architecture (backbone + head)
    model = CLIPClassifier(freeze_backbone=False)  # unfreeze so all weights load

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

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
    print("=" * 50)
    print("Test: CLIP Classifier (frozen backbone)")
    print("=" * 50)
    model = create_model(freeze_backbone=True)

    # Forward pass with dummy input
    dummy_input = torch.randn(4, 3, 224, 224).to(DEVICE)
    with torch.no_grad():
        outputs = model(pixel_values=dummy_input)
    probs = torch.softmax(outputs.logits, dim=1)
    print(f"\nInput shape:  {dummy_input.shape}")
    print(f"Output shape: {outputs.logits.shape}")   # Expected: [4, 2]
    print(f"Predictions:  {torch.argmax(probs, dim=1)}")  # 0=REAL, 1=FAKE
