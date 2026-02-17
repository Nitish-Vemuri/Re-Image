"""
Inference script — predict if a single image is Real or AI-Generated.

Usage:
    python inference.py --image path/to/image.jpg
    python inference.py --image path/to/image.png --model path/to/model.pth
    python inference.py --dir path/to/folder/   # Run on all images in a folder
"""

import argparse
import os

import torch
from PIL import Image

from config import CLASSES, DEVICE, MODEL_SAVE_PATH
from dataset import get_eval_transforms
from model import load_model


def predict_image(model, image_path):
    """
    Predict whether a single image is Real or AI-Generated.

    Args:
        model: Trained ViT model (in eval mode).
        image_path: Path to the image file.

    Returns:
        dict: {
            "image": filename,
            "prediction": "REAL" or "FAKE",
            "confidence": float (0-1),
            "probabilities": {"FAKE": float, "REAL": float}
        }
    """
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    transform = get_eval_transforms()
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)  # Add batch dim

    # Predict
    with torch.no_grad():
        outputs = model(pixel_values=input_tensor)
        probs = torch.softmax(outputs.logits, dim=1)[0]

    predicted_idx = torch.argmax(probs).item()
    predicted_class = CLASSES[predicted_idx]
    confidence = probs[predicted_idx].item()

    result = {
        "image": os.path.basename(image_path),
        "prediction": predicted_class,
        "confidence": confidence,
        "probabilities": {
            CLASSES[i]: round(probs[i].item(), 4) for i in range(len(CLASSES))
        },
    }

    return result


def predict_directory(model, dir_path):
    """
    Run inference on all images in a directory.

    Args:
        model: Trained ViT model.
        dir_path: Path to directory containing images.

    Returns:
        list: List of prediction dicts.
    """
    supported_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
    results = []

    image_files = [
        f for f in os.listdir(dir_path)
        if os.path.splitext(f)[1].lower() in supported_ext
    ]

    if not image_files:
        print(f"No images found in {dir_path}")
        return results

    print(f"Running inference on {len(image_files)} images...\n")

    for filename in sorted(image_files):
        image_path = os.path.join(dir_path, filename)
        result = predict_image(model, image_path)
        results.append(result)

        icon = "🤖" if result["prediction"] == "FAKE" else "📷"
        print(
            f"  {icon} {result['image']:30s} → "
            f"{result['prediction']:4s} ({result['confidence']*100:.1f}%)"
        )

    # Summary
    fake_count = sum(1 for r in results if r["prediction"] == "FAKE")
    real_count = len(results) - fake_count
    print(f"\nSummary: {fake_count} FAKE, {real_count} REAL out of {len(results)} images")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict if an image is Real or AI-Generated"
    )
    parser.add_argument(
        "--image", type=str, default=None,
        help="Path to a single image file",
    )
    parser.add_argument(
        "--dir", type=str, default=None,
        help="Path to a directory of images",
    )
    parser.add_argument(
        "--model", type=str, default=MODEL_SAVE_PATH,
        help=f"Path to model checkpoint (default: {MODEL_SAVE_PATH})",
    )
    args = parser.parse_args()

    if not args.image and not args.dir:
        parser.error("Provide either --image or --dir")

    # Load model
    model = load_model(args.model)

    if args.image:
        # Single image
        result = predict_image(model, args.image)
        print(f"\n{'='*50}")
        print(f"  Image:      {result['image']}")
        print(f"  Prediction: {result['prediction']}")
        print(f"  Confidence: {result['confidence']*100:.1f}%")
        print(f"  FAKE prob:  {result['probabilities']['FAKE']*100:.1f}%")
        print(f"  REAL prob:  {result['probabilities']['REAL']*100:.1f}%")
        print(f"{'='*50}")

    elif args.dir:
        # Directory of images
        predict_directory(model, args.dir)
