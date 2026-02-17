"""
Evaluation script for the trained AI-Generated Image Detection model.

Runs the model on the test set and generates:
    1. Accuracy, Precision, Recall, F1-Score
    2. Confusion Matrix (saved as image)
    3. Training history plots (loss & accuracy curves)
    4. Per-class metrics

Usage:
    python evaluate.py                          # Use default best_model.pth
    python evaluate.py --model path/to/model.pth  # Use specific checkpoint
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from config import (
    CLASSES,
    DEVICE,
    LOGS_DIR,
    MODEL_SAVE_PATH,
    PLOTS_DIR,
    create_dirs,
)
from dataset import get_dataloaders
from model import load_model


# =============================================================================
# EVALUATION ON TEST SET
# =============================================================================
@torch.no_grad()
def evaluate_model(model, test_loader):
    """
    Run model on test set and collect predictions.

    Returns:
        all_labels: Ground truth labels
        all_preds: Predicted labels
        all_probs: Prediction probabilities (for confidence analysis)
    """
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    print(f"Evaluating on {len(test_loader.dataset)} test images...")

    for images, labels in test_loader:
        images = images.to(DEVICE)

        outputs = model(pixel_values=images)
        probs = torch.softmax(outputs.logits, dim=1)
        _, predicted = torch.max(probs, 1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    return all_labels, all_preds, all_probs


# =============================================================================
# METRICS
# =============================================================================
def print_metrics(labels, preds):
    """Print detailed classification metrics."""
    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average="weighted")
    recall = recall_score(labels, preds, average="weighted")
    f1 = f1_score(labels, preds, average="weighted")

    print(f"\n{'='*50}")
    print(f"  TEST SET RESULTS")
    print(f"{'='*50}")
    print(f"  Accuracy:  {acc*100:.2f}%")
    print(f"  Precision: {precision*100:.2f}%")
    print(f"  Recall:    {recall*100:.2f}%")
    print(f"  F1-Score:  {f1*100:.2f}%")
    print(f"{'='*50}")

    print(f"\n  Per-class report:")
    print(classification_report(labels, preds, target_names=CLASSES))

    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


# =============================================================================
# PLOTS
# =============================================================================
def plot_confusion_matrix(labels, preds):
    """Generate and save confusion matrix plot."""
    cm = confusion_matrix(labels, preds)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        ylabel="True Label",
        xlabel="Predicted Label",
        title="Confusion Matrix",
    )

    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=14,
            )

    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, "confusion_matrix.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Confusion matrix saved: {save_path}")


def plot_training_history():
    """Plot training loss and accuracy curves from saved history."""
    history_path = os.path.join(LOGS_DIR, "training_history.json")

    if not os.path.exists(history_path):
        print(f"  No training history found at {history_path}")
        return

    with open(history_path, "r") as f:
        history = json.load(f)

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot
    ax1.plot(epochs, history["train_loss"], "b-o", label="Train Loss", markersize=4)
    ax1.plot(epochs, history["val_loss"], "r-o", label="Val Loss", markersize=4)
    ax1.set_title("Loss over Epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy plot
    ax2.plot(epochs, history["train_acc"], "b-o", label="Train Acc", markersize=4)
    ax2.plot(epochs, history["val_acc"], "r-o", label="Val Acc", markersize=4)
    ax2.set_title("Accuracy over Epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, "training_history.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Training history plot saved: {save_path}")


def plot_confidence_distribution(labels, probs):
    """Plot the distribution of prediction confidence scores."""
    # Get the confidence of the predicted class
    max_probs = np.max(probs, axis=1)

    correct_mask = labels == np.argmax(probs, axis=1)
    correct_conf = max_probs[correct_mask]
    incorrect_conf = max_probs[~correct_mask]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(correct_conf, bins=50, alpha=0.7, label=f"Correct ({len(correct_conf)})", color="green")
    ax.hist(incorrect_conf, bins=50, alpha=0.7, label=f"Incorrect ({len(incorrect_conf)})", color="red")
    ax.set_title("Prediction Confidence Distribution")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, "confidence_distribution.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Confidence distribution saved: {save_path}")


# =============================================================================
# MAIN
# =============================================================================
def evaluate(model_path):
    """Full evaluation pipeline."""
    create_dirs()

    # Load model
    model = load_model(model_path)

    # Load test data
    _, _, test_loader = get_dataloaders()

    # Run evaluation
    labels, preds, probs = evaluate_model(model, test_loader)

    # Print metrics
    metrics = print_metrics(labels, preds)

    # Generate plots
    print("\nGenerating plots...")
    plot_confusion_matrix(labels, preds)
    plot_training_history()
    plot_confidence_distribution(labels, probs)

    # Save metrics to file
    metrics_path = os.path.join(LOGS_DIR, "test_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)
    print(f"  Metrics saved: {metrics_path}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained model on test set")
    parser.add_argument(
        "--model", type=str, default=MODEL_SAVE_PATH,
        help=f"Path to model checkpoint (default: {MODEL_SAVE_PATH})",
    )
    args = parser.parse_args()

    evaluate(args.model)
