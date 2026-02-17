"""
Training script for ViT-based AI-Generated Image Detection.

Usage:
    python train.py                    # Full fine-tuning
    python train.py --freeze           # Frozen backbone (faster, for testing)
    python train.py --epochs 5         # Override num epochs
    python train.py --resume checkpoint.pth  # Resume from checkpoint
"""

import argparse
import json
import os
import time

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from config import (
    BATCH_SIZE,
    DEVICE,
    EARLY_STOPPING_PATIENCE,
    LEARNING_RATE,
    LOG_INTERVAL,
    LOGS_DIR,
    MODEL_SAVE_PATH,
    NUM_EPOCHS,
    OUTPUT_DIR,
    SEED,
    WEIGHT_DECAY,
    create_dirs,
)
from dataset import get_dataloaders
from model import create_model


def set_seed(seed):
    """Set seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, train_loader, optimizer, scheduler, criterion, epoch, num_epochs):
    """
    Train the model for one epoch.

    Returns:
        tuple: (average_loss, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        # Forward pass
        outputs = model(pixel_values=images)
        loss = criterion(outputs.logits, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # Track metrics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Log progress
        if (batch_idx + 1) % LOG_INTERVAL == 0:
            avg_loss = running_loss / (batch_idx + 1)
            acc = 100.0 * correct / total
            lr = scheduler.get_last_lr()[0]
            print(
                f"  Epoch [{epoch+1}/{num_epochs}] "
                f"Batch [{batch_idx+1}/{len(train_loader)}] "
                f"Loss: {avg_loss:.4f} | Acc: {acc:.2f}% | LR: {lr:.2e}"
            )

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def validate(model, val_loader, criterion):
    """
    Evaluate the model on the validation set.

    Returns:
        tuple: (average_loss, accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in val_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(pixel_values=images)
        loss = criterion(outputs.logits, labels)

        running_loss += loss.item()
        _, predicted = torch.max(outputs.logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    val_loss = running_loss / len(val_loader)
    val_acc = 100.0 * correct / total
    return val_loss, val_acc


def save_checkpoint(model, optimizer, scheduler, epoch, val_loss, val_acc, path):
    """Save model checkpoint with all training state."""
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_loss": val_loss,
            "val_acc": val_acc,
        },
        path,
    )
    print(f"  Checkpoint saved: {path}")


def train(args):
    """
    Full training pipeline.

    Steps:
        1. Set seed for reproducibility
        2. Load data
        3. Create model
        4. Set up optimizer, scheduler, loss
        5. Training loop with validation & early stopping
        6. Save best model
    """
    # Setup
    set_seed(SEED)
    create_dirs()

    num_epochs = args.epochs
    print(f"\n{'='*60}")
    print(f"  AI-Generated Image Detection — Training")
    print(f"{'='*60}")
    print(f"  Device:         {DEVICE}")
    print(f"  Epochs:         {num_epochs}")
    print(f"  Batch size:     {BATCH_SIZE}")
    print(f"  Learning rate:  {LEARNING_RATE}")
    print(f"  Freeze backbone: {args.freeze}")
    print(f"{'='*60}\n")

    # Step 1: Data
    print("Loading data...")
    train_loader, val_loader, _ = get_dataloaders()

    # Step 2: Model
    print("\nCreating model...")
    model = create_model(freeze_backbone=args.freeze)

    # Step 3: Optimizer & Scheduler
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    total_steps = len(train_loader) * num_epochs
    scheduler = OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        total_steps=total_steps,
        pct_start=0.1,       # 10% warmup
        anneal_strategy="cos",
    )

    criterion = nn.CrossEntropyLoss()

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        print(f"\nResuming from: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resuming from epoch {start_epoch}")

    # Step 4: Training loop
    best_val_acc = 0.0
    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    print(f"\nStarting training...\n")
    total_start = time.time()

    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.time()

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, epoch, num_epochs
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion)

        epoch_time = time.time() - epoch_start

        # Log
        print(
            f"\n  Epoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s) — "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%"
        )

        # Save history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_loss, val_acc, MODEL_SAVE_PATH
            )
            patience_counter = 0
            print(f"  ★ New best model! Val Acc: {val_acc:.2f}%")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{EARLY_STOPPING_PATIENCE})")

        # Early stopping
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\n  Early stopping triggered after {epoch+1} epochs.")
            break

        print()

    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  Training complete!")
    print(f"  Total time:   {total_time/60:.1f} minutes")
    print(f"  Best Val Acc: {best_val_acc:.2f}%")
    print(f"  Best Val Loss: {best_val_loss:.4f}")
    print(f"  Model saved:  {MODEL_SAVE_PATH}")
    print(f"{'='*60}")

    # Save training history
    history_path = os.path.join(LOGS_DIR, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"  Training history saved: {history_path}")

    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ViT for AI Image Detection")
    parser.add_argument(
        "--freeze", action="store_true",
        help="Freeze ViT backbone, only train classification head",
    )
    parser.add_argument(
        "--epochs", type=int, default=NUM_EPOCHS,
        help=f"Number of training epochs (default: {NUM_EPOCHS})",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume training from",
    )
    args = parser.parse_args()

    train(args)
