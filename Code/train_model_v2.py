"""
RidgeDx — ResNet-18 Training Pipeline
=======================================

Features:
    - ResNet-18 modified for grayscale input (1 channel)
    - 8-class ABO+Rh blood group classification
    - 70/15/15 train/val/test split with stratification
    - Data augmentation: rotation, flip, affine transforms
    - Early stopping with patience
    - Learning rate scheduling (ReduceLROnPlateau)
    - Training logs and accuracy/loss plots

Usage:
    python train_model_v2.py

Output:
    - resnet18_bloodgroup.pth (best model weights)
    - training_plots.png (accuracy/loss curves)
"""

import os
import sys
import time
import copy
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from sklearn.model_selection import train_test_split

from config import get_config

# ============================================================
# Configuration
# ============================================================
config = get_config()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR = os.environ.get('DATA_DIR', os.path.join(config.BASE_DIR, 'dataset_blood_group'))
MODEL_SAVE_PATH = config.MODEL_PATH
NUM_CLASSES = config.NUM_CLASSES
IMAGE_SIZE = config.IMAGE_SIZE
BATCH_SIZE = 32
NUM_EPOCHS = 30
LEARNING_RATE = 1e-3
PATIENCE = 7  # Early stopping patience
RANDOM_SEED = 42

print(f"Device: {DEVICE}")
print(f"Data directory: {DATA_DIR}")
print(f"Model save path: {MODEL_SAVE_PATH}")
print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Max epochs: {NUM_EPOCHS}")
print(f"Early stopping patience: {PATIENCE}")
print()


# ============================================================
# Data Augmentation & Transforms
# ============================================================
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.1, 0.1),
        scale=(0.9, 1.1),
        shear=5
    ),
    transforms.ToTensor(),
])

val_test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])


# ============================================================
# Dataset & Split
# ============================================================
print("Loading dataset...")
full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=train_transform)

classes = full_dataset.classes
print(f"Detected classes ({len(classes)}): {classes}")
print(f"Total samples: {len(full_dataset)}")

# Get labels for stratified split
labels = [s[1] for s in full_dataset.samples]

# 70/15/15 split with stratification
train_idx, temp_idx = train_test_split(
    range(len(full_dataset)),
    test_size=0.30,
    stratify=labels,
    random_state=RANDOM_SEED
)

temp_labels = [labels[i] for i in temp_idx]
val_idx, test_idx = train_test_split(
    temp_idx,
    test_size=0.50,
    stratify=temp_labels,
    random_state=RANDOM_SEED
)

print(f"Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")
print()

# Create subsets (apply different transforms for val/test)
train_dataset = Subset(full_dataset, train_idx)

# For val/test, use the non-augmented transform
val_dataset_base = datasets.ImageFolder(root=DATA_DIR, transform=val_test_transform)
val_dataset = Subset(val_dataset_base, val_idx)
test_dataset = Subset(val_dataset_base, test_idx)

pin_mem = DEVICE.type == 'cuda'
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=pin_mem)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=pin_mem)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=pin_mem)


# ============================================================
# Model
# ============================================================
def build_resnet18(num_classes, pretrained=True):
    """Build ResNet-18 modified for 1-channel grayscale input."""
    if pretrained:
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)
    else:
        model = models.resnet18(weights=None)

    # Modify first conv layer: 3-channel -> 1-channel
    original_conv = model.conv1
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    if pretrained:
        with torch.no_grad():
            model.conv1.weight = nn.Parameter(
                original_conv.weight.mean(dim=1, keepdim=True)
            )

    # Modify final FC layer for N classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


model = build_resnet18(NUM_CLASSES, pretrained=True).to(DEVICE)
print(f"Model: ResNet-18 (modified for grayscale)")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
print()


# ============================================================
# Training Setup
# ============================================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3
)

# Training history
history = {
    'train_loss': [], 'train_acc': [],
    'val_loss': [], 'val_acc': []
}


# ============================================================
# Training Loop
# ============================================================
best_val_loss = float('inf')
best_model_wts = copy.deepcopy(model.state_dict())
patience_counter = 0
start_time = time.time()

print("=" * 60)
print("Starting training...")
print("=" * 60)

for epoch in range(NUM_EPOCHS):
    epoch_start = time.time()

    # --- Training Phase ---
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels_batch in train_loader:
        inputs, labels_batch = inputs.to(DEVICE), labels_batch.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels_batch.size(0)
        correct += (predicted == labels_batch).sum().item()

    train_loss = running_loss / total
    train_acc = 100.0 * correct / total

    # --- Validation Phase ---
    model.eval()
    val_running_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for inputs, labels_batch in val_loader:
            inputs, labels_batch = inputs.to(DEVICE), labels_batch.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels_batch)

            val_running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            val_total += labels_batch.size(0)
            val_correct += (predicted == labels_batch).sum().item()

    val_loss = val_running_loss / val_total
    val_acc = 100.0 * val_correct / val_total

    # Record history
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)

    # Learning rate scheduling
    scheduler.step(val_loss)

    # Early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_wts = copy.deepcopy(model.state_dict())
        patience_counter = 0
        marker = ' * (best)'
    else:
        patience_counter += 1
        marker = f' (patience: {patience_counter}/{PATIENCE})'

    epoch_time = time.time() - epoch_start
    current_lr = optimizer.param_groups[0]['lr']

    print(
        f"Epoch [{epoch+1:2d}/{NUM_EPOCHS}] "
        f"Train Loss: {train_loss:.4f} Acc: {train_acc:.1f}% | "
        f"Val Loss: {val_loss:.4f} Acc: {val_acc:.1f}% | "
        f"LR: {current_lr:.6f} | {epoch_time:.1f}s{marker}"
    )

    if patience_counter >= PATIENCE:
        print(f"\nEarly stopping triggered at epoch {epoch+1}")
        break

total_time = time.time() - start_time
print(f"\nTraining complete in {total_time:.1f}s ({total_time/60:.1f} min)")


# ============================================================
# Load Best Weights & Save
# ============================================================
model.load_state_dict(best_model_wts)
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"Best model saved to {MODEL_SAVE_PATH}")


# ============================================================
# Test Evaluation
# ============================================================
print("\n" + "=" * 60)
print("Evaluating on test set...")
print("=" * 60)

model.eval()
test_correct = 0
test_total = 0
class_correct = [0] * NUM_CLASSES
class_total = [0] * NUM_CLASSES

with torch.no_grad():
    for inputs, labels_batch in test_loader:
        inputs, labels_batch = inputs.to(DEVICE), labels_batch.to(DEVICE)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        test_total += labels_batch.size(0)
        test_correct += (predicted == labels_batch).sum().item()

        for i in range(labels_batch.size(0)):
            label = labels_batch[i].item()
            class_total[label] += 1
            if predicted[i] == label:
                class_correct[label] += 1

test_acc = 100.0 * test_correct / test_total
print(f"\nTest Accuracy: {test_acc:.2f}% ({test_correct}/{test_total})")
print("\nPer-class accuracy:")
for i, cls in enumerate(classes):
    if class_total[i] > 0:
        cls_acc = 100.0 * class_correct[i] / class_total[i]
        print(f"  {cls:>4s}: {cls_acc:5.1f}% ({class_correct[i]}/{class_total[i]})")
    else:
        print(f"  {cls:>4s}: N/A (0 samples)")


# ============================================================
# Save Training Plots
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

epochs_range = range(1, len(history['train_loss']) + 1)

# Loss plot
ax1.plot(epochs_range, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
ax1.plot(epochs_range, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training & Validation Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Accuracy plot
ax2.plot(epochs_range, history['train_acc'], 'b-', label='Train Accuracy', linewidth=2)
ax2.plot(epochs_range, history['val_acc'], 'r-', label='Val Accuracy', linewidth=2)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Training & Validation Accuracy')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(config.BASE_DIR, 'training_plots.png')
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nTraining plots saved to {plot_path}")
print("\nDone!")
