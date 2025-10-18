"""
VERSION 2: IMPROVED MODEL (CORRECTED WITH TRAIN/VAL/TEST SPLIT)
================================================================
Purpose: Improve over baseline with deeper architecture and dropout

Changes from Version 1:
- 3 convolutional layers (instead of 2)
- 3 fully connected layers (instead of 1)
- Added Dropout (0.5) for regularization
- More parameters for better feature extraction

CORRECTED METHODOLOGY:
- Train/Validation split: 85%/15% of training data
- Test set: Evaluated ONLY ONCE at the end
- No information leakage from test set
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import warnings
import os
import json
warnings.filterwarnings("ignore")

# ===========================
# DEVICE SETUP
# ===========================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# ===========================
# DATASET SETUP WITH TRAIN/VAL/TEST SPLIT
# ===========================
print("Loading Brain Tumor MRI dataset...")

# Basic transformation (no augmentation)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_path = '/kaggle/input/brain-tumor-mri-dataset/Training'
test_path = '/kaggle/input//brain-tumor-mri-dataset/Testing'

# Load full training data
full_train_dataset = datasets.ImageFolder(root=train_path, transform=transform)

# Split into train (85%) and validation (15%)
train_size = int(0.85 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size

train_dataset, val_dataset = random_split(
    full_train_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

# Load test set (for final evaluation only)
test_dataset = datasets.ImageFolder(root=test_path, transform=transform)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

classes = full_train_dataset.classes
print(f"Classes: {classes}")
print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Testing samples: {len(test_dataset)}")

# ===========================
# CNN MODEL
# ===========================
class BrainTumorCNN(nn.Module):
    def __init__(self):
        super(BrainTumorCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.dropout = nn.Dropout(0.5)
        self._to_linear = None
        self._initialize_fc()
        self.fc1 = nn.Linear(self._to_linear, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 4)  # 4 classes

    def _initialize_fc(self):
        x = torch.randn(1, 3, 128, 128)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        self._to_linear = x.view(1, -1).size(1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

model = BrainTumorCNN().to(device)
print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

# ===========================
# TRAINING CONFIG
# ===========================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 10

train_losses, train_accuracies = [], []
val_losses, val_accuracies = [], []
best_val_acc = 0
best_model_state = None

# ===========================
# TRAINING LOOP (USING VALIDATION SET)
# ===========================
print("\nStarting training...")
print("=" * 60)
for epoch in range(epochs):
    # Training phase
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    train_acc = 100 * correct / total
    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    
    # Validation phase (NOT test!)
    model.eval()
    val_correct, val_total, val_running_loss = 0, 0, 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    
    val_acc = 100 * val_correct / val_total
    val_loss = val_running_loss / len(val_loader)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)
    
    print(f"Epoch [{epoch+1}/{epochs}]")
    print(f"  Train - Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")
    print(f"  Val   - Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")
    
    # Save best model based on validation accuracy
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_state = model.state_dict().copy()
        print(f"  ✓ Best validation accuracy updated!")

# Load best model
model.load_state_dict(best_model_state)

# ===========================
# FINAL TEST SET EVALUATION (ONLY ONCE!)
# ===========================
print("\n" + "="*60)
print("FINAL TEST SET EVALUATION")
print("="*60)

model.eval()
test_correct, test_total, test_running_loss = 0, 0, 0.0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

test_acc = 100 * test_correct / test_total
test_loss = test_running_loss / len(test_loader)

# ===========================
# FINAL EVALUATION
# ===========================
print("\n" + "="*60)
print("FINAL RESULTS - IMPROVED MODEL (Version 2)")
print("="*60)
print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
print(f"Final Test Accuracy: {test_acc:.2f}%")
print(f"Final Train Accuracy: {train_accuracies[-1]:.2f}%")
print(f"Train-Val Gap: {train_accuracies[-1] - best_val_acc:.2f}%")
print(f"Val-Test Gap: {best_val_acc - test_acc:.2f}%")

# ===========================
# VISUALIZATION
# ===========================
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Loss plot
axes[0].plot(train_losses, label='Train Loss', marker='o')
axes[0].plot(val_losses, label='Val Loss', marker='s')
axes[0].axhline(y=test_loss, color='red', linestyle='--', 
                label=f'Test Loss: {test_loss:.4f}', linewidth=2)
axes[0].set_title("Loss Comparison")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(True)

# Accuracy plot
axes[1].plot(train_accuracies, label='Train Accuracy', marker='o')
axes[1].plot(val_accuracies, label='Val Accuracy', marker='s')
axes[1].axhline(y=test_acc, color='red', linestyle='--', 
                label=f'Test Acc: {test_acc:.2f}%', linewidth=2)
axes[1].set_title("Accuracy Comparison")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy (%)")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig('/kaggle/working/improved_model_results.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nPlot saved as 'improved_model_results.png'")

# ===========================
# SAVE RESULTS TO JSON
# ===========================
results = {
    'model_name': 'Version 2 - Improved Model',
    'methodology': 'Proper train/val/test split - test evaluated once',
    'train_losses': train_losses,
    'train_accuracies': train_accuracies,
    'val_losses': val_losses,
    'val_accuracies': val_accuracies,
    'best_val_acc': best_val_acc,
    'final_test_acc': test_acc,
    'test_loss': test_loss,
    'train_val_gap': train_accuracies[-1] - best_val_acc,
    'val_test_gap': best_val_acc - test_acc,
    'epochs': len(train_accuracies)
}

with open('/kaggle/working/improved_model_results.json', 'w') as f:
    json.dump(results, f, indent=4)

print("Results saved to 'improved_model_results.json'")