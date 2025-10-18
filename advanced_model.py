"""
VERSION 3: ADVANCED MODEL WITH REGULARIZATION
==============================================
Purpose: Maximize generalization and reduce overfitting

Changes from Version 2 (Improved):
----------------------------------
Data Augmentation (NEW):
- Random horizontal flips (medical images can appear flipped)
- Random rotation (±15 degrees) for orientation invariance
- Random affine transformations for slight deformations
- Color jitter for brightness/contrast variations
- Applied only to training data

Regularization Techniques:
- L2 Regularization (weight_decay=0.0001) added to optimizer
- Maintained Dropout (0.5)
- Combination of both regularization methods

Model Architecture:
- Same as Version 2 (3 conv layers, 3 FC layers)
- Architecture is already good, focus on regularization

Training Configuration:
- Learning Rate Scheduler: StepLR (decay by 0.5 every 5 epochs)
- Increased epochs: 10 → 15 (to benefit from LR scheduling)
- L2 regularization through weight_decay
- Better convergence with adaptive learning rate

Expected Improvements:
- Significantly reduced overfitting
- Better generalization to unseen data
- More robust to image variations
- Smaller train-test accuracy gap
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
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
# DATASET SETUP WITH AUGMENTATION
# ===========================
print("Loading Brain Tumor MRI dataset...")

# TRAINING: With data augmentation
train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# TESTING: No augmentation (only normalization)
test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

script_dir = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(script_dir, 'brain_tumor_dataset', 'training')
test_path = os.path.join(script_dir, 'brain_tumor_dataset', 'testing')

train_dataset = datasets.ImageFolder(root=train_path, transform=train_transform)
test_dataset = datasets.ImageFolder(root=test_path, transform=test_transform)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

classes = train_dataset.classes
print(f"Classes: {classes}")
print(f"Training samples: {len(train_dataset)}")
print(f"Testing samples: {len(test_dataset)}")

# ===========================
# ADVANCED CNN MODEL
# ===========================
class AdvancedCNN(nn.Module):
    """
    Advanced model with same architecture as Version 2
    but trained with better regularization techniques:
    - Data augmentation (applied in dataset)
    - L2 regularization (applied in optimizer)
    - Dropout (0.5)
    """
    def __init__(self):
        super(AdvancedCNN, self).__init__()
        # 3 conv layers
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
        
        # Calculate flattened size
        self._to_linear = None
        self._initialize_fc()
        
        # 3 FC layers
        self.fc1 = nn.Linear(self._to_linear, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 4)

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

model = AdvancedCNN().to(device)
print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

# ===========================
# TRAINING CONFIG WITH L2 & SCHEDULER
# ===========================
criterion = nn.CrossEntropyLoss()

# L2 Regularization via weight_decay
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

# Learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

epochs = 15  # More epochs to benefit from LR scheduling

train_losses, train_accuracies = [], []
test_losses, test_accuracies = [], []
learning_rates = []

# ===========================
# TRAINING LOOP
# ===========================
print("\nStarting training with augmentation, L2, and LR scheduling...")
for epoch in range(epochs):
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
    
    # Evaluate on test set
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
    test_losses.append(test_loss)
    test_accuracies.append(test_acc)
    
    # Record learning rate
    current_lr = optimizer.param_groups[0]['lr']
    learning_rates.append(current_lr)
    
    print(f"Epoch [{epoch+1}/{epochs}] - LR: {current_lr:.6f}")
    print(f"  Train - Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")
    print(f"  Test  - Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%")
    
    # Step the scheduler
    scheduler.step()

# ===========================
# FINAL EVALUATION
# ===========================
print("\n" + "="*50)
print("FINAL RESULTS - ADVANCED MODEL")
print("="*50)
print(f"Final Train Accuracy: {train_accuracies[-1]:.2f}%")
print(f"Final Test Accuracy: {test_accuracies[-1]:.2f}%")
print(f"Overfitting Gap: {train_accuracies[-1] - test_accuracies[-1]:.2f}%")
print(f"\nBest Test Accuracy: {max(test_accuracies):.2f}% (Epoch {test_accuracies.index(max(test_accuracies))+1})")

# ===========================
# VISUALIZATION
# ===========================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Loss plot
axes[0, 0].plot(train_losses, label='Train Loss', marker='o')
axes[0, 0].plot(test_losses, label='Test Loss', marker='s')
axes[0, 0].set_title("Loss Comparison")
axes[0, 0].set_xlabel("Epoch")
axes[0, 0].set_ylabel("Loss")
axes[0, 0].legend()
axes[0, 0].grid(True)

# Accuracy plot
axes[0, 1].plot(train_accuracies, label='Train Accuracy', marker='o')
axes[0, 1].plot(test_accuracies, label='Test Accuracy', marker='s')
axes[0, 1].set_title("Accuracy Comparison")
axes[0, 1].set_xlabel("Epoch")
axes[0, 1].set_ylabel("Accuracy (%)")
axes[0, 1].legend()
axes[0, 1].grid(True)

# Learning rate schedule
axes[1, 0].plot(learning_rates, marker='o', color='green')
axes[1, 0].set_title("Learning Rate Schedule")
axes[1, 0].set_xlabel("Epoch")
axes[1, 0].set_ylabel("Learning Rate")
axes[1, 0].grid(True)

# Overfitting gap over epochs
gaps = [train_accuracies[i] - test_accuracies[i] for i in range(len(train_accuracies))]
axes[1, 1].plot(gaps, marker='o', color='red')
axes[1, 1].set_title("Overfitting Gap (Train - Test)")
axes[1, 1].set_xlabel("Epoch")
axes[1, 1].set_ylabel("Gap (%)")
axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.3)
axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig('advanced_model_results.png', dpi=300, bbox_inches='tight')
plt.show()

results = {
    'model_name': 'Version 3 - Advanced with Regularization',
    'train_losses': train_losses,
    'train_accuracies': train_accuracies,
    'test_losses': test_losses,
    'test_accuracies': test_accuracies,
    'learning_rates': learning_rates,
    'final_train_acc': train_accuracies[-1],
    'final_test_acc': test_accuracies[-1],
    'overfitting_gap': train_accuracies[-1] - test_accuracies[-1],
    'best_test_acc': max(test_accuracies),
    'best_epoch': test_accuracies.index(max(test_accuracies)) + 1,
    'epochs': len(train_accuracies)
}

with open('advanced_results.json', 'w') as f:
    json.dump(results, f, indent=4)

print("\nResults saved to 'advanced_results.json'")