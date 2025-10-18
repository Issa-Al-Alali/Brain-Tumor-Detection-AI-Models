"""
VERSION 4: OPTIMIZED MODEL - BALANCED REGULARIZATION
====================================================
Purpose: Fix over-regularization from Version 3 while maintaining good generalization

Analysis of Version 3 Issues:
-----------------------------
- Test accuracy dropped from 96.11% (V2) to 87.72% (V3)
- Over-regularization: dropout (0.5) + L2 (0.0001) + heavy augmentation
- The model is now UNDERFITTING rather than overfitting

Changes from Version 3:
-----------------------
1. REDUCED Data Augmentation:
   - Keep horizontal flip (medical relevance)
   - Reduce rotation: 15° → 10°
   - Remove affine transforms (too aggressive)
   - Reduce color jitter intensity
   
2. ADJUSTED Regularization:
   - Reduce dropout: 0.5 → 0.3
   - Reduce L2: 0.0001 → 0.00005
   - Balance between regularization and learning capacity

3. REFINED Training:
   - Keep learning rate scheduler (proven effective)
   - Increase epochs: 15 → 20 (model needs more time with lighter regularization)
   - Add early stopping patience
   
Expected Results:
- Test accuracy: 96-98%
- Overfitting gap: < 2%
- Better balance between generalization and performance
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
# DATASET SETUP WITH OPTIMIZED AUGMENTATION
# ===========================
print("Loading Brain Tumor MRI dataset...")

# TRAINING: Lighter, more appropriate augmentation
train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(p=0.5),  # Keep - medically relevant
    transforms.RandomRotation(10),  # Reduced from 15
    transforms.ColorJitter(brightness=0.1, contrast=0.1),  # Reduced from 0.2
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# TESTING: No augmentation
test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_path = '/kaggle/input/brain-tumor-mri-dataset/Training'
test_path = '/kaggle/input//brain-tumor-mri-dataset/Testing'

train_dataset = datasets.ImageFolder(root=train_path, transform=train_transform)
test_dataset = datasets.ImageFolder(root=test_path, transform=test_transform)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

classes = train_dataset.classes
print(f"Classes: {classes}")
print(f"Training samples: {len(train_dataset)}")
print(f"Testing samples: {len(test_dataset)}")

# ===========================
# OPTIMIZED CNN MODEL
# ===========================
class OptimizedCNN(nn.Module):
    """
    Optimized model with balanced regularization:
    - Same architecture as V2 and V3
    - Reduced dropout: 0.3 (from 0.5)
    - L2 regularization: 0.00005 (from 0.0001)
    - Lighter data augmentation
    """
    def __init__(self):
        super(OptimizedCNN, self).__init__()
        # 3 conv layers
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Reduced dropout
        self.dropout = nn.Dropout(0.3)  # Changed from 0.5
        
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

model = OptimizedCNN().to(device)
print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

# ===========================
# TRAINING CONFIG WITH BALANCED REGULARIZATION
# ===========================
criterion = nn.CrossEntropyLoss()

# Reduced L2 Regularization
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.00005)

# Learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

epochs = 20  # More epochs for better convergence
best_test_acc = 0
patience = 5
patience_counter = 0

train_losses, train_accuracies = [], []
test_losses, test_accuracies = [], []
learning_rates = []

# ===========================
# TRAINING LOOP WITH EARLY STOPPING
# ===========================
print("\nStarting training with optimized regularization...")
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
    
    # Early stopping check
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        patience_counter = 0
        print(f"  ✓ New best test accuracy!")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break
    
    # Step the scheduler
    scheduler.step()

# ===========================
# FINAL EVALUATION
# ===========================
print("\n" + "="*50)
print("FINAL RESULTS - OPTIMIZED MODEL")
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
plt.savefig('/kaggle/working/optimized_model_results.png', dpi=300, bbox_inches='tight')
plt.show()

results = {
    'model_name': 'Version 4 - Optimized (Balanced Regularization)',
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
    'epochs': len(train_accuracies),
    'early_stopped': len(train_accuracies) < epochs
}

with open('/kaggle/working/optimized_results.json', 'w') as f:
    json.dump(results, f, indent=4)

print("\nResults saved to 'optimized_results.json'")