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
# DATASET SETUP
# ===========================
print("Loading Brain Tumor MRI dataset...")

# Basic transformation (no augmentation)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Dataset paths - Using absolute path from script location
train_path = '/kaggle/input/brain-tumor-mri-dataset/Training'
test_path = '/kaggle/input//brain-tumor-mri-dataset/Testing'

train_dataset = datasets.ImageFolder(root=train_path, transform=transform)
test_dataset = datasets.ImageFolder(root=test_path, transform=transform)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

classes = train_dataset.classes
print(f"Classes: {classes}")
print(f"Training samples: {len(train_dataset)}")
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
test_losses, test_accuracies = [], []

# ===========================
# TRAINING LOOP
# ===========================
print("\nStarting training...")
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
    
    # Evaluate on test set each epoch
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
    
    print(f"Epoch [{epoch+1}/{epochs}]")
    print(f"  Train - Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")
    print(f"  Test  - Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%")

# ===========================
# FINAL EVALUATION
# ===========================
print("\n" + "="*50)
print("FINAL RESULTS - Improved MODEL (Version 2)")
print("="*50)
print(f"Final Train Accuracy: {train_accuracies[-1]:.2f}%")
print(f"Final Test Accuracy: {test_accuracies[-1]:.2f}%")
print(f"Overfitting Gap: {train_accuracies[-1] - test_accuracies[-1]:.2f}%")

# ===========================
# VISUALIZATION
# ===========================
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Loss plot
axes[0].plot(train_losses, label='Train Loss', marker='o')
axes[0].plot(test_losses, label='Test Loss', marker='s')
axes[0].set_title("Loss Comparison")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(True)

# Accuracy plot
axes[1].plot(train_accuracies, label='Train Accuracy', marker='o')
axes[1].plot(test_accuracies, label='Test Accuracy', marker='s')
axes[1].set_title("Accuracy Comparison")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy (%)")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig('/kaggle/working/imporved_model_results.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nPlot saved as 'improved_model_results.png'")

# ===========================
# SAVE RESULTS TO JSON
# ===========================
results = {
    'model_name': 'Version 2 - Improved Model',
    'train_losses': train_losses,
    'train_accuracies': train_accuracies,
    'test_losses': test_losses,
    'test_accuracies': test_accuracies,
    'final_train_acc': train_accuracies[-1],
    'final_test_acc': test_accuracies[-1],
    'overfitting_gap': train_accuracies[-1] - test_accuracies[-1],
    'epochs': len(train_accuracies)
}

with open('/kaggle/working/improved_model_results.json', 'w') as f:
    json.dump(results, f, indent=4)

print("Results saved to 'improved_model_results.json'")