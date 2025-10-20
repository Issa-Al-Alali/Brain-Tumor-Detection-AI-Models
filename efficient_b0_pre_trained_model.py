"""
VERSION 5: TRANSFER LEARNING WITH EFFICIENTNET-B0
================================================================
Purpose: Surpass the optimized model's accuracy using a state-of-the-art
         pre-trained model and a fine-tuning strategy.

Model Architecture:
- EfficientNet-B0: A highly efficient and powerful pre-trained model.
- The base model's weights are frozen initially to preserve learned features.
- The final classifier "head" is replaced with a new one for our 4 classes.

Training Strategy:
- Two-Phase Fine-Tuning:
  1. Feature Extraction (5 Epochs): Train ONLY the new classifier head with a
     standard learning rate. This adapts the model to our dataset.
  2. Fine-Tuning (15 Epochs): Unfreeze all layers and train the entire network
     with a very low learning rate to specialize it for tumor classification
     without "catastrophic forgetting".
- Data Augmentation: Uses the same balanced augmentation from Version 4.
- Image Size: Increased to 224x224, the standard for EfficientNet, to
  provide more detail to the model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np
import warnings
import json
import time
from tqdm import tqdm

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

# EfficientNet expects 224x224 images and specific normalization
# We use the same 'lighter' augmentation from the V4 optimized model
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

train_path = '/kaggle/input/brain-tumor-mri-dataset/Training'
test_path = '/kaggle/input//brain-tumor-mri-dataset/Testing'

# Load full training data with validation transform first
full_train_dataset = datasets.ImageFolder(root=train_path, transform=data_transforms['val'])

# Split into train (85%) and validation (15%)
train_size = int(0.85 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size
indices = list(range(len(full_train_dataset)))
np.random.seed(42)
np.random.shuffle(indices)
train_indices, val_indices = indices[:train_size], indices[train_size:]

# Create datasets with the correct transforms
train_dataset_aug = datasets.ImageFolder(root=train_path, transform=data_transforms['train'])
train_dataset = Subset(train_dataset_aug, train_indices)
val_dataset = Subset(full_train_dataset, val_indices)
test_dataset = datasets.ImageFolder(root=test_path, transform=data_transforms['val'])

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Testing samples: {len(test_dataset)}")

# ===========================
# MODEL SETUP (EFFICIENTNET-B0)
# ===========================

# 1. Initialize the model architecture WITHOUT weights first
model = models.efficientnet_b0(weights=None)

# 2. Define the path to the local .pth file
WEIGHTS_FILE = '/kaggle/input/efficient-b-0-weights/pytorch/default/1/efficientnet_b0_rwightman-7f5810bc.pth'

print(f"Loading weights from: {WEIGHTS_FILE}")

# 3. Load the state dictionary (CORRECTED VERSION)
try:
    # Load the checkpoint file
    checkpoint = torch.load(WEIGHTS_FILE, map_location=device)
    
    # Extract the state_dict (it might be nested)
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            # If none of the above, assume the checkpoint IS the state_dict
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # Remove 'module.' prefix if it exists (from DataParallel)
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    # Load weights with strict=False to ignore mismatched classifier layers
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    print("✓ EfficientNet-B0 weights loaded successfully from local file.")
    if missing_keys:
        print(f"  Missing keys (expected for classifier): {len(missing_keys)} keys")
    if unexpected_keys:
        print(f"  Unexpected keys: {len(unexpected_keys)} keys")
    
except Exception as e:
    print(f"ERROR loading weights: {e}")
    print("\nTrying alternative: loading with weights_only=True")
    try:
        # For newer PyTorch versions, try weights_only parameter
        state_dict = torch.load(WEIGHTS_FILE, map_location=device, weights_only=True)
        model.load_state_dict(state_dict, strict=False)
        print("✓ Weights loaded successfully with weights_only=True")
    except:
        print("FATAL ERROR: Could not load weights.")
        print("Consider using PyTorch's built-in weights instead:")
        print("  model = models.efficientnet_b0(weights='IMAGENET1K_V1')")
        raise

# Freeze all the parameters in the feature extraction layers
for param in model.parameters():
    param.requires_grad = False

# Replace the classifier
# Get input features from the original classifier's last layer
num_ftrs = model.classifier[1].in_features 
model.classifier = nn.Sequential(
    nn.Dropout(p=0.3, inplace=True),
    nn.Linear(num_ftrs, 4) # Your 4 classes
)

model = model.to(device)
print(f"Model: EfficientNet-B0 (Tuned on Melanoma/ImageNet)")
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# ===========================
# TRAINING CONFIG
# ===========================
criterion = nn.CrossEntropyLoss()
epochs_feature_extraction = 5
epochs_fine_tuning = 15
total_epochs = epochs_feature_extraction + epochs_fine_tuning

# We will define the optimizer inside the training loop

train_losses, train_accuracies = [], []
val_losses, val_accuracies = [], []
best_val_acc = 0
best_model_state = None

# ===========================
# TRAINING LOOP
# ===========================
start_time = time.time()

for phase in ['feature_extraction', 'fine_tuning']:
    if phase == 'feature_extraction':
        print("\n" + "="*70)
        print("PHASE 1: FEATURE EXTRACTION (Training the classifier head)")
        print("="*70)
        # Only train the classifier head
        optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
        epochs = epochs_feature_extraction
    else:
        print("\n" + "="*70)
        print("PHASE 2: FINE-TUNING (Training the entire network)")
        print("="*70)
        # Unfreeze all layers
        for param in model.parameters():
            param.requires_grad = True
        # Use a very low learning rate
        optimizer = optim.Adam(model.parameters(), lr=1e-5)
        epochs = epochs_fine_tuning

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    for epoch in range(epochs):
        epoch_num = len(train_accuracies) + 1
        # --- TRAINING ---
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch_num}/{total_epochs} [Train]")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        train_loss = running_loss / total
        train_acc = 100 * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # --- VALIDATION ---
        model.eval()
        val_running_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch_num}/{total_epochs} [Val]")
            for inputs, labels in pbar_val:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss = val_running_loss / val_total
        val_acc = 100 * val_correct / val_total
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch_num}/{total_epochs} -> Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            print(f"  ✓ New best validation accuracy: {best_val_acc:.2f}%")

        scheduler.step()

# Load best model
model.load_state_dict(best_model_state)

# ===========================
# FINAL TEST SET EVALUATION
# ===========================
print("\n" + "="*70)
print("FINAL TEST SET EVALUATION")
print("="*70)

model.eval()
test_correct, test_total = 0, 0
with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Testing"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

test_acc = 100 * test_correct / test_total

# ===========================
# FINAL RESULTS
# ===========================
end_time = time.time()
print("\n" + "="*70)
print("FINAL RESULTS - EFFICIENTNET-B0 TRANSFER LEARNING")
print("="*70)
print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
print(f"Final Test Accuracy: {test_acc:.2f}%")
print(f"Total Training Time: {(end_time - start_time) / 60:.2f} minutes")

# ===========================
# VISUALIZATION
# ===========================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
epochs_range = range(1, len(train_accuracies) + 1)

# Accuracy Plot
axes[0].plot(epochs_range, train_accuracies, label='Train Accuracy', marker='o')
axes[0].plot(epochs_range, val_accuracies, label='Val Accuracy', marker='s')
axes[0].axvline(x=epochs_feature_extraction, color='grey', linestyle='--', label='Fine-Tuning Start')
axes[0].axhline(y=test_acc, color='red', linestyle='--', label=f'Test Acc: {test_acc:.2f}%')
axes[0].set_title("Accuracy vs. Epochs", fontweight='bold')
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy (%)")
axes[0].legend()
axes[0].grid(True)

# Loss Plot
axes[1].plot(epochs_range, train_losses, label='Train Loss', marker='o')
axes[1].plot(epochs_range, val_losses, label='Val Loss', marker='s')
axes[1].axvline(x=epochs_feature_extraction, color='grey', linestyle='--', label='Fine-Tuning Start')
axes[1].set_title("Loss vs. Epochs", fontweight='bold')
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig('transfer_learning_results.png', dpi=300)
plt.show()

results = {
    'model_name': 'Version 5 - Transfer Learning (EfficientNet-B0)',
    'best_val_acc': best_val_acc,
    'final_test_acc': test_acc,
    'train_accuracies': train_accuracies,
    'val_accuracies': val_accuracies,
    'train_losses': train_losses,
    'val_losses': val_losses,
    'total_epochs': len(train_accuracies)
}

with open('transfer_learning_results.json', 'w') as f:
    json.dump(results, f, indent=4)

print("\n✓ Results saved to 'transfer_learning_results.json'")
print("✓ Plot saved as 'transfer_learning_results.png'")

