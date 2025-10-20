"""
resnet18 TRANSFER LEARNING FOR BRAIN TUMOR CLASSIFICATION
================================================================
ResNet architecture often works better for medical imaging than EfficientNet
because of its simpler, more interpretable residual connections.

Training Strategy:
- Progressive unfreezing (4 phases)
- Enhanced augmentation tailored for medical images
- Discriminative learning rates
- Deeper custom classifier head
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

# Enhanced augmentation for medical images
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((240, 240)),
        transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
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

# Load datasets
full_train_dataset = datasets.ImageFolder(root=train_path, transform=data_transforms['val'])

# Split into train (85%) and validation (15%)
train_size = int(0.85 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size
indices = list(range(len(full_train_dataset)))
np.random.seed(42)
np.random.shuffle(indices)
train_indices, val_indices = indices[:train_size], indices[train_size:]

train_dataset_aug = datasets.ImageFolder(root=train_path, transform=data_transforms['train'])
train_dataset = Subset(train_dataset_aug, train_indices)
val_dataset = Subset(full_train_dataset, val_indices)
test_dataset = datasets.ImageFolder(root=test_path, transform=data_transforms['val'])

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Testing samples: {len(test_dataset)}")

# ===========================
# MODEL SETUP (resnet18)
# ===========================

# 1. Initialize the model architecture WITHOUT weights first
model = models.resnet18(weights=None)

# 2. IMPORTANT: Update this path to your local weights file
WEIGHTS_FILE = '/kaggle/input/resnet-18/pytorch/default/1/resnet18-f37072fd.pth'
# Alternative common names:
# '/kaggle/input/resnet18-weights/resnet18.pth'
# '/kaggle/input/resnet18-imagenet/resnet18-19c8e357.pth'

print(f"Loading weights from: {WEIGHTS_FILE}")

# 3. Load the state dictionary
try:
    checkpoint = torch.load(WEIGHTS_FILE, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # Remove 'module.' prefix if exists (from DataParallel)
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    # Load weights (strict=False to ignore mismatched fc layer)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    print("✓ resnet18 weights loaded successfully from local file.")
    if missing_keys:
        print(f"  Missing keys (expected for fc layer): {missing_keys}")
    if unexpected_keys:
        print(f"  Unexpected keys: {unexpected_keys}")
    
except Exception as e:
    print(f"ERROR loading weights: {e}")
    print("\nTrying alternative: loading with weights_only=True")
    try:
        state_dict = torch.load(WEIGHTS_FILE, map_location=device, weights_only=True)
        model.load_state_dict(state_dict, strict=False)
        print("✓ Weights loaded successfully with weights_only=True")
    except Exception as e2:
        print(f"FATAL ERROR: Could not load weights - {e2}")
        print("\nIf this fails, you can use PyTorch's built-in weights:")
        print("  model = models.resnet18(weights='IMAGENET1K_V1')")
        raise

# Freeze all parameters initially
for param in model.parameters():
    param.requires_grad = False

# Replace the final fully connected layer with a custom classifier
num_ftrs = model.fc.in_features  # resnet18: 2048 features

model.fc = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(num_ftrs, 512),
    nn.ReLU(),
    nn.BatchNorm1d(512),
    nn.Dropout(p=0.4),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.BatchNorm1d(256),
    nn.Dropout(p=0.3),
    nn.Linear(256, 4)
)

model = model.to(device)
print(f"\nModel: resnet18 with Custom Classifier")
print(f"Initial trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# ===========================
# PROGRESSIVE UNFREEZING HELPER
# ===========================
def unfreeze_resnet_layers(model, phase):
    """
    Progressively unfreeze ResNet layers
    resnet18 structure: conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4, fc
    """
    # Always freeze everything first except fc
    for name, param in model.named_parameters():
        if 'fc' not in name:
            param.requires_grad = False
    
    if phase == 1:
        # Phase 1: Only fc (already unfrozen by default)
        pass
    elif phase == 2:
        # Phase 2: Unfreeze layer4 (last residual block)
        for param in model.layer4.parameters():
            param.requires_grad = True
    elif phase == 3:
        # Phase 3: Unfreeze layer3 and layer4
        for param in model.layer3.parameters():
            param.requires_grad = True
        for param in model.layer4.parameters():
            param.requires_grad = True
    elif phase == 4:
        # Phase 4: Unfreeze all layers
        for param in model.parameters():
            param.requires_grad = True
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable

# ===========================
# TRAINING CONFIGURATION
# ===========================
criterion = nn.CrossEntropyLoss()

# Progressive unfreezing strategy (4 phases)
training_phases = [
    {'name': 'Phase 1: Train FC Only', 'phase': 1, 'epochs': 5, 'lr_backbone': 0, 'lr_fc': 1e-3},
    {'name': 'Phase 2: Unfreeze Layer4', 'phase': 2, 'epochs': 5, 'lr_backbone': 5e-5, 'lr_fc': 5e-4},
    {'name': 'Phase 3: Unfreeze Layer3-4', 'phase': 3, 'epochs': 5, 'lr_backbone': 3e-5, 'lr_fc': 3e-4},
    {'name': 'Phase 4: Full Fine-tuning', 'phase': 4, 'epochs': 15, 'lr_backbone': 1e-5, 'lr_fc': 1e-4},
]

train_losses, train_accuracies = [], []
val_losses, val_accuracies = [], []
best_val_acc = 0
best_model_state = None

# ===========================
# TRAINING LOOP
# ===========================
print("\n" + "="*70)
print("STARTING PROGRESSIVE TRAINING")
print("="*70)

start_time = time.time()
total_epochs = sum(phase['epochs'] for phase in training_phases)
epoch_counter = 0

for phase_config in training_phases:
    print("\n" + "="*70)
    print(f"{phase_config['name']}")
    print("="*70)
    
    # Unfreeze appropriate layers
    trainable_params = unfreeze_resnet_layers(model, phase_config['phase'])
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Setup optimizer with discriminative learning rates
    if phase_config['phase'] == 1:
        # Only FC layer
        optimizer = optim.Adam(model.fc.parameters(), lr=phase_config['lr_fc'])
    else:
        # Discriminative learning rates for backbone and FC
        optimizer = optim.Adam([
            {'params': [p for n, p in model.named_parameters() if 'fc' not in n and p.requires_grad], 
             'lr': phase_config['lr_backbone']},
            {'params': model.fc.parameters(), 'lr': phase_config['lr_fc']}
        ])
    
    # Cosine annealing scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=phase_config['epochs'], eta_min=1e-6)
    
    for epoch in range(phase_config['epochs']):
        epoch_counter += 1
        
        # --- TRAINING ---
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch_counter}/{total_epochs} [Train]")
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
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch_counter}/{total_epochs} [Val]")
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

        print(f"Epoch {epoch_counter}/{total_epochs} -> Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

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
print("FINAL RESULTS - resnet18 TRANSFER LEARNING")
print("="*70)
print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
print(f"Final Test Accuracy: {test_acc:.2f}%")
print(f"Your Custom Model Baseline: 97.10%")
print(f"Difference: {test_acc - 97.10:.2f}%")
print(f"Total Training Time: {(end_time - start_time) / 60:.2f} minutes")

if test_acc >= 97.0:
    print("\n✓ SUCCESS! resnet18 matches or beats your custom model!")
elif test_acc >= 96.0:
    print("\n⚠ Close! resnet18 is competitive but slightly below your custom model.")
else:
    print("\n⚠ resnet18 with ImageNet weights still underperforms your custom architecture.")

# ===========================
# VISUALIZATION
# ===========================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
epochs_range = range(1, len(train_accuracies) + 1)

# Mark phase transitions
phase_transitions = [0]
for phase in training_phases:
    phase_transitions.append(phase_transitions[-1] + phase['epochs'])

# Accuracy Plot
axes[0].plot(epochs_range, train_accuracies, label='Train Accuracy', marker='o', markersize=3)
axes[0].plot(epochs_range, val_accuracies, label='Val Accuracy', marker='s', markersize=3)
for i, trans in enumerate(phase_transitions[1:-1], 1):
    axes[0].axvline(x=trans, color='grey', linestyle='--', alpha=0.5)
axes[0].axhline(y=test_acc, color='red', linestyle='--', label=f'Test Acc: {test_acc:.2f}%')
axes[0].axhline(y=97.10, color='green', linestyle='--', label='Your Model: 97.10%', alpha=0.7)
axes[0].set_title("resnet18 - Accuracy vs. Epochs", fontweight='bold')
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy (%)")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Loss Plot
axes[1].plot(epochs_range, train_losses, label='Train Loss', marker='o', markersize=3)
axes[1].plot(epochs_range, val_losses, label='Val Loss', marker='s', markersize=3)
for i, trans in enumerate(phase_transitions[1:-1], 1):
    axes[1].axvline(x=trans, color='grey', linestyle='--', alpha=0.5)
axes[1].set_title("resnet18 - Loss vs. Epochs", fontweight='bold')
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('resnet18_transfer_learning_results.png', dpi=300)
plt.show()

results = {
    'model_name': 'resnet18 Transfer Learning',
    'best_val_acc': best_val_acc,
    'final_test_acc': test_acc,
    'baseline_acc': 97.10,
    'train_accuracies': train_accuracies,
    'val_accuracies': val_accuracies,
    'train_losses': train_losses,
    'val_losses': val_losses,
    'total_epochs': len(train_accuracies),
    'training_phases': training_phases
}

with open('resnet18_transfer_learning_results.json', 'w') as f:
    json.dump(results, f, indent=4)

print("\n✓ Results saved to 'resnet18_transfer_learning_results.json'")
print("✓ Plot saved as 'resnet18_transfer_learning_results.png'")