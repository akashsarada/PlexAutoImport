from matplotlib import pyplot as plt
import torch, torchvision
from torchvision import transforms
import os
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {device} for computation")
if device.type == 'cuda':
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

# Image size (128x128 keeps compute load very low on NAS while retaining details)
IMG_SIZE = 128

# Validation/Test transform: deterministic center cropping
transform_val = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Training transform: data augmentation to combat overfitting on small dataset
transform_train = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.RandomCrop(IMG_SIZE, padding=8, padding_mode='reflect'),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Map val_set and test_set to use the validation transform (no augmentation)
class SubsetWrapper(torch.utils.data.Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
    def __getitem__(self, index):
        path, label = self.subset.dataset.samples[self.subset.indices[index]]
        img = self.subset.dataset.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, label
    def __len__(self):
        return len(self.subset)

if __name__ == '__main__':
    # Preprocess any videos in the labeled folder first
    from preprocess_videos import preprocess_labeled_folder
    preprocess_labeled_folder('labeled')

    def is_image_file(filename):
        return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))

    # Load dataset twice to apply different transforms to train and val/test sets
    full_train_dataset = torchvision.datasets.ImageFolder(
        root='labeled', transform=transform_train, is_valid_file=is_image_file
    )
    full_val_dataset = torchvision.datasets.ImageFolder(
        root='labeled', transform=transform_val, is_valid_file=is_image_file
    )
    
    num_classes = len(full_train_dataset.classes)
    print(f"Classes: {full_train_dataset.classes}")
    print(f"Total images: {len(full_train_dataset)}")

    # Deterministic subset split indices
    train_size = int(len(full_train_dataset) * 0.70)
    val_size = int(len(full_train_dataset) * 0.15)
    test_size = len(full_train_dataset) - train_size - val_size
    
    # Use random_split with generator for reproducibility
    generator = torch.Generator().manual_seed(42)
    train_set, val_set, test_set = torch.utils.data.random_split(
        full_train_dataset, [train_size, val_size, test_size], generator=generator
    )
    
    val_set = SubsetWrapper(val_set, transform_val)
    test_set = SubsetWrapper(test_set, transform_val)

    print(f"Split: {len(train_set)} train / {len(val_set)} val / {len(test_set)} test")

    # Larger batch size to fully utilize GPU memory and parallel processing power
    batch = 64
    num_workers_train = min(4, os.cpu_count() or 1) if device.type == 'cuda' else 0
    pin_mem = (device.type == 'cuda')

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch, shuffle=True, drop_last=True, num_workers=num_workers_train, pin_memory=pin_mem)
    # val_loader and test_loader run in the main process (num_workers=0) to bypass pickling/multiprocessing errors with SubsetWrapper
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch, shuffle=False, drop_last=False, num_workers=0, pin_memory=pin_mem)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch, shuffle=False, drop_last=False, num_workers=0, pin_memory=pin_mem)

    # Lightweight 4-block CNN — Still tiny (~105K parameters, ~400 KB) suitable for NAS
    net = torch.nn.Sequential(
        # Block 1: 3 -> 16 channels, 128x128 -> 64x64
        torch.nn.Conv2d(3, 16, kernel_size=3, padding=1),
        torch.nn.BatchNorm2d(16),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2),

        # Block 2: 16 -> 32 channels, 64x64 -> 32x32
        torch.nn.Conv2d(16, 32, kernel_size=3, padding=1),
        torch.nn.BatchNorm2d(32),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2),

        # Block 3: 32 -> 64 channels, 32x32 -> 16x16
        torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
        torch.nn.BatchNorm2d(64),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2),

        # New Block 4: 64 -> 128 channels, 16x16 -> 8x8 (improves capacity and validation performance)
        torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
        torch.nn.BatchNorm2d(128),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2),

        # Global average pool collapses spatial dims -> 128-dim vector
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),

        # Classifier head (increased dropout to 0.4 to prevent overfitting)
        torch.nn.Dropout(0.4),
        torch.nn.Linear(128, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, num_classes),
    ).to(device)

    total_params = sum(p.numel() for p in net.parameters())
    print(f"Model parameters: {total_params:,} (~{total_params * 4 / 1024:.1f} KB)")

    loss_fn = torch.nn.CrossEntropyLoss()
    # Added weight_decay=1e-4 (L2 regularization) to prevent large weights
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    # Learning rate scheduler to reduce LR when validation loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    # Scaler for AMP (mixed precision) to speed up training on compatible GPUs
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    train_losses = []
    val_losses = []

    # Early stopping parameters
    best_val_loss = float('inf')
    patience = 15  # Increased patience slightly to allow learning rate decay to take effect
    epochs_no_improve = 0
    best_model_state = None

    for epoch in range(100):
        # --- Training ---
        net.train()
        epoch_train_loss = 0.0
        train_batches = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            # Use AMP mixed precision for faster training
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    outputs = net(inputs)
                    loss = loss_fn(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = net(inputs)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
                
            epoch_train_loss += loss.item()
            train_batches += 1
        train_losses.append(epoch_train_loss / max(train_batches, 1))

        # --- Validation ---
        net.eval()
        epoch_val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                if scaler is not None:
                    with torch.amp.autocast('cuda'):
                        outputs = net(inputs)
                        loss = loss_fn(outputs, labels)
                else:
                    outputs = net(inputs)
                    loss = loss_fn(outputs, labels)
                epoch_val_loss += loss.item()
                val_batches += 1
        val_losses.append(epoch_val_loss / max(val_batches, 1))

        current_val_loss = val_losses[-1]
        print(f'Epoch {epoch+1:3d}/100 | Train Loss: {train_losses[-1]:.4f} | Val Loss: {current_val_loss:.4f}')

        # Step the learning rate scheduler with the validation loss
        scheduler.step(current_val_loss)

        # Early Stopping check
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            epochs_no_improve = 0
            best_model_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}. Restoring best weights from epoch {epoch+1-patience}.")
                if best_model_state is not None:
                    net.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
                break

    # Final accuracy on test set
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    outputs = net(images)
            else:
                outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_accuracy = 100 * correct / total
    print(f'Accuracy on test images: {test_accuracy:.1f}%')

    # Save training history and metadata to a log file
    history_log = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'test_accuracy': test_accuracy,
        'classes': full_train_dataset.classes,
        'model_parameters': total_params
    }
    with open('training_history.json', 'w') as f:
        json.dump(history_log, f, indent=4)
    print('Training history saved to training_history.json')

    # Save model for inference on the NAS
    torch.save({
        'model_state_dict': net.state_dict(),
        'classes': full_train_dataset.classes,
        'img_size': IMG_SIZE,
    }, 'sorter_model.pth')
    print('Model saved to sorter_model.pth')

    # Plot and save curve to disk
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_loss_plot.png')
    print('Training loss curve saved to training_loss_plot.png')
    plt.show()
