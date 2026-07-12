"""
MobileNetV3-Small transfer learning trainer.

Uses pretrained ImageNet weights with a fine-tuned classifier head.
Same data pipeline, video preprocessing, and output format as sorter.py.

Model size: ~9 MB (vs ~400 KB for the custom CNN)
Expected accuracy: ~90-95% even with small datasets thanks to transfer learning.
"""

from matplotlib import pyplot as plt
import torch, torchvision
from torchvision import transforms
import os
import json
import datetime
from preprocess_videos import preprocess_labeled_folder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {device} for computation")
if device.type == 'cuda':
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

# MobileNetV3 expects 224x224 input (ImageNet standard)
IMG_SIZE = 224

# ImageNet normalization (required for pretrained weights)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Validation/Test transform: deterministic resize + center crop
transform_val = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# Training transform: heavier augmentation to combat overfitting on small dataset
transform_train = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


class SubsetWrapper(torch.utils.data.Dataset):
    """Wraps a Subset so we can apply a different transform than the parent dataset."""
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


def build_mobilenet_v3(num_classes, freeze_backbone=True):
    """
    Build MobileNetV3-Small with pretrained ImageNet weights.

    Strategy:
      Phase 1 (freeze_backbone=True): Freeze all backbone layers, only train
              the classifier head. Fast convergence, prevents catastrophic
              forgetting of pretrained features.
      Phase 2 (freeze_backbone=False): Unfreeze everything for fine-tuning
              with a lower learning rate.
    """
    model = torchvision.models.mobilenet_v3_small(
        weights=torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
    )

    # Freeze backbone if requested
    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    # Replace the classifier head
    # Original: Linear(576, 1024) -> Hardswish -> Dropout(0.2) -> Linear(1024, 1000)
    in_features = model.classifier[0].in_features  # 576
    model.classifier = torch.nn.Sequential(
        torch.nn.Linear(in_features, 256),
        torch.nn.Hardswish(),
        torch.nn.Dropout(0.3),
        torch.nn.Linear(256, num_classes),
    )

    return model


if __name__ == '__main__':
    # Preprocess any videos in the labeled folder first
    preprocess_labeled_folder('labeled')

    EXCLUDED_CLASSES = {'Misc', 'Unsortable'}

    def is_image_file(filename):
        return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))

    def make_dataset_filtered(root, transform):
        full_ds = torchvision.datasets.ImageFolder(
            root=root, transform=transform,
            is_valid_file=is_image_file, allow_empty=True
        )
        kept_classes = [c for c in full_ds.classes if c not in EXCLUDED_CLASSES]
        old_to_new = {full_ds.class_to_idx[c]: i for i, c in enumerate(kept_classes)}
        filtered_samples = [
            (path, old_to_new[old_idx])
            for path, old_idx in full_ds.samples
            if full_ds.classes[old_idx] not in EXCLUDED_CLASSES
        ]
        full_ds.samples = filtered_samples
        full_ds.targets = [s[1] for s in filtered_samples]
        full_ds.classes = kept_classes
        full_ds.class_to_idx = {c: i for i, c in enumerate(kept_classes)}
        return full_ds

    # Load dataset
    full_train_dataset = make_dataset_filtered('labeled', transform_train)
    full_val_dataset   = make_dataset_filtered('labeled', transform_val)

    num_classes = len(full_train_dataset.classes)
    print(f"Classes: {full_train_dataset.classes}")
    print(f"Total images: {len(full_train_dataset)}")

    # 70/15/15 split with deterministic seed
    train_size = int(len(full_train_dataset) * 0.70)
    val_size = int(len(full_train_dataset) * 0.15)
    test_size = len(full_train_dataset) - train_size - val_size

    generator = torch.Generator().manual_seed(42)
    train_set, val_set, test_set = torch.utils.data.random_split(
        full_train_dataset, [train_size, val_size, test_size], generator=generator
    )

    # Apply val transform to val and test sets
    val_set = SubsetWrapper(val_set, transform_val)
    test_set = SubsetWrapper(test_set, transform_val)

    print(f"Split: {len(train_set)} train / {len(val_set)} val / {len(test_set)} test")

    # DataLoader config
    batch = 64
    num_workers = min(8, os.cpu_count() or 1)
    pin_mem = (device.type == 'cuda')

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch, shuffle=True, drop_last=True,
        num_workers=num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=4,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch, shuffle=False, drop_last=False,
        num_workers=num_workers, pin_memory=pin_mem, persistent_workers=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch, shuffle=False, drop_last=False,
        num_workers=num_workers, pin_memory=pin_mem, persistent_workers=True,
    )

    # =========================================================================
    # Phase 1: Train classifier head only (backbone frozen)
    # =========================================================================
    print("\n" + "=" * 60)
    print("PHASE 1: Training classifier head (backbone frozen)")
    print("=" * 60)

    net = build_mobilenet_v3(num_classes, freeze_backbone=True).to(device)

    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    total = sum(p.numel() for p in net.parameters())
    print(f"Total parameters: {total:,} (~{total * 4 / 1024 / 1024:.1f} MB)")
    print(f"Trainable parameters (Phase 1): {trainable:,}")

    loss_fn = torch.nn.CrossEntropyLoss()
    # Higher LR is fine since we're only training the head
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, net.parameters()),
        lr=1e-3, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 25
    epochs_no_improve = 0
    best_model_state = None

    PHASE1_EPOCHS = 50

    for epoch in range(PHASE1_EPOCHS):
        # --- Training ---
        net.train()
        epoch_train_loss = 0.0
        train_batches = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

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
        correct = 0
        total_val = 0
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
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_losses.append(epoch_val_loss / max(val_batches, 1))
        val_acc = 100 * correct / max(total_val, 1)

        current_val_loss = val_losses[-1]
        print(f'  Phase 1 Epoch {epoch+1:3d}/{PHASE1_EPOCHS} | '
              f'Train Loss: {train_losses[-1]:.4f} | '
              f'Val Loss: {current_val_loss:.4f} | '
              f'Val Acc: {val_acc:.1f}%')

        scheduler.step(current_val_loss)

        # Early stopping
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            epochs_no_improve = 0
            best_model_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  Early stopping at epoch {epoch+1}.")
                if best_model_state is not None:
                    net.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
                break

    # Restore best Phase 1 weights before starting Phase 2
    if best_model_state is not None:
        net.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})

    # =========================================================================
    # Phase 2: Fine-tune entire network (backbone unfrozen, low LR)
    # =========================================================================
    print("\n" + "=" * 60)
    print("PHASE 2: Fine-tuning entire network (backbone unfrozen)")
    print("=" * 60)

    # Unfreeze all layers
    for param in net.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Trainable parameters (Phase 2): {trainable:,}")

    # Much lower LR to avoid destroying pretrained features
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Reset early stopping for phase 2
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    PHASE2_EPOCHS = 100

    for epoch in range(PHASE2_EPOCHS):
        # --- Training ---
        net.train()
        epoch_train_loss = 0.0
        train_batches = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

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
        correct = 0
        total_val = 0
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
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_losses.append(epoch_val_loss / max(val_batches, 1))
        val_acc = 100 * correct / max(total_val, 1)

        current_val_loss = val_losses[-1]
        print(f'  Phase 2 Epoch {epoch+1:3d}/{PHASE2_EPOCHS} | '
              f'Train Loss: {train_losses[-1]:.4f} | '
              f'Val Loss: {current_val_loss:.4f} | '
              f'Val Acc: {val_acc:.1f}%')

        scheduler.step(current_val_loss)

        # Early stopping
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            epochs_no_improve = 0
            best_model_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  Early stopping at epoch {epoch+1}.")
                if best_model_state is not None:
                    net.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
                break

    # Restore best weights
    if best_model_state is not None:
        net.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})

    # =========================================================================
    # Final evaluation on test set
    # =========================================================================
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)

    net.eval()
    correct = 0
    total_test = 0
    # Per-class tracking
    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    outputs = net(images)
            else:
                outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            total_test += labels.size(0)
            correct += (predicted == labels).sum().item()
            for i in range(labels.size(0)):
                label = labels[i].item()
                class_correct[label] += (predicted[i] == labels[i]).item()
                class_total[label] += 1

    test_accuracy = 100 * correct / max(total_test, 1)
    print(f'\nOverall test accuracy: {test_accuracy:.1f}%')
    print(f'\nPer-class accuracy:')
    for i, class_name in enumerate(full_train_dataset.classes):
        if class_total[i] > 0:
            acc = 100 * class_correct[i] / class_total[i]
            print(f'  {class_name:>10s}: {acc:5.1f}% ({class_correct[i]}/{class_total[i]})')
        else:
            print(f'  {class_name:>10s}: N/A (0 test samples)')

    # Save training history
    history_log = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'test_accuracy': test_accuracy,
        'classes': full_train_dataset.classes,
        'per_class_accuracy': {
            full_train_dataset.classes[i]: (
                100 * class_correct[i] / class_total[i] if class_total[i] > 0 else None
            )
            for i in range(num_classes)
        },
        'class_counts': {cls: int(sum(1 for _, l in full_train_dataset.samples if l == i))
                         for i, cls in enumerate(full_train_dataset.classes)},
        'hyperparameters': {
            'model':               'MobileNetV3-Small (transfer learning)',
            'total_parameters':    sum(p.numel() for p in net.parameters()),
            'img_size':            IMG_SIZE,
            'batch_size':          batch,
            'phase1_max_epochs':   PHASE1_EPOCHS,
            'phase2_max_epochs':   PHASE2_EPOCHS,
            'epochs_trained':      len(train_losses),
            'early_stop_patience': patience,
            'phase1_lr':           1e-3,
            'phase2_lr':           1e-4,
            'lr_final':            optimizer.param_groups[0]['lr'],
            'lr_scheduler':        'ReduceLROnPlateau(factor=0.5, patience=5)',
            'weight_decay':        1e-4,
            'dropout':             0.3,
            'optimizer':           'Adam',
            'loss':                'CrossEntropyLoss (uniform)',
            'train_split':         0.70,
            'val_split':           0.15,
            'test_split':          0.15,
            'freeze_backbone_phase1': True,
            'augmentation': {
                'random_resized_crop_scale': (0.7, 1.0),
                'horizontal_flip':           True,
                'rotation_degrees':          20,
                'color_jitter':              {'brightness': 0.3, 'contrast': 0.3, 'saturation': 0.3, 'hue': 0.15},
                'random_grayscale_p':        0.05,
            },
        },
    }
    dt_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    score_rounded = round(test_accuracy)
    history_filename = f"{score_rounded}_{dt_str}_mobilenet_training_history.json"
    with open(history_filename, 'w') as f:
        json.dump(history_log, f, indent=4)
    print(f'\nTraining history saved to {history_filename}')

    # Save model for inference
    model_filename = f"{score_rounded}_{dt_str}_mobilenet_model.pth"
    torch.save({
        'model_state_dict': net.state_dict(),
        'classes': full_train_dataset.classes,
        'img_size': IMG_SIZE,
        'model_type': 'mobilenet_v3_small',
    }, model_filename)
    print(f'Model saved to {model_filename}')

    # Plot training curves
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', alpha=0.8)
    plt.plot(val_losses, label='Val Loss', alpha=0.8)
    # Mark phase boundary
    plt.axvline(x=len(train_losses) - len(val_losses) + PHASE1_EPOCHS - 1,
                color='gray', linestyle='--', alpha=0.5, label='Phase 1→2')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('MobileNetV3 Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    class_names = full_train_dataset.classes
    class_accs = [
        100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
        for i in range(num_classes)
    ]
    bars = plt.bar(class_names, class_accs, color=['#4285F4', '#EA4335', '#FBBC04', '#34A853'])
    plt.ylabel('Accuracy (%)')
    plt.title(f'Per-Class Test Accuracy (Overall: {test_accuracy:.1f}%)')
    plt.ylim(0, 105)
    for bar, acc in zip(bars, class_accs):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f'{acc:.0f}%', ha='center', va='bottom', fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('mobilenet_training_plot.png', dpi=150)
    print('Training plot saved to mobilenet_training_plot.png')
    plt.show()
