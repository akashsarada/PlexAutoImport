from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import os
import json
import numpy as np
from preprocess_videos import preprocess_labeled_folder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {device} for computation")
if device.type == 'cuda':
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

# Image size (128x128 keeps compute load low on NAS)
IMG_SIZE = 128

# Classes to EXCLUDE from training
EXCLUDED_CLASSES = {'Misc', 'Unsortable'}

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
    transforms.RandomVerticalFlip(p=0.1),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ── SubsetWrapper ─────────────────────────────────────────────────────────────
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


# ── MobileNetV3 Core Modules ──────────────────────────────────────────────────
class SqueezeExcitation(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        reduced_channels = max(1, channels // reduction)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, reduced_channels, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, channels, 1, bias=True),
            nn.Hardsigmoid()
        )
    def forward(self, x):
        return x * self.se(x)


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, exp_channels, out_channels, kernel_size, stride, use_se, act_layer):
        super().__init__()
        self.use_res_connect = stride == 1 and in_channels == out_channels
        
        layers = []
        # Expand 1x1
        if exp_channels != in_channels:
            layers.append(nn.Conv2d(in_channels, exp_channels, 1, stride=1, padding=0, bias=False))
            layers.append(nn.BatchNorm2d(exp_channels))
            layers.append(act_layer())
        
        # Depthwise
        padding = (kernel_size - 1) // 2
        layers.append(nn.Conv2d(exp_channels, exp_channels, kernel_size, stride, padding, groups=exp_channels, bias=False))
        layers.append(nn.BatchNorm2d(exp_channels))
        layers.append(act_layer())
        
        # Squeeze-and-Excitation
        if use_se:
            layers.append(SqueezeExcitation(exp_channels))
            
        # Project 1x1
        layers.append(nn.Conv2d(exp_channels, out_channels, 1, stride=1, padding=0, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        
        self.conv = nn.Sequential(*layers)
        
    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV3Mini(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        # Stem layer
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.Hardswish()
        )
        
        # Core blocks (MobileNetV3 style: kernel, expansion, output, SE, activation, stride)
        self.bneck = nn.Sequential(
            InvertedResidual(16, 16, 16, kernel_size=3, stride=1, use_se=True, act_layer=nn.ReLU),
            InvertedResidual(16, 48, 24, kernel_size=3, stride=2, use_se=False, act_layer=nn.ReLU),
            InvertedResidual(24, 72, 24, kernel_size=3, stride=1, use_se=False, act_layer=nn.ReLU),
            InvertedResidual(24, 72, 40, kernel_size=5, stride=2, use_se=True, act_layer=nn.Hardswish),
            InvertedResidual(40, 120, 40, kernel_size=5, stride=1, use_se=True, act_layer=nn.Hardswish),
            InvertedResidual(40, 120, 80, kernel_size=3, stride=2, use_se=False, act_layer=nn.Hardswish),
            InvertedResidual(80, 240, 80, kernel_size=3, stride=1, use_se=False, act_layer=nn.Hardswish),
        )
        
        # Last conv stage
        self.conv_last = nn.Sequential(
            nn.Conv2d(80, 240, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(240),
            nn.Hardswish()
        )
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(240, 120),
            nn.Hardswish(),
            nn.Dropout(0.3),
            nn.Linear(120, num_classes)
        )
        
    def forward(self, x):
        x = self.stem(x)
        x = self.bneck(x)
        x = self.conv_last(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    # ── Step 0: Extract frames from any labeled videos ────────────────────────
    preprocess_labeled_folder('labeled')

    def is_image_file(filename):
        return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))

    # ── Step 1: Load dataset, excluding Misc/Unsortable ───────────────────────
    def make_dataset_filtered(root, transform):
        full_ds = torchvision.datasets.ImageFolder(
            root=root, transform=transform,
            is_valid_file=is_image_file, allow_empty=True
        )
        kept_classes = [c for c in full_ds.classes if c not in EXCLUDED_CLASSES]
        old_to_new = {
            full_ds.class_to_idx[c]: new_idx
            for new_idx, c in enumerate(kept_classes)
        }
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

    full_train_dataset = make_dataset_filtered('labeled', transform_train)
    full_val_dataset   = make_dataset_filtered('labeled', transform_val)

    num_classes = len(full_train_dataset.classes)
    print(f"\nClasses: {full_train_dataset.classes}")

    class_counts = np.zeros(num_classes, dtype=np.int64)
    for _, label in full_train_dataset.samples:
        class_counts[label] += 1

    print(f"Per-class image counts:")
    for cls, cnt in zip(full_train_dataset.classes, class_counts):
        print(f"  {cls:>12s}: {cnt}")
    print(f"Total images: {sum(class_counts)}")

    # ── Step 2: Train / Val / Test split ─────────────────────────────────────
    train_size = int(len(full_train_dataset) * 0.70)
    val_size   = int(len(full_train_dataset) * 0.15)
    test_size  = len(full_train_dataset) - train_size - val_size

    generator = torch.Generator().manual_seed(42)
    train_set, val_set, test_set = torch.utils.data.random_split(
        full_train_dataset, [train_size, val_size, test_size], generator=generator
    )

    val_set  = SubsetWrapper(val_set,  transform_val)
    test_set = SubsetWrapper(test_set, transform_val)

    print(f"\nSplit: {len(train_set)} train / {len(val_set)} val / {len(test_set)} test")

    # ── Step 3: WeightedRandomSampler to balance class frequency ─────────────
    sample_weights = np.array([
        1.0 / class_counts[full_train_dataset.samples[i][1]]
        for i in train_set.indices
    ], dtype=np.float32)
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights),
        num_samples=len(train_set),
        replacement=True,
    )

    # ── Step 4: Class-weighted CrossEntropyLoss ───────────────────────────────
    class_weights = torch.tensor(
        1.0 / (class_counts / class_counts.sum()),
        dtype=torch.float32
    )
    class_weights = class_weights / class_weights.sum() * num_classes
    class_weights = class_weights.to(device)
    print(f"\nClass loss weights: {dict(zip(full_train_dataset.classes, class_weights.cpu().tolist()))}")

    # ── Step 5: DataLoaders ───────────────────────────────────────────────────
    batch = 64
    num_workers_train = min(16, os.cpu_count() or 1) if device.type == 'cuda' else 0
    pin_mem = (device.type == 'cuda')

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch, sampler=sampler,
        drop_last=True, num_workers=num_workers_train, pin_memory=pin_mem
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch, shuffle=False,
        drop_last=False, num_workers=0, pin_memory=pin_mem
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch, shuffle=False,
        drop_last=False, num_workers=0, pin_memory=pin_mem
    )

    # ── Step 6: Model ─────────────────────────────────────────────────────────
    net = MobileNetV3Mini(num_classes=num_classes).to(device)

    total_params = sum(p.numel() for p in net.parameters())
    print(f"\nModel parameters: {total_params:,} (~{total_params * 4 / 1024:.1f} KB)")

    # ── Step 7: Training setup ────────────────────────────────────────────────
    loss_fn  = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    train_losses, val_losses = [], []
    best_val_loss    = float('inf')
    patience         = 15
    epochs_no_improve = 0
    best_model_state  = None

    # ── Step 8: Training loop ─────────────────────────────────────────────────
    for epoch in range(250):
        # --- Training ---
        net.train()
        epoch_train_loss = 0.0
        train_batches    = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    outputs = net(inputs)
                    loss    = loss_fn(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = net(inputs)
                loss    = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()

            epoch_train_loss += loss.item()
            train_batches    += 1
        train_losses.append(epoch_train_loss / max(train_batches, 1))

        # --- Validation ---
        net.eval()
        epoch_val_loss = 0.0
        val_batches    = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                if scaler is not None:
                    with torch.amp.autocast('cuda'):
                        outputs = net(inputs)
                        loss    = loss_fn(outputs, labels)
                else:
                    outputs = net(inputs)
                    loss    = loss_fn(outputs, labels)
                epoch_val_loss += loss.item()
                val_batches    += 1
        val_losses.append(epoch_val_loss / max(val_batches, 1))

        current_val_loss = val_losses[-1]
        lr_now = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:3d}/250 | Train Loss: {train_losses[-1]:.4f} | "
              f"Val Loss: {current_val_loss:.4f} | LR: {lr_now:.2e}")

        scheduler.step(current_val_loss)

        # Early stopping
        if current_val_loss < best_val_loss:
            best_val_loss      = current_val_loss
            epochs_no_improve  = 0
            best_model_state   = {k: v.cpu().clone() for k, v in net.state_dict().items()}
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}. "
                      f"Restoring best weights (epoch {epoch+1 - patience}).")
                if best_model_state is not None:
                    net.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
                break

    # ── Step 9: Final evaluation ──────────────────────────────────────────────
    net.eval()
    correct        = 0
    total          = 0
    per_class_correct = np.zeros(num_classes, dtype=np.int64)
    per_class_total   = np.zeros(num_classes, dtype=np.int64)

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    outputs = net(images)
            else:
                outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            total   += labels.size(0)
            correct += (predicted == labels).sum().item()
            for t, p in zip(labels.cpu().numpy(), predicted.cpu().numpy()):
                per_class_total[t]   += 1
                per_class_correct[t] += int(t == p)

    test_accuracy = 100 * correct / total
    print(f"\nOverall test accuracy : {test_accuracy:.1f}%")
    print(f"Per-class accuracy:")
    for cls, c, t in zip(full_train_dataset.classes, per_class_correct, per_class_total):
        pct = 100 * c / t if t > 0 else 0
        print(f"  {cls:>12s}: {pct:.1f}%  ({c}/{t})")

    # ── Step 10: Save artifacts ───────────────────────────────────────────────
    history_log = {
        'train_losses':      train_losses,
        'val_losses':        val_losses,
        'test_accuracy':     test_accuracy,
        'per_class_accuracy': {
            cls: float(100 * c / t) if t > 0 else 0
            for cls, c, t in zip(full_train_dataset.classes, per_class_correct, per_class_total)
        },
        'class_counts':      {cls: int(cnt) for cls, cnt in zip(full_train_dataset.classes, class_counts)},
        'classes':           full_train_dataset.classes,
        'model_parameters':  total_params,
    }
    with open('sorter_mini_history.json', 'w') as f:
        json.dump(history_log, f, indent=4)
    print('Training history saved to sorter_mini_history.json')

    torch.save({
        'model_state_dict': net.state_dict(),
        'classes':          full_train_dataset.classes,
        'img_size':         IMG_SIZE,
    }, 'sorter_mini_model.pth')
    print('Model saved to sorter_mini_model.pth')

    # ── Step 11: Plot ─────────────────────────────────────────────────────────
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses,   label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress (MobileNetV3-Mini)')
    plt.legend()
    plt.grid(True)
    plt.savefig('sorter_mini_loss_plot.png')
    print('Training loss curve saved to sorter_mini_loss_plot.png')
    plt.show()
