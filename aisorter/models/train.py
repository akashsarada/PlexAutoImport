import argparse
import json
import os
import signal
import sys
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from aisorter.dataset import MultiLabelPhotoDataset, build_transforms
from aisorter.models.DSC import CategorySorter
from aisorter.models.MobileNetV3_Mini import MobileNetV3Mini
from aisorter.models.MobileNetV3_Small import MobileNetV3Small
from aisorter.models.small_CNN import CustomCNN
from constants import TRAINING_IMAGE_SIZE


def parse_args() -> argparse.Namespace:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description="Train CategorySorter CNN")
    parser.add_argument("--data", required=True, help="Path to dataset root")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(__file__), "..", "results", f"run_{timestamp}"),
    )
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--in-memory", action="store_true", help="Load and resize all images to RAM at startup")
    parser.add_argument(
        "--early-stopping-tolerance",
        type=int,
        default=50,
        help="Number of epochs without validation improvement before stopping (default: 50)",
    )
    parser.add_argument(
        "--model-type",
        choices=["category_sorter", "sorter_mini", "sorter_mobilenet", "sorter"],
        default="category_sorter",
        help="Model architecture: category_sorter | MobileNetV3_Mini | MobileNetV3_Small | CustomCNN",
    )
    return parser.parse_args()


def compute_val_accuracy(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> list[float]:
    model.eval()
    correct = torch.zeros(num_classes, device=device)
    total = torch.zeros(num_classes, device=device)
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            preds = torch.sigmoid(model(images)) >= 0.5
            correct += (preds == labels.bool()).float().sum(dim=0)
            total += labels.size(0)
    per_class = (correct / total.clamp(min=1)).cpu().tolist()
    return per_class


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    train: bool,
) -> float:
    model.train(train)
    total_loss = 0.0
    with torch.set_grad_enabled(train):
        for images, labels in loader:
            images = images.to(device)
            labels = labels.float().to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
    return total_loss / max(len(loader.dataset), 1)


def save_checkpoint(model: nn.Module, path: str, class_names: list[str], model_type: str) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": class_names,
            "num_classes": len(class_names),
            "img_size": TRAINING_IMAGE_SIZE,
            "model_type": model_type,
        },
        path,
    )


class _TransformSubset(torch.utils.data.Dataset):
    def __init__(self, subset: torch.utils.data.Subset, transform: object) -> None:
        self._subset = subset
        self._transform = transform

    def __len__(self) -> int:
        return len(self._subset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, label = self._subset[idx]
        if self._transform is not None:
            image = self._transform(image)
        return image, label


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    train_transforms = build_transforms(train=True)
    val_transforms = build_transforms(train=False)
    full_dataset = MultiLabelPhotoDataset(args.data, transform=None, in_memory=args.in_memory)

    num_classes = full_dataset.num_classes
    class_names = full_dataset.class_names

    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    generator = torch.Generator().manual_seed(42)
    train_subset, val_subset = random_split(
        full_dataset, [train_size, val_size], generator=generator
    )

    train_dataset = _TransformSubset(train_subset, train_transforms)
    val_dataset = _TransformSubset(val_subset, val_transforms)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    if args.model_type == "category_sorter":
        model = CategorySorter(num_classes=full_dataset.num_classes).to(device)
    elif args.model_type == "sorter_mini":
        model = MobileNetV3Mini(num_classes=full_dataset.num_classes).to(device)
    elif args.model_type == "sorter_mobilenet":
        model = MobileNetV3Small(num_classes=full_dataset.num_classes, pretrained=True).to(device)
    elif args.model_type == "sorter":
        model = CustomCNN(num_classes=full_dataset.num_classes).to(device)

    # Startup message
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print("\n" + "=" * 50)
    print("      PHOTO AI SORTER: TRAINING STARTUP")
    print("=" * 50)
    print(f"Model Architecture: {model.__class__.__name__}")
    print(f"Total Parameters:    {total_params:,}")
    print(f"Trainable Params:    {trainable_params:,}")
    print(f"Dataset Directory:   {args.data}")
    print(f"Training Samples:    {len(train_dataset)}")
    print(f"Validation Samples:  {len(val_dataset)}")
    print(f"Number of Classes:   {num_classes} ({', '.join(class_names)})")
    print(f"Batch Size:          {args.batch_size}")
    print(f"Initial LR:          {args.lr}")
    print(f"Target Device:       {device} ({device_name})")
    print("=" * 50 + "\n")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float("inf")
    best_val_acc = [0.0] * num_classes
    best_val_epoch = 0
    best_checkpoint_path = os.path.join(args.output_dir, "best_model.pth")
    history: dict = {"timestamp": [], "train_loss": [], "val_loss": [], "val_accuracy_per_class": []}

    def _save_and_exit(signum, frame) -> None:
        print("\nInterrupted — saving current model …")
        interrupted_path = os.path.join(args.output_dir, "interrupted_model.pth")
        save_checkpoint(model, interrupted_path, class_names, args.model_type)
        _flush_history()
        print(f"Saved to {interrupted_path}")
        sys.exit(0)

    def _flush_history() -> None:
        history_path = os.path.join(args.output_dir, "history.json")
        with open(history_path, "w") as fh:
            json.dump(history, fh, indent=2)

    signal.signal(signal.SIGINT, _save_and_exit)
    signal.signal(signal.SIGTERM, _save_and_exit)

    for epoch in range(1, args.epochs + 1):
        if (epoch - best_val_epoch) >= args.early_stopping_tolerance:
            print(f"\nEarly stopping triggered: No validation improvement for {args.early_stopping_tolerance} epochs.")
            break
        timestamp = int(datetime.now().timestamp() * 1000)
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        val_acc = compute_val_accuracy(model, val_loader, device, num_classes)

        scheduler.step()

        history["timestamp"].append(timestamp)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy_per_class"].append(val_acc)

        acc_str = "  ".join(f"{name}:{acc:.3f}" for name, acc in zip(class_names, val_acc))
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f} | {acc_str}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_val_epoch = epoch
            save_checkpoint(model, best_checkpoint_path, class_names, args.model_type)

    _flush_history()

    # Rename output directory to append the best val accuracy
    best_score_pct = round(sum(best_val_acc) / len(best_val_acc) * 100)
    final_output_dir = f"{args.output_dir}_{best_score_pct}"
    try:
        if os.path.exists(args.output_dir):
            os.rename(args.output_dir, final_output_dir)
            best_checkpoint_path = os.path.join(final_output_dir, "best_model.pth")
            history_json_path = os.path.join(final_output_dir, "history.json")
        else:
            history_json_path = os.path.join(args.output_dir, "history.json")
    except Exception as e:
        print(f"Warning: Could not rename directory to include score: {e}")
        history_json_path = os.path.join(args.output_dir, "history.json")

    print(f"\nBest model  → {best_checkpoint_path}")
    print(f"History     → {history_json_path}")


if __name__ == "__main__":
    main()
