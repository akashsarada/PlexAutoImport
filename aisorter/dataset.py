"""Multi-label image dataset for the photo AI sorter.

Labels (index → concept):
  0 → car
  1 → person
  2 → scenery

Folder names are matched by substring so any combination folder
(e.g. 'cars_and_people') is decoded automatically.
"""

import os
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

_LABEL_KEYWORDS: list[tuple[str, int]] = [
    ("cars",    0),
    ("people",  1),
    ("scenery", 2),
]

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


def _folder_to_label(folder_name: str) -> torch.Tensor:
    name = folder_name.lower()
    label = torch.zeros(3, dtype=torch.float32)
    for keyword, idx in _LABEL_KEYWORDS:
        if keyword in name:
            label[idx] = 1.0
    return label


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in _IMAGE_EXTENSIONS


class MultiLabelPhotoDataset(Dataset):
    """Folder-based multi-label image dataset.

    Each immediate subdirectory of *root_dir* is mapped to a 3-element
    multi-hot label vector via substring matching on the folder name.
    """

    def __init__(self, root_dir: str, transform: Optional[object] = None, in_memory: bool = False) -> None:
        self.root = Path(root_dir)
        self.transform = transform
        self.in_memory = in_memory
        self.samples: list[tuple[Path, torch.Tensor]] = []
        self.class_names = [keyword for keyword, _ in _LABEL_KEYWORDS]
        self.num_classes = len(self.class_names)

        temp_samples = []
        for folder in sorted(self.root.iterdir()):
            if not folder.is_dir():
                continue
            label = _folder_to_label(folder.name)
            for img_path in sorted(folder.iterdir()):
                if _is_image(img_path):
                    temp_samples.append((img_path, label))

        if self.in_memory:
            self.cached_images = []
            print(f"Pre-loading {len(temp_samples)} images into RAM (pre-resized to 128x128)...")
            from tqdm import tqdm
            for img_path, label in tqdm(temp_samples, desc="Caching to RAM"):
                try:
                    with Image.open(img_path) as img:
                        cached_img = img.convert("RGB").resize((128, 128))
                        # Load to force reading pixel data fully into memory
                        cached_img.load()
                        self.cached_images.append(cached_img)
                        self.samples.append((img_path, label))
                except Exception as e:
                    print(f"Warning: Failed to load {img_path}: {e}")
        else:
            self.samples = temp_samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.in_memory:
            image = self.cached_images[index]
            label = self.samples[index][1]
        else:
            img_path, label = self.samples[index]
            image = Image.open(img_path).convert("RGB")
            
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def build_transforms(train: bool) -> transforms.Compose:
    """Return torchvision transforms for training or validation."""
    normalize = transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD)
    if train:
        return transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.ToTensor(),
            normalize,
        ])
    return transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        normalize,
    ])
