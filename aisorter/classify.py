"""
Inference & Sorting Pipeline
=============================
1. Asks which model to use (Custom CNN or MobileNetV3)
2. Loads the trained model
3. Preprocesses all videos in the dataset folder (extract frames for classification)
4. Classifies every image (and video frame) in dataset/
5. Moves originals into test/<ClassName>/ subdirectories
6. Cleans up temporary extracted frames
"""

import os
import sys
import shutil
import glob
import torch
import torchvision
from torchvision import transforms
from PIL import Image

# Try to import HEIC support — optional dependency
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HEIC_SUPPORTED = True
except ImportError:
    HEIC_SUPPORTED = False

from preprocess_videos import extract_frames_from_video

# ─── Constants ────────────────────────────────────────────────────────────────

VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v')
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
HEIC_EXTENSIONS = ('.heic', '.heif')
RAW_EXTENSIONS = ('.dng', '.cr2', '.nef', '.arw', '.orf', '.rw2')

# All extensions we can classify (images the model can read)
CLASSIFIABLE_EXTENSIONS = IMAGE_EXTENSIONS + (HEIC_EXTENSIONS if HEIC_SUPPORTED else ())

# All media we want to sort (including videos and RAW files we can't classify directly)
ALL_MEDIA_EXTENSIONS = IMAGE_EXTENSIONS + HEIC_EXTENSIONS + RAW_EXTENSIONS + VIDEO_EXTENSIONS

DATASET_DIR = 'dataset'
OUTPUT_DIR = 'test'

# ImageNet normalization (shared by both models)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ─── Model Loading ───────────────────────────────────────────────────────────

def load_custom_cnn(checkpoint_path, device):
    """Load the custom tiny CNN from sorter.py."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    classes = checkpoint['classes']
    img_size = checkpoint['img_size']
    num_classes = len(classes)

    net = torch.nn.Sequential(
        # Block 1
        torch.nn.Conv2d(3, 16, kernel_size=3, padding=1),
        torch.nn.BatchNorm2d(16),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2),
        # Block 2
        torch.nn.Conv2d(16, 32, kernel_size=3, padding=1),
        torch.nn.BatchNorm2d(32),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2),
        # Block 3
        torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
        torch.nn.BatchNorm2d(64),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2),
        # Block 4
        torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
        torch.nn.BatchNorm2d(128),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2),
        # Head
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
        torch.nn.Dropout(0.4),
        torch.nn.Linear(128, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, num_classes),
    )

    net.load_state_dict(checkpoint['model_state_dict'])
    net.to(device)
    net.eval()
    return net, classes, img_size



def load_mobilenet(checkpoint_path, device):
    """Load the MobileNetV3-Small model from sorter_mobilenet.py."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    classes = checkpoint['classes']
    img_size = checkpoint['img_size']
    num_classes = len(classes)

    model = torchvision.models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[0].in_features
    model.classifier = torch.nn.Sequential(
        torch.nn.Linear(in_features, 256),
        torch.nn.Hardswish(),
        torch.nn.Dropout(0.3),
        torch.nn.Linear(256, num_classes),
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model, classes, img_size


# ─── MobileNetV3 Mini Model Definition ────────────────────────────────────────

class SqueezeExcitation(torch.nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        reduced_channels = max(1, channels // reduction)
        self.se = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Conv2d(channels, reduced_channels, 1, bias=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(reduced_channels, channels, 1, bias=True),
            torch.nn.Hardsigmoid()
        )
    def forward(self, x):
        return x * self.se(x)


class InvertedResidual(torch.nn.Module):
    def __init__(self, in_channels, exp_channels, out_channels, kernel_size, stride, use_se, act_layer):
        super().__init__()
        self.use_res_connect = stride == 1 and in_channels == out_channels
        
        layers = []
        # Expand 1x1
        if exp_channels != in_channels:
            layers.append(torch.nn.Conv2d(in_channels, exp_channels, 1, stride=1, padding=0, bias=False))
            layers.append(torch.nn.BatchNorm2d(exp_channels))
            layers.append(act_layer())
        
        # Depthwise
        padding = (kernel_size - 1) // 2
        layers.append(torch.nn.Conv2d(exp_channels, exp_channels, kernel_size, stride, padding, groups=exp_channels, bias=False))
        layers.append(torch.nn.BatchNorm2d(exp_channels))
        layers.append(act_layer())
        
        # Squeeze-and-Excitation
        if use_se:
            layers.append(SqueezeExcitation(exp_channels))
            
        # Project 1x1
        layers.append(torch.nn.Conv2d(exp_channels, out_channels, 1, stride=1, padding=0, bias=False))
        layers.append(torch.nn.BatchNorm2d(out_channels))
        
        self.conv = torch.nn.Sequential(*layers)
        
    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV3Mini(torch.nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        # Stem layer
        self.stem = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(16),
            torch.nn.Hardswish()
        )
        
        # Core blocks (MobileNetV3 style: kernel, expansion, output, SE, activation, stride)
        self.bneck = torch.nn.Sequential(
            InvertedResidual(16, 16, 16, kernel_size=3, stride=1, use_se=True, act_layer=torch.nn.ReLU),
            InvertedResidual(16, 48, 24, kernel_size=3, stride=2, use_se=False, act_layer=torch.nn.ReLU),
            InvertedResidual(24, 72, 24, kernel_size=3, stride=1, use_se=False, act_layer=torch.nn.ReLU),
            InvertedResidual(24, 72, 40, kernel_size=5, stride=2, use_se=True, act_layer=torch.nn.Hardswish),
            InvertedResidual(40, 120, 40, kernel_size=5, stride=1, use_se=True, act_layer=torch.nn.Hardswish),
            InvertedResidual(40, 120, 80, kernel_size=3, stride=2, use_se=False, act_layer=torch.nn.Hardswish),
            InvertedResidual(80, 240, 80, kernel_size=3, stride=1, use_se=False, act_layer=torch.nn.Hardswish),
        )
        
        # Last conv stage
        self.conv_last = torch.nn.Sequential(
            torch.nn.Conv2d(80, 240, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(240),
            torch.nn.Hardswish()
        )
        
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        
        # Classifier
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(240, 120),
            torch.nn.Hardswish(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(120, num_classes)
        )
        
    def forward(self, x):
        x = self.stem(x)
        x = self.bneck(x)
        x = self.conv_last(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def load_mobilenet_mini(checkpoint_path, device):
    """Load the custom MobileNetV3-Mini model."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    classes = checkpoint['classes']
    img_size = checkpoint['img_size']
    num_classes = len(classes)
    
    model = MobileNetV3Mini(num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model, classes, img_size



# ─── Inference ────────────────────────────────────────────────────────────────

def get_transform(img_size, model_type):
    """Build the inference transform (matches the validation transform from training)."""
    if model_type == 'mobilenet':
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


def classify_image(image_path, model, transform, classes, device):
    """Classify a single image and return the predicted class name and confidence."""
    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"  ⚠ Could not open {os.path.basename(image_path)}: {e}")
        return None, 0.0

    input_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)

    return classes[predicted_idx.item()], confidence.item()


def classify_video_via_frames(video_path, model, transform, classes, device):
    """
    Classify a video by extracting temporary frames, classifying each,
    and returning the majority-vote class.
    Returns (predicted_class, avg_confidence, list_of_temp_frame_paths).
    """
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    temp_dir = os.path.join(DATASET_DIR, f'.tmp_frames_{base_name}')
    os.makedirs(temp_dir, exist_ok=True)

    frame_paths = extract_frames_from_video(video_path, temp_dir, base_name)

    if not frame_paths:
        shutil.rmtree(temp_dir, ignore_errors=True)
        return None, 0.0, []

    # Classify each frame
    votes = {}
    total_conf = {}
    for fp in frame_paths:
        cls, conf = classify_image(fp, model, transform, classes, device)
        if cls is not None:
            votes[cls] = votes.get(cls, 0) + 1
            total_conf[cls] = total_conf.get(cls, 0.0) + conf

    if not votes:
        shutil.rmtree(temp_dir, ignore_errors=True)
        return None, 0.0, []

    # Majority vote (tie-break by total confidence)
    best_class = max(votes, key=lambda c: (votes[c], total_conf[c]))
    avg_conf = total_conf[best_class] / votes[best_class]

    return best_class, avg_conf, temp_dir


# ─── Main Pipeline ───────────────────────────────────────────────────────────

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device} for inference")

    # ── Step 1: Ask which model to use ──────────────────────────────────────
    print("\n" + "=" * 50)
    print("  SELECT MODEL")
    print("=" * 50)

    # Detect available model files
    custom_available = os.path.exists('sorter_model.pth')
    mobilenet_available = os.path.exists('mobilenet_model.pth')
    mini_available = os.path.exists('sorter_mini_model.pth')

    if not custom_available and not mobilenet_available and not mini_available:
        print("\n✗ No trained models found!")
        print("  Run 'python sorter.py', 'python sorter_mobilenet.py' or 'python sorter_mini.py' first.")
        sys.exit(1)

    options = []
    if custom_available:
        options.append(('1', 'Custom CNN (sorter_model.pth)', 'custom'))
    if mobilenet_available:
        options.append(('2', 'MobileNetV3-Small (mobilenet_model.pth)', 'mobilenet'))
    if mini_available:
        options.append(('3', 'MobileNetV3-Mini (sorter_mini_model.pth)', 'mini'))

    for key, label, _ in options:
        print(f"  [{key}] {label}")

    while True:
        choice = input("\nEnter choice: ").strip()
        valid_keys = [o[0] for o in options]
        if choice in valid_keys:
            model_type = next(o[2] for o in options if o[0] == choice)
            break
        print(f"  Invalid choice. Enter one of: {', '.join(valid_keys)}")

    # ── Step 2: Load the model ──────────────────────────────────────────────
    print(f"\nLoading model...")
    if model_type == 'custom':
        model, classes, img_size = load_custom_cnn('sorter_model.pth', device)
    elif model_type == 'mini':
        model, classes, img_size = load_mobilenet_mini('sorter_mini_model.pth', device)
    else:
        model, classes, img_size = load_mobilenet('mobilenet_model.pth', device)

    print(f"  Model: {model_type}")
    print(f"  Classes: {classes}")
    print(f"  Input size: {img_size}x{img_size}")

    transform = get_transform(img_size, model_type)

    # ── Step 3: Create output directories ───────────────────────────────────
    for cls in classes:
        os.makedirs(os.path.join(OUTPUT_DIR, cls), exist_ok=True)
    # Folder for files we cannot classify (HEIC, RAW, unrecognized, or failed classification)
    os.makedirs(os.path.join(OUTPUT_DIR, 'Unsortable'), exist_ok=True)

    # ── Step 4: Scan dataset folder ─────────────────────────────────────────
    if not os.path.exists(DATASET_DIR):
        print(f"\n✗ '{DATASET_DIR}' directory not found!")
        sys.exit(1)

    all_files = [
        f for f in os.listdir(DATASET_DIR)
        if os.path.isfile(os.path.join(DATASET_DIR, f))
        and not f.startswith('.')
    ]

    # Categorize files
    images = [f for f in all_files if f.lower().endswith(CLASSIFIABLE_EXTENSIONS)]
    videos = [f for f in all_files if f.lower().endswith(VIDEO_EXTENSIONS)]
    heic_unsupported = (
        [f for f in all_files if f.lower().endswith(HEIC_EXTENSIONS)]
        if not HEIC_SUPPORTED else []
    )
    raw_files = [f for f in all_files if f.lower().endswith(RAW_EXTENSIONS)]
    unrecognized = [
        f for f in all_files
        if not f.lower().endswith(ALL_MEDIA_EXTENSIONS)
        and f not in ('Thumbs.db', 'desktop.ini', '.DS_Store')
    ]

    print(f"\n{'=' * 50}")
    print(f"  DATASET SUMMARY")
    print(f"{'=' * 50}")
    print(f"  Classifiable images : {len(images)}")
    print(f"  Videos              : {len(videos)}")
    if heic_unsupported:
        print(f"  HEIC (no support)   : {len(heic_unsupported)}")
    if raw_files:
        print(f"  RAW files           : {len(raw_files)}")
    if unrecognized:
        print(f"  Unrecognized        : {len(unrecognized)}")

    if heic_unsupported:
        print(f"\n  ⚠ {len(heic_unsupported)} HEIC files cannot be classified.")
        print(f"    Install 'pillow-heif' (pip install pillow-heif) for HEIC support.")
        print(f"    These files will be moved to '{OUTPUT_DIR}/Unsortable/'.")

    total = len(images) + len(videos) + len(heic_unsupported) + len(raw_files)
    if total == 0:
        print("\n  Nothing to sort!")
        sys.exit(0)

    print(f"\n  Total files to sort: {total}")
    input("\nPress Enter to start sorting...")

    # ── Step 5: Classify and sort images ────────────────────────────────────
    print(f"\n{'=' * 50}")
    print(f"  SORTING IMAGES")
    print(f"{'=' * 50}")

    sorted_count = 0
    failed_count = 0
    class_counts = {cls: 0 for cls in classes}

    for i, filename in enumerate(images, 1):
        filepath = os.path.join(DATASET_DIR, filename)
        predicted_class, confidence = classify_image(filepath, model, transform, classes, device)

        if predicted_class is not None:
            dest = os.path.join(OUTPUT_DIR, predicted_class, filename)
            # Handle filename collisions
            if os.path.exists(dest):
                base, ext = os.path.splitext(filename)
                counter = 1
                while os.path.exists(dest):
                    dest = os.path.join(OUTPUT_DIR, predicted_class, f"{base}_{counter}{ext}")
                    counter += 1
            shutil.move(filepath, dest)
            class_counts[predicted_class] = class_counts.get(predicted_class, 0) + 1
            sorted_count += 1
            print(f"  [{i}/{len(images)}] {filename} → {predicted_class}/ ({confidence:.0%})")
        else:
            # If we can't classify, dump to Unsortable
            dest = os.path.join(OUTPUT_DIR, 'Unsortable', filename)
            if os.path.exists(dest):
                base, ext = os.path.splitext(filename)
                counter = 1
                while os.path.exists(dest):
                    dest = os.path.join(OUTPUT_DIR, 'Unsortable', f"{base}_{counter}{ext}")
                    counter += 1
            shutil.move(filepath, dest)
            failed_count += 1
            print(f"  [{i}/{len(images)}] {filename} → Unsortable/ (failed to classify)")

    # ── Step 6: Classify and sort videos ────────────────────────────────────
    if videos:
        print(f"\n{'=' * 50}")
        print(f"  SORTING VIDEOS")
        print(f"{'=' * 50}")

        for i, filename in enumerate(videos, 1):
            filepath = os.path.join(DATASET_DIR, filename)
            predicted_class, confidence, temp_dir = classify_video_via_frames(
                filepath, model, transform, classes, device
            )

            # Clean up temporary frames
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)

            if predicted_class is not None:
                dest = os.path.join(OUTPUT_DIR, predicted_class, filename)
                if os.path.exists(dest):
                    base, ext = os.path.splitext(filename)
                    counter = 1
                    while os.path.exists(dest):
                        dest = os.path.join(OUTPUT_DIR, predicted_class, f"{base}_{counter}{ext}")
                        counter += 1
                shutil.move(filepath, dest)
                class_counts[predicted_class] = class_counts.get(predicted_class, 0) + 1
                sorted_count += 1
                print(f"  [{i}/{len(videos)}] {filename} → {predicted_class}/ ({confidence:.0%})")
            else:
                dest = os.path.join(OUTPUT_DIR, 'Unsortable', filename)
                if os.path.exists(dest):
                    base, ext = os.path.splitext(filename)
                    counter = 1
                    while os.path.exists(dest):
                        dest = os.path.join(OUTPUT_DIR, 'Unsortable', f"{base}_{counter}{ext}")
                        counter += 1
                shutil.move(filepath, dest)
                failed_count += 1
                print(f"  [{i}/{len(videos)}] {filename} → Unsortable/ (failed to classify)")

    # ── Step 7: Move unsupported HEIC files to Unsortable ──────────────────
    if heic_unsupported:
        print(f"\n{'=' * 50}")
        print(f"  MOVING UNSUPPORTED HEIC FILES → Unsortable/")
        print(f"{'=' * 50}")

        for filename in heic_unsupported:
            filepath = os.path.join(DATASET_DIR, filename)
            dest = os.path.join(OUTPUT_DIR, 'Unsortable', filename)
            if os.path.exists(dest):
                base, ext = os.path.splitext(filename)
                counter = 1
                while os.path.exists(dest):
                    dest = os.path.join(OUTPUT_DIR, 'Unsortable', f"{base}_{counter}{ext}")
                    counter += 1
            shutil.move(filepath, dest)
            print(f"  {filename} → Unsortable/")

    # ── Step 8: Move RAW files to Unsortable (can't classify) ────────────────
    if raw_files:
        print(f"\n{'=' * 50}")
        print(f"  MOVING RAW FILES → Unsortable/")
        print(f"{'=' * 50}")

        for filename in raw_files:
            filepath = os.path.join(DATASET_DIR, filename)
            dest = os.path.join(OUTPUT_DIR, 'Unsortable', filename)
            if os.path.exists(dest):
                base, ext = os.path.splitext(filename)
                counter = 1
                while os.path.exists(dest):
                    dest = os.path.join(OUTPUT_DIR, 'Unsortable', f"{base}_{counter}{ext}")
                    counter += 1
            shutil.move(filepath, dest)
            print(f"  {filename} → Unsortable/")
    # ── Step 9: Move unrecognized files to Unsortable ───────────────────────
    if unrecognized:
        print(f"\n{'=' * 50}")
        print(f"  MOVING UNRECOGNIZED FILES → Unsortable/")
        print(f"{'=' * 50}")
        for filename in unrecognized:
            filepath = os.path.join(DATASET_DIR, filename)
            dest = os.path.join(OUTPUT_DIR, 'Unsortable', filename)
            if os.path.exists(dest):
                base, ext = os.path.splitext(filename)
                counter = 1
                while os.path.exists(dest):
                    dest = os.path.join(OUTPUT_DIR, 'Unsortable', f"{base}_{counter}{ext}")
                    counter += 1
            shutil.move(filepath, dest)
            print(f"  {filename} → Unsortable/")
    # ── Step 9: Final cleanup ───────────────────────────────────────────────
    # Remove any leftover temp frame directories
    for item in os.listdir(DATASET_DIR):
        item_path = os.path.join(DATASET_DIR, item)
        if os.path.isdir(item_path) and item.startswith('.tmp_frames_'):
            shutil.rmtree(item_path, ignore_errors=True)

    # ── Summary ─────────────────────────────────────────────────────────────
    print(f"\n{'=' * 50}")
    print(f"  SORTING COMPLETE")
    print(f"{'=' * 50}")
    print(f"  Successfully sorted : {sorted_count}")
    print(f"  Failed / Unsortable : {failed_count + len(heic_unsupported) + len(raw_files) + len(unrecognized)}")
    print(f"\n  Per-class breakdown:")
    for cls in classes:
        count = class_counts.get(cls, 0)
        print(f"    {cls:>10s}: {count}")
    print(f"\n  Output directory: {os.path.abspath(OUTPUT_DIR)}/")


if __name__ == '__main__':
    main()
