"""
classify.py — Interactive image sorter
=======================================
1. Select model architecture
2. Select a .pth weights file
3. Select input directory
4. Sort images into class folders (skips files with confidence < 85%)
5. Display timing summary
6. Pause for review — press Enter to UNDO and move everything back
"""

import os
import sys
import shutil
import time
import glob

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import pillow_heif

pillow_heif.register_heif_opener()

from aisorter.models.DSC import CategorySorter
from aisorter.models.MobileNetV3_Mini import MobileNetV3Mini
from aisorter.models.MobileNetV3_Small import MobileNetV3Small
from aisorter.models.small_CNN import CustomCNN
from aisorter.preprocess_videos import extract_frames_from_video
from constants import (
    CLASSIFY_CONFIDENCE_THRESHOLD as CONFIDENCE_THRESHOLD,
    IMAGE_EXTENSIONS,
    IMAGENET_MEAN,
    IMAGENET_STD,
    MULTILABEL_MODEL_TYPES as MULTILABEL_TYPES,
    VIDEO_EXTENSIONS,
)

# ─── Architecture registry ────────────────────────────────────────────────────

ARCHITECTURES = [
    {
        "key":        "1",
        "label":      "CategorySorter — depthwise separable CNN, multi-label (DSC.py)",
        "model_type": "category_sorter",
        "cls":        CategorySorter,
        "img_size":   128,
    },
    {
        "key":        "2",
        "label":      "MobileNetV3-Mini — lightweight inverted residuals, single-label",
        "model_type": "sorter_mini",
        "cls":        MobileNetV3Mini,
        "img_size":   160,
    },
    {
        "key":        "3",
        "label":      "MobileNetV3-Small — pretrained transfer learning, single-label",
        "model_type": "sorter_mobilenet",
        "cls":        MobileNetV3Small,
        "img_size":   224,
    },
    {
        "key":        "4",
        "label":      "CustomCNN — small 4-block CNN, single-label",
        "model_type": "sorter",
        "cls":        CustomCNN,
        "img_size":   128,
    },
]

# ─── Helpers ──────────────────────────────────────────────────────────────────

def hr(char="─", width=56):
    print(char * width)


def header(title):
    hr()
    print(f"  {title}")
    hr()


def load_model(arch_info, pth_path, device):
    """Load a .pth checkpoint, auto-detecting architecture from saved metadata."""
    checkpoint = torch.load(pth_path, map_location=device, weights_only=True)

    # Prefer metadata in checkpoint, fall back to the user-selected architecture
    model_type  = checkpoint.get("model_type",  arch_info["model_type"])
    class_names = checkpoint.get("class_names", None)
    img_size    = checkpoint.get("img_size",    arch_info["img_size"])

    # Map model_type → class
    type_to_cls = {a["model_type"]: a["cls"] for a in ARCHITECTURES}
    ModelCls = type_to_cls.get(model_type, arch_info["cls"])

    num_classes = len(class_names) if class_names else 3

    if model_type == "sorter_mobilenet":
        model = ModelCls(num_classes=num_classes, pretrained=False)
    else:
        model = ModelCls(num_classes=num_classes)

    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    return model, class_names, img_size, model_type


def build_transform(img_size, model_type):
    """Deterministic validation transform matching the training pipeline."""
    if model_type == "sorter_mobilenet":
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def predict(image_path, model, transform, class_names, device, model_type):
    """
    Run inference on one image.

    Returns:
        For single-label: list of at most one (class_name, confidence) tuple.
        For multi-label:  list of (class_name, confidence) tuples for every
                          class whose sigmoid probability ≥ CONFIDENCE_THRESHOLD.
        Returns [] if nothing crosses the threshold.
    """
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"  ⚠  Cannot open {os.path.basename(image_path)}: {e}")
        return []

    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)

        if model_type in MULTILABEL_TYPES:
            probs = torch.sigmoid(logits)[0]
            results = [
                (class_names[i], float(probs[i]))
                for i in range(len(class_names))
                if float(probs[i]) >= CONFIDENCE_THRESHOLD
            ]
        else:
            probs = torch.softmax(logits, dim=1)[0]
            conf, idx = torch.max(probs, 0)
            conf = float(conf)
            if conf >= CONFIDENCE_THRESHOLD:
                results = [(class_names[idx.item()], conf)]
            else:
                results = []

    return results


def classify_video(video_path, model, transform, class_names, device, model_type):
    """
    Extract frames via preprocess_videos, classify each frame, and return
    the majority-vote result as a list of (class_name, avg_confidence) tuples.

    For single-label models: returns at most one tuple (the winning class).
    For multi-label models:  returns all classes that won a majority across frames.
    Returns [] if no frames could be extracted or nothing clears the threshold.
    """
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    tmp_dir   = os.path.join(os.path.dirname(video_path), f".tmp_frames_{base_name}")
    os.makedirs(tmp_dir, exist_ok=True)

    frame_paths = extract_frames_from_video(video_path, tmp_dir, base_name)

    if not frame_paths:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return []

    # Tally votes and cumulative confidence per class
    votes      = {cls: 0   for cls in class_names}
    total_conf = {cls: 0.0 for cls in class_names}

    for fp in frame_paths:
        hits = predict(fp, model, transform, class_names, device, model_type)
        for cls_name, conf in hits:
            votes[cls_name]      += 1
            total_conf[cls_name] += conf

    shutil.rmtree(tmp_dir, ignore_errors=True)

    n_frames = len(frame_paths)

    if model_type in MULTILABEL_TYPES:
        # Multi-label: keep any class that received votes from a majority of frames
        results = [
            (cls, total_conf[cls] / votes[cls])
            for cls in class_names
            if votes[cls] > n_frames / 2
        ]
    else:
        # Single-label: pick the class with the most votes (tie-break: higher avg conf)
        best = max(class_names, key=lambda c: (votes[c], total_conf[c]))
        if votes[best] == 0:
            results = []
        else:
            avg_conf = total_conf[best] / votes[best]
            results = [(best, avg_conf)] if avg_conf >= CONFIDENCE_THRESHOLD else []

    return results


def safe_dest(folder, filename):
    """Return a collision-free destination path inside folder."""
    dest = os.path.join(folder, filename)
    if not os.path.exists(dest):
        return dest
    base, ext = os.path.splitext(filename)
    counter = 1
    while os.path.exists(dest):
        dest = os.path.join(folder, f"{base}_{counter}{ext}")
        counter += 1
    return dest


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Step 1: Choose architecture ──────────────────────────────────────────
    header("STEP 1 — SELECT ARCHITECTURE")
    for arch in ARCHITECTURES:
        print(f"  [{arch['key']}] {arch['label']}")
    print()

    valid_keys = [a["key"] for a in ARCHITECTURES]
    while True:
        choice = input("Architecture: ").strip()
        if choice in valid_keys:
            arch_info = next(a for a in ARCHITECTURES if a["key"] == choice)
            break
        print(f"  Enter one of: {', '.join(valid_keys)}")

    # ── Step 2: Select .pth file ─────────────────────────────────────────────
    header("STEP 2 — SELECT WEIGHTS FILE (.pth)")

    # Scan for .pth files to offer as suggestions
    pth_files = sorted(glob.glob("**/*.pth", recursive=True) + glob.glob("*.pth"))
    if pth_files:
        print("  Found checkpoints:")
        for i, f in enumerate(pth_files[:10], 1):
            print(f"    [{i}] {f}")
        print()
        print("  Enter a number to select, or paste a full path.")
    else:
        print("  No .pth files found in current directory tree.")
        print("  Paste the full path to your weights file.")

    print()
    while True:
        raw = input("Weights: ").strip()
        # Allow selecting by index from the list
        if raw.isdigit() and pth_files and 1 <= int(raw) <= len(pth_files[:10]):
            pth_path = pth_files[int(raw) - 1]
        else:
            pth_path = raw
        if os.path.isfile(pth_path) and pth_path.endswith(".pth"):
            break
        print(f"  File not found or not a .pth: {pth_path}")

    # ── Step 3: Select input directory ───────────────────────────────────────
    header("STEP 3 — SELECT INPUT DIRECTORY")
    while True:
        input_dir = input("Input folder: ").strip()
        if os.path.isdir(input_dir):
            break
        print(f"  Directory not found: {input_dir}")

    # ── Step 4: Load model ───────────────────────────────────────────────────
    header("LOADING MODEL")
    model, class_names, img_size, model_type = load_model(arch_info, pth_path, device)
    is_multilabel = model_type in MULTILABEL_TYPES
    total_params  = sum(p.numel() for p in model.parameters())

    print(f"  Architecture : {arch_info['cls'].__name__}")
    print(f"  Model type   : {'multi-label' if is_multilabel else 'single-label'}")
    print(f"  Parameters   : {total_params:,}")
    print(f"  Classes      : {class_names}")
    print(f"  Input size   : {img_size}×{img_size}")
    print(f"  Device       : {device}" +
          (f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else ""))
    print(f"  Threshold    : {CONFIDENCE_THRESHOLD:.0%}")

    transform = build_transform(img_size, model_type)

    # ── Step 5: Collect images ───────────────────────────────────────────────
    all_images = [
        os.path.join(root, f)
        for root, _, files in os.walk(input_dir)
        for f in files
        if f.lower().endswith(IMAGE_EXTENSIONS) and not f.startswith(".")
        and not os.path.join(root, f).startswith(os.path.join(input_dir, "sorted"))
    ]

    all_videos = [
        os.path.join(root, f)
        for root, _, files in os.walk(input_dir)
        for f in files
        if f.lower().endswith(VIDEO_EXTENSIONS) and not f.startswith(".")
        and not os.path.join(root, f).startswith(os.path.join(input_dir, "sorted"))
    ]

    if not all_images and not all_videos:
        print(f"\n  No images or videos found in {input_dir}")
        sys.exit(0)

    print(f"  Found {len(all_images)} image(s) and {len(all_videos)} video(s).")

    # ── Step 6: Create output folders ────────────────────────────────────────
    output_root = os.path.join(input_dir, "sorted")
    for cls in class_names:
        os.makedirs(os.path.join(output_root, cls), exist_ok=True)
    os.makedirs(os.path.join(output_root, "Unsorted"), exist_ok=True)

    # ── Step 7: Sort ─────────────────────────────────────────────────────────
    header(f"SORTING {len(all_images)} IMAGE(S) + {len(all_videos)} VIDEO(S)")
    input("  Press Enter to start...\n")

    # Undo reverses moves and deletes copies, in reverse order.
    ops: list[dict] = []   # {"src": str, "dst": str, "op": "move"|"copy"}

    start_time = time.perf_counter()

    sorted_count   = 0
    unsorted_count = 0
    class_tally    = {cls: 0 for cls in class_names}

    for i, img_path in enumerate(all_images, 1):
        filename = os.path.basename(img_path)
        hits = predict(img_path, model, transform, class_names, device, model_type)

        prefix = f"  [{i:>{len(str(len(all_images)))}}/{len(all_images)}]"

        if not hits:
            # Below threshold → move to Unsorted
            dst = safe_dest(os.path.join(output_root, "Unsorted"), filename)
            shutil.move(img_path, dst)
            ops.append({"src": img_path, "dst": dst, "op": "move"})
            unsorted_count += 1
            print(f"{prefix} {filename}  →  Unsorted/  (below threshold)")
        elif len(hits) == 1:
            cls_name, conf = hits[0]
            dst = safe_dest(os.path.join(output_root, cls_name), filename)
            shutil.move(img_path, dst)
            ops.append({"src": img_path, "dst": dst, "op": "move"})
            class_tally[cls_name] += 1
            sorted_count += 1
            print(f"{prefix} {filename}  →  {cls_name}/  ({conf:.1%})")
        else:
            # Multi-label: copy to all but last class folder, move to last
            labels_str = ", ".join(f"{cls}({conf:.0%})" for cls, conf in hits)
            dst_paths = [
                safe_dest(os.path.join(output_root, cls_name), filename)
                for cls_name, _ in hits
            ]
            for dst in dst_paths[:-1]:
                shutil.copy2(img_path, dst)
                ops.append({"src": img_path, "dst": dst, "op": "copy"})
            shutil.move(img_path, dst_paths[-1])
            ops.append({"src": img_path, "dst": dst_paths[-1], "op": "move"})
            for cls_name, _ in hits:
                class_tally[cls_name] += 1
            sorted_count += 1
            print(f"{prefix} {filename}  →  [{labels_str}]")

    # ── Video sorting ─────────────────────────────────────────────────────────
    if all_videos:
        print()
        hr()
        print(f"  SORTING {len(all_videos)} VIDEO(S)  [extracting frames + majority vote]")
        hr()

    for i, vid_path in enumerate(all_videos, 1):
        filename = os.path.basename(vid_path)
        prefix   = f"  [V{i:>{len(str(len(all_videos)))}}/{len(all_videos)}]"
        print(f"{prefix} {filename}  — extracting frames...", end="", flush=True)

        hits = classify_video(vid_path, model, transform, class_names, device, model_type)

        if not hits:
            dst = safe_dest(os.path.join(output_root, "Unsorted"), filename)
            shutil.move(vid_path, dst)
            ops.append({"src": vid_path, "dst": dst, "op": "move"})
            unsorted_count += 1
            print(f"  →  Unsorted/  (below threshold or unreadable)")
        elif len(hits) == 1:
            cls_name, conf = hits[0]
            dst = safe_dest(os.path.join(output_root, cls_name), filename)
            shutil.move(vid_path, dst)
            ops.append({"src": vid_path, "dst": dst, "op": "move"})
            class_tally[cls_name] = class_tally.get(cls_name, 0) + 1
            sorted_count += 1
            print(f"  →  {cls_name}/  ({conf:.1%} avg, majority vote)")
        else:
            labels_str = ", ".join(f"{cls}({conf:.0%})" for cls, conf in hits)
            dst_paths  = [
                safe_dest(os.path.join(output_root, cls_name), filename)
                for cls_name, _ in hits
            ]
            for dst in dst_paths[:-1]:
                shutil.copy2(vid_path, dst)
                ops.append({"src": vid_path, "dst": dst, "op": "copy"})
            shutil.move(vid_path, dst_paths[-1])
            ops.append({"src": vid_path, "dst": dst_paths[-1], "op": "move"})
            for cls_name, _ in hits:
                class_tally[cls_name] = class_tally.get(cls_name, 0) + 1
            sorted_count += 1
            print(f"  →  [{labels_str}]  (majority vote)")

    elapsed = time.perf_counter() - start_time

    # ── Step 8: Summary ───────────────────────────────────────────────────────
    header("SORT COMPLETE")
    print(f"  Total images   : {len(all_images)}")
    print(f"  Sorted         : {sorted_count}")
    print(f"  Unsorted (<{CONFIDENCE_THRESHOLD:.0%}): {unsorted_count}")
    print(f"  Time elapsed   : {elapsed:.1f}s  ({len(all_images)/elapsed:.1f} img/s)")
    print()
    print("  Per-class breakdown:")
    for cls in class_names:
        print(f"    {cls:>14s}: {class_tally[cls]}")
    print()
    print(f"  Sorted output  : {os.path.abspath(output_root)}/")

    # ── Step 9: Pause for review then undo ───────────────────────────────────
    print()
    hr("═")
    print("  Review the sorted folders now.")
    print("  When done, press Enter to UNDO all moves and restore originals.")
    hr("═")
    input()

    header("UNDOING SORT")
    undo_start = time.perf_counter()
    errors = 0

    # Process in reverse order so multi-label moves restore correctly
    for record in reversed(ops):
        src, dst, op = record["src"], record["dst"], record["op"]
        try:
            if op == "move":
                # Move back: dst → original src location
                os.makedirs(os.path.dirname(src), exist_ok=True)
                shutil.move(dst, src)
            elif op == "copy":
                # Delete the copy
                if os.path.exists(dst):
                    os.remove(dst)
        except Exception as e:
            print(f"  ⚠  Could not undo {dst}: {e}")
            errors += 1

    # Remove empty sorted sub-folders
    for cls in list(class_names) + ["Unsorted"]:
        folder = os.path.join(output_root, cls)
        try:
            if os.path.isdir(folder) and not os.listdir(folder):
                os.rmdir(folder)
        except Exception:
            pass
    try:
        if os.path.isdir(output_root) and not os.listdir(output_root):
            os.rmdir(output_root)
    except Exception:
        pass

    undo_elapsed = time.perf_counter() - undo_start
    print(f"  Restored {len(ops) - errors} operations in {undo_elapsed:.1f}s.")
    if errors:
        print(f"  ⚠  {errors} operations could not be undone — check output above.")
    else:
        print("  ✓  All files restored to original locations.")


if __name__ == "__main__":
    main()
