"""
AISorterPipeline — cascading 4-stage photo inference executor.

Stage 1: Face detection (always)
Stage 2: Face identification (only when faces found and references loaded)
Stage 3: Category classification via ONNX (always)
Stage 4: EXIF keyword write + folder routing
"""

import argparse
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import onnxruntime as ort
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent / "models"))
from face_detector import FaceDetector
from face_identifier import FaceIdentifier

sys.path.insert(0, str(Path(__file__).parent))
from exif_writer import write_keywords
from router import PhotoRouter

CATEGORY_LABELS: list[str] = ["cars", "people", "scenery"]

_CATEGORY_INPUT_SIZE = (128, 128)
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_CATEGORY_THRESHOLD = 0.75
_DEFAULT_MODEL_PATH = Path(__file__).parent / "models" / "category_sorter.onnx"

logger = logging.getLogger(__name__)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class AISorterPipeline:
    """End-to-end photo sorting pipeline."""

    def __init__(
        self,
        output_root: str,
        reference_dir: Optional[str] = None,
        category_model_path: Optional[str] = None,
        face_confidence: float = 0.7,
        identity_threshold: float = 0.4,
    ) -> None:
        self._face_detector = FaceDetector(confidence_threshold=face_confidence)

        self._face_identifier: Optional[FaceIdentifier] = None
        if reference_dir is not None:
            self._face_identifier = FaceIdentifier(distance_threshold=identity_threshold)
            self._face_identifier.load_references(reference_dir)

        model_path = Path(category_model_path) if category_model_path else _DEFAULT_MODEL_PATH
        self._category_session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )
        self._category_input_name: str = self._category_session.get_inputs()[0].name

        self._router = PhotoRouter(output_root)

    def process_image(self, image_path: str) -> dict:
        """Run all 4 stages on a single image and route it to its destination."""
        bgr = cv2.imread(image_path)
        if bgr is None:
            raise ValueError(f"Could not read image: {image_path}")

        face_boxes = self._face_detector.detect(bgr)

        identities: list[str] = []
        if face_boxes and self._face_identifier is not None:
            for face in face_boxes:
                x1, y1, x2, y2 = (int(v) for v in face["bbox"])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(bgr.shape[1], x2), min(bgr.shape[0], y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = bgr[y1:y2, x1:x2]
                name = self._face_identifier.identify(crop)
                if name is not None and name not in identities:
                    identities.append(name)

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, _CATEGORY_INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
        normalized = (resized.astype(np.float32) / 255.0 - _IMAGENET_MEAN) / _IMAGENET_STD
        blob = np.expand_dims(normalized.transpose(2, 0, 1), axis=0)

        logits = self._category_session.run(None, {self._category_input_name: blob})[0]
        probs = _sigmoid(logits[0])
        categories = [CATEGORY_LABELS[i] for i, p in enumerate(probs) if p >= _CATEGORY_THRESHOLD]

        labels = list(identities) + [c for c in categories if c not in identities]
        write_keywords(image_path, labels)
        destinations = self._router.route(image_path, labels)

        return {
            "image": image_path,
            "faces": len(face_boxes),
            "identities": identities,
            "categories": categories,
            "labels": labels,
            "destinations": destinations,
        }

    def process_directory(
        self,
        input_dir: str,
        extensions: tuple[str, ...] = (".jpg", ".jpeg", ".JPG", ".JPEG"),
    ) -> list[dict]:
        """Recursively process all matching images in *input_dir*."""
        image_paths: list[str] = []
        for root, _, files in os.walk(input_dir):
            for fname in files:
                if any(fname.endswith(ext) for ext in extensions):
                    image_paths.append(os.path.join(root, fname))

        results: list[dict] = []
        for path in tqdm(image_paths, desc="Sorting photos", unit="img"):
            try:
                results.append(self.process_image(path))
            except Exception as exc:
                warnings.warn(f"Skipping {path}: {exc}", stacklevel=2)
                logger.warning("Skipping %s: %s", path, exc)

        return results


def main() -> None:
    parser = argparse.ArgumentParser(description="AI-powered photo sorter")
    parser.add_argument("--input", required=True, help="Input directory to scan")
    parser.add_argument("--output", required=True, help="Output root for sorted/ folders")
    parser.add_argument("--references", default=None, help="Path to reference faces directory")
    parser.add_argument("--model", default=None, help="Path to category_sorter.onnx")
    parser.add_argument("--face-confidence", type=float, default=0.7)
    parser.add_argument("--identity-threshold", type=float, default=0.4)
    args = parser.parse_args()

    pipeline = AISorterPipeline(
        output_root=args.output,
        reference_dir=args.references,
        category_model_path=args.model,
        face_confidence=args.face_confidence,
        identity_threshold=args.identity_threshold,
    )
    results = pipeline.process_directory(args.input)
    print(f"\nProcessed {len(results)} images.")


if __name__ == "__main__":
    main()
