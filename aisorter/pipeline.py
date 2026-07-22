"""
AISorterPipeline — cascading photo inference executor.

Stage 1: Category classification via ONNX (always)
Stage 2: Face detection (only when "people" category predicted)
Stage 3: Face identification (only when faces found and references loaded)
Stage 4: EXIF keyword write
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import onnxruntime as ort
import pillow_heif
from PIL import Image
from tqdm import tqdm

from aisorter.exif_writer import write_keywords
from aisorter.models.face_detector import FaceDetector
from aisorter.models.face_identifier import FaceIdentifier
from constants import (
    CATEGORY_INPUT_SIZE,
    CATEGORY_LABELS,
    CATEGORY_THRESHOLD,
    DEFAULT_CATEGORY_MODEL_FILENAME,
    IMAGE_EXTENSIONS,
    IMAGENET_MEAN,
    IMAGENET_STD,
)

pillow_heif.register_heif_opener()
ort.set_default_logger_severity(3)

_IMAGENET_MEAN = np.array(IMAGENET_MEAN, dtype=np.float32)
_IMAGENET_STD = np.array(IMAGENET_STD, dtype=np.float32)
_DEFAULT_MODEL_PATH = Path(__file__).parent / "models" / DEFAULT_CATEGORY_MODEL_FILENAME

logger = logging.getLogger(__name__)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class AISorterPipeline:
    """End-to-end photo sorting pipeline."""

    def __init__(
        self,
        reference_dir: Optional[str] = None,
        category_model_path: Optional[str] = None,
        face_detector_model_path: Optional[str] = None,
        face_identifier_model_path: Optional[str] = None,
        face_confidence: float = 0.7,
        identity_threshold: float = 0.4,
    ) -> None:
        self._face_detector = FaceDetector(
            confidence_threshold=face_confidence,
            model_path=face_detector_model_path,
        )

        self._face_identifier: Optional[FaceIdentifier] = None
        if reference_dir is not None:
            self._face_identifier = FaceIdentifier(
                model_path=face_identifier_model_path,
                distance_threshold=identity_threshold,
            )
            self._face_identifier.load_references(reference_dir)

        model_path = Path(category_model_path) if category_model_path else _DEFAULT_MODEL_PATH
        self._category_session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )
        self._category_input_name: str = self._category_session.get_inputs()[0].name

    def process_image(self, image_path: str) -> dict:
        """Run all stages on a single image; returns categories, identities, and labels written."""
        rgb, bgr = self._load_image(image_path)

        resized = cv2.resize(rgb, CATEGORY_INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
        normalized = (resized.astype(np.float32) / 255.0 - _IMAGENET_MEAN) / _IMAGENET_STD
        blob = np.expand_dims(normalized.transpose(2, 0, 1), axis=0)

        logits = self._category_session.run(None, {self._category_input_name: blob})[0]
        probs = _sigmoid(logits[0])
        categories = [CATEGORY_LABELS[i] for i, p in enumerate(probs) if p >= CATEGORY_THRESHOLD]
        logger.info(
            "Stage 1/4 category classification complete for %s: categories=%s",
            image_path,
            categories,
        )

        face_boxes = []
        identities: list[str] = []
        if any(c.lower() == "people" for c in categories):
            face_boxes = self._face_detector.detect(bgr)
            logger.info(
                "Stage 2/4 face detection complete for %s: faces=%d",
                image_path,
                len(face_boxes),
            )
            identities = self._identify_faces(bgr, face_boxes)
            if self._face_identifier is None:
                logger.info(
                    "Stage 3/4 face identification skipped for %s: identifier not configured",
                    image_path,
                )
            else:
                logger.info(
                    "Stage 3/4 face identification complete for %s: identities=%s",
                    image_path,
                    identities,
                )
        else:
            logger.info(
                "Stage 2/4 face detection skipped for %s: people category not detected",
                image_path,
            )
            logger.info(
                "Stage 3/4 face identification skipped for %s: face detection not run",
                image_path,
            )

        labels = list(identities) + [c for c in categories if c not in identities]
        write_keywords(image_path, labels)
        logger.info(
            "Stage 4/4 keyword write complete for %s: labels=%s",
            image_path,
            labels,
        )

        return {
            "image": image_path,
            "faces": len(face_boxes),
            "identities": identities,
            "categories": categories,
            "labels": labels,
        }

    def _load_image(self, image_path: str) -> tuple[np.ndarray, np.ndarray]:
        if image_path.lower().endswith(".dng"):
            try:
                import rawpy
                with rawpy.imread(image_path) as raw:
                    rgb = raw.postprocess()
                return rgb, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            except Exception as raw_err:
                logger.warning("Failed to load DNG using rawpy: %s. Falling back to default loader.", raw_err)

        try:
            with Image.open(image_path) as pil_img:
                rgb = np.array(pil_img.convert("RGB"))
                return rgb, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except Exception as e:
            bgr = cv2.imread(image_path)
            if bgr is None:
                raise ValueError(f"Could not read image {image_path}: {e}") from e
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), bgr

    def _identify_faces(self, bgr: np.ndarray, face_boxes: list[dict]) -> list[str]:
        if not face_boxes or self._face_identifier is None:
            return []
        identities: list[str] = []
        for face in face_boxes:
            x1, y1, x2, y2 = (int(v) for v in face["bbox"])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(bgr.shape[1], x2), min(bgr.shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                continue
            name = self._face_identifier.identify(bgr[y1:y2, x1:x2])
            if name is not None and name not in identities:
                identities.append(name)
        return identities

    def process_directory(
        self,
        input_dir: str,
        extensions: tuple[str, ...] = IMAGE_EXTENSIONS
    ) -> list[dict]:
        """Recursively process all matching images in *input_dir*."""
        image_paths: list[str] = []
        for root, _, files in os.walk(input_dir):
            for fname in files:
                if fname.lower().endswith(extensions):
                    image_paths.append(os.path.join(root, fname))

        results: list[dict] = []
        for path in tqdm(image_paths, desc="Sorting photos", unit="img"):
            try:
                results.append(self.process_image(path))
            except Exception as exc:
                logger.warning("Skipping %s: %s", path, exc)

        return results


def main() -> None:
    parser = argparse.ArgumentParser(description="AI-powered photo sorter")
    parser.add_argument("--input", required=True, help="Input directory to scan")
    parser.add_argument("--references", default=None, help="Path to reference faces directory")
    parser.add_argument("--model", default=None, help="Path to the category classifier ONNX model")
    parser.add_argument("--face-confidence", type=float, default=0.7)
    parser.add_argument("--identity-threshold", type=float, default=0.4)
    args = parser.parse_args()

    pipeline = AISorterPipeline(
        reference_dir=args.references,
        category_model_path=args.model,
        face_confidence=args.face_confidence,
        identity_threshold=args.identity_threshold,
    )
    results = pipeline.process_directory(args.input)
    print(f"\nProcessed {len(results)} images.")


if __name__ == "__main__":
    main()
