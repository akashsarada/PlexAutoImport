"""
AISorterPipeline — cascading photo inference executor.

Stage 1: Face detection (always)
Stage 2: Face identification (only when faces found and references loaded)
Stage 3: Category classification via ONNX (always)
Stage 4: EXIF keyword write (skipped when write_metadata=False)
"""

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import onnxruntime as ort
import pillow_heif
from PIL import Image
from tqdm import tqdm

from aisorter.exif_writer import ExifToolKeywordWriter, WriteKeywords, write_keywords
from aisorter.models.face_detector import FaceDetector
from aisorter.models.face_identifier import FaceIdentifier
from constants import (
    CATEGORY_INPUT_SIZE,
    CATEGORY_LABELS,
    CATEGORY_THRESHOLD,
    DEFAULT_CATEGORY_MODEL_FILENAME,
    DEFAULT_FAMILY_GROUP,
    IMAGE_EXTENSIONS,
    IMAGENET_MEAN,
    IMAGENET_STD,
    MAX_WORKING_RESOLUTION,
)

pillow_heif.register_heif_opener()
ort.set_default_logger_severity(3)

_IMAGENET_MEAN = np.array(IMAGENET_MEAN, dtype=np.float32)
_IMAGENET_STD = np.array(IMAGENET_STD, dtype=np.float32)
_DEFAULT_MODEL_PATH = Path(__file__).parent / "models" / DEFAULT_CATEGORY_MODEL_FILENAME

logger = logging.getLogger(__name__)


def merge_labels(identities: list[str], categories: list[str]) -> list[str]:
    """Combine identity and category labels, identities first, without duplicates."""
    return list(identities) + [c for c in categories if c not in identities]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _cap_resolution(image: np.ndarray) -> np.ndarray:
    """Downscale *image* so its long edge is at most MAX_WORKING_RESOLUTION."""
    long_edge = max(image.shape[:2])
    if long_edge <= MAX_WORKING_RESOLUTION:
        return image
    scale = MAX_WORKING_RESOLUTION / long_edge
    new_size = (round(image.shape[1] * scale), round(image.shape[0] * scale))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


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
        family_group: str = DEFAULT_FAMILY_GROUP,
    ) -> None:
        self._family_group = family_group
        self._face_detector = FaceDetector(
            confidence_threshold=face_confidence,
            model_path=face_detector_model_path,
        )

        self._face_identifier: Optional[FaceIdentifier] = None
        self._family_identities: set[str] = set()
        if reference_dir is not None:
            self._face_identifier = FaceIdentifier(
                model_path=face_identifier_model_path,
                distance_threshold=identity_threshold,
            )
            self._face_identifier.load_references(reference_dir)

            # Collect names of people inside the family_group subfolder
            family_dir = Path(reference_dir) / family_group
            if family_dir.is_dir():
                self._family_identities = {
                    d.name for d in family_dir.iterdir() if d.is_dir()
                }
                logger.info(
                    "Family group '%s' loaded: %d member(s): %s",
                    family_group,
                    len(self._family_identities),
                    sorted(self._family_identities),
                )

        model_path = Path(category_model_path) if category_model_path else _DEFAULT_MODEL_PATH
        self._category_session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )
        self._category_input_name: str = self._category_session.get_inputs()[0].name

    @property
    def new_people_found(self) -> int:
        """Number of new unknown_x person clusters discovered this session."""
        if self._face_identifier is None:
            return 0
        return self._face_identifier.new_people_found

    def process_image(
        self,
        image_path: str,
        write_metadata: bool = True,
        keyword_writer: Optional[WriteKeywords] = None,
    ) -> dict:
        """Run inference and optionally write labels through *keyword_writer*."""
        bgr = self._load_image(image_path)

        resized = cv2.resize(bgr, CATEGORY_INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = (rgb.astype(np.float32) / 255.0 - _IMAGENET_MEAN) / _IMAGENET_STD
        blob = np.expand_dims(normalized.transpose(2, 0, 1), axis=0)

        identities: list[str] = []
        t0 = time.perf_counter()
        face_boxes = self._face_detector.detect(bgr)
        logger.info(
            "Stage 1/4 face detection complete for %s: faces=%d elapsed_ms=%.1f",
            image_path,
            len(face_boxes),
            (time.perf_counter() - t0) * 1000,
        )

        t1 = time.perf_counter()
        if self._face_identifier is None:
            logger.info(
                "Stage 2/4 face identification skipped for %s: identifier not configured elapsed_ms=0.0",
                image_path,
            )
        else:
            identities = self._identify_faces(bgr, face_boxes, image_path)
            logger.info(
                "Stage 2/4 face identification complete for %s: identities=%s elapsed_ms=%.1f",
                image_path,
                identities,
                (time.perf_counter() - t1) * 1000,
            )

        t2 = time.perf_counter()
        logits = self._category_session.run(None, {self._category_input_name: blob})[0]
        probs = _sigmoid(logits[0])
        categories = [CATEGORY_LABELS[i] for i, p in enumerate(probs) if p >= CATEGORY_THRESHOLD]
        logger.info(
            "Stage 3/4 category classification complete for %s: categories=%s elapsed_ms=%.1f",
            image_path,
            categories,
            (time.perf_counter() - t2) * 1000,
        )

        is_family_photo = len(set(identities) & self._family_identities) >= 2

        labels = merge_labels(identities, categories)
        if is_family_photo and self._family_group not in labels:
            labels.append(self._family_group)

        t3 = time.perf_counter()
        if write_metadata:
            (keyword_writer or write_keywords)(image_path, labels)
            logger.info(
                "Stage 4/4 keyword write complete for %s: labels=%s elapsed_ms=%.1f",
                image_path,
                labels,
                (time.perf_counter() - t3) * 1000,
            )
        else:
            logger.info(
                "Stage 4/4 keyword write skipped for %s: write_metadata=False elapsed_ms=0.0",
                image_path,
            )

        return {
            "image": image_path,
            "faces": len(face_boxes),
            "identities": identities,
            "categories": categories,
            "labels": labels,
            "is_family_photo": is_family_photo,
        }

    def _load_image(self, image_path: str) -> np.ndarray:
        """Load *image_path* as BGR, capped to MAX_WORKING_RESOLUTION on the long edge."""
        if image_path.lower().endswith(".dng"):
            try:
                import rawpy
                with rawpy.imread(image_path) as raw:
                    rgb = raw.postprocess(half_size=True)
                return cv2.cvtColor(_cap_resolution(rgb), cv2.COLOR_RGB2BGR)
            except Exception as raw_err:
                logger.warning("Failed to load DNG using rawpy: %s. Falling back to default loader.", raw_err)

        try:
            with Image.open(image_path) as pil_img:
                pil_img.draft("RGB", (MAX_WORKING_RESOLUTION, MAX_WORKING_RESOLUTION))
                rgb = _cap_resolution(np.array(pil_img.convert("RGB")))
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except Exception as e:
            bgr = cv2.imread(image_path)
            if bgr is None:
                raise ValueError(f"Could not read image {image_path}: {e}") from e
            return _cap_resolution(bgr)

    def _identify_faces(self, bgr: np.ndarray, face_boxes: list[dict], image_path: str) -> list[str]:
        if not face_boxes or self._face_identifier is None:
            return []
        identities: list[str] = []
        for i, face in enumerate(face_boxes):
            x1, y1, x2, y2 = (int(v) for v in face["bbox"])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(bgr.shape[1], x2), min(bgr.shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                continue
            name = self._face_identifier.identify(bgr[y1:y2, x1:x2], image_path, i)
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
        with ExifToolKeywordWriter() as keyword_writer:
            for path in tqdm(image_paths, desc="Sorting photos", unit="img"):
                try:
                    results.append(
                        self.process_image(path, keyword_writer=keyword_writer)
                    )
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
