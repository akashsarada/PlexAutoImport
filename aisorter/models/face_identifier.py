"""
Face identity matching via MobileFaceNet ONNX.

Runs only after face_detector has confirmed at least one face in the image.
Produces 128-dim L2-normalised embeddings, compares against a reference set
using cosine distance, and returns the best-matching identity name (or None).

Targets legacy NAS x86 CPU via onnxruntime — no torch dependency.
"""

import logging
import urllib.request
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import onnxruntime as ort

from constants import (
    FACE_IDENTIFIER_INPUT_SIZE as _INPUT_SIZE,
    FACE_IDENTIFIER_MODEL_URL as _MODEL_URL,
    IMAGE_EXTENSIONS as _SUPPORTED_EXTENSIONS,
    MODEL_CACHE_DIR as _CACHE_DIR,
)

_MODEL_FILENAME = "face_identifier.onnx"

logger = logging.getLogger(__name__)


def _download_if_missing(model_path: Path) -> None:
    """Download the ONNX model to *model_path* if it does not already exist."""
    if model_path.exists():
        return
    model_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading MobileFaceNet ONNX -> %s", model_path)
    urllib.request.urlretrieve(_MODEL_URL, model_path)
    logger.info("Download complete.")


def _is_unknown_dir(name: str) -> bool:
    """Return True for 'unknown' or 'unknown_<number>' directory names."""
    if name == "unknown":
        return True
    if name.startswith("unknown_") and name[len("unknown_"):].isdigit():
        return True
    return False


class FaceIdentifier:
    """Identity matcher using MobileFaceNet 128-dim face embeddings.

    Assumes the caller has already cropped each face region from the source
    image (face_detector's output) before calling identify().
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        distance_threshold: float = 0.4,
    ) -> None:
        resolved = (
            Path(model_path) if model_path else _CACHE_DIR / _MODEL_FILENAME
        )
        _download_if_missing(resolved)

        sess_opts = ort.SessionOptions()
        sess_opts.inter_op_num_threads = 1
        sess_opts.intra_op_num_threads = 2
        # Prefer SSE4 kernels over heavier graph optimisations on legacy CPUs.
        sess_opts.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        )
        self._session = ort.InferenceSession(
            str(resolved),
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )
        self._input_name: str = self._session.get_inputs()[0].name
        self._distance_threshold = distance_threshold
        self._references: dict[str, np.ndarray] = {}
        self._new_people_found: int = 0
        self._reference_dir: Optional[Path] = None

    @property
    def new_people_found(self) -> int:
        """Number of new unknown_x clusters created during this session."""
        return self._new_people_found

    def load_references(self, reference_dir: str) -> None:
        """Build mean-embedding lookup from a directory of per-person folders.

        Supports flat and nested (grouped) layouts::

            reference_dir/          # flat
            ├── Alice/
            │   └── photo1.jpg
            └── Bob/
                └── photo1.jpg

            reference_dir/          # nested groups
            ├── Family/
            │   ├── Alice/
            │   │   └── photo1.jpg
            │   └── Bob/
            │       └── photo1.jpg
            └── Friends/
                └── Charlie/
                    └── photo1.jpg

        The ``unknown/`` directory (and any ``unknown_<n>/`` directories) at
        the root of *reference_dir* are reserved for clustering and are never
        loaded as known identities.
        """
        ref_path = Path(reference_dir)
        if not ref_path.is_dir():
            raise ValueError(f"reference_dir does not exist: {reference_dir}")

        self._reference_dir = ref_path
        self._references.clear()
        self._new_people_found = 0

        self._load_dir(ref_path, depth=0)
        logger.info(
            "Loaded %d known identit%s from %s",
            len(self._references),
            "y" if len(self._references) == 1 else "ies",
            reference_dir,
        )

    def _load_dir(self, directory: Path, depth: int) -> None:
        """Recursively scan *directory* for person folders containing images.

        Directories whose names are 'unknown' or 'unknown_<n>' are always
        skipped regardless of depth.
        """
        for entry in sorted(directory.iterdir()):
            if not entry.is_dir():
                continue

            # Always skip clustering directories
            if _is_unknown_dir(entry.name):
                continue

            image_files = [
                p for p in entry.iterdir()
                if p.is_file() and p.suffix.lower() in _SUPPORTED_EXTENSIONS
            ]

            if image_files:
                # This directory contains images — treat it as a person folder
                embeddings: list[np.ndarray] = []
                for img_path in image_files:
                    crop = cv2.imread(str(img_path))
                    if crop is None:
                        continue
                    embeddings.append(self._embed(crop))

                if embeddings:
                    mean_vec = np.mean(np.stack(embeddings, axis=0), axis=0)
                    norm = np.linalg.norm(mean_vec)
                    self._references[entry.name] = (
                        mean_vec / norm if norm > 1e-8 else mean_vec
                    )
                    logger.debug("Loaded identity '%s' (%d image(s))", entry.name, len(embeddings))
            else:
                # No images at this level — treat as a group folder and recurse
                has_subdirs = any(e.is_dir() for e in entry.iterdir())
                if has_subdirs:
                    self._load_dir(entry, depth=depth + 1)

    def identify(
        self,
        face_crop: np.ndarray,
        source_path: Optional[str] = None,
        face_index: int = 0,
    ) -> Optional[str]:
        """Return the best-matching identity name, or None if no match.

        Args:
            face_crop: BGR face crop as returned by the face detector.
            source_path: Path of the source image file containing the face.
            face_index: Index of the face detected in the source image.

        Returns:
            Person name string when matched, otherwise None.
        """
        # Embed the query face
        query = self._embed(face_crop)

        # 1. Match against known references (includes existing folders like unknown_1, unknown_2)
        best_name: Optional[str] = None
        best_dist = float("inf")

        for name, ref_vec in self._references.items():
            dist = self._cosine_distance(query, ref_vec)
            if dist < best_dist:
                best_dist = dist
                best_name = name

        if best_dist <= self._distance_threshold:
            return best_name

        # If reference directory is not set, we cannot do clustering
        if not hasattr(self, "_reference_dir") or self._reference_dir is None:
            return None

        # 2. Match against single unrecognized headshots in references/unknown/
        unknown_dir = self._reference_dir / "unknown"
        matched_unknown_path: Optional[Path] = None
        matched_embedding: Optional[np.ndarray] = None
        best_unknown_dist = float("inf")

        if unknown_dir.is_dir():
            for img_path in sorted(unknown_dir.iterdir()):
                if img_path.suffix.lower() in _SUPPORTED_EXTENSIONS:
                    crop = cv2.imread(str(img_path))
                    if crop is None:
                        continue
                    emb = self._embed(crop)
                    dist = self._cosine_distance(query, emb)
                    if dist < best_unknown_dist:
                        best_unknown_dist = dist
                        matched_unknown_path = img_path
                        matched_embedding = emb

        # Generate unique filename for the current face crop
        if source_path:
            base = Path(source_path).stem
            out_filename = f"{base}_face_{face_index}.jpg"
        else:
            import uuid
            out_filename = f"face_{uuid.uuid4().hex[:8]}.jpg"

        # 3. If matched with an existing headshot in unknown/
        if best_unknown_dist <= self._distance_threshold and matched_unknown_path is not None:
            # Find the next available unknown_x directory name
            x = 1
            while (self._reference_dir / f"unknown_{x}").exists():
                x += 1
            new_person_dir = self._reference_dir / f"unknown_{x}"
            new_person_dir.mkdir(parents=True, exist_ok=True)

            # Move the matched headshot from unknown/ to unknown_x/
            target_matched_path = new_person_dir / matched_unknown_path.name
            try:
                matched_unknown_path.rename(target_matched_path)
            except Exception as e:
                logger.error("Failed to move matched headshot %s: %s", matched_unknown_path, e)

            # Save the current face crop to unknown_x/
            new_face_path = new_person_dir / out_filename
            cv2.imwrite(str(new_face_path), face_crop)

            # Update our in-memory references lookup table with the mean embedding of both crops
            if matched_embedding is not None:
                mean_vec = np.mean(np.stack([query, matched_embedding], axis=0), axis=0)
                norm = np.linalg.norm(mean_vec)
                self._references[new_person_dir.name] = (
                    mean_vec / norm if norm > 1e-8 else mean_vec
                )

            self._new_people_found += 1
            logger.info(
                "Created new clustering identity %s: grouped %s and %s",
                new_person_dir.name,
                target_matched_path.name,
                out_filename
            )
            return new_person_dir.name

        # 4. If no match anywhere, save the headshot to unknown/
        unknown_dir.mkdir(parents=True, exist_ok=True)
        new_face_path = unknown_dir / out_filename
        cv2.imwrite(str(new_face_path), face_crop)
        logger.info("Saved unrecognized face to %s", new_face_path)
        return None

    def _embed(self, face_crop: np.ndarray) -> np.ndarray:
        """Preprocess *face_crop* and return an L2-normalised 128-dim vector.

        Pipeline: resize → BGR→RGB → normalise to [-1, 1] → CHW → batch dim.
        """
        resized = cv2.resize(face_crop, _INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalised = (rgb.astype(np.float32) - 127.5) / 127.5
        chw = np.transpose(normalised, (2, 0, 1))
        batch = np.expand_dims(chw, axis=0)

        outputs = self._session.run(None, {self._input_name: batch})
        embedding: np.ndarray = outputs[0][0]

        norm = np.linalg.norm(embedding)
        return embedding / norm if norm > 1e-8 else embedding

    def _cosine_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine distance in [0, 2]; lower means more similar."""
        dot = float(np.dot(a, b))
        return 1.0 - dot
