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
_EMBEDDING_SIZE = 128

logger = logging.getLogger(__name__)


def _download_if_missing(model_path: Path) -> None:
    """Download the ONNX model to *model_path* if it does not already exist."""
    if model_path.exists():
        return
    model_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading MobileFaceNet ONNX -> %s", model_path)
    urllib.request.urlretrieve(_MODEL_URL, model_path)
    logger.info("Download complete.")


def _is_unclustered_dir(name: str) -> bool:
    """Return whether *name* is the directory of unclustered faces."""
    return name == "unknown"


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
        self._unknown_paths: list[Path] = []
        self._unknown_matrix: np.ndarray = np.empty((0, _EMBEDDING_SIZE), dtype=np.float32)

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

        The ``unknown/`` directory contains unclustered faces and is cached
        separately. Completed ``unknown_<n>/`` clusters are loaded as identities.
        """
        ref_path = Path(reference_dir)
        if not ref_path.is_dir():
            raise ValueError(f"reference_dir does not exist: {reference_dir}")

        self._reference_dir = ref_path
        self._references.clear()
        self._new_people_found = 0

        self._load_dir(ref_path, depth=0)
        self._rebuild_unknown_cache(ref_path / "unknown")
        logger.info(
            "Loaded %d known identit%s from %s",
            len(self._references),
            "y" if len(self._references) == 1 else "ies",
            reference_dir,
        )

    def _rebuild_unknown_cache(self, unknown_dir: Path) -> None:
        """(Re)build the in-memory cache of embeddings from *unknown_dir*."""
        self._unknown_paths = []
        embeddings: list[np.ndarray] = []

        if unknown_dir.is_dir():
            for img_path in sorted(unknown_dir.iterdir()):
                if img_path.suffix.lower() in _SUPPORTED_EXTENSIONS:
                    crop = cv2.imread(str(img_path))
                    if crop is None:
                        continue
                    embeddings.append(self._embed(crop))
                    self._unknown_paths.append(img_path)

        if embeddings:
            self._unknown_matrix = np.stack(embeddings, axis=0).astype(np.float32)
        else:
            self._unknown_matrix = np.empty((0, _EMBEDDING_SIZE), dtype=np.float32)

    def _append_unknown_cache(self, image_path: Path, embedding: np.ndarray) -> None:
        """Add one face embedding to the unclustered cache."""
        self._unknown_paths.append(image_path)
        self._unknown_matrix = np.vstack(
            [self._unknown_matrix, embedding.reshape(1, -1).astype(np.float32)]
        )

    def _remove_unknown_cache(self, image_path: Path) -> None:
        """Remove one face embedding from the unclustered cache."""
        keep = [index for index, path in enumerate(self._unknown_paths) if path != image_path]
        self._unknown_paths = [self._unknown_paths[index] for index in keep]
        self._unknown_matrix = (
            self._unknown_matrix[keep]
            if keep
            else np.empty((0, _EMBEDDING_SIZE), dtype=np.float32)
        )

    def _save_unknown_face(
        self,
        image_path: Path,
        face_crop: np.ndarray,
        embedding: np.ndarray,
    ) -> bool:
        """Persist and cache one unclustered face."""
        if not cv2.imwrite(str(image_path), face_crop):
            logger.error("Failed to save unrecognized face to %s", image_path)
            return False
        self._append_unknown_cache(image_path, embedding)
        logger.info("Saved unrecognized face to %s", image_path)
        return True

    def _load_dir(self, directory: Path, depth: int) -> None:
        """Recursively scan *directory* for person folders containing images.

        The unclustered ``unknown`` directory is skipped at every depth.
        """
        for entry in sorted(directory.iterdir()):
            if not entry.is_dir():
                continue

            # Always skip clustering directories
            if _is_unclustered_dir(entry.name):
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

        # 2. Match against cached embeddings from references/unknown/
        matched_unknown_path: Optional[Path] = None
        matched_embedding: Optional[np.ndarray] = None
        best_unknown_dist = float("inf")

        if self._unknown_matrix.shape[0] > 0:
            dists: np.ndarray = 1.0 - (self._unknown_matrix @ query)
            best_idx = int(np.argmin(dists))
            best_unknown_dist = float(dists[best_idx])
            matched_unknown_path = self._unknown_paths[best_idx]
            matched_embedding = self._unknown_matrix[best_idx]

        unknown_dir = self._reference_dir / "unknown"

        # Generate unique filename for the current face crop
        if source_path:
            base = Path(source_path).stem
            out_filename = f"{base}_face_{face_index}.jpg"
        else:
            import uuid
            out_filename = f"face_{uuid.uuid4().hex[:8]}.jpg"

        # 3. If matched with an existing headshot in unknown/
        if best_unknown_dist <= self._distance_threshold and matched_unknown_path is not None:
            x = 1
            while (self._reference_dir / f"unknown_{x}").exists():
                x += 1
            new_person_dir = self._reference_dir / f"unknown_{x}"
            new_person_dir.mkdir(parents=True, exist_ok=True)

            target_matched_path = new_person_dir / matched_unknown_path.name
            try:
                matched_unknown_path.rename(target_matched_path)
            except OSError as error:
                logger.error("Failed to move matched headshot %s: %s", matched_unknown_path, error)
                unknown_dir.mkdir(parents=True, exist_ok=True)
                self._save_unknown_face(unknown_dir / out_filename, face_crop, query)
                return None

            new_face_path = new_person_dir / out_filename
            if not cv2.imwrite(str(new_face_path), face_crop):
                logger.error("Failed to save clustered face to %s", new_face_path)
                try:
                    target_matched_path.rename(matched_unknown_path)
                except OSError as rollback_error:
                    logger.error(
                        "Failed to restore matched headshot %s: %s",
                        target_matched_path,
                        rollback_error,
                    )
                    self._remove_unknown_cache(matched_unknown_path)
                return None

            if matched_embedding is not None:
                mean_vec = np.mean(np.stack([query, matched_embedding], axis=0), axis=0)
                norm = np.linalg.norm(mean_vec)
                self._references[new_person_dir.name] = (
                    mean_vec / norm if norm > 1e-8 else mean_vec
                )

            self._remove_unknown_cache(matched_unknown_path)
            self._new_people_found += 1
            logger.info(
                "Created new clustering identity %s: grouped %s and %s",
                new_person_dir.name,
                target_matched_path.name,
                out_filename,
            )
            return new_person_dir.name

        # 4. If no match anywhere, save the headshot to unknown/
        unknown_dir.mkdir(parents=True, exist_ok=True)
        self._save_unknown_face(unknown_dir / out_filename, face_crop, query)
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
