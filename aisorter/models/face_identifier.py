"""
Face identity matching via MobileFaceNet ONNX.

Runs only after face_detector has confirmed at least one face in the image.
Produces 128-dim L2-normalised embeddings, compares against a reference set
using cosine distance, and returns the best-matching identity name (or None).

Targets legacy NAS x86 CPU via onnxruntime — no torch dependency.
"""

import os
import urllib.request
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import onnxruntime as ort

_MODEL_URL = (
    "https://github.com/deepinsight/insightface/raw/master"
    "/model_zoo/models/buffalo_sc/w600k_mbf.onnx"
)
_CACHE_DIR = Path.home() / ".cache" / "aisorter"
_MODEL_FILENAME = "face_identifier.onnx"
_INPUT_SIZE = (112, 112)
_EMBEDDING_DIM = 128
_SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _download_if_missing(model_path: Path) -> None:
    """Download the ONNX model to *model_path* if it does not already exist."""
    if model_path.exists():
        return
    model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[face_identifier] Downloading MobileFaceNet ONNX → {model_path}")
    urllib.request.urlretrieve(_MODEL_URL, model_path)
    print("[face_identifier] Download complete.")


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

    def load_references(self, reference_dir: str) -> None:
        """Build mean-embedding lookup from a directory of per-person folders.

        Expected layout::

            reference_dir/
            ├── Alice/
            │   ├── photo1.jpg
            │   └── photo2.jpg
            └── Bob/
                └── photo1.jpg
        """
        ref_path = Path(reference_dir)
        if not ref_path.is_dir():
            raise ValueError(f"reference_dir does not exist: {reference_dir}")

        self._references.clear()
        for person_dir in sorted(ref_path.iterdir()):
            if not person_dir.is_dir():
                continue
            image_files = [
                p for p in person_dir.iterdir()
                if p.suffix.lower() in _SUPPORTED_EXTENSIONS
            ]
            if not image_files:
                continue

            embeddings: list[np.ndarray] = []
            for img_path in image_files:
                crop = cv2.imread(str(img_path))
                if crop is None:
                    continue
                embeddings.append(self._embed(crop))

            if embeddings:
                mean_vec = np.mean(np.stack(embeddings, axis=0), axis=0)
                norm = np.linalg.norm(mean_vec)
                self._references[person_dir.name] = (
                    mean_vec / norm if norm > 1e-8 else mean_vec
                )

    def identify(self, face_crop: np.ndarray) -> Optional[str]:
        """Return the best-matching identity name, or None if no match.

        Args:
            face_crop: BGR face crop as returned by the face detector.

        Returns:
            Person name string when the closest reference is within
            *distance_threshold*, otherwise None.
        """
        if not self._references:
            return None

        query = self._embed(face_crop)
        best_name: Optional[str] = None
        best_dist = float("inf")

        for name, ref_vec in self._references.items():
            dist = self._cosine_distance(query, ref_vec)
            if dist < best_dist:
                best_dist = dist
                best_name = name

        return best_name if best_dist <= self._distance_threshold else None

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
