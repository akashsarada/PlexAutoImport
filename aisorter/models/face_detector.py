"""
Ultra-lightweight face detection via the Ultra-Light Generic Face Detector (RFB-320).

Model: ~400 K parameters, ~1 MB ONNX weight.  Runs on SSE4-only CPUs (no AVX/AVX2).
Input resolution: 320 × 240 pixels.
Weight source: https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB
"""

import logging
import pathlib
import urllib.request
from typing import Optional

import cv2
import numpy as np
import onnxruntime as ort
from tqdm import tqdm

from runtime_output import is_verbose
from constants import (
    FACE_DETECTOR_INPUT_HEIGHT as _INPUT_H,
    FACE_DETECTOR_INPUT_WIDTH as _INPUT_W,
    FACE_DETECTOR_MODEL_URL as _MODEL_URL,
    FACE_NMS_IOU_THRESHOLD as _NMS_IOU_THRESHOLD,
    MODEL_CACHE_DIR,
)

_DEFAULT_CACHE_PATH = MODEL_CACHE_DIR / "face_detector.onnx"

logger = logging.getLogger(__name__)


def _download_if_missing(url: str, path: pathlib.Path) -> None:
    """Download *url* to *path* with progress unless verbose logging is enabled."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = path.with_suffix(f"{path.suffix}.part")
    logger.info("Downloading face detector model -> %s", path)
    try:
        with urllib.request.urlopen(url, timeout=60) as response, open(partial_path, "wb") as output:
            total = int(response.headers.get("content-length", 0))
            with tqdm(
                desc=path.name,
                total=total,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                disable=is_verbose(),
            ) as bar:
                while True:
                    chunk = response.read(8192)
                    if not chunk:
                        break
                    output.write(chunk)
                    bar.update(len(chunk))
        partial_path.replace(path)
        logger.info("Face detector model download complete")
    except Exception:
        partial_path.unlink(missing_ok=True)
        raise


def _iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Return IoU between *box* [x1,y1,x2,y2] and each row of *boxes*."""
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_box = (box[2] - box[0]) * (box[3] - box[1])
    area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area_box + area_boxes - inter
    return inter / np.where(union > 0, union, 1e-9)


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """Non-maximum suppression; returns kept indices sorted by descending score."""
    order = scores.argsort()[::-1]
    kept: list[int] = []
    while order.size > 0:
        idx = int(order[0])
        kept.append(idx)
        if order.size == 1:
            break
        rest = order[1:]
        ious = _iou(boxes[idx], boxes[rest])
        order = rest[ious <= iou_threshold]
    return kept


class FaceDetector:
    """
    Detects faces in BGR images using the Ultra-Light Generic Face Detector (RFB-320).

    The ONNX weights are downloaded once to ~/.cache/aisorter/face_detector.onnx on
    first instantiation.  Inference always runs on CPU via onnxruntime, making this
    safe on SSE4-only hardware.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.7,
        model_path: Optional[str] = None,
    ) -> None:
        resolved = pathlib.Path(model_path) if model_path else _DEFAULT_CACHE_PATH
        _download_if_missing(_MODEL_URL, resolved)
        # Force CPU provider; avoids AVX/AVX2 dispatching that breaks on legacy NAS
        self._session = ort.InferenceSession(
            str(resolved),
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name
        self._confidence_threshold = confidence_threshold

    def detect(self, image: np.ndarray) -> list[dict]:
        """
        Detect faces in a BGR numpy array of any resolution.

        Returns a list of dicts, each with:
          'bbox'       – (x1, y1, x2, y2) in original image pixel coordinates
          'confidence' – float in [0, 1]
        """
        orig_h, orig_w = image.shape[:2]
        blob = self._preprocess(image)
        confidences, boxes = self._session.run(None, {self._input_name: blob})
        # confidences: [1, N, 2]  boxes: [1, N, 4] in normalised [0,1] coords
        scores = confidences[0, :, 1]  # class-1 is "face"
        raw_boxes = boxes[0]           # shape [N, 4]: cx,cy,w,h  →  already x1y1x2y2 per model

        mask = scores >= self._confidence_threshold
        scores = scores[mask]
        raw_boxes = raw_boxes[mask]

        if scores.size == 0:
            return []

        # Rescale normalised [0,1] coordinates to original image pixels
        scale = np.array([orig_w, orig_h, orig_w, orig_h], dtype=np.float32)
        pixel_boxes = raw_boxes * scale

        kept = _nms(pixel_boxes, scores, _NMS_IOU_THRESHOLD)

        return [
            {
                "bbox": (
                    float(pixel_boxes[i, 0]),
                    float(pixel_boxes[i, 1]),
                    float(pixel_boxes[i, 2]),
                    float(pixel_boxes[i, 3]),
                ),
                "confidence": float(scores[i]),
            }
            for i in kept
        ]

    def _preprocess(self, bgr: np.ndarray) -> np.ndarray:
        """BGR uint8 → normalised float32 blob [1, 3, 240, 320]."""
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (_INPUT_W, _INPUT_H), interpolation=cv2.INTER_LINEAR)
        # Normalise to [-1, 1]
        blob = (resized.astype(np.float32) - 127.0) / 128.0
        blob = blob.transpose(2, 0, 1)   # HWC → CHW
        blob = np.expand_dims(blob, 0)   # → [1, 3, H, W]
        return blob
