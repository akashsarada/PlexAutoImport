"""Face model download locations and inference parameters."""

from pathlib import Path

MODEL_CACHE_DIR = Path.home() / ".cache" / "aisorter"

FACE_DETECTOR_MODEL_URL = (
    "https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB"
    "/raw/master/models/onnx/version-RFB-320.onnx"
)
FACE_DETECTOR_INPUT_WIDTH = 320
FACE_DETECTOR_INPUT_HEIGHT = 240
FACE_NMS_IOU_THRESHOLD = 0.4

FACE_IDENTIFIER_MODEL_URL = (
    "https://huggingface.co/deepghs/insightface/resolve/main"
    "/buffalo_s/w600k_mbf.onnx"
)
FACE_IDENTIFIER_INPUT_SIZE = (112, 112)
