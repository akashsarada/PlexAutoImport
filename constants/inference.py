"""Classification and training constants."""

CATEGORY_LABELS: list[str] = ["cars", "people", "scenery"]

# (keyword, label index) pairs used to decode training folder names into multi-hot labels.
LABEL_KEYWORDS: list[tuple[str, int]] = [
    ("cars", 0),
    ("people", 1),
    ("scenery", 2),
]

CATEGORY_THRESHOLD = 0.75
CLASSIFY_CONFIDENCE_THRESHOLD = 0.85

CATEGORY_INPUT_SIZE = (128, 128)
TRAINING_IMAGE_SIZE = 128

# 2560px keeps any face detectable at 320px wide at >=112px for the identifier.
MAX_WORKING_RESOLUTION = 2560

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# model_type values that emit sigmoid multi-label outputs; all others are softmax single-label.
MULTILABEL_MODEL_TYPES = {"category_sorter"}

DEFAULT_CATEGORY_MODEL_FILENAME = "category_sorter.onnx"
