"""Export a trained MobileNetV3Mini .pth checkpoint to ONNX for onnxruntime inference."""

import argparse
import os
import sys

import numpy as np
import torch
import onnxruntime as ort

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "models"))
from models.DSC import CategorySorter
from models.MobileNetV3_Mini import MobileNetV3Mini
from models.MobileNetV3_Small import MobileNetV3Small
from models.small_CNN import CustomCNN


def _resolve_output_path(model_path: str, output_arg: str | None) -> str:
    if output_arg:
        return output_arg
    base = os.path.splitext(model_path)[0]
    return base + ".onnx"


def _param_count(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def export(model_path: str, output_path: str, opset: int) -> str:
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    class_names = checkpoint.get("class_names", ["cars", "people", "scenery"])
    num_classes = len(class_names)
    model_type = checkpoint.get("model_type", "category_sorter")

    if model_type == "category_sorter":
        model = CategorySorter(num_classes=num_classes)
    elif model_type == "sorter_mini":
        model = MobileNetV3Mini(num_classes=num_classes)
    elif model_type == "sorter_mobilenet":
        model = MobileNetV3Small(num_classes=num_classes, pretrained=False)
    elif model_type == "sorter":
        model = CustomCNN(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model_type in checkpoint: {model_type}")

    model.load_state_dict(checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint)
    model.eval()

    img_size = checkpoint.get("img_size", 128)
    dummy = torch.zeros(1, 3, img_size, img_size)

    torch.onnx.export(
        model,
        dummy,
        output_path,
        opset_version=opset,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input":  {0: "batch"},
            "output": {0: "batch"},
        },
    )


    session = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
    result = session.run(None, {"input": np.zeros((1, 3, img_size, img_size), dtype=np.float32)})
    output_shape = result[0].shape

    params = _param_count(model)
    print(f"Exported: {output_path}")
    print(f"Classes:  {class_names}")
    print(f"Params:   {params:,}")
    print(f"Output shape (batch=1): {output_shape}")

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a .pth checkpoint to ONNX.")
    parser.add_argument("--model",  required=True, help="Path to .pth checkpoint")
    parser.add_argument("--output", default=None,  help="Path for .onnx output (default: same dir, same name)")
    parser.add_argument("--opset",  type=int, default=13, help="ONNX opset version (default: 13)")
    args = parser.parse_args()

    output_path = _resolve_output_path(args.model, args.output)
    export(args.model, output_path, args.opset)


if __name__ == "__main__":
    main()
