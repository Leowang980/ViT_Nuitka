import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torchvision import transforms

from models import VisionTransformer, vit_base, vit_small, vit_tiny


MODEL_FACTORY = {
    "vit_tiny": vit_tiny,
    "vit_small": vit_small,
    "vit_base": vit_base,
}

DEFAULT_CIFAR10_CLASSES: List[str] = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run inference with a Vision Transformer checkpoint.")
    parser.add_argument("inputs", type=str, nargs="+", help="Image file(s) or directories for inference.")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to model checkpoint (.pth).")
    parser.add_argument("--model", type=str, default="vit_tiny", choices=MODEL_FACTORY.keys(), help="Model variant.")
    parser.add_argument("--image-size", type=int, default=224, help="Input resolution expected by the model.")
    parser.add_argument("--num-classes", type=int, default=None, help="Override number of classes if checkpoint absent.")
    parser.add_argument("--class-names", type=str, default="", help="Optional text file containing class names.")
    parser.add_argument("--device", type=str, default="auto", help="Torch device string or 'auto'.")
    parser.add_argument("--topk", type=int, default=5, help="Number of top predictions to display.")
    parser.add_argument("--amp", action="store_true", help="Enable AMP during inference on CUDA.")
    parser.add_argument("--quiet", action="store_true", help="Suppress console output and only return results.")
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = build_parser()
    if argv is None:
        return parser.parse_args()
    return parser.parse_args(list(argv))


def load_class_names(path: str) -> List[str]:
    if not path:
        return DEFAULT_CIFAR10_CLASSES
    class_file = Path(path)
    if not class_file.is_file():
        raise FileNotFoundError(f"Class names file '{class_file}' not found.")
    with class_file.open("r", encoding="utf-8") as handle:
        names = [line.strip() for line in handle if line.strip()]
    if not names:
        raise ValueError("Class names file is empty.")
    return names


def collect_images(inputs: List[str]) -> List[Path]:
    images: List[Path] = []
    for item in inputs:
        path = Path(item)
        if path.is_dir():
            for file in sorted(path.rglob("*")):
                if file.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                    images.append(file)
        elif path.is_file():
            images.append(path)
        else:
            raise FileNotFoundError(f"Input path '{path}' does not exist.")
    if not images:
        raise ValueError("No image files found in the provided inputs.")
    return images


def get_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(args: argparse.Namespace, example_classes: int) -> VisionTransformer:
    if args.model not in MODEL_FACTORY:
        raise ValueError(f"Unknown model '{args.model}'. Available: {list(MODEL_FACTORY.keys())}")
    num_classes = args.num_classes or example_classes
    model = MODEL_FACTORY[args.model](num_classes=num_classes, image_size=args.image_size)
    return model


def load_checkpoint(path: Path, model: VisionTransformer) -> Dict:
    checkpoint = torch.load(path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(state_dict)
    print(f"Loaded checkpoint from '{path}'.")
    return checkpoint


def build_transform(image_size: int) -> transforms.Compose:
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    return transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.1)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def perform_inference(args: argparse.Namespace) -> Dict[str, List[Tuple[str, float]]]:
    device = get_device(args.device)
    if not args.quiet:
        print(f"Using device: {device}")

    images = collect_images(args.inputs)
    class_names = load_class_names(args.class_names)
    model = build_model(args, example_classes=len(class_names))

    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if checkpoint_path.is_file():
            load_checkpoint(checkpoint_path, model)
        else:
            raise FileNotFoundError(f"Checkpoint '{checkpoint_path}' not found.")

    model.to(device)
    model.eval()

    preprocess = build_transform(args.image_size)
    topk = max(1, min(args.topk, len(class_names)))
    predictions: Dict[str, List[Tuple[str, float]]] = {}

    with torch.inference_mode():
        amp_enabled = args.amp and device.type == "cuda"
        for image_path in images:
            image = Image.open(image_path).convert("RGB")
            tensor = preprocess(image).unsqueeze(0).to(device)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                logits = model(tensor)
            probabilities = torch.softmax(logits, dim=-1)
            values, indices = probabilities.topk(topk, dim=-1)

            entries: List[Tuple[str, float]] = []
            for rank in range(topk):
                score = values[0, rank].item()
                label_idx = indices[0, rank].item()
                label = class_names[label_idx] if label_idx < len(class_names) else f"class_{label_idx}"
                entries.append((label, float(score)))

            predictions[str(image_path)] = entries
            if not args.quiet:
                print(f"\nResults for '{image_path}':")
                for rank, (label, score) in enumerate(entries, start=1):
                    print(f"  Top {rank}: {label} ({score*100:.2f}%)")

    return predictions


def main() -> None:
    args = parse_args()
    perform_inference(args)


if __name__ == "__main__":
    main()
