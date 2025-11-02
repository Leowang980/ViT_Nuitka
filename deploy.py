import argparse
import io
import json
import os
import struct
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torchvision import transforms

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from models import VisionTransformer, vit_base, vit_small, vit_tiny
from seal import EncryptedTarStream, HEADER_LEN_FMT, MAGIC, b64d, unwrap_kc_with_passphrase


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

CLASS_NAME_TXT_CANDIDATES = ("class_names.txt", "labels.txt")
CLASS_NAME_JSON_CANDIDATES = ("class_names.json", "labels.json")
CHECKPOINT_EXTENSIONS = (".pth", ".pt", ".pth.tar")


@dataclass
class BundleArtifacts:
    checkpoint: Optional[Dict[str, Any]]
    checkpoint_name: Optional[str]
    class_names: Optional[List[str]]
    header: Dict[str, Any]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run inference with a Vision Transformer checkpoint.")
    parser.add_argument("inputs", type=str, nargs="+", help="Image file(s) or directories for inference.")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to model checkpoint (.pth).")
    parser.add_argument("--sealed", type=str, default="", help="Encrypted bundle (.enc) containing model assets.")
    parser.add_argument(
        "--sealed-passphrase",
        type=str,
        default=None,
        help="Passphrase for decrypting the sealed bundle (if passphrase-protected).",
    )
    parser.add_argument(
        "--sealed-key",
        type=str,
        default=None,
        help="Base64 content key for sealed bundles without a passphrase (falls back to MODEL_SEALER_KC_B64).",
    )
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


def _resolve_content_key(
    header: Dict[str, Any],
    passphrase: Optional[str],
    content_key_b64: Optional[str],
) -> bytes:
    if header.get("kc_wrapped"):
        if not passphrase:
            raise ValueError("Passphrase is required to decrypt this sealed bundle.")
        return unwrap_kc_with_passphrase(header["kc_wrapped"], passphrase, header["kdf"], header["kc_nonce"])

    key_source = content_key_b64 or os.environ.get("MODEL_SEALER_KC_B64")
    if not key_source:
        raise ValueError(
            "Bundle is not passphrase-protected. Provide --sealed-key or MODEL_SEALER_KC_B64 with the base64 content key."
        )
    return b64d(key_source)


def load_sealed_bundle(
    bundle_path: Path,
    passphrase: Optional[str],
    content_key_b64: Optional[str],
) -> BundleArtifacts:
    if not bundle_path.is_file():
        raise FileNotFoundError(f"Sealed bundle '{bundle_path}' not found.")

    bundle_size = bundle_path.stat().st_size
    with bundle_path.open("rb") as handle:
        magic = handle.read(len(MAGIC))
        if magic != MAGIC:
            raise ValueError("Invalid sealed bundle magic/version.")

        header_len_bytes = handle.read(struct.calcsize(HEADER_LEN_FMT))
        if len(header_len_bytes) != struct.calcsize(HEADER_LEN_FMT):
            raise ValueError("Malformed sealed bundle header length.")
        header_len = struct.unpack(HEADER_LEN_FMT, header_len_bytes)[0]
        header_raw = handle.read(header_len)
        if len(header_raw) != header_len:
            raise ValueError("Malformed sealed bundle header payload.")

        header: Dict[str, Any] = json.loads(header_raw.decode("utf-8"))

        chunk_size = int(header["chunk_size"])
        total_chunks = int(header["total_chunks"])
        tar_sha = header["tar_sha256"]

        kc = _resolve_content_key(header, passphrase, content_key_b64)
        aes = AESGCM(kc)

        stream = EncryptedTarStream(
            handle,
            file_size=bundle_size,
            chunk_size=chunk_size,
            total_chunks=total_chunks,
            tar_sha=tar_sha,
            aes=aes,
        )

        checkpoint: Optional[Dict[str, Any]] = None
        checkpoint_name: Optional[str] = None
        class_names: Optional[List[str]] = None

        with tarfile.open(fileobj=stream, mode="r|*") as archive:
            for member in archive:
                if member.isdir():
                    continue
                if member.issym() or member.islnk():
                    raise ValueError(f"Blocked link entry in sealed bundle: {member.name}")

                member_path = Path(member.name)
                if member_path.is_absolute() or ".." in member_path.parts:
                    raise ValueError(f"Unsafe path '{member.name}' encountered in sealed bundle.")
                if not member.isfile():
                    raise ValueError(f"Unsupported tar entry type: {member.name}")

                extracted = archive.extractfile(member)
                if extracted is None:
                    continue

                with extracted:
                    lower_name = member.name.lower()
                    if checkpoint is None and any(lower_name.endswith(ext) for ext in CHECKPOINT_EXTENSIONS):
                        # torch.load requires a seekable file-like object, so we need to
                        # read the stream into memory first
                        checkpoint_data = extracted.read()
                        checkpoint = torch.load(io.BytesIO(checkpoint_data), map_location="cpu")
                        checkpoint_name = member.name
                        # Release the raw bytes data immediately after loading
                        # (torch.load has already deserialized it into Python objects,
                        # and the BytesIO object will be garbage collected automatically)
                        del checkpoint_data
                    elif class_names is None and (
                        lower_name in CLASS_NAME_JSON_CANDIDATES
                        or lower_name.endswith(".json")
                        or lower_name in CLASS_NAME_TXT_CANDIDATES
                        or lower_name.endswith(".txt")
                    ):
                        data = extracted.read()
                        parsed = _parse_class_names_json(data) if lower_name.endswith(".json") else None
                        if parsed is None:
                            parsed = _parse_class_names_text(data)
                        class_names = parsed or class_names
                    else:
                        # Drain the file to keep the tar stream aligned.
                        while extracted.read(1024 * 1024):
                            pass

        stream.verify(tar_sha)

        if checkpoint is None:
            raise FileNotFoundError("No checkpoint (.pth/.pt) found in sealed bundle.")

        return BundleArtifacts(
            checkpoint=checkpoint,
            checkpoint_name=checkpoint_name,
            class_names=class_names,
            header=header,
        )


def _parse_class_names_text(data: bytes) -> Optional[List[str]]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return None
    names = [line.strip() for line in text.splitlines() if line.strip()]
    return names or None


def _parse_class_names_json(data: bytes) -> Optional[List[str]]:
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if isinstance(payload, list):
        names = [str(item).strip() for item in payload if str(item).strip()]
        return names or None
    return None


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
    if args.checkpoint and args.sealed:
        raise ValueError("Please specify either --checkpoint or --sealed, not both.")

    device = get_device(args.device)
    if not args.quiet:
        print(f"Using device: {device}")

    images = collect_images(args.inputs)
    bundle_artifacts: Optional[BundleArtifacts] = None
    if args.sealed:
        bundle_artifacts = load_sealed_bundle(
            Path(args.sealed),
            args.sealed_passphrase,
            args.sealed_key,
        )
        if not args.quiet:
            created = bundle_artifacts.header.get("created_at", "unknown")
            print(f"Decrypted sealed bundle '{args.sealed}' (created_at={created}).")

    if args.class_names:
        class_names = load_class_names(args.class_names)
    elif bundle_artifacts and bundle_artifacts.class_names:
        class_names = bundle_artifacts.class_names
    else:
        class_names = load_class_names("")

    model = build_model(args, example_classes=len(class_names))

    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if checkpoint_path.is_file():
            load_checkpoint(checkpoint_path, model)
        else:
            raise FileNotFoundError(f"Checkpoint '{checkpoint_path}' not found.")
    elif bundle_artifacts:
        checkpoint_obj = bundle_artifacts.checkpoint
        if checkpoint_obj is None:
            raise FileNotFoundError("Sealed bundle did not contain a checkpoint.")
        state_dict = checkpoint_obj.get("model", checkpoint_obj)
        model.load_state_dict(state_dict)
        if not args.quiet:
            src = bundle_artifacts.checkpoint_name or "<unknown>"
            print(f"Loaded checkpoint '{src}' from sealed bundle.")
        # Drop decrypted checkpoint to avoid keeping an extra plaintext copy in memory.
        del state_dict
        del checkpoint_obj
        bundle_artifacts.checkpoint = None

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
