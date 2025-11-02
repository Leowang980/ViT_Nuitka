# ViT Nuitka Project Guide

This repository contains a Vision Transformer (ViT) training and inference stack that can be compiled with Nuitka into a Python extension module. The compiled artifact exposes a minimal API surface (`train_vit`, `infer_vit`, `encrypt`) while keeping the implementation details sealed.

---

## Environment
- Python 3.12 (matches the sample binary `api.cpython-312-*.so`)
- Dependencies:
  ```bash
  python -m pip install torch torchvision numpy pillow cryptography
  ```

---

## 1. Rebuilding with Nuitka
1. Install Nuitka if necessary:
   ```bash
   python -m pip install nuitka
   ```
1. Build from the project root (outputs to `dist/` by default):
   ```bash
   bash build_nuitka.sh
   ```
   You can pass extra Nuitka flags, for example:
   ```bash
   bash build_nuitka.sh dist --onefile
   ```
2. The compiled module exports `train_vit`, `infer_vit`, and `encrypt`. Example usage (assuming the binary is emitted as `dist/api.so`):
   ```python
   import api

   api.train_vit(epochs=1, batch_size=32)
   results = api.infer_vit("tests/image.png", checkpoint="outputs/best.pth")
   meta = api.encrypt(["outputs/best.pth"], out="dist/model_bundle.enc")
   ```
3. Distribute only the contents of `dist/` alongside the runtime dependencies.

---

## 2. Binary Module Usage

### 2.1 Importing the Module
Place the Nuitka-generated binary (e.g., `api.cpython-312-darwin.so`) on the Python path (project root or `site-packages`), then:
```python
import api
```

### 2.2 Exposed APIs
- `train_vit(**options) -> float`: run the full training loop and return the best validation accuracy.
- `infer_vit(inputs, **options) -> Dict[str, List[Tuple[str, float]]]`: run inference on one or more images and return top-k predictions.
- `encrypt(inputs, **options) -> Dict[str, Any]`: package files into an encrypted `.enc` bundle and return metadata (including the base64 content key when no passphrase is supplied).

All other implementation details remain embedded inside the compiled module.

### 2.3 Encrypting Artifacts
Python usage:
```python
import api

meta = api.encrypt(
    ["outputs/best.pth", "class_names.txt"],
    out="dist/model_bundle.enc",
    passphrase="keep-me-safe",       # optional
    chunk_size_mb=32,                # optional, default 64
)
print(meta["output"], meta["content_key_b64"])
```

Command-line equivalent:
```bash
python encrypt.py --out dist/model_bundle.enc outputs/best.pth class_names.txt --passphrase keep-me-safe
```

If you skip the passphrase, provide the base64 content key during inference via `--sealed-key` or the `MODEL_SEALER_KC_B64` environment variable.

### 2.4 Training Example
```python
import api

best_acc = api.train_vit(
    data_path="./data",
    epochs=5,
    batch_size=64,
    model="vit_small",
    amp=True,
    output_dir="./outputs",
)
print(f"Best validation accuracy: {best_acc * 100:.2f}%")
```

Common options:
- `data_path`: dataset root (default `./data`).
- `num_classes`: override class count when not inferred from CIFAR-10.
- `device`: `"cuda"`, `"cpu"`, `"mps"`, or `"auto"` (default).
- `resume`: checkpoint to resume training.

### 2.5 Inference Example
```python
import api

results = api.infer_vit(
    ["examples/dog.png", "examples/cat.jpg"],
    sealed="dist/model_bundle.enc",
    sealed_passphrase="keep-me-safe",  # or sealed_key="base64-of-content-key"
    model="vit_small",
    image_size=224,
    topk=3,
)

for path, predictions in results.items():
    print(path)
    for idx, (label, score) in enumerate(predictions, start=1):
        print(f"  Top {idx}: {label} ({score * 100:.2f}%)")
```

Command-line usage:
```bash
python deploy.py ./images --sealed dist/model_bundle.enc --sealed-passphrase keep-me-safe
```

Useful options:
- `class_names`: label text file (one class per line); defaults to CIFAR-10 labels.
- `amp`: enable mixed-precision inference on CUDA.
- `quiet`: suppress console output (default `True`).
- `sealed` / `sealed_passphrase` / `sealed_key`: load encrypted bundles without touching disk with plaintext files.

### 2.6 Security Notes
- AES-256-GCM is used for chunked encryption. Decryption streams chunks directly into memory; plaintext TAR data and checkpoints are never written to disk.
- Tar extraction validates paths, forbids symlinks/hardlinks, and rejects unexplored entry types.
- When no passphrase is supplied, safeguard the returned `content_key_b64` and transport it out of band.
- Training outputs (`last.pth`, `best.pth`) are still written to `output_dir`. Encrypt them before distribution if confidentiality is required.
- The compiled module does not bundle third-party dependencies. Install `torch`, `torchvision`, `pillow`, `cryptography`, etc., on the target system.
