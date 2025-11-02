"""
Utilities for packaging model artifacts into encrypted bundles.

This module reuses the sealing primitives defined in ``seal.py`` but exposes
an easier to consume API along with a small CLI.  The public helper
``encrypt_model_bundle`` is intended to be used by ``api.encrypt`` so Nuitka
builds only expose a minimal surface area.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from seal import (  # type: ignore
    b64e,
    build_tar,
    encrypt_tar_to_enc,
    human_bytes,
    rand_bytes,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Encrypt model checkpoints and metadata into a sealed bundle."
    )
    parser.add_argument("--out", required=True, help="Output .enc path.")
    parser.add_argument(
        "--chunk-size-mb",
        type=int,
        default=64,
        help="Chunk size for streaming AES-GCM encryption (default: 64 MiB).",
    )
    parser.add_argument(
        "--passphrase",
        default=None,
        help="Optional passphrase used to wrap the content key via scrypt + AES-256-GCM.",
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Input files to include in the sealed bundle (e.g., model.pth class_names.txt).",
    )
    return parser


def _validate_inputs(inputs: Iterable[str]) -> List[str]:
    normalized: List[str] = []
    for item in inputs:
        path = Path(item)
        if not path.is_file():
            raise FileNotFoundError(f"Input file '{path}' does not exist.")
        normalized.append(str(path))
    return normalized


def encrypt_model_bundle(
    inputs: Iterable[str],
    output_path: str,
    *,
    passphrase: Optional[str] = None,
    chunk_size_mb: int = 64,
) -> Dict[str, object]:
    """Encrypt the provided files into ``output_path``.

    Returns metadata describing the produced bundle, including the base64
    content key (if no passphrase is supplied).
    """
    normalized_inputs = _validate_inputs(inputs)
    if chunk_size_mb <= 0:
        raise ValueError("chunk_size_mb must be a positive integer.")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tar_path = tmp.name
    try:
        tar_size, tar_sha, files_meta = build_tar(normalized_inputs, tar_path)
        kc = rand_bytes(32)
        encrypt_tar_to_enc(
            tar_path=tar_path,
            out_path=str(output),
            chunk_size=chunk_size_mb * 1024 * 1024,
            kc=kc,
            passphrase=passphrase,
        )
    finally:
        try:
            os.unlink(tar_path)
        except FileNotFoundError:
            pass

    metadata: Dict[str, object] = {
        "output": str(output),
        "content_key_b64": b64e(kc),
        "tar_size_bytes": tar_size,
        "tar_size_human": human_bytes(tar_size),
        "tar_sha256": tar_sha,
        "files": files_meta,
        "passphrase_protected": bool(passphrase),
    }
    if passphrase:
        # If a passphrase is used the caller does not need the raw content key,
        # but we still return it in case they choose to escrow it separately.
        metadata["note"] = (
            "Content key is wrapped with the provided passphrase. Keep the passphrase safe."
        )
    return metadata


def perform_encryption(args: argparse.Namespace) -> Dict[str, object]:
    return encrypt_model_bundle(
        inputs=args.inputs,
        output_path=args.out,
        passphrase=args.passphrase,
        chunk_size_mb=args.chunk_size_mb,
    )


def main() -> None:
    args = build_parser().parse_args()
    metadata = perform_encryption(args)

    print("[+] Inputs:")
    for entry in metadata["files"]:  # type: ignore[index]
        path = entry["path"]
        size = entry["size"]
        sha256 = entry["sha256"]
        sha_suffix = sha256[:16] + "..." if isinstance(sha256, str) else "n/a"
        print(f"    - {path} ({human_bytes(size)}) sha256={sha_suffix}")

    print(f"[+] TAR size: {metadata['tar_size_human']} (sha256={metadata['tar_sha256'][:16]}...)")
    if metadata.get("passphrase_protected"):
        print("[+] Using passphrase-based wrapping for content key.")
    else:
        print("[!] No passphrase provided. Content key must be protected separately.")

    print(f"[✓] Wrote sealed bundle to {metadata['output']}")
    print(f"[→] Content key (base64): {metadata['content_key_b64']}")


if __name__ == "__main__":
    main()
