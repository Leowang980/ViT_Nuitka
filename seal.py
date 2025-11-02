#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import base64
import binascii
import hashlib
import io
import json
import os
import shutil
import struct
import sys
import tarfile
import tempfile
import time
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
from cryptography.hazmat.primitives import constant_time
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend

MAGIC = b"ENC1"  # file magic/version
HEADER_LEN_FMT = "!Q"  # 8-byte unsigned big-endian

# ---------- helpers ----------

def sha256_file(fp, chunk_size=1024 * 1024) -> str:
    h = hashlib.sha256()
    while True:
        data = fp.read(chunk_size)
        if not data:
            break
        h.update(data)
    fp.seek(0)
    return h.hexdigest()

def sha256_path(path, chunk_size=1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def rand_bytes(n: int) -> bytes:
    return os.urandom(n)

def b64e(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")

def b64d(s: str) -> bytes:
    return base64.b64decode(s.encode("ascii"))

def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def human_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    v = float(n)
    while v >= 1024 and i < len(units) - 1:
        v /= 1024.0
        i += 1
    return f"{v:.2f} {units[i]}"

# ---------- header dataclass ----------

@dataclass
class EncHeader:
    version: int
    created_at: str
    algorithm: str  # "AES-256-GCM-chunked"
    chunk_size: int
    total_chunks: int
    tar_size: int
    tar_sha256: str
    files: List[dict]          # [{path, size, sha256}, ...]
    kc_wrapped: Optional[str]  # base64 of wrapped Kc (if passphrase used)
    kc_wrap_alg: Optional[str] # "AES-256-GCM"
    kdf: Optional[dict]        # {"name":"scrypt", "salt":b64, "n":..., "r":..., "p":...}

# ---------- tar utilities ----------

def build_tar(inputs: List[str], tar_path: str) -> Tuple[int, str, List[dict]]:
    """
    Build a tar at tar_path containing all inputs.
    Returns: (tar_size, tar_sha256, files_meta)
    """
    files_meta = []
    with tarfile.open(tar_path, "w") as tf:
        for p in inputs:
            arcname = os.path.basename(p)
            st = os.stat(p)
            file_sha = sha256_path(p)
            files_meta.append({"path": arcname, "size": st.st_size, "sha256": file_sha})
            tf.add(p, arcname=arcname)
    # compute tar hash
    with open(tar_path, "rb") as f:
        tar_sha = sha256_file(f)
        f.seek(0, os.SEEK_END)
        tar_size = f.tell()
    return tar_size, tar_sha, files_meta

# ---------- key wrap (passphrase optional) ----------

def derive_kek_from_passphrase(passphrase: str, salt: bytes, n=2**15, r=8, p=1, length=32) -> bytes:
    kdf = Scrypt(salt=salt, length=length, n=n, r=r, p=p, backend=default_backend())
    return kdf.derive(passphrase.encode("utf-8"))

def wrap_kc_with_passphrase(kc: bytes, passphrase: str) -> Tuple[dict, bytes, bytes, bytes]:
    """
    Returns (kdf_info, nonce, wrapped, salt)
    wrapped = AESGCM(KEK).encrypt(nonce, kc, aad=None)
    """
    salt = rand_bytes(16)
    kek = derive_kek_from_passphrase(passphrase, salt)
    nonce = rand_bytes(12)
    wrapped = AESGCM(kek).encrypt(nonce, kc, None)
    kdf_info = {"name": "scrypt", "salt": b64e(salt), "n": 2**15, "r": 8, "p": 1}
    return kdf_info, nonce, wrapped, salt

def unwrap_kc_with_passphrase(kc_wrapped_b64: str, passphrase: str, kdf_info: dict, nonce_b64: str) -> bytes:
    salt = b64d(kdf_info["salt"])
    kek = derive_kek_from_passphrase(passphrase, salt, n=kdf_info["n"], r=kdf_info["r"], p=kdf_info["p"])
    nonce = b64d(nonce_b64)
    wrapped = b64d(kc_wrapped_b64)
    return AESGCM(kek).decrypt(nonce, wrapped, None)

# ---------- chunked AES-GCM ----------

def encrypt_tar_to_enc(
    tar_path: str,
    out_path: str,
    chunk_size: int,
    kc: bytes,
    passphrase: Optional[str]
):
    tar_size = os.path.getsize(tar_path)
    total_chunks = (tar_size + chunk_size - 1) // chunk_size

    # Prepare header
    with open(tar_path, "rb") as f:
        tar_sha = sha256_file(f)

    # Optional key wrapping
    if passphrase:
        kdf_info, kc_nonce, kc_wrapped, _salt = wrap_kc_with_passphrase(kc, passphrase)
        kc_wrapped_b64 = b64e(kc_wrapped)
        kc_nonce_b64 = b64e(kc_nonce)
        kc_wrap_alg = "AES-256-GCM"
    else:
        kdf_info = None
        kc_wrapped_b64 = None
        kc_nonce_b64 = None
        kc_wrap_alg = None

    # File list metadata (recompute here for header)
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.close()
    try:
        # We rebuilt tar earlier to get file list; here reuse metadata by re-tarring for correctness
        pass
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass

    # Since we already computed files_meta in build_tar, re-read quickly:
    # We'll reconstruct the list by a light pass: this info is best returned by build_tar
    # -> We'll re-build it again to get files list; but simpler: call tarfile to list members.
    files_meta = []
    with tarfile.open(tar_path, "r") as tf:
        for m in tf.getmembers():
            if not m.isfile():
                continue
            files_meta.append({"path": m.name, "size": m.size})

    # For each file we didn't compute sha in this quick path; it's optional but nice to have.
    # To keep header compact for huge archives, we can omit per-file sha here.
    for entry in files_meta:
        entry.setdefault("sha256", None)

    header = EncHeader(
        version=1,
        created_at=now_iso(),
        algorithm="AES-256-GCM-chunked",
        chunk_size=chunk_size,
        total_chunks=total_chunks,
        tar_size=tar_size,
        tar_sha256=tar_sha,
        files=files_meta,
        kc_wrapped=kc_wrapped_b64,
        kc_wrap_alg=kc_wrap_alg,
        kdf=kdf_info,
    )
    header_dict = asdict(header)
    # store also kc_nonce if present
    if passphrase:
        header_dict["kc_nonce"] = kc_nonce_b64

    header_json = json.dumps(header_dict, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

    aes = AESGCM(kc)

    with open(out_path, "wb") as out_f, open(tar_path, "rb") as in_f:
        # write magic + header
        out_f.write(MAGIC)
        out_f.write(struct.pack(HEADER_LEN_FMT, len(header_json)))
        out_f.write(header_json)

        # stream encrypt per chunk
        idx = 0
        while True:
            chunk = in_f.read(chunk_size)
            if not chunk:
                break
            nonce = rand_bytes(12)
            # AAD binds to (idx, total_chunks, tar_sha256) to prevent reordering/truncation
            aad = json.dumps(
                {"i": idx, "n": total_chunks, "h": tar_sha},
                separators=(",", ":")
            ).encode("utf-8")
            ct = aes.encrypt(nonce, chunk, aad)  # ciphertext||tag
            out_f.write(nonce)
            out_f.write(ct)
            idx += 1

class EncryptedTarStream(io.RawIOBase):
    """Lazy reader that decrypts chunked TAR payloads on demand."""

    def __init__(
        self,
        file_handle,
        *,
        file_size: int,
        chunk_size: int,
        total_chunks: int,
        tar_sha: str,
        aes: AESGCM,
    ) -> None:
        super().__init__()
        self._fh = file_handle
        self._file_size = file_size
        self._chunk_size = chunk_size
        self._total_chunks = total_chunks
        self._tar_sha = tar_sha
        self._aes = aes
        self._buffer = bytearray()
        self._chunk_index = 0
        self._digest = hashlib.sha256()
        self._finished = False

    def readable(self) -> bool:
        return True

    def _aad(self, idx: int) -> bytes:
        return json.dumps({"i": idx, "n": self._total_chunks, "h": self._tar_sha}, separators=(",", ":")).encode(
            "utf-8"
        )

    def _fill_buffer(self) -> None:
        if self._finished or self._chunk_index >= self._total_chunks:
            self._finished = True
            return

        nonce = self._fh.read(12)
        if len(nonce) != 12:
            raise ValueError("Broken ENC: nonce missing or truncated.")

        if self._chunk_index < self._total_chunks - 1:
            to_read = self._chunk_size + 16
        else:
            current_pos = self._fh.tell()
            to_read = self._file_size - current_pos
            if to_read <= 0:
                raise ValueError("Broken ENC: final chunk length invalid.")

        ciphertext = self._fh.read(to_read)
        if len(ciphertext) != to_read:
            raise ValueError("Broken ENC: ciphertext truncated.")

        plaintext = self._aes.decrypt(nonce, ciphertext, self._aad(self._chunk_index))
        self._digest.update(plaintext)
        self._buffer.extend(plaintext)
        self._chunk_index += 1
        if self._chunk_index >= self._total_chunks:
            self._finished = True

    def read(self, size: int = -1) -> bytes:
        if size == 0:
            return b""

        if size < 0:
            while not self._finished:
                self._fill_buffer()
            data = bytes(self._buffer)
            self._buffer.clear()
            return data

        while len(self._buffer) < size and not self._finished:
            self._fill_buffer()
        if len(self._buffer) == 0 and self._finished:
            return b""
        data = bytes(self._buffer[:size])
        del self._buffer[:size]
        return data

    def readinto(self, b) -> int:
        data = self.read(len(b))
        if not data:
            return 0
        b[: len(data)] = data
        return len(data)

    def verify(self, expected_sha: str) -> None:
        while not self._finished:
            self._fill_buffer()
            self._buffer.clear()

        if self._buffer:
            # Tarfile should have consumed the entire stream; leftover implies malformed archive.
            raise ValueError("Sealed bundle contains trailing data after TAR extraction.")
        if self._chunk_index != self._total_chunks:
            raise ValueError("Sealed bundle truncated during decryption.")
        calc = self._digest.hexdigest()
        if not constant_time.bytes_eq(calc.encode("ascii"), expected_sha.encode("ascii")):
            raise ValueError("Integrity check failed: tar hash mismatch.")


def decrypt_enc_to_dir(enc_path: str, out_dir: str, passphrase: Optional[str]):
    bundle_size = os.path.getsize(enc_path)
    with open(enc_path, "rb") as f:
        magic = f.read(len(MAGIC))
        if magic != MAGIC:
            raise ValueError("Invalid magic/version")
        header_len_bytes = f.read(struct.calcsize(HEADER_LEN_FMT))
        if len(header_len_bytes) != struct.calcsize(HEADER_LEN_FMT):
            raise ValueError("Malformed header length")
        header_len = struct.unpack(HEADER_LEN_FMT, header_len_bytes)[0]
        header_json = f.read(header_len)
        if len(header_json) != header_len:
            raise ValueError("Malformed header payload")
        header = json.loads(header_json.decode("utf-8"))

        chunk_size = int(header["chunk_size"])
        total_chunks = int(header["total_chunks"])
        tar_sha = header["tar_sha256"]

        if header.get("kc_wrapped"):
            if not passphrase:
                raise ValueError("This file requires a passphrase to unwrap Kc.")
            kc = unwrap_kc_with_passphrase(
                header["kc_wrapped"],
                passphrase,
                header["kdf"],
                header["kc_nonce"],
            )
        else:
            kc_b64 = os.environ.get("MODEL_SEALER_KC_B64")
            if not kc_b64:
                raise ValueError("No wrapped Kc in header. Provide Kc via env MODEL_SEALER_KC_B64 (base64).")
            kc = b64d(kc_b64)

        aes = AESGCM(kc)
        stream = EncryptedTarStream(
            f,
            file_size=bundle_size,
            chunk_size=chunk_size,
            total_chunks=total_chunks,
            tar_sha=tar_sha,
            aes=aes,
        )

        os.makedirs(out_dir, exist_ok=True)
        with tarfile.open(fileobj=stream, mode="r|*") as tf:
            safe_extract_tar(tf, out_dir)
        stream.verify(tar_sha)


def _validate_member_path(base_dir: str, member_name: str) -> str:
    dest_path = os.path.abspath(os.path.join(base_dir, member_name))
    base_dir_abs = os.path.abspath(base_dir)
    if not dest_path.startswith(base_dir_abs + os.sep) and dest_path != base_dir_abs:
        raise ValueError(f"Blocked path traversal in tar member: {member_name}")
    return dest_path


def safe_extract_tar(tar: tarfile.TarFile, path: str):
    for member in tar:
        if member.issym() or member.islnk():
            raise ValueError(f"Blocked link entry in tar member: {member.name}")

        dest_path = _validate_member_path(path, member.name)

        if member.isdir():
            os.makedirs(dest_path, exist_ok=True)
            continue
        if not member.isfile():
            raise ValueError(f"Unsupported tar entry type for member: {member.name}")

        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        extracted = tar.extractfile(member)
        if extracted is None:
            raise ValueError(f"Failed to read member: {member.name}")
        with extracted:
            with open(dest_path, "wb") as out_f:
                shutil.copyfileobj(extracted, out_f)
        if member.mode is not None:
            try:
                os.chmod(dest_path, member.mode)
            except OSError:
                pass

# ---------- CLI ----------

def cmd_encrypt(args):
    inputs = args.inputs
    if not inputs:
        print("No input files.", file=sys.stderr)
        sys.exit(2)
    for p in inputs:
        if not os.path.isfile(p):
            print(f"Not a file: {p}", file=sys.stderr)
            sys.exit(2)

    # 1) build tar (on disk, to support huge size)
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tar_path = tmp.name
    try:
        print("[+] Building TAR from inputs...")
        tar_size, tar_sha, files_meta = build_tar(inputs, tar_path)
        print(f"    TAR size = {human_bytes(tar_size)}, sha256={tar_sha[:16]}...")

        # 2) generate content key (Kc)
        kc = rand_bytes(32)
        print(f"[+] Generated content key Kc (base64): {b64e(kc)}")
        if args.passphrase:
            print("[+] Wrapping Kc with passphrase using scrypt + AES-256-GCM")
        else:
            print("[!] No passphrase used. Keep Kc safe and inject at runtime!")

        # 3) encrypt chunked
        chunk_size = args.chunk_size_mb * 1024 * 1024
        print(f"[+] Encrypting in chunks of {args.chunk_size_mb} MiB ...")
        encrypt_tar_to_enc(
            tar_path=tar_path,
            out_path=args.out,
            chunk_size=chunk_size,
            kc=kc,
            passphrase=args.passphrase,
        )
        print(f"[✓] Wrote ENC: {args.out}")
    finally:
        try:
            os.unlink(tar_path)
        except Exception:
            pass

def cmd_decrypt(args):
    enc_path = args.input
    out_dir = args.out_dir
    print(f"[+] Decrypting {enc_path} -> {out_dir}")
    decrypt_enc_to_dir(enc_path, out_dir, args.passphrase)
    print("[✓] Done")

def main():
    ap = argparse.ArgumentParser(description="Large-file model/config encrypter (.enc) with chunked AES-256-GCM.")
    sub = ap.add_subparsers(dest="cmd")

    ap_e = sub.add_parser("encrypt", help="Encrypt files into a single .enc")
    ap_e.add_argument("--out", required=True, help="Output .enc path")
    ap_e.add_argument("--chunk-size-mb", type=int, default=64, help="Chunk size in MiB (default: 64)")
    ap_e.add_argument("--passphrase", default=None, help="Optional passphrase to wrap Kc (use scrypt + AES-GCM)")
    ap_e.add_argument("inputs", nargs="+", help="Input files (e.g., model.pth config.yaml)")
    ap_e.set_defaults(func=cmd_encrypt)

    ap_d = sub.add_parser("decrypt", help="Decrypt .enc into a directory (for verification)")
    ap_d.add_argument("--in", dest="input", required=True, help="Input .enc path")
    ap_d.add_argument("--out-dir", required=True, help="Output directory to extract files")
    ap_d.add_argument("--passphrase", default=None, help="Passphrase if Kc was wrapped")
    ap_d.set_defaults(func=cmd_decrypt)

    args = ap.parse_args()
    if not args.cmd:
        ap.print_help()
        sys.exit(2)
    args.func(args)

if __name__ == "__main__":
    main()
