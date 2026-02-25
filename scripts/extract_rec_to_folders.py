#!/usr/bin/env python3
"""
Extract CASIA-WebFace from record format (train.rec / train.lst) into folder-per-identity
with .jpg files: data/casia_webface_extracted/0000045/001.jpg etc.

No extra dependencies (reads MXNet-style .rec sequentially). Run from project root:
  python scripts/extract_rec_to_folders.py
  python scripts/extract_rec_to_folders.py --limit 1000
"""

from __future__ import annotations

import argparse
import re
import struct
from pathlib import Path

# MXNet recordio magic as stored in file (big-endian)
RECORDIO_MAGIC_BYTES = bytes([0x0A, 0x23, 0xD7, 0xCE])


def project_root() -> Path:
    root = Path(__file__).resolve().parent.parent
    if not (root / "README.md").exists():
        raise SystemExit("Run from project root (directory containing README.md).")
    return root


def find_jpeg_in_record(data: bytes) -> bytes | None:
    start = data.find(b"\xff\xd8\xff")
    if start < 0:
        return None
    end = data.find(b"\xff\xd9", start)
    if end < 0:
        return None
    return data[start : end + 2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract train.rec/train.lst to folder-per-identity .jpg")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Directory containing train.rec, train.lst (default: data/casia_webface)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output root for extracted folders (default: data/casia_webface_extracted)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of images to extract (default: all)",
    )
    args = parser.parse_args()

    root = project_root()
    input_dir = args.input_dir or root / "data" / "casia_webface"
    output_root = args.output or root / "data" / "casia_webface_extracted"

    rec_path = input_dir / "train.rec"
    lst_path = input_dir / "train.lst"
    if not rec_path.exists():
        raise SystemExit(f"Not found: {rec_path}. Run get_dataset.py --kaggle first.")
    if not lst_path.exists():
        raise SystemExit(f"Not found: {lst_path}.")

    # Parse .lst: index, path, label_id, ...
    lines = lst_path.read_text(encoding="utf-8", errors="replace").strip().splitlines()
    entries = []
    for line in lines:
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        path_str = parts[1]
        match = re.search(r"/([^/]+)/([^/]+\.(?:jpg|jpeg|png))$", path_str, re.I)
        if match:
            id_folder, filename = match.group(1), match.group(2)
        else:
            id_folder = f"id_{len(entries):07d}"
            filename = f"img_{len(entries):05d}.jpg"
        entries.append((id_folder, filename))

    print(f"LST entries: {len(entries)}. Output: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    limit = args.limit or len(entries)

    magic_bytes = RECORDIO_MAGIC_BYTES
    rec_file = open(rec_path, "rb")
    try:
        record_index = 0
        extracted = 0
        chunk_size = 2 * 1024 * 1024
        buffer = b""
        while record_index < limit:
            if len(buffer) < 8:
                more = rec_file.read(chunk_size)
                if not more:
                    break
                buffer += more
            pos = buffer.find(magic_bytes)
            if pos < 0:
                buffer = buffer[-7:]
                continue
            buffer = buffer[pos:]
            if len(buffer) < 8:
                buffer += rec_file.read(chunk_size)
                if len(buffer) < 8:
                    break
            jpeg_start = buffer.find(b"\xff\xd8\xff", 8)
            if jpeg_start < 0:
                buffer += rec_file.read(chunk_size)
                if len(buffer) > 8 * 1024 * 1024:
                    buffer = buffer[-8:]
                continue
            jpeg_end = buffer.find(b"\xff\xd9", jpeg_start)
            if jpeg_end < 0:
                buffer += rec_file.read(chunk_size)
                if len(buffer) > 8 * 1024 * 1024:
                    buffer = buffer[: jpeg_start + 8]
                continue
            jpeg_data = buffer[jpeg_start : jpeg_end + 2]
            buffer = buffer[jpeg_end + 2 :]
            if record_index >= len(entries):
                break
            id_folder, filename = entries[record_index]
            out_dir = output_root / id_folder
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / filename).write_bytes(jpeg_data)
            extracted += 1
            record_index += 1
            if extracted % 5000 == 0:
                print(f"  Extracted {extracted} images...")
    finally:
        rec_file.close()

    print(f"Done. Extracted {extracted} images under {output_root}")


if __name__ == "__main__":
    main()
