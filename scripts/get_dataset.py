#!/usr/bin/env python3
"""
Setup CASIA-WebFace dataset for the Face Recognition Under Adversarial Attacks project.

The dataset is NOT hosted in this repo. Each team member must obtain it themselves
(see docs/DATASET.md). This script can download via Kaggle (kagglehub), unpack a
local archive, or copy an existing directory into data/casia_webface/.

Usage:
  # From project root (aibs_project/):
  python scripts/get_dataset.py --kaggle
  python scripts/get_dataset.py --kaggle --symlink
  python scripts/get_dataset.py --archive path/to/casia_webface.zip
  python scripts/get_dataset.py --extracted path/to/casia_webface_folder
  python scripts/get_dataset.py --extracted path/to/casia_webface_folder --symlink
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path

# Expected approximate counts (CASIA-WebFace)
EXPECTED_IDS = 10_575
EXPECTED_IMAGES = 494_414
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}


def project_root() -> Path:
    root = Path(__file__).resolve().parent.parent
    if not (root / "README.md").exists():
        raise SystemExit("Expected to run from repo with README.md (project root).")
    return root


def target_dir(root: Path) -> Path:
    return root / "data" / "casia_webface"


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def count_identities_and_images(dir_path: Path) -> tuple[int, int]:
    """Count identity folders and total image files. Only one level of identity folders."""
    n_ids = 0
    n_images = 0
    for item in dir_path.iterdir():
        if not item.is_dir():
            continue
        n_ids += 1
        for f in item.iterdir():
            if f.is_file() and is_image(f):
                n_images += 1
    return n_ids, n_images


def find_identity_root(archive_root: Path) -> Path:
    """
    Archive may be:
      - root/id_1/, root/id_2/, ...  -> return root
      - root/casia_webface/id_1/, ... -> return root/casia_webface
      - root/casia-webface/ (Kaggle) may hold identity dirs or record files
    """
    subdirs = [d for d in archive_root.iterdir() if d.is_dir()]
    if not subdirs:
        return archive_root
    # If exactly one subdir and it looks like a container (many subdirs), use it
    if len(subdirs) == 1:
        one = subdirs[0]
        subsub = [x for x in one.iterdir() if x.is_dir()]
        if len(subsub) > 10:  # likely identity folders
            return one
    # Kaggle: root may have "casia-webface" (and "eval"); prefer that as content root
    for d in subdirs:
        if d.name in ("casia-webface", "casia_webface"):
            return d
    return archive_root


def is_record_format(path: Path) -> bool:
    """True if this path contains MXNet/RecordIO style files (train.rec, etc.)."""
    return (path / "train.rec").exists()


def validate_layout(path: Path) -> bool:
    path = path.resolve()
    if not path.is_dir():
        print(f"  Error: {path} is not a directory.")
        return False
    root = find_identity_root(path)
    n_ids, n_images = count_identities_and_images(root)
    print(f"  Found {n_ids} identities and {n_images} images.")
    if n_ids < 100:
        print(f"  Warning: expected ~{EXPECTED_IDS} identities. Layout might be wrong.")
    if n_images < 1000:
        print(f"  Warning: expected ~{EXPECTED_IMAGES} images. Layout might be wrong.")
    return n_ids >= 10 and n_images >= 100


def unpack_archive(archive_path: Path, dest: Path) -> Path:
    """Unpack zip or tar.* into dest. Returns path to unpacked root (may be dest or dest/subdir)."""
    archive_path = archive_path.resolve()
    if not archive_path.exists():
        raise SystemExit(f"Archive not found: {archive_path}")

    dest = dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)

    if zipfile.is_zipfile(archive_path):
        print(f"Unpacking zip: {archive_path}")
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(dest)
    elif tarfile.is_tarfile(archive_path):
        print(f"Unpacking tar: {archive_path}")
        with tarfile.open(archive_path, "r:*") as tf:
            tf.extractall(dest)
    else:
        raise SystemExit(f"Unknown archive format: {archive_path}")

    # Unpacked content may be dest/ or dest/single_folder/
    unpacked_root = find_identity_root(dest)
    return unpacked_root


def copy_or_link_source_to_target(source_root: Path, target: Path, symlink: bool) -> None:
    """Copy or symlink identity folders from source_root into target/casia_webface layout."""
    target = target.resolve()
    target.mkdir(parents=True, exist_ok=True)
    identity_root = find_identity_root(source_root)

    for item in identity_root.iterdir():
        if not item.is_dir():
            continue
        dest_item = target / item.name
        if dest_item.exists():
            print(f"  Skip (exists): {dest_item.name}")
            continue
        if symlink:
            dest_item.symlink_to(item.resolve(), target_is_directory=True)
        else:
            shutil.copytree(item, dest_item)
    print(f"  Done. Dataset at: {target}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Set up CASIA-WebFace in data/casia_webface/ (see docs/DATASET.md)."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--archive",
        type=Path,
        metavar="PATH",
        help="Path to CASIA-WebFace zip or tar.gz (after you obtain it from official source).",
    )
    group.add_argument(
        "--extracted",
        type=Path,
        metavar="PATH",
        help="Path to already-extracted CASIA-WebFace folder (one folder per identity).",
    )
    group.add_argument(
        "--kaggle",
        action="store_true",
        help="Download via Kaggle (kagglehub) from debarghamitraroy/casia-webface and set up in data/.",
    )
    parser.add_argument(
        "--symlink",
        action="store_true",
        help="With --extracted or --kaggle: create symlinks instead of copying (saves disk space).",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip validation of identity/image counts.",
    )
    args = parser.parse_args()

    root = project_root()
    target = target_dir(root)

    if args.archive:
        # Unpack into staging, then move identity folders into data/casia_webface
        staging = root / "data" / "casia_webface_staging"
        staging.mkdir(parents=True, exist_ok=True)
        try:
            unpacked_root = unpack_archive(args.archive, staging)
            target.mkdir(parents=True, exist_ok=True)
            for item in unpacked_root.iterdir():
                dest = target / item.name
                if dest.exists():
                    shutil.rmtree(dest) if dest.is_dir() else dest.unlink()
                shutil.move(str(item), str(dest))
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    elif args.kaggle:
        try:
            import kagglehub
        except ImportError:
            raise SystemExit(
                "kagglehub is required for --kaggle. Install with: pip install kagglehub"
            )
        print("Downloading CASIA-WebFace from Kaggle (debarghamitraroy/casia-webface)...")
        kaggle_path = Path(kagglehub.dataset_download("debarghamitraroy/casia-webface"))
        print("Path to dataset files:", kaggle_path)
        content_root = find_identity_root(kaggle_path)
        if is_record_format(content_root):
            # Dataset is in record format (train.rec); copy whole dir to target
            target.mkdir(parents=True, exist_ok=True)
            for item in content_root.iterdir():
                dest_item = target / item.name
                if dest_item.exists():
                    if dest_item.is_symlink():
                        dest_item.unlink()
                    elif dest_item.is_dir():
                        shutil.rmtree(dest_item)
                    else:
                        dest_item.unlink()
                if args.symlink:
                    dest_item.symlink_to(item.resolve(), target_is_directory=item.is_dir())
                elif item.is_dir():
                    shutil.copytree(item, dest_item)
                else:
                    shutil.copy2(item, dest_item)
            # Also copy eval/ from Kaggle root to data/eval (sibling of casia_webface)
            eval_src = kaggle_path / "eval"
            if eval_src.is_dir():
                eval_dest = root / "data" / "eval"
                if eval_dest.exists():
                    if eval_dest.is_symlink():
                        eval_dest.unlink()
                    else:
                        shutil.rmtree(eval_dest)
                if args.symlink:
                    eval_dest.symlink_to(eval_src.resolve(), target_is_directory=True)
                else:
                    shutil.copytree(eval_src, eval_dest)
                print("  Copied eval/ to data/eval (evaluation bins).")
            print("  Done. Dataset at:", target)
            print("  Note: This Kaggle version uses record format (train.rec). Use an MXNet/RecordIO loader or see docs for folder-per-identity alternatives.")
            if not args.no_validate:
                args.no_validate = True
                print("  Validation skipped (record format).")
        else:
            copy_or_link_source_to_target(kaggle_path, target, args.symlink)
    else:
        copy_or_link_source_to_target(args.extracted, target, args.symlink)

    if not args.no_validate:
        print("Validating layout...")
        if not validate_layout(target):
            print("  Validation had warnings. Check docs/DATASET.md for expected layout.")
            sys.exit(1)
    print("Dataset ready at:", target)


if __name__ == "__main__":
    main()
