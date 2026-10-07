#!/usr/bin/env python3
"""Preflight-check a class-folder image dataset without decoding image data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
DEFAULT_DATASET_DIR = Path(__file__).resolve().parents[1] / "data" / "pokemon-128aug"


def _sort_key(path: Path):
    return (path.name.casefold(), path.name)


def is_image_file(path: Path) -> bool:
    """Return whether a file has an accepted image extension."""
    return path.suffix.casefold() in IMAGE_EXTENSIONS


def scan_dataset(dataset_dir=DEFAULT_DATASET_DIR):
    """Return a deterministic summary of the dataset directory layout.

    Immediate subdirectories are classes. The scan only checks file names and
    directory structure; it never opens or decodes image contents.
    """
    root = Path(dataset_dir).expanduser()
    result = {
        "dataset_dir": str(root),
        "valid": False,
        "error": None,
        "classes": [],
        "ignored_files": [],
        "total_images": 0,
        "total_ignored": 0,
    }

    if not root.exists():
        result["error"] = "missing path: %s" % root
        return result
    if not root.is_dir():
        result["error"] = "invalid path (not a directory): %s" % root
        return result

    entries = sorted(root.iterdir(), key=_sort_key)
    if not entries:
        result["error"] = "empty dataset: %s has no entries" % root
        return result

    class_dirs = [entry for entry in entries if entry.is_dir()]
    result["ignored_files"] = [
        entry.name for entry in entries if entry.is_file()
    ]
    result["total_ignored"] += len(result["ignored_files"])

    if not class_dirs:
        result["error"] = "no classes: %s contains no class subdirectories" % root
        return result

    empty_classes = []
    for class_dir in class_dirs:
        files = sorted(
            (entry for entry in class_dir.iterdir() if entry.is_file()),
            key=_sort_key,
        )
        images = [entry for entry in files if is_image_file(entry)]
        ignored = [entry.name for entry in files if not is_image_file(entry)]
        result["classes"].append(
            {
                "name": class_dir.name,
                "image_count": len(images),
                "ignored_files": ignored,
            }
        )
        result["total_images"] += len(images)
        result["total_ignored"] += len(ignored)
        if not images:
            empty_classes.append(class_dir.name)

    if empty_classes:
        result["error"] = "empty classes: %s" % ", ".join(empty_classes)
        return result

    result["valid"] = True
    return result


def render_text(result) -> str:
    """Render a deterministic human-readable summary."""
    lines = ["Dataset preflight: %s" % result["dataset_dir"]]
    if result["valid"]:
        lines.append("Status: VALID")
    else:
        lines.extend(("Status: INVALID", "Error: %s" % result["error"]))

    lines.append("Classes: %d" % len(result["classes"]))
    lines.append("Total images: %d" % result["total_images"])
    lines.append("Total ignored files: %d" % result["total_ignored"])

    if result["ignored_files"]:
        lines.append("Ignored at dataset root: %s" % ", ".join(result["ignored_files"]))
    for class_info in result["classes"]:
        suffix = ""
        if class_info["ignored_files"]:
            suffix = " (ignored: %s)" % ", ".join(class_info["ignored_files"])
        lines.append(
            "  %s: %d image(s)%s"
            % (class_info["name"], class_info["image_count"], suffix)
        )
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Check class-folder dataset structure and image extensions. "
            "Image files are not decoded."
        ),
        allow_abbrev=False,
    )
    parser.add_argument(
        "dataset_dir",
        nargs="?",
        default=str(DEFAULT_DATASET_DIR),
        help="dataset root (default: %(default)s)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print a stable JSON summary",
    )
    args = parser.parse_args(argv)
    result = scan_dataset(args.dataset_dir)

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(render_text(result))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
