"""Inspect hierarchical SigLIP routing for a directory of local images."""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ruff: noqa: E402 - repository root must be added before local imports.
from app.models.loader import get_engine
from app.services.image_validation import InvalidImage, validate_image

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, dest="directory")
    args = parser.parse_args()

    root = Path(args.directory)
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")

    engine = get_engine()
    files = sorted(path for path in root.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if not files:
        raise SystemExit("No JPG/PNG/WebP images found.")

    for path in files:
        print(f"\n=== {path.name} ===")
        mime, _ = mimetypes.guess_type(path)
        try:
            validated = validate_image(path.read_bytes(), mime or "")
        except InvalidImage as exc:
            print(json.dumps({"error": str(exc)}, indent=2))
            continue

        diagnostics = engine.gate.diagnose(
            validated.rgb, quality_weak=validated.quality_weak
        )
        print(f"dimensions: {validated.rgb.size[0]}x{validated.rgb.size[1]}")
        print(f"quality_weak: {validated.quality_weak}")
        print(json.dumps(diagnostics, indent=2, ensure_ascii=False))
        result = engine.predict(validated)
        print("public_result:")
        print(json.dumps(result.model_dump(), indent=2, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
