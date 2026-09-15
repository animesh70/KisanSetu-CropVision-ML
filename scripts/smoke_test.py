"""Exercise an already running service with a local image, without exposing the key."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import httpx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--key", required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--crop-hint")
    args = parser.parse_args()
    image_path: Path = args.image
    suffix_mime = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }
    mime = suffix_mime.get(image_path.suffix.lower(), "application/octet-stream")
    with httpx.Client(timeout=60) as client:
        health = client.get(f"{args.url.rstrip('/')}/health")
        health.raise_for_status()
        response = client.post(
            f"{args.url.rstrip('/')}/predict",
            headers={"X-KisanSetu-Key": args.key},
            files={"image": (image_path.name, image_path.read_bytes(), mime)},
            data={"cropHint": args.crop_hint or ""},
        )
    print(
        json.dumps(
            {"health": health.json(), "status": response.status_code, "result": response.json()},
            indent=2,
        )
    )
    response.raise_for_status()


if __name__ == "__main__":
    main()
