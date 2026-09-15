"""Validate pixels, not filenames, before any model sees an upload."""

from __future__ import annotations

import io
import warnings
from dataclasses import dataclass

from PIL import Image, ImageOps, UnidentifiedImageError

from app.config import ACCEPTED_MIME, MAX_IMAGE_PIXELS, MAX_IMAGE_SIDE

Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS

FORMAT_MIME = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}


class InvalidImage(ValueError):
    pass


@dataclass(frozen=True)
class ValidatedImage:
    rgb: Image.Image
    mime: str
    quality_weak: bool


def _quality_weak(image: Image.Image) -> bool:
    # This is only an image-quality warning; it never identifies a crop or disease.
    if min(image.size) < 128:
        return True
    sample = image.convert("L")
    sample.thumbnail((256, 256))
    histogram = sample.histogram()
    count = sum(histogram)
    mean = sum(index * value for index, value in enumerate(histogram)) / count
    variance = sum((index - mean) ** 2 * value for index, value in enumerate(histogram)) / count
    return mean < 24 or mean > 235 or variance < 80


def validate_image(data: bytes, declared_mime: str) -> ValidatedImage:
    if declared_mime not in ACCEPTED_MIME:
        raise InvalidImage("Only JPG, PNG, and WebP images are accepted.")
    if not data:
        raise InvalidImage("The image is empty.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as opened:
                actual_mime = FORMAT_MIME.get(opened.format)
                if actual_mime != declared_mime:
                    raise InvalidImage("The image bytes do not match the declared image format.")
                width, height = opened.size
                if (
                    width <= 0
                    or height <= 0
                    or width > MAX_IMAGE_SIDE
                    or height > MAX_IMAGE_SIDE
                    or width * height > MAX_IMAGE_PIXELS
                ):
                    raise InvalidImage("The image dimensions are too large.")
                opened.load()
                oriented = ImageOps.exif_transpose(opened)
                rgb = oriented.convert("RGB")
                return ValidatedImage(rgb=rgb, mime=actual_mime, quality_weak=_quality_weak(rgb))
    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as exc:
        if isinstance(exc, InvalidImage):
            raise
        raise InvalidImage("The uploaded image is corrupt or unreadable.") from exc
