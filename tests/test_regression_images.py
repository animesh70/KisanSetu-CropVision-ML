"""Opt-in tests use caller-provided image paths; no private photos are committed."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.models.loader import get_engine
from app.services.image_validation import validate_image


@pytest.mark.parametrize(
    ("path_variable", "image_type", "crop", "message_code"),
    [
        ("CROPVISION_ANIME_IMAGE", "non_crop", None, "NON_CROP"),
        ("CROPVISION_HARVESTED_ONION_IMAGE", "harvested_produce", "Onion", "HARVESTED_PRODUCE"),
    ],
)
def test_local_images_relevance_and_classifier_bypass(
    path_variable: str, image_type: str, crop: str | None, message_code: str, monkeypatch
) -> None:
    path = os.getenv(path_variable)
    if not path:
        pytest.skip(f"Set {path_variable} to run this local-image regression test.")
    model_root = os.getenv("CROPVISION_MODEL_ROOT")
    if not model_root:
        pytest.skip("Set CROPVISION_MODEL_ROOT to the pinned local checkpoints.")

    image = validate_image(Path(path).read_bytes(), "image/jpeg")
    engine = get_engine()
    calls = []

    def forbidden_classifier(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("The disease classifier must not run for a non-living image.")

    monkeypatch.setattr(engine.classifier, "classify", forbidden_classifier)
    result = engine.predict(image, crop_hint="Onion")
    assert result.imageType == image_type
    assert result.crop == crop
    assert result.assessment == "not_applicable"
    assert result.condition is None
    assert result.confidence is None
    assert result.messageCode == message_code
    assert calls == []
