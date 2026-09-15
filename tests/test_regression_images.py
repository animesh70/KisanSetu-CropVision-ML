"""Opt-in tests use caller-provided image paths; no private photos are committed."""

from __future__ import annotations

import mimetypes
import os
from pathlib import Path

import pytest

from app.models.loader import get_engine
from app.services.image_validation import validate_image


@pytest.mark.parametrize(
    ("path_variable", "image_type", "crop", "message_code", "classifier_must_be_unused"),
    [
        ("CROPVISION_ANIME_IMAGE", "non_crop", None, "NON_CROP", True),
        ("CROPVISION_PERSON_IMAGE", "non_crop", None, "NON_CROP", True),
        (
            "CROPVISION_HARVESTED_ONION_IMAGE",
            "harvested_produce",
            "Onion",
            "HARVESTED_PRODUCE",
            True,
        ),
        (
            "CROPVISION_HARVESTED_TOMATO_IMAGE",
            "harvested_produce",
            "Tomato",
            "HARVESTED_PRODUCE",
            True,
        ),
    ],
)
def test_local_images_relevance_and_classifier_bypass(
    path_variable: str,
    image_type: str,
    crop: str | None,
    message_code: str,
    classifier_must_be_unused: bool,
    monkeypatch,
) -> None:
    path = os.getenv(path_variable)
    if not path:
        pytest.skip(f"Set {path_variable} to run this local-image regression test.")
    model_root = os.getenv("CROPVISION_MODEL_ROOT")
    if not model_root:
        pytest.skip("Set CROPVISION_MODEL_ROOT to the pinned local checkpoints.")

    mime, _ = mimetypes.guess_type(path)
    image = validate_image(Path(path).read_bytes(), mime or "")
    engine = get_engine()
    calls = []

    if classifier_must_be_unused:
        original_classifier = engine.classifier.classify

        def forbidden_classifier(*args, **kwargs):
            calls.append((args, kwargs))
            raise AssertionError("The disease classifier must not run for this image.")

        monkeypatch.setattr(engine.classifier, "classify", forbidden_classifier)
        try:
            result = engine.predict(image, crop_hint="Onion")
        finally:
            monkeypatch.setattr(engine.classifier, "classify", original_classifier)
    else:
        result = engine.predict(image, crop_hint="Onion")

    assert result.imageType == image_type
    assert result.crop == crop
    assert result.condition is None
    assert result.confidence is None
    assert result.messageCode == message_code
    assert calls == []


def test_local_living_tomato_is_living_and_does_not_require_forced_diagnosis(monkeypatch):
    path = os.getenv("CROPVISION_LIVING_TOMATO_IMAGE")
    if not path:
        pytest.skip("Set CROPVISION_LIVING_TOMATO_IMAGE to run this regression test.")
    if not os.getenv("CROPVISION_MODEL_ROOT"):
        pytest.skip("Set CROPVISION_MODEL_ROOT to the pinned local checkpoints.")

    mime, _ = mimetypes.guess_type(path)
    image = validate_image(Path(path).read_bytes(), mime or "")
    engine = get_engine()
    decision = engine.gate.screen(image.rgb, quality_weak=image.quality_weak)
    assert decision.image_type == "living_crop"
    assert decision.crop == "Tomato"

    # If the gate says the image is not disease-screenable, the leaf classifier must be bypassed.
    if decision.screenable is not True:
        calls = []

        def forbidden_classifier(*args, **kwargs):
            calls.append((args, kwargs))
            raise AssertionError("Non-screenable living crops must not reach the leaf classifier.")

        monkeypatch.setattr(engine.classifier, "classify", forbidden_classifier)
        result = engine.predict(image)
        assert result.assessment == "condition_unclear"
        assert result.condition is None
        assert calls == []
