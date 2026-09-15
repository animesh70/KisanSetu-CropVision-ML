"""Visual gate owns relevance; leaf classification is gated behind living-crop evidence."""

from __future__ import annotations

from typing import Protocol

from PIL import Image

from app.schemas import DiseaseDecision, GateDecision, ModelIdentity, Prediction
from app.services.image_validation import ValidatedImage


class Gate(Protocol):
    model_id: str

    def screen(self, image: Image.Image, *, quality_weak: bool = False) -> GateDecision: ...


class Classifier(Protocol):
    model_id: str
    supported_crops: set[str]

    def classify(self, image: Image.Image, crop: str) -> DiseaseDecision: ...


class InferenceEngine:
    def __init__(self, gate: Gate, classifier: Classifier) -> None:
        self.gate = gate
        self.classifier = classifier

    def predict(self, validated: ValidatedImage, crop_hint: str | None = None) -> Prediction:
        # crop_hint is deliberately not used to assign or overwrite any visual result.
        # It remains optional request metadata for a future, validated crop-specific gate.
        _ = crop_hint
        decision = self.gate.screen(validated.rgb, quality_weak=validated.quality_weak)
        identity = ModelIdentity(gate=self.gate.model_id)

        if decision.image_type == "non_crop":
            return Prediction(
                imageType="non_crop",
                crop=None,
                assessment="not_applicable",
                condition=None,
                messageCode="NON_CROP",
                model=identity,
            )
        if decision.image_type == "crop_related_unclear":
            return Prediction(
                imageType="crop_related_unclear",
                crop=None,
                assessment="condition_unclear",
                condition=None,
                messageCode="UNCLEAR",
                model=identity,
            )
        if decision.image_type == "harvested_produce":
            return Prediction(
                imageType="harvested_produce",
                crop=decision.crop,
                assessment="not_applicable",
                condition=None,
                messageCode="HARVESTED_PRODUCE",
                model=identity,
            )
        if decision.crop is None:
            return Prediction(
                imageType="living_crop",
                crop=None,
                assessment="condition_unclear",
                condition=None,
                messageCode="UNCLEAR",
                model=identity,
            )
        if decision.crop == "Onion" or decision.crop not in self.classifier.supported_crops:
            return Prediction(
                imageType="living_crop",
                crop=decision.crop,
                assessment="condition_unclear",
                condition=None,
                messageCode="UNSUPPORTED_CROP",
                model=identity,
            )

        disease = self.classifier.classify(validated.rgb, decision.crop)
        identity.classifier = disease.model_id
        if disease.weak:
            return Prediction(
                imageType="living_crop",
                crop=decision.crop,
                assessment="condition_unclear",
                condition=None,
                messageCode="UNCLEAR",
                model=identity,
            )
        if disease.healthy:
            return Prediction(
                imageType="living_crop",
                crop=decision.crop,
                assessment="healthy",
                condition=None,
                messageCode="HEALTHY",
                model=identity,
            )
        return Prediction(
            imageType="living_crop",
            crop=decision.crop,
            assessment="possibly_diseased",
            condition=disease.condition,
            messageCode="POSSIBLE_DISEASE",
            model=identity,
        )
