"""Language-neutral public API contract."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

ImageType = Literal["non_crop", "harvested_produce", "living_crop", "crop_related_unclear"]
Assessment = Literal["healthy", "possibly_diseased", "condition_unclear", "not_applicable"]
MessageCode = Literal[
    "NON_CROP",
    "HARVESTED_PRODUCE",
    "HEALTHY",
    "POSSIBLE_DISEASE",
    "UNCLEAR",
    "UNSUPPORTED_CROP",
]


class ModelIdentity(BaseModel):
    gate: str
    classifier: str | None = None


class Prediction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    imageType: ImageType
    crop: str | None = None
    assessment: Assessment
    condition: str | None = None
    confidence: None = None
    messageCode: MessageCode
    model: ModelIdentity


class GateDecision(BaseModel):
    image_type: ImageType
    crop: str | None = None
    weak: bool = False


class DiseaseDecision(BaseModel):
    crop: str | None = None
    condition: str | None = None
    healthy: bool = False
    weak: bool = True
    model_id: str | None = None
