"""FastAPI boundary: authentication and upload validation precede local inference."""

from __future__ import annotations

import hmac
from collections.abc import Callable
from typing import Annotated

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile

from app.config import MAX_IMAGE_BYTES, api_key
from app.models.loader import get_engine
from app.schemas import Prediction
from app.services.image_validation import InvalidImage, validate_image
from app.services.inference import InferenceEngine


def create_app(engine_factory: Callable[[], InferenceEngine] = get_engine) -> FastAPI:
    service = FastAPI(title="KisanSetu CropVision ML", version="0.1.0")

    @service.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "kisansetu-cropvision-ml"}

    @service.post("/predict", response_model=Prediction)
    async def predict(
        image: Annotated[UploadFile, File()],
        crop_hint: Annotated[str | None, Form(alias="cropHint")] = None,
        supplied_key: Annotated[str | None, Header(alias="X-KisanSetu-Key")] = None,
    ) -> Prediction:
        configured = api_key()
        if not configured:
            raise HTTPException(status_code=503, detail="Service authentication is not configured.")
        if not supplied_key or not hmac.compare_digest(supplied_key, configured):
            raise HTTPException(status_code=401, detail="Invalid service key.")
        payload = await image.read(MAX_IMAGE_BYTES + 1)
        if len(payload) > MAX_IMAGE_BYTES:
            raise HTTPException(status_code=413, detail="Image exceeds the 6 MB limit.")
        try:
            validated = validate_image(payload, image.content_type or "")
        except InvalidImage as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        try:
            # Neither input images nor result objects are cached between requests.
            return engine_factory().predict(validated, crop_hint)
        except Exception as exc:
            # Model identifiers, network errors, and stack traces remain server-side.
            raise HTTPException(
                status_code=503, detail="Crop screening is temporarily unavailable."
            ) from exc

    return service


app = create_app()
