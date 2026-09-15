import asyncio
import io

import httpx
import pytest
from PIL import Image

from app.api import create_app
from app.schemas import GateDecision
from app.services.inference import InferenceEngine
from tests.test_result_mapper import StubClassifier, StubGate


def png_bytes():
    output = io.BytesIO()
    Image.new("RGB", (256, 256), (20, 140, 50)).save(output, format="PNG")
    return output.getvalue()


def request(app, method, path, **kwargs):
    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            return await client.request(method, path, **kwargs)

    return asyncio.run(run())


@pytest.fixture(autouse=True)
def key(monkeypatch):
    monkeypatch.setenv("KISANSETU_ML_API_KEY", "test-secret")


def app_with(decisions):
    engine = InferenceEngine(StubGate(decisions), StubClassifier())
    return create_app(lambda: engine)


def test_health_is_public():
    response = request(app_with([]), "GET", "/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "kisansetu-cropvision-ml"}


def test_missing_or_wrong_api_key_is_rejected():
    app = app_with([])
    for headers in ({}, {"X-KisanSetu-Key": "wrong"}):
        response = request(
            app, "POST", "/predict", headers=headers,
            files={"image": ("crop.png", png_bytes(), "image/png")},
        )
        assert response.status_code == 401


def test_valid_key_and_non_crop_hinted_onion_returns_safe_result():
    app = app_with([GateDecision(image_type="non_crop")])
    response = request(
        app, "POST", "/predict",
        headers={"X-KisanSetu-Key": "test-secret"},
        files={"image": ("travel.png", png_bytes(), "image/png")},
        data={"cropHint": "Onion"},
    )
    assert response.status_code == 200
    result = response.json()
    assert result["imageType"] == "non_crop"
    assert result["crop"] is None
    assert result["condition"] is None
    assert result["confidence"] is None
    assert result["assessment"] == "not_applicable"


def test_missing_image_is_rejected():
    response = request(
        app_with([]), "POST", "/predict", headers={"X-KisanSetu-Key": "test-secret"}
    )
    assert response.status_code == 422


def test_invalid_image_bytes_are_rejected():
    response = request(
        app_with([]), "POST", "/predict",
        headers={"X-KisanSetu-Key": "test-secret"},
        files={"image": ("bad.png", b"garbage", "image/png")},
    )
    assert response.status_code == 400


def test_image_exceeding_six_megabytes_is_rejected_before_inference():
    response = request(
        app_with([]), "POST", "/predict",
        headers={"X-KisanSetu-Key": "test-secret"},
        files={"image": ("large.png", b"x" * (6 * 1024 * 1024 + 1), "image/png")},
    )
    assert response.status_code == 413


def test_missing_server_secret_fails_safely(monkeypatch):
    monkeypatch.delenv("KISANSETU_ML_API_KEY")
    response = request(
        app_with([]), "POST", "/predict",
        headers={"X-KisanSetu-Key": "test-secret"},
        files={"image": ("crop.png", png_bytes(), "image/png")},
    )
    assert response.status_code == 503


def test_sequential_api_requests_have_independent_results():
    app = app_with(
        [
            GateDecision(image_type="non_crop"),
            GateDecision(image_type="harvested_produce", crop="Onion"),
        ]
    )
    headers = {"X-KisanSetu-Key": "test-secret"}
    files = {"image": ("picture.png", png_bytes(), "image/png")}
    first = request(app, "POST", "/predict", headers=headers, files=files)
    second = request(app, "POST", "/predict", headers=headers, files=files)
    assert first.json()["imageType"] == "non_crop"
    assert second.json()["imageType"] == "harvested_produce"
