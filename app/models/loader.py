"""Fetch pinned checkpoint files once; reuse both model objects in a warm process."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from app.config import (
    DISEASE_MODEL_ID,
    DISEASE_REVISION,
    GATE_MODEL_ID,
    GATE_REVISION,
    model_cache_root,
)
from app.models.disease_classifier import MobileNetLeafClassifier
from app.models.relevance_gate import SiglipRelevanceGate
from app.services.inference import InferenceEngine


def checkpoint_dirs() -> tuple[str, str]:
    from huggingface_hub import snapshot_download

    root = model_cache_root()
    if root:
        gate = Path(root) / "gate"
        disease = Path(root) / "disease"
        if not (gate / "model.safetensors").is_file() or not (
            disease / "newplant_model_final.pth"
        ).is_file():
            raise RuntimeError("Pinned model files are not present in CROPVISION_MODEL_ROOT.")
        return str(gate), str(disease)

    gate = snapshot_download(
        repo_id=GATE_MODEL_ID,
        revision=GATE_REVISION,
        allow_patterns=["*.json", "*.txt", "*.model", "*.safetensors"],
    )
    disease = snapshot_download(
        repo_id=DISEASE_MODEL_ID,
        revision=DISEASE_REVISION,
        allow_patterns=["labels.txt", "newplant_model_final.pth"],
    )
    return gate, disease


@lru_cache(maxsize=1)
def get_engine() -> InferenceEngine:
    gate_path, disease_path = checkpoint_dirs()
    return InferenceEngine(
        SiglipRelevanceGate(gate_path), MobileNetLeafClassifier(disease_path)
    )
