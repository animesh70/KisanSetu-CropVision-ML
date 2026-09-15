"""Pinned MobileNetV2 PlantVillage leaf classifier; no onion checkpoint is implied."""

from __future__ import annotations

import re
from pathlib import Path

from PIL import Image

from app.config import DISEASE_MIN_MARGIN, DISEASE_MIN_SHARE, DISEASE_MODEL_ID
from app.schemas import DiseaseDecision


def parse_label(label: str) -> tuple[str, str]:
    # The model's labels.txt is the authoritative ordered class list.
    parts = re.split(r"_{2,3}", label.strip(), maxsplit=1)
    if len(parts) != 2:
        return "", ""
    return parts[0].replace("_", " ").strip().title(), parts[1].replace("_", " ").strip()


def map_logits(labels: list[str], logits: list[float], crop: str) -> DiseaseDecision:
    import torch

    identity = DISEASE_MODEL_ID
    if crop.lower() == "onion":
        return DiseaseDecision(crop="Onion", weak=True, model_id=None)
    if not logits or len(logits) != len(labels):
        return DiseaseDecision(crop=crop, weak=True, model_id=identity)

    ranked = sorted(
        zip(labels, torch.tensor(logits).softmax(dim=0).tolist(), strict=True),
        key=lambda item: item[1],
        reverse=True,
    )
    top_crop, top_condition = parse_label(ranked[0][0])
    if (
        not top_crop
        or top_crop.lower() != crop.lower()
        or ranked[0][1] < DISEASE_MIN_SHARE
        or ranked[0][1] - ranked[1][1] < DISEASE_MIN_MARGIN
    ):
        return DiseaseDecision(crop=crop, weak=True, model_id=identity)
    if "healthy" in top_condition.lower():
        return DiseaseDecision(crop=crop, healthy=True, weak=False, model_id=identity)
    return DiseaseDecision(
        crop=crop, condition=top_condition, healthy=False, weak=False, model_id=identity
    )


class MobileNetLeafClassifier:
    model_id = DISEASE_MODEL_ID

    def __init__(self, checkpoint_dir: str) -> None:
        import torch
        from torchvision import models, transforms

        root = Path(checkpoint_dir)
        self.labels = [
            item.strip()
            for item in (root / "labels.txt").read_text(encoding="utf-8").splitlines()
            if item.strip()
        ]
        if len(self.labels) != 38:
            raise RuntimeError("The pinned disease checkpoint's class list is invalid.")
        self.supported_crops = {
            crop
            for label in self.labels
            if (crop := parse_label(label)[0]) and crop.lower() != "onion"
        }
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = torch.nn.Linear(
            model.classifier[1].in_features, len(self.labels)
        )
        # weights_only avoids executing arbitrary objects from a third-party .pth.
        weights = torch.load(
            root / "newplant_model_final.pth", map_location="cpu", weights_only=True
        )
        model.load_state_dict(weights)
        self.model = model.eval()
        self.preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def classify(self, image: Image.Image, crop: str) -> DiseaseDecision:
        import torch

        if crop.lower() == "onion" or crop not in self.supported_crops:
            return DiseaseDecision(crop=crop, weak=True, model_id=None)
        with torch.inference_mode():
            logits = self.model(self.preprocess(image).unsqueeze(0))[0].tolist()
        return map_logits(self.labels, logits, crop)
