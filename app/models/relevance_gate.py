"""Conservative, image-based relevance and crop-type screening with local SigLIP."""

from __future__ import annotations

from collections import defaultdict

from PIL import Image

from app.config import (
    CROP_MIN_MARGIN,
    CROP_MIN_SHARE,
    GATE_MIN_MARGIN,
    GATE_MIN_SHARE,
    GATE_MODEL_ID,
)
from app.schemas import GateDecision

# English prompts match the model card's training setup. Equal prompt counts prevent
# a category winning just because it has more candidate text.
RELEVANCE_PROMPTS = {
    "non_crop": [
        "A screenshot of a webpage with text, tables, and interface elements.",
        "An anime illustration or poster, not a photograph of a plant.",
        "A photograph of a person, animal, vehicle, or building.",
        "A document, logo, chart, or meme without an agricultural subject.",
    ],
    "harvested_produce": [
        "A photograph of harvested onion bulbs separated from their plant.",
        "A photograph of harvested tomatoes or potatoes for sale.",
        "A pile of harvested vegetables on a market table.",
        "A close-up product photograph of picked agricultural produce.",
    ],
    "living_crop": [
        "A photograph of green leaves attached to a living agricultural crop plant.",
        "A close-up of a diseased leaf growing on a tomato plant.",
        "An onion crop with leaves growing in soil.",
        "A living crop plant growing in a farm field.",
    ],
    "crop_related_unclear": [
        "An extremely blurry photo that may contain an agricultural plant.",
        "A dark or distant photo of a possible crop field.",
        "An unclear close-up of a possible plant or vegetable.",
        "An obscured photograph of an agricultural subject.",
    ],
}

CROP_PROMPTS = {
    "Onion": "A photograph of onions or an onion crop plant.",
    "Tomato": "A photograph of tomatoes or a tomato crop plant.",
    "Potato": "A photograph of potatoes or a potato crop plant.",
    "Soybean": "A photograph of soybeans or a soybean crop plant.",
}


def _relative_shares(group_logits: dict[str, float]) -> dict[str, float]:
    import torch

    names = list(group_logits)
    values = torch.tensor([group_logits[name] for name in names], dtype=torch.float32)
    shares = values.softmax(dim=0).tolist()
    return dict(zip(names, shares, strict=True))


def conservative_choice(
    group_logits: dict[str, float], *, min_share: float, min_margin: float
) -> str | None:
    if not group_logits:
        return None
    shares = sorted(_relative_shares(group_logits).items(), key=lambda item: item[1], reverse=True)
    if shares[0][1] < min_share or shares[0][1] - shares[1][1] < min_margin:
        return None
    return shares[0][0]


class SiglipRelevanceGate:
    model_id = GATE_MODEL_ID

    def __init__(self, model_path: str) -> None:
        from transformers import AutoModelForZeroShotImageClassification, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForZeroShotImageClassification.from_pretrained(
            model_path, local_files_only=True
        ).eval()

    def _logits(self, image: Image.Image, prompts: list[str]) -> list[float]:
        import torch

        # SigLIP was trained on fixed-length text; dynamic batch padding changes
        # the score of a prompt depending on the other prompts in its batch.
        inputs = self.processor(
            text=prompts, images=image, padding="max_length", truncation=True, return_tensors="pt"
        )
        with torch.inference_mode():
            scores = self.model(**inputs).logits_per_image[0]
        return [float(value) for value in scores]

    def screen(self, image: Image.Image, *, quality_weak: bool = False) -> GateDecision:
        prompts = [prompt for group in RELEVANCE_PROMPTS.values() for prompt in group]
        values = self._logits(image, prompts)
        grouped: dict[str, list[float]] = defaultdict(list)
        for group, value in zip(
            (group for group, items in RELEVANCE_PROMPTS.items() for _ in items),
            values,
            strict=True,
        ):
            grouped[group].append(value)
        group_logits = {group: sum(scores) / len(scores) for group, scores in grouped.items()}
        chosen = conservative_choice(
            group_logits, min_share=GATE_MIN_SHARE, min_margin=GATE_MIN_MARGIN
        )
        if chosen is None or chosen == "crop_related_unclear":
            return GateDecision(image_type="crop_related_unclear", weak=True)
        if chosen == "non_crop":
            return GateDecision(image_type="non_crop")
        if quality_weak:
            return GateDecision(image_type="crop_related_unclear", weak=True)

        crop_logits = dict(
            zip(CROP_PROMPTS, self._logits(image, list(CROP_PROMPTS.values())), strict=True)
        )
        crop = conservative_choice(
            crop_logits, min_share=CROP_MIN_SHARE, min_margin=CROP_MIN_MARGIN
        )
        return GateDecision(image_type=chosen, crop=crop, weak=crop is None)
