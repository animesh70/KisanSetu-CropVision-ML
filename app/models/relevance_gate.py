"""Hierarchical, image-based crop relevance screening with local SigLIP."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from PIL import Image

from app.config import (
    AGRICULTURE_MIN_MARGIN,
    AGRICULTURE_MIN_SHARE,
    CROP_MIN_MARGIN,
    CROP_MIN_SHARE,
    GATE_MODEL_ID,
    SCREENABILITY_MIN_MARGIN,
    SCREENABILITY_MIN_SHARE,
    STATE_MIN_MARGIN,
    STATE_MIN_SHARE,
)
from app.schemas import GateDecision

# SigLIP works best when each stage compares semantically parallel concepts.
# "Unclear" is deliberately not a prompt class; it is produced by abstention.
AGRICULTURE_PROMPTS = {
    "agricultural": [
        "A photograph whose main subject is an agricultural crop, fruit, vegetable, or farm plant.",
        "A photograph of fresh harvested agricultural produce such as vegetables or fruits.",
        "A photograph of a living crop plant growing in soil or in a farm field.",
        "A close-up photograph of crop leaves, stems, fruits, bulbs, pods, or vegetables.",
        "A studio or market photograph whose main subject is fresh crop produce.",
        "A photograph of agricultural produce even if a watermark or text overlay is present.",
    ],
    "non_crop": [
        "A screenshot of a website, software interface, table, document, or presentation.",
        "An anime, cartoon, movie poster, entertainment illustration, or digital artwork.",
        "A portrait or fashion photograph whose main subject is a person.",
        "A photograph of a vehicle, building, room, or object unrelated to agriculture.",
        "A logo, meme, chart, advertisement, or graphic design without crop produce "
        "as the subject.",
        "A non-agricultural image where crops, fruits, vegetables, or farm plants are "
        "not the subject.",
    ],
}

STATE_PROMPTS = {
    "harvested_produce": [
        "Harvested vegetables or fruits that have already been picked from the growing plant.",
        "Fresh produce displayed after harvest on a table, basket, crate, market, or "
        "studio background.",
        "Onion bulbs, potatoes, tomatoes, peppers, or other produce removed from soil or plant.",
        "Tomatoes or other fruits attached only to a short cut or detached stem after harvest.",
        "A product photograph of picked agricultural produce, including produce on a "
        "white background.",
        "Cut or whole harvested vegetables prepared for sale, storage, transport, or eating.",
    ],
    "living_crop": [
        "A crop plant actively growing in soil, a field, greenhouse, or garden.",
        "Fruit still attached to an intact living agricultural plant with growing leaves "
        "and stems.",
        "Crop leaves and stems connected to a plant that is visibly growing from soil.",
        "A living farm crop photographed in its growing environment.",
        "A close-up of living crop leaves or stems still attached to the growing plant.",
        "Agricultural vegetation forming part of an intact living crop plant.",
    ],
}

CROP_PROMPTS = {
    "Onion": [
        "Harvested onion bulbs or red onions.",
        "An onion crop plant with narrow tubular leaves.",
        "Onions or an onion plant as the main agricultural subject.",
    ],
    "Tomato": [
        "Red or green tomato fruits as the main agricultural subject.",
        "Harvested tomatoes or cherry tomatoes.",
        "A tomato crop plant with tomato fruits, stems, and leaves.",
    ],
    "Potato": [
        "Harvested potato tubers.",
        "A potato crop plant growing in soil.",
        "Potatoes or a potato plant as the main agricultural subject.",
    ],
    "Soybean": [
        "Soybean pods or harvested soybeans.",
        "A soybean crop plant growing in a field.",
        "Soybean leaves and pods as the main agricultural subject.",
    ],
}

SCREENABILITY_PROMPTS = {
    "screenable": [
        "A clear close-up of crop leaves suitable for plant disease screening.",
        "Affected living plant tissue with visible spots, lesions, discoloration, or damage.",
        "A close photograph of crop leaves or stems where disease symptoms could be inspected.",
        "Living crop foliage shown large and clearly enough to inspect leaf health.",
        "A close-up agricultural image with useful symptomatic leaf or stem detail.",
    ],
    "not_screenable": [
        "A crop photograph mainly showing fruit rather than detailed leaves or disease symptoms.",
        "A whole crop plant or field view where leaf details are too small for disease screening.",
        "A living plant photograph with no clear close-up of affected leaf or stem tissue.",
        "A crop image where disease symptoms cannot be inspected reliably from visible "
        "plant detail.",
        "A distant, obstructed, fruit-dominant, or otherwise unsuitable image for leaf "
        "disease screening.",
    ],
}


def _relative_shares(group_logits: dict[str, float]) -> dict[str, float]:
    import torch

    names = list(group_logits)
    values = torch.tensor([group_logits[name] for name in names], dtype=torch.float32)
    shares = values.softmax(dim=0).tolist()
    return dict(zip(names, shares, strict=True))


def _ranked_shares(group_logits: dict[str, float]) -> list[tuple[str, float]]:
    return sorted(_relative_shares(group_logits).items(), key=lambda item: item[1], reverse=True)


def conservative_choice(
    group_logits: dict[str, float], *, min_share: float, min_margin: float
) -> str | None:
    if len(group_logits) < 2:
        return None
    shares = _ranked_shares(group_logits)
    if shares[0][1] < min_share or shares[0][1] - shares[1][1] < min_margin:
        return None
    return shares[0][0]


def aggregate_prompt_logits(scores: list[float], *, top_k: int = 2) -> float:
    """Robustly combine prompt logits without letting weak paraphrases erase a strong match.

    We average the strongest two prompt matches (or all available prompts when fewer than two
    exist). This still requires more than one prompt to support a category while avoiding the
    dilution caused by averaging every broad paraphrase equally.
    """

    if not scores:
        raise ValueError("At least one prompt score is required.")
    count = min(max(top_k, 1), len(scores))
    strongest = sorted(scores, reverse=True)[:count]
    return sum(strongest) / count


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

    def _evaluate_stage(
        self,
        image: Image.Image,
        prompt_groups: dict[str, list[str]],
        *,
        min_share: float,
        min_margin: float,
    ) -> dict[str, Any]:
        prompts = [prompt for group in prompt_groups.values() for prompt in group]
        values = self._logits(image, prompts)
        grouped: dict[str, list[float]] = defaultdict(list)
        labels = (group for group, items in prompt_groups.items() for _ in items)
        for group, value in zip(labels, values, strict=True):
            grouped[group].append(value)

        aggregates = {
            group: aggregate_prompt_logits(scores) for group, scores in grouped.items()
        }
        shares = _relative_shares(aggregates)
        ranked = sorted(shares.items(), key=lambda item: item[1], reverse=True)
        choice = conservative_choice(
            aggregates,
            min_share=min_share,
            min_margin=min_margin,
        )
        return {
            "raw": dict(grouped),
            "aggregates": aggregates,
            "shares": shares,
            "winner": ranked[0][0],
            "margin": ranked[0][1] - ranked[1][1],
            "choice": choice,
        }

    def _screen_with_diagnostics(
        self, image: Image.Image, *, quality_weak: bool = False
    ) -> tuple[GateDecision, dict[str, Any]]:
        diagnostics: dict[str, Any] = {"quality_weak": quality_weak}

        relevance = self._evaluate_stage(
            image,
            AGRICULTURE_PROMPTS,
            min_share=AGRICULTURE_MIN_SHARE,
            min_margin=AGRICULTURE_MIN_MARGIN,
        )
        diagnostics["agriculture"] = relevance
        if relevance["choice"] is None:
            return GateDecision(image_type="crop_related_unclear", weak=True), diagnostics
        if relevance["choice"] == "non_crop":
            return GateDecision(image_type="non_crop"), diagnostics

        state = self._evaluate_stage(
            image,
            STATE_PROMPTS,
            min_share=STATE_MIN_SHARE,
            min_margin=STATE_MIN_MARGIN,
        )
        diagnostics["state"] = state
        if state["choice"] is None:
            return GateDecision(image_type="crop_related_unclear", weak=True), diagnostics

        crop_stage = self._evaluate_stage(
            image,
            CROP_PROMPTS,
            min_share=CROP_MIN_SHARE,
            min_margin=CROP_MIN_MARGIN,
        )
        diagnostics["crop"] = crop_stage
        crop = crop_stage["choice"]

        if state["choice"] == "harvested_produce":
            return (
                GateDecision(
                    image_type="harvested_produce",
                    crop=crop,
                    weak=crop is None,
                    screenable=None,
                ),
                diagnostics,
            )

        screenability = self._evaluate_stage(
            image,
            SCREENABILITY_PROMPTS,
            min_share=SCREENABILITY_MIN_SHARE,
            min_margin=SCREENABILITY_MIN_MARGIN,
        )
        diagnostics["screenability"] = screenability
        screenable = screenability["choice"] == "screenable"
        if screenability["choice"] is None:
            screenable = False
            diagnostics["screenability_abstained"] = True
        if quality_weak:
            # Image-quality warnings do not erase strong semantic evidence. They only prevent
            # a leaf-disease diagnosis when the visual detail is not trustworthy enough.
            screenable = False
            diagnostics["quality_forced_unscreenable"] = True

        return (
            GateDecision(
                image_type="living_crop",
                crop=crop,
                weak=crop is None,
                screenable=screenable,
            ),
            diagnostics,
        )

    def screen(self, image: Image.Image, *, quality_weak: bool = False) -> GateDecision:
        decision, _ = self._screen_with_diagnostics(image, quality_weak=quality_weak)
        return decision

    def diagnose(self, image: Image.Image, *, quality_weak: bool = False) -> dict[str, Any]:
        decision, diagnostics = self._screen_with_diagnostics(image, quality_weak=quality_weak)
        diagnostics["decision"] = decision.model_dump()
        return diagnostics
