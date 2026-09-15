from types import SimpleNamespace

import torch
from PIL import Image

from app.models.relevance_gate import (
    AGRICULTURE_PROMPTS,
    CROP_PROMPTS,
    SCREENABILITY_PROMPTS,
    STATE_PROMPTS,
    SiglipRelevanceGate,
    aggregate_prompt_logits,
    conservative_choice,
)


def _flat(groups):
    return [prompt for prompts in groups.values() for prompt in prompts]


def test_weak_or_ambiguous_logits_abstain():
    assert conservative_choice(
        {"non_crop": 0.0, "agricultural": 0.0},
        min_share=0.60,
        min_margin=0.18,
    ) is None


def test_relative_non_crop_visual_evidence_selects_non_crop():
    assert conservative_choice(
        {"non_crop": 6.0, "agricultural": 0.0},
        min_share=0.60,
        min_margin=0.18,
    ) == "non_crop"


def test_prompt_aggregation_uses_more_than_one_supporting_prompt():
    # A single outlier cannot fully determine the category because the top two are averaged.
    assert aggregate_prompt_logits([10.0, 2.0, 0.0, -1.0]) == 6.0


def test_crop_hint_is_not_an_input_to_visual_gate():
    assert "cropHint" not in SiglipRelevanceGate.screen.__code__.co_varnames


def test_siglip_text_uses_fixed_training_length_not_batch_padding():
    gate = SiglipRelevanceGate.__new__(SiglipRelevanceGate)
    seen = {}

    def processor(**kwargs):
        seen.update(kwargs)
        return {"input_ids": torch.zeros((2, 64), dtype=torch.int64)}

    gate.processor = processor
    gate.model = lambda **kwargs: SimpleNamespace(logits_per_image=torch.tensor([[1.0, 2.0]]))
    assert gate._logits(Image.new("RGB", (224, 224)), ["a crop", "a poster"]) == [1.0, 2.0]
    assert seen["padding"] == "max_length"
    assert seen["truncation"] is True


def test_unclear_is_abstention_not_a_prompt_category():
    assert "crop_related_unclear" not in AGRICULTURE_PROMPTS
    assert "crop_related_unclear" not in STATE_PROMPTS


def _scripted_gate(stage_values):
    gate = SiglipRelevanceGate.__new__(SiglipRelevanceGate)

    def logits(image, prompts):
        del image
        if prompts == _flat(AGRICULTURE_PROMPTS):
            return stage_values["agriculture"]
        if prompts == _flat(STATE_PROMPTS):
            return stage_values["state"]
        if prompts == _flat(CROP_PROMPTS):
            return stage_values["crop"]
        if prompts == _flat(SCREENABILITY_PROMPTS):
            return stage_values["screenability"]
        raise AssertionError("Unexpected prompt batch")

    gate._logits = logits
    return gate


def _scores_for(groups, winner, high=8.0, low=0.0):
    values = []
    for group, prompts in groups.items():
        values.extend([high if group == winner else low] * len(prompts))
    return values


def test_confident_non_crop_stops_before_state_and_crop_stages():
    gate = SiglipRelevanceGate.__new__(SiglipRelevanceGate)
    calls = []

    def logits(image, prompts):
        del image
        calls.append(prompts)
        return _scores_for(AGRICULTURE_PROMPTS, "non_crop")

    gate._logits = logits
    decision = gate.screen(Image.new("RGB", (256, 256)))
    assert decision.image_type == "non_crop"
    assert len(calls) == 1


def test_harvested_type_survives_failed_crop_naming():
    gate = _scripted_gate(
        {
            "agriculture": _scores_for(AGRICULTURE_PROMPTS, "agricultural"),
            "state": _scores_for(STATE_PROMPTS, "harvested_produce"),
            "crop": [0.0] * len(_flat(CROP_PROMPTS)),
            "screenability": _scores_for(SCREENABILITY_PROMPTS, "not_screenable"),
        }
    )
    decision = gate.screen(Image.new("RGB", (256, 256)))
    assert decision.image_type == "harvested_produce"
    assert decision.crop is None


def test_quality_warning_does_not_erase_strong_harvested_semantics():
    gate = _scripted_gate(
        {
            "agriculture": _scores_for(AGRICULTURE_PROMPTS, "agricultural"),
            "state": _scores_for(STATE_PROMPTS, "harvested_produce"),
            "crop": _scores_for(CROP_PROMPTS, "Tomato"),
            "screenability": _scores_for(SCREENABILITY_PROMPTS, "not_screenable"),
        }
    )
    decision = gate.screen(Image.new("RGB", (256, 256)), quality_weak=True)
    assert decision.image_type == "harvested_produce"
    assert decision.crop == "Tomato"


def test_living_crop_can_be_retained_but_marked_not_screenable():
    gate = _scripted_gate(
        {
            "agriculture": _scores_for(AGRICULTURE_PROMPTS, "agricultural"),
            "state": _scores_for(STATE_PROMPTS, "living_crop"),
            "crop": _scores_for(CROP_PROMPTS, "Tomato"),
            "screenability": _scores_for(SCREENABILITY_PROMPTS, "not_screenable"),
        }
    )
    decision = gate.screen(Image.new("RGB", (256, 256)))
    assert decision.image_type == "living_crop"
    assert decision.crop == "Tomato"
    assert decision.screenable is False


def test_living_crop_can_be_screenable():
    gate = _scripted_gate(
        {
            "agriculture": _scores_for(AGRICULTURE_PROMPTS, "agricultural"),
            "state": _scores_for(STATE_PROMPTS, "living_crop"),
            "crop": _scores_for(CROP_PROMPTS, "Soybean"),
            "screenability": _scores_for(SCREENABILITY_PROMPTS, "screenable"),
        }
    )
    decision = gate.screen(Image.new("RGB", (256, 256)))
    assert decision.image_type == "living_crop"
    assert decision.crop == "Soybean"
    assert decision.screenable is True
