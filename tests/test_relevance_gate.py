from PIL import Image

from app.models.relevance_gate import SiglipRelevanceGate, conservative_choice


def test_weak_or_ambiguous_logits_abstain():
    assert conservative_choice(
        {"non_crop": 0.0, "living_crop": 0.0},
        min_share=0.46,
        min_margin=0.12,
    ) is None


def test_relative_non_crop_visual_evidence_selects_non_crop():
    assert conservative_choice(
        {"non_crop": 6.0, "living_crop": 0.0},
        min_share=0.46,
        min_margin=0.12,
    ) == "non_crop"


def test_crop_hint_is_not_an_input_to_visual_gate():
    assert "cropHint" not in SiglipRelevanceGate.screen.__code__.co_varnames


def test_low_quality_crop_related_photo_abstains():
    gate = SiglipRelevanceGate.__new__(SiglipRelevanceGate)
    gate._logits = lambda image, prompts: (
        [0.0] * 4 + [1.0] * 4 + [9.0] * 4 + [0.0] * 4
        if len(prompts) == 16
        else [1.0, 8.0, 0.0, 0.0]
    )
    decision = gate.screen(Image.new("RGB", (256, 256)), quality_weak=True)
    assert decision.image_type == "crop_related_unclear"
