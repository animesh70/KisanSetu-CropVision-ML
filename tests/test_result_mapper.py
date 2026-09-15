from PIL import Image

from app.models.disease_classifier import map_logits, parse_label
from app.schemas import DiseaseDecision, GateDecision
from app.services.image_validation import ValidatedImage
from app.services.inference import InferenceEngine


class StubGate:
    model_id = "test-local-gate"

    def __init__(self, decisions):
        self.decisions = iter(decisions)

    def screen(self, image, *, quality_weak=False):
        return next(self.decisions)


class StubClassifier:
    model_id = "test-local-leaf"
    supported_crops = {"Tomato", "Potato"}

    def __init__(self, decision=None):
        self.calls = 0
        self.decision = decision or DiseaseDecision(
            crop="Tomato", condition="Early blight", weak=False, model_id=self.model_id
        )

    def classify(self, image, crop):
        self.calls += 1
        return self.decision


def validated_image():
    return ValidatedImage(Image.new("RGB", (256, 256)), "image/png", False)


def test_non_crop_forces_all_non_applicable_fields_even_with_onion_hint():
    engine = InferenceEngine(StubGate([GateDecision(image_type="non_crop")]), StubClassifier())
    result = engine.predict(validated_image(), crop_hint="Onion")
    assert result.imageType == "non_crop"
    assert result.crop is None
    assert result.assessment == "not_applicable"
    assert result.condition is None
    assert result.confidence is None
    assert result.messageCode == "NON_CROP"


def test_harvested_onion_never_reaches_leaf_disease_classifier():
    classifier = StubClassifier()
    engine = InferenceEngine(
        StubGate([GateDecision(image_type="harvested_produce", crop="Onion")]), classifier
    )
    result = engine.predict(validated_image())
    assert result.imageType == "harvested_produce"
    assert result.crop == "Onion"
    assert result.assessment == "not_applicable"
    assert result.condition is None
    assert classifier.calls == 0


def test_living_supported_crop_reaches_disease_model_once():
    classifier = StubClassifier()
    engine = InferenceEngine(
        StubGate([GateDecision(image_type="living_crop", crop="Tomato")]), classifier
    )
    result = engine.predict(validated_image())
    assert classifier.calls == 1
    assert result.assessment == "possibly_diseased"
    assert result.condition == "Early blight"
    assert result.confidence is None


def test_living_onion_is_safely_unsupported_without_checkpoint():
    classifier = StubClassifier()
    engine = InferenceEngine(
        StubGate([GateDecision(image_type="living_crop", crop="Onion")]), classifier
    )
    result = engine.predict(validated_image())
    assert result.messageCode == "UNSUPPORTED_CROP"
    assert result.assessment == "condition_unclear"
    assert classifier.calls == 0


def test_weak_or_unclear_crop_cannot_be_diagnosed():
    classifier = StubClassifier()
    engine = InferenceEngine(
        StubGate([GateDecision(image_type="crop_related_unclear", weak=True)]), classifier
    )
    result = engine.predict(validated_image())
    assert result.condition is None
    assert result.confidence is None
    assert classifier.calls == 0


def test_healthy_plant_does_not_force_a_condition():
    classifier = StubClassifier(
        DiseaseDecision(crop="Tomato", healthy=True, weak=False, model_id="test-local-leaf")
    )
    engine = InferenceEngine(
        StubGate([GateDecision(image_type="living_crop", crop="Tomato")]), classifier
    )
    result = engine.predict(validated_image())
    assert result.assessment == "healthy"
    assert result.condition is None


def test_sequential_predictions_do_not_reuse_previous_results():
    engine = InferenceEngine(
        StubGate(
            [
                GateDecision(image_type="non_crop"),
                GateDecision(image_type="harvested_produce", crop="Onion"),
            ]
        ),
        StubClassifier(),
    )
    first = engine.predict(validated_image(), crop_hint="Onion")
    second = engine.predict(validated_image(), crop_hint="Onion")
    assert first.imageType == "non_crop"
    assert second.imageType == "harvested_produce"


def test_classifier_label_mapping_abstains_on_crop_mismatch():
    assert parse_label("Tomato___Early_blight") == ("Tomato", "Early blight")
    result = map_logits(
        ["Tomato___Early_blight", "Potato___Early_blight", "Tomato___healthy"],
        [9.0, 1.0, 0.0],
        "Potato",
    )
    assert result.weak
    assert result.condition is None
